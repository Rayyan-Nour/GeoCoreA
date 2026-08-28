"""
Pipeline orchestration: config in -> validated, provenance-tracked outputs.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import __version__
from .config import PipelineConfig
from .raster_io import Grid, load_aligned, load_dem, save_geotiff
from .features import (FeatureStack, csv_points, idw_raster, proximity_raster,
                       terrain_features)
from .sampling import build_samples, load_deposits_csv, spatial_holdout
from .validation import holdout_evaluation, spatial_cv, synthesize_verdict
from .model import save_model, train
from .grade import grade_context, prospectivity_classes
from .depth import euler_deconvolution, euler_depth_map, spectral_depth_map
from .projectdb import ProjectDB
from .report import write_report

log = logging.getLogger("geocore")


@dataclass
class PipelineResult:
    prob_map: np.ndarray
    uncertainty_map: np.ndarray
    grid: Grid
    cv_report: object
    holdout_metrics: Dict
    importances: List
    depth_spectral: Optional[np.ndarray] = None
    depth_euler: Optional[np.ndarray] = None
    n_euler_solutions: int = 0
    artifacts: Dict[str, str] = field(default_factory=dict)
    n_deposits_train: int = 0
    n_deposits_holdout: int = 0
    warnings: List[str] = field(default_factory=list)
    targets: List[Dict] = field(default_factory=list)


def run_pipeline(cfg: PipelineConfig,
                 progress: Optional[Callable[[int, str], None]] = None,
                 db: Optional[ProjectDB] = None,
                 project_name: str = "default") -> PipelineResult:
    """Run the full prospectivity analysis. Raises on invalid config."""
    def tick(pct: int, msg: str):
        log.info("[%3d%%] %s", pct, msg)
        if progress:
            progress(pct, msg)

    problems = cfg.validate()
    if problems:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(problems))

    spec = cfg.spec()
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    run_id = None
    if db is not None:
        proj = db.get_project(project_name)
        pid = proj["id"] if proj else db.create_project(project_name, cfg.commodity)
        run_id = db.start_run(pid, cfg.__dict__, __version__)

    try:
        # ---- 1. Grid & terrain ----
        tick(5, "Loading DEM")
        grid = load_dem(cfg.dem_path, cfg.crop_mode, cfg.analysis_max_px)
        fs = FeatureStack(grid)
        gh, gw = grid.shape
        px_x, px_y = grid.pixel_size_m()
        tick(8, f"Analysis grid {gw} x {gh} @ {px_x:.0f} m pixels")
        tick(12, "Terrain features")
        for name, layer in terrain_features(grid).items():
            fs.add(name, layer)

        # ---- 2. Evidence layers ----
        tick(20, "Evidence layers")
        magnetics_grid: Optional[Grid] = None
        for name, path in cfg.feature_rasters.items():
            if not path:
                continue
            g = load_aligned(path, grid)
            if g is None or g.mask.mean() < cfg.min_valid_fraction:
                warnings.append(f"raster '{name}' skipped: no/low overlap with AOI")
                continue
            fs.add(name, g.data)
            if name == cfg.magnetics_layer:
                magnetics_grid = g

        for name, path in cfg.feature_vectors.items():
            if not path:
                continue
            low = path.lower()
            if low.endswith(".shp"):
                try:
                    from .vector_io import shapefile_to_raster
                    fs.add(name, shapefile_to_raster(path, grid))
                except Exception as e:
                    warnings.append(f"vector '{name}' skipped: {e}")
                continue
            pts = csv_points(path) if low.endswith(".csv") else []
            if not pts:
                warnings.append(f"vector '{name}' skipped: no usable points")
                continue
            if any(abs(v - 1.0) > 1e-9 for _, _, v in pts):
                fs.add(name, idw_raster(pts, grid))
            else:
                fs.add(name, proximity_raster([(x, y) for x, y, _ in pts], grid))

        if len(fs.names) < 3:
            raise RuntimeError(
                f"Only {len(fs.names)} usable features - need >=3 for a "
                "meaningful model. Check evidence layer paths/overlap.")

        X_flat, valid_flat = fs.matrix()

        # ---- 3. Deposits ----
        tick(30, "Loading deposits")
        if not cfg.deposit_csv:
            raise RuntimeError("deposit_csv is required for training")
        try:
            if cfg.deposit_csv.lower().endswith(".shp"):
                from .sampling import load_deposits_shp
                deposits = load_deposits_shp(cfg.deposit_csv, grid,
                                             cfg.commodity)
            else:
                deposits = load_deposits_csv(cfg.deposit_csv, grid,
                                             cfg.commodity)
        except PermissionError as e:
            raise RuntimeError(
                f"Cannot read the deposits file - it appears to be open in "
                f"another program (Excel locks files while open). Close it "
                f"and run again. ({cfg.deposit_csv})") from e
        if len(deposits) < spec.min_deposits_for_training:
            raise RuntimeError(
                f"Found {len(deposits)} {spec.label} deposits in the AOI; "
                f"need >= {spec.min_deposits_for_training}. Expand the AOI, "
                "check commodity filters, or use transfer prediction with a "
                "model trained elsewhere (geocore.model.load_model).")

        train_deps, holdout_deps = spatial_holdout(
            deposits, grid.shape, cfg.holdout_fraction, cfg.random_seed)
        tick(35, f"{len(train_deps)} training / {len(holdout_deps)} holdout deposits")

        # ---- 4. Samples & spatial CV ----
        tick(40, "Building training samples")
        samples = build_samples(
            X_flat, valid_flat, grid, train_deps, deposits,
            buffer_m=spec.negative_buffer_m,
            negative_ratio=cfg.negative_ratio, seed=cfg.random_seed,
            feature_names=fs.names)

        tick(50, "Spatial block cross-validation")
        from .model import adaptive_rf
        cv = spatial_cv(adaptive_rf(int(samples.y.sum()), cfg.n_trees,
                                    cfg.random_seed),
                        samples, grid.shape, k=cfg.spatial_blocks,
                        seed=cfg.random_seed)

        # ---- 5. Train & predict ----
        tick(60, "Training calibrated model")
        model = train(samples, cfg.commodity, cfg.n_trees, cfg.random_seed)
        tick(70, "Predicting full map")
        prob_map, unc_map = model.predict_map(
            X_flat, valid_flat, grid.shape, chunk=1_000_000,
            progress=lambda f: tick(70 + 10 * f,
                                      f"Predicting full map ({f*100:.0f}%)"))

        if cfg.blend_system_prior > 0:
            warnings.append(
                f"probability blended with hand-crafted prior at weight "
                f"{cfg.blend_system_prior} - DISCLOSED, not silent")

        _slope = None
        if "slope" in fs.names:
            _slope = fs.layers[fs.names.index("slope")]
        holdout_metrics = holdout_evaluation(
            prob_map, holdout_deps, grid.mask, cfg.random_seed,
            dem=grid.data, slope=_slope)

        # ---- 6. Depth-to-source ----
        depth_spec = depth_euler = None
        n_sols = 0
        if cfg.estimate_depth and magnetics_grid is not None:
            tick(80, "Depth-to-source (spectral)")
            depth_spec = spectral_depth_map(magnetics_grid, cfg.depth_window_px)
            tick(85, "Depth-to-source (Euler deconvolution)")
            from .depth import multiscale_euler
            sols = multiscale_euler(
                magnetics_grid, structural_index=cfg.euler_structural_index)
            n_sols = len(sols)
            if sols:
                depth_euler = euler_depth_map(magnetics_grid, sols)
        elif cfg.estimate_depth:
            warnings.append(
                f"depth estimation skipped: no '{cfg.magnetics_layer}' raster")

        # ---- 7. Outputs ----
        tick(92, "Saving outputs")
        artifacts: Dict[str, str] = {}

        def save(kind: str, arr: np.ndarray, fname: str):
            p = str(out_dir / fname)
            save_geotiff(arr, p, grid)
            artifacts[kind] = p

        save("probability", prob_map, "geocore_probability.tif")
        save("uncertainty", unc_map, "geocore_uncertainty.tif")
        _cls_for_save = prospectivity_classes(prob_map)
        save("classes", _cls_for_save, "geocore_classes.tif")
        save("dem", grid.data, "geocore_dem.tif")
        if depth_spec is not None:
            save("depth_spectral", depth_spec, "geocore_depth_spectral_m.tif")
        if depth_euler is not None:
            save("depth_euler", depth_euler, "geocore_depth_euler_m.tif")

        # ---- 7b. Ranked target list ----
        from .targets import extract_targets, write_targets_csv
        cls_map = prospectivity_classes(prob_map)
        targets = extract_targets(
            prob_map, unc_map, cls_map, grid,
            X_flat=X_flat, valid_flat=valid_flat,
            feature_names=fs.names, importances=model.importances(),
            depth_spectral=depth_spec, depth_euler=depth_euler,
            deposits=deposits)
        tpath = write_targets_csv(out_dir / "geocore_targets.csv", targets,
                                  grade_context(spec).statement)
        artifacts["targets"] = str(tpath)

        mpath = str(out_dir / f"geocore_{cfg.commodity}_model.joblib")
        save_model(model, mpath)
        artifacts["model"] = mpath
        cfg.to_json(str(out_dir / "run_config.json"))
        artifacts["config"] = str(out_dir / "run_config.json")

        gc = grade_context(spec)
        run_verdict = synthesize_verdict(cv, holdout_metrics or {})
        rpath = str(out_dir / "geocore_validation_report.md")
        write_report(rpath, cfg, spec, cv, holdout_metrics,
                     model.importances(), gc, len(train_deps),
                     len(holdout_deps), n_sols, warnings,
                     targets=targets, verdict=run_verdict)
        artifacts["report"] = rpath

        from .report import write_executive_summary
        epath = str(out_dir / "geocore_executive_summary.md")
        write_executive_summary(epath, cfg, spec, cv, holdout_metrics,
                                gc, targets,
                                len(train_deps) + len(holdout_deps))
        artifacts["executive_summary"] = epath

        if db is not None and run_id is not None:
            db.log_metric(run_id, "cv_auc_mean", cv.auc_mean)
            db.log_metric(run_id, "cv_auc_std", cv.auc_std)
            if holdout_metrics:
                db.log_metric(run_id, "holdout_contrast_auc",
                              holdout_metrics["contrast_auc"])
            for k, p in artifacts.items():
                db.log_artifact(run_id, k, p)
            db.finish_run(run_id, "complete")

        tick(100, "Complete")
        return PipelineResult(
            prob_map=prob_map, uncertainty_map=unc_map, grid=grid,
            cv_report=cv, holdout_metrics=holdout_metrics,
            importances=model.importances(),
            depth_spectral=depth_spec, depth_euler=depth_euler,
            n_euler_solutions=n_sols, targets=targets, artifacts=artifacts,
            n_deposits_train=len(train_deps),
            n_deposits_holdout=len(holdout_deps), warnings=warnings)

    except Exception:
        if db is not None and run_id is not None:
            db.finish_run(run_id, "failed")
        raise
