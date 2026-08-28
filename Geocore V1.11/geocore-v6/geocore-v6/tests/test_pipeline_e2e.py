"""
End-to-end pipeline test on a synthetic world with planted ground truth.

The world: deposits occur where (high magnetics anomaly) AND (close to a
"fault") AND (moderate TPI). Magnetic sources are buried at a known depth.
A working engine must (a) rank planted-deposit areas above background under
honest spatial validation, (b) recover the planted source depth, and
(c) produce every promised artifact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
from scipy import ndimage

from synthetic import (deposits_to_csv, field_from_sources_at_depth,
                       make_grid, plant_deposits, synthetic_dem, write_tif)
from geocore.config import PipelineConfig
from geocore.pipeline import run_pipeline
from geocore.projectdb import ProjectDB


TRUE_SOURCE_DEPTH = 600.0
H = W = 220
PIXEL_M = 100.0


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Build the synthetic world once for all tests in this module."""
    root = tmp_path_factory.mktemp("world")
    rng = np.random.default_rng(7)
    grid = make_grid(H, W, pixel_m=PIXEL_M)

    dem = synthetic_dem(H, W, seed=1)
    write_tif(root / "dem.tif", dem, grid)

    # Magnetics: spectrum consistent with sources at TRUE_SOURCE_DEPTH,
    # plus a strong coherent anomaly belt where mineralization "is".
    mag = field_from_sources_at_depth(H, W, PIXEL_M, TRUE_SOURCE_DEPTH, seed=2)
    mag = (mag - mag.mean()) / mag.std()
    belt = np.zeros((H, W))
    rr, cc = np.mgrid[0:H, 0:W]
    belt += np.exp(-(((rr - 60) ** 2 + (cc - 70) ** 2) / (2 * 25 ** 2)))
    belt += np.exp(-(((rr - 150) ** 2 + (cc - 160) ** 2) / (2 * 25 ** 2)))
    mag_total = (mag + 3.0 * belt).astype(np.float32)
    write_tif(root / "magnetics.tif", mag_total, grid)

    # "Fault" proximity truth: a diagonal corridor
    fault = np.exp(-np.abs(rr - cc) / 15.0)
    write_tif(root / "fault_proximity.tif", fault.astype(np.float32), grid)

    # Hidden prospectivity truth and planted deposits
    truth = 0.6 * belt / belt.max() + 0.4 * fault
    deposits = plant_deposits(truth, n=60, threshold_q=0.93, seed=3)
    deposits_to_csv(root / "deposits.csv", deposits, grid)

    cfg = PipelineConfig(
        dem_path=str(root / "dem.tif"),
        deposit_csv=str(root / "deposits.csv"),
        feature_rasters={
            "magnetics": str(root / "magnetics.tif"),
            "fault_proximity": str(root / "fault_proximity.tif"),
        },
        commodity="copper",
        results_dir=str(root / "results"),
        n_trees=150,
        depth_window_px=64,
        random_seed=42,
    )
    db = ProjectDB(str(root / "geocore.db"))
    result = run_pipeline(cfg, db=db, project_name="SyntheticWorld")
    return {"cfg": cfg, "result": result, "root": root, "db": db,
            "deposits": deposits, "truth": truth}


class TestEndToEnd:
    def test_spatial_cv_detects_real_signal(self, world):
        cv = world["result"].cv_report
        assert cv.n_folds_used >= 3
        assert cv.auc_mean > 0.75, \
            f"spatial CV AUC {cv.auc_mean:.3f} too low on a learnable world"

    def test_holdout_deposits_outrank_background(self, world):
        hm = world["result"].holdout_metrics
        assert hm, "expected a spatial holdout with 60 deposits"
        assert hm["contrast_auc"] > 0.75
        assert hm["mean"] > hm["background_mean"]

    def test_probability_map_well_formed(self, world):
        p = world["result"].prob_map
        ok = np.isfinite(p)
        assert ok.mean() > 0.95
        assert ((p[ok] >= 0) & (p[ok] <= 1)).all()
        # planted truth-high region must score above truth-low region
        truth = world["truth"]
        hi = p[truth >= np.quantile(truth, 0.95)]
        lo = p[truth <= np.quantile(truth, 0.30)]
        assert np.nanmean(hi) > np.nanmean(lo) + 0.1

    def test_uncertainty_nonnegative_and_bounded(self, world):
        u = world["result"].uncertainty_map
        ok = np.isfinite(u)
        assert (u[ok] >= 0).all() and (u[ok] <= 0.6).all()

    def test_depth_recovered_within_tolerance(self, world):
        d = world["result"].depth_spectral
        assert d is not None
        est = float(np.nanmedian(d))
        assert est == pytest.approx(TRUE_SOURCE_DEPTH, rel=0.5), \
            f"median spectral depth {est:.0f} m vs planted {TRUE_SOURCE_DEPTH} m"

    def test_all_artifacts_written(self, world):
        arts = world["result"].artifacts
        for kind in ("probability", "uncertainty", "classes", "dem",
                     "model", "config", "report"):
            assert kind in arts, f"missing artifact: {kind}"
            assert os.path.exists(arts[kind]), f"artifact not on disk: {kind}"

    def test_report_contains_honest_language(self, world):
        text = open(world["result"].artifacts["report"]).read()
        assert "Spatial block cross-validation" in text
        assert "does not produce grade maps" in text
        assert "Limitations" in text

    def test_db_provenance_recorded(self, world):
        db = world["db"]
        projects = db.list_projects()
        assert projects and projects[0]["name"] == "SyntheticWorld"
        # find the run and check metrics landed
        rid = db.conn.execute("SELECT id FROM runs ORDER BY id DESC").fetchone()[0]
        s = db.run_summary(rid)
        assert s["run"]["status"] == "complete"
        assert "cv_auc_mean" in s["metrics"]

    def test_reproducibility_same_seed_same_map(self, world):
        cfg = world["cfg"]
        cfg2 = PipelineConfig(**{**cfg.__dict__,
                                 "results_dir": cfg.results_dir + "_rerun",
                                 "estimate_depth": False})
        r2 = run_pipeline(cfg2)
        p1 = world["result"].prob_map
        p2 = r2.prob_map
        both = np.isfinite(p1) & np.isfinite(p2)
        assert np.allclose(p1[both], p2[both], atol=1e-6), \
            "same config + seed must give identical maps"


class TestFailureModes:
    def test_too_few_deposits_clear_error(self, tmp_path):
        grid = make_grid(80, 80, pixel_m=100.0)
        write_tif(tmp_path / "dem.tif", synthetic_dem(80, 80), grid)
        write_tif(tmp_path / "mag.tif",
                  np.random.default_rng(0).normal(size=(80, 80)).astype(np.float32),
                  grid)
        deposits_to_csv(tmp_path / "deps.csv", [(10, 10), (20, 20)], grid)
        cfg = PipelineConfig(
            dem_path=str(tmp_path / "dem.tif"),
            deposit_csv=str(tmp_path / "deps.csv"),
            feature_rasters={"magnetics": str(tmp_path / "mag.tif")},
            results_dir=str(tmp_path / "res"))
        with pytest.raises(RuntimeError, match="need >="):
            run_pipeline(cfg)

    def test_invalid_config_rejected_before_work(self, tmp_path):
        cfg = PipelineConfig(dem_path=str(tmp_path / "nope.tif"))
        with pytest.raises(ValueError, match="Invalid configuration"):
            run_pipeline(cfg)

    def test_nonoverlapping_raster_warned_not_crashed(self, tmp_path):
        grid = make_grid(100, 100, pixel_m=100.0)
        write_tif(tmp_path / "dem.tif", synthetic_dem(100, 100), grid)
        far = make_grid(100, 100, pixel_m=100.0, x0=9_000_000.0)
        write_tif(tmp_path / "far.tif",
                  np.random.default_rng(0).normal(size=(100, 100)).astype(np.float32),
                  far)
        write_tif(tmp_path / "mag.tif",
                  field_from_sources_at_depth(100, 100, 100.0, 500.0),
                  grid)
        rng = np.random.default_rng(5)
        deps = [(int(r), int(c)) for r, c in
                zip(rng.integers(0, 100, 30), rng.integers(0, 100, 30))]
        deposits_to_csv(tmp_path / "deps.csv", deps, grid)
        cfg = PipelineConfig(
            dem_path=str(tmp_path / "dem.tif"),
            deposit_csv=str(tmp_path / "deps.csv"),
            feature_rasters={"magnetics": str(tmp_path / "mag.tif"),
                             "orphan": str(tmp_path / "far.tif")},
            results_dir=str(tmp_path / "res"), estimate_depth=False)
        result = run_pipeline(cfg)
        assert any("orphan" in w for w in result.warnings)
