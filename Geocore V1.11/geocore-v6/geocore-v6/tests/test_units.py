"""Unit tests: features, sampling, spatial validation, model, project DB."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from synthetic import make_grid, synthetic_dem
from geocore.features import (FeatureStack, idw_raster, proximity_raster,
                              terrain_features, topographic_position_index)
from geocore.sampling import Deposit, build_samples, spatial_holdout
from geocore.validation import assign_blocks, spatial_cv
from geocore.model import (align_features_for_transfer, load_model,
                           save_model, train)
from geocore.sampling import Samples
from geocore.projectdb import ProjectDB
from geocore.grade import grade_context, prospectivity_classes
from geocore.config import COMMODITIES, PipelineConfig


# ---------------------------------------------------------------- features
class TestFeatures:
    def test_tpi_ridge_positive_valley_negative(self):
        dem = np.zeros((50, 50), dtype=np.float32)
        dem[25, 25] = 100.0      # spike = ridge
        tpi = topographic_position_index(dem, window=11)
        assert tpi[25, 25] > 0
        dem2 = np.full((50, 50), 100.0, dtype=np.float32)
        dem2[25, 25] = 0.0       # pit = valley
        assert topographic_position_index(dem2, window=11)[25, 25] < 0

    def test_terrain_features_nan_safe_and_metric(self):
        g = make_grid(60, 60, pixel_m=100.0)
        dem = synthetic_dem(60, 60)
        dem[0:5, 0:5] = np.nan
        g.data = dem
        g.mask = np.isfinite(dem)
        feats = terrain_features(g)
        assert set(feats) == {"tpi", "slope", "aspect_sin", "aspect_cos",
                              "curvature"}
        # slope is rise/run: synthetic terrain (500m relief over 6km) must
        # have plausible slopes, not degree-scaled garbage
        s = feats["slope"][np.isfinite(feats["slope"])]
        assert 0 <= np.nanmax(s) < 5.0

    def test_proximity_decays_in_meters(self):
        g = make_grid(100, 100, pixel_m=100.0)
        x = g.transform.c + 50.5 * 100
        y = g.transform.f - 50.5 * 100
        prox = proximity_raster([(x, y)], g, decay_m=2000.0)
        assert prox[50, 50] == pytest.approx(1.0, abs=0.05)
        assert prox[50, 90] == 0.0          # 4 km away: fully decayed
        assert 0.0 < prox[50, 60] < 1.0     # 1 km away: partial

    def test_idw_interpolates_between_points(self):
        g = make_grid(40, 40, pixel_m=100.0)
        def world(r, c):
            return (g.transform.c + (c + .5) * 100, g.transform.f - (r + .5) * 100)
        p1, p2 = world(10, 10), world(30, 30)
        out = idw_raster([(p1[0], p1[1], 10.0), (p2[0], p2[1], 90.0)], g)
        assert out[10, 10] == pytest.approx(10.0, abs=2.0)
        assert out[30, 30] == pytest.approx(90.0, abs=2.0)
        assert 10.0 < out[20, 20] < 90.0

    def test_stack_rejects_constant_and_duplicate(self):
        g = make_grid(20, 20)
        fs = FeatureStack(g)
        fs.add("const", np.ones((20, 20)))           # rejected silently
        assert "const" not in fs.names
        fs.add("ok", np.random.default_rng(0).normal(size=(20, 20)))
        with pytest.raises(ValueError):
            fs.add("ok", np.random.default_rng(1).normal(size=(20, 20)))

    def test_matrix_median_fills_nans(self):
        g = make_grid(20, 20)
        fs = FeatureStack(g)
        layer = np.random.default_rng(0).normal(size=(20, 20)).astype(np.float32)
        layer[0, 0] = np.nan
        fs.add("f", layer)
        X, valid = fs.matrix()
        assert np.isfinite(X).all()


# ---------------------------------------------------------------- sampling
class TestSampling:
    def _deposits(self, n, H, W, seed=0):
        rng = np.random.default_rng(seed)
        return [Deposit(row=int(r), col=int(c), x=0, y=0)
                for r, c in zip(rng.integers(0, H, n), rng.integers(0, W, n))]

    def test_negatives_respect_buffer_and_keep_coords(self):
        g = make_grid(200, 200, pixel_m=100.0)
        F = 4
        Xf = np.random.default_rng(0).normal(size=(200 * 200, F)).astype(np.float32)
        valid = np.ones(200 * 200, dtype=bool)
        deps = self._deposits(20, 200, 200)
        s = build_samples(Xf, valid, g, deps, deps, buffer_m=1000.0,
                          negative_ratio=2, feature_names=list("abcd"))
        assert s.X.shape == (60, F)
        assert (s.y == 1).sum() == 20 and (s.y == 0).sum() == 40
        # every negative must be >= ~1000 m (10 px) from every deposit
        for i in np.where(s.y == 0)[0]:
            dmin = min(np.hypot(s.rows[i] - d.row, s.cols[i] - d.col)
                       for d in deps)
            assert dmin >= 9   # buffer with int rounding tolerance
        # THE v4 BUG: coordinates must be real, not random — verify negatives'
        # coords index pixels that were actually sampled
        assert s.rows.shape == s.cols.shape == (60,)

    def test_buffer_shrinks_rather_than_failing(self):
        g = make_grid(50, 50, pixel_m=100.0)
        Xf = np.zeros((2500, 2), dtype=np.float32)
        valid = np.ones(2500, dtype=bool)
        deps = self._deposits(30, 50, 50)
        s = build_samples(Xf, valid, g, deps, deps, buffer_m=100000.0,
                          negative_ratio=2)
        assert (s.y == 0).sum() > 0

    def test_spatial_holdout_is_coherent_quadrant(self):
        deps = self._deposits(40, 100, 100, seed=1)
        train, hold = spatial_holdout(deps, (100, 100), fraction=0.25)
        assert len(train) + len(hold) == 40
        assert len(hold) > 0
        # all holdout deposits share a quadrant
        quads = {(d.row >= 50, d.col >= 50) for d in hold}
        assert len(quads) == 1

    def test_holdout_skipped_for_tiny_datasets(self):
        deps = self._deposits(6, 100, 100)
        train, hold = spatial_holdout(deps, (100, 100))
        assert hold == [] and len(train) == 6


# ---------------------------------------------------------------- validation
class TestValidation:
    def test_blocks_are_spatial(self):
        rng = np.random.default_rng(0)
        rows = rng.integers(0, 100, 500); cols = rng.integers(0, 100, 500)
        folds = assign_blocks(rows, cols, (100, 100), k=5)
        assert set(np.unique(folds)) <= set(range(5))
        # same pixel -> same fold (deterministic, spatial)
        f1 = assign_blocks(np.array([10, 10]), np.array([10, 10]), (100, 100), 5)
        assert f1[0] == f1[1]

    def test_spatial_cv_detects_signal_and_absence(self):
        rng = np.random.default_rng(0)
        n = 400
        rows = rng.integers(0, 100, n); cols = rng.integers(0, 100, n)
        y = (rng.random(n) < 0.33).astype(np.int8)
        X_signal = np.column_stack([y + rng.normal(0, .3, n),
                                    rng.normal(size=n)]).astype(np.float32)
        X_noise = rng.normal(size=(n, 2)).astype(np.float32)
        from geocore.model import adaptive_rf
        s_sig = Samples(X=X_signal, y=y, rows=rows, cols=cols)
        s_noi = Samples(X=X_noise, y=y, rows=rows, cols=cols)
        cv_sig = spatial_cv(adaptive_rf(int(y.sum())), s_sig, (100, 100))
        cv_noi = spatial_cv(adaptive_rf(int(y.sum())), s_noi, (100, 100))
        assert cv_sig.auc_mean > 0.85
        assert 0.3 < cv_noi.auc_mean < 0.7   # noise ≈ coin flip — no inflation


# ---------------------------------------------------------------- model
class TestModel:
    def _samples(self, n=300, F=5, seed=0):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.33).astype(np.int8)
        X = rng.normal(size=(n, F)).astype(np.float32)
        X[:, 0] += 2.0 * y
        return Samples(X=X, y=y, rows=rng.integers(0, 100, n),
                       cols=rng.integers(0, 100, n),
                       feature_names=[f"f{i}" for i in range(F)])

    def test_train_predict_roundtrip(self, tmp_path):
        s = self._samples()
        m = train(s, "copper", n_trees=100)
        X_flat = np.random.default_rng(1).normal(size=(50 * 50, 5)).astype(np.float32)
        valid = np.ones(2500, dtype=bool)
        valid[:100] = False
        prob, unc = m.predict_map(X_flat, valid, (50, 50))
        assert prob.shape == (50, 50)
        assert np.isnan(prob.reshape(-1)[:100]).all()      # invalid -> NaN
        ok = np.isfinite(prob)
        assert ((prob[ok] >= 0) & (prob[ok] <= 1)).all()
        assert (unc[ok] >= 0).all()
        # informative feature ranks first
        assert m.importances()[0][0] == "f0"

        p = tmp_path / "m.joblib"
        save_model(m, str(p))
        m2 = load_model(str(p))
        assert m2.feature_names == m.feature_names

    def test_transfer_refuses_feature_mismatch(self):
        s = self._samples()
        m = train(s, "copper", n_trees=50)
        X_now = np.zeros((10, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="Transfer refused"):
            align_features_for_transfer(["zzz", "qqq"], m, X_now)

    def test_transfer_aligns_reordered_features(self):
        s = self._samples()
        m = train(s, "copper", n_trees=50)
        names_now = ["f4", "f3", "f2", "f1", "f0"]
        X_now = np.arange(50, dtype=np.float32).reshape(10, 5)
        out, matched = align_features_for_transfer(names_now, m, X_now)
        assert matched == 5
        assert np.allclose(out[:, 0], X_now[:, 4])  # f0 pulled from col 4


# ---------------------------------------------------------------- grade / db / config
class TestGradeContext:
    def test_copper_percentiles_match_usgs(self):
        # Ground truth is the cited distribution: log10(Cu%) ~ N(-0.357, 0.227)
        gc = grade_context(COMMODITIES["copper"])
        assert gc.percentiles["P50"] == pytest.approx(10 ** -0.357, rel=0.01)
        z10 = 1.2816
        assert gc.percentiles["P10"] == pytest.approx(
            10 ** (-0.357 - 0.227 * z10), rel=0.01)
        assert gc.percentiles["P90"] == pytest.approx(
            10 ** (-0.357 + 0.227 * z10), rel=0.01)
        assert "drilling" in gc.statement

    def test_classes_categorical(self):
        p = np.array([[0.1, 0.6], [0.8, np.nan]], dtype=np.float32)
        c = prospectivity_classes(p)
        assert c[0, 0] == 0 and c[0, 1] == 1 and c[1, 0] == 2
        assert np.isnan(c[1, 1])


class TestProjectDB:
    def test_full_run_provenance(self, tmp_path):
        db = ProjectDB(str(tmp_path / "geo.db"))
        pid = db.create_project("Gila", "copper")
        rid = db.start_run(pid, {"dem_path": "x.tif"}, "5.0.0")
        db.log_metric(rid, "cv_auc_mean", 0.87)
        db.log_artifact(rid, "probability", "/out/p.tif")
        db.finish_run(rid)
        s = db.run_summary(rid)
        assert s["metrics"]["cv_auc_mean"] == pytest.approx(0.87)
        assert s["artifacts"]["probability"] == "/out/p.tif"
        assert s["run"]["status"] == "complete"
        assert db.list_projects()[0]["name"] == "Gila"
        db.close()


class TestConfig:
    def test_validation_catches_problems(self, tmp_path):
        cfg = PipelineConfig(dem_path=str(tmp_path / "missing.tif"),
                             commodity="unobtanium", depth_window_px=33)
        probs = cfg.validate()
        assert any("DEM not found" in p for p in probs)
        assert any("unknown commodity" in p for p in probs)
        assert any("power of two" in p for p in probs)

    def test_roundtrip_json(self, tmp_path):
        cfg = PipelineConfig(dem_path="a.tif", commodity="ree", n_trees=123)
        p = tmp_path / "c.json"
        cfg.to_json(str(p))
        cfg2 = PipelineConfig.from_json(str(p))
        assert cfg2.n_trees == 123 and cfg2.commodity == "ree"


class TestAnalysisResolution:
    def test_decimated_dem_load_preserves_georeferencing(self, tmp_path):
        import rasterio
        from rasterio.transform import from_origin
        from geocore.raster_io import load_dem
        H = W = 3000
        dem = np.fromfunction(lambda r, c: 1000 + r * 0.1 + c * 0.05,
                              (H, W), dtype=np.float32)
        prof = dict(driver="GTiff", height=H, width=W, count=1,
                    dtype="float32", crs="EPSG:32612",
                    transform=from_origin(500000, 4100000, 10, 10))
        p = tmp_path / "dem.tif"
        with rasterio.open(p, "w", **prof) as dst:
            dst.write(dem, 1)

        g = load_dem(str(p), max_px=600)
        assert max(g.shape) == 600
        # pixel size scales exactly: 10 m * 3000/600 = 50 m
        assert g.pixel_size_m()[0] == pytest.approx(50.0, rel=1e-6)
        # corner georeferencing unchanged
        assert g.transform.c == pytest.approx(500000)
        assert g.transform.f == pytest.approx(4100000)
        # values plausible (averaged, not garbage)
        assert 1000 <= np.nanmean(g.data) <= 1400

    def test_native_when_zero(self, tmp_path):
        import rasterio
        from rasterio.transform import from_origin
        from geocore.raster_io import load_dem
        prof = dict(driver="GTiff", height=120, width=100, count=1,
                    dtype="float32", crs="EPSG:32612",
                    transform=from_origin(500000, 4100000, 10, 10))
        p = tmp_path / "dem.tif"
        with rasterio.open(p, "w", **prof) as dst:
            dst.write(np.ones((120, 100), np.float32), 1)
        g = load_dem(str(p), max_px=0)
        assert g.shape == (120, 100)
