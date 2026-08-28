"""
Depth estimation tests: the estimators must recover KNOWN planted depths.
If these pass, the depth feature is evidence-backed, not marketing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from synthetic import make_grid, field_from_sources_at_depth, point_pole_field
from geocore.depth import (euler_deconvolution, euler_depth_map,
                           spectral_depth, spectral_depth_map,
                           vertical_derivative)


class TestSpectralDepth:
    @pytest.mark.parametrize("true_depth", [300.0, 800.0, 1500.0])
    def test_recovers_known_ensemble_depth(self, true_depth):
        # Field = white noise upward-continued by h  =>  ln P ~ -2 h |k|
        pixel = 100.0
        field = field_from_sources_at_depth(256, 256, pixel, true_depth, seed=3)
        est, r2 = spectral_depth(field, pixel, pixel)
        assert r2 > 0.8
        assert est == pytest.approx(true_depth, rel=0.25), \
            f"estimated {est:.0f} m vs true {true_depth:.0f} m"

    def test_flat_field_returns_nan(self):
        est, _ = spectral_depth(np.zeros((64, 64)), 100.0, 100.0)
        assert np.isnan(est)

    def test_depth_map_shape_and_spatial_variation(self):
        pixel = 100.0
        H = W = 256
        shallow = field_from_sources_at_depth(H, W, pixel, 300.0, seed=1)
        deep = field_from_sources_at_depth(H, W, pixel, 1500.0, seed=2)
        field = np.concatenate([shallow[:, :W // 2], deep[:, W // 2:]], axis=1)
        g = make_grid(H, W, pixel_m=pixel)
        g.data = field
        g.mask = np.ones((H, W), bool)
        dmap = spectral_depth_map(g, window_px=64)
        assert dmap.shape == (H, W)
        left = np.nanmedian(dmap[:, : W // 4])
        right = np.nanmedian(dmap[:, 3 * W // 4:])
        assert left < right, "shallow half must map shallower than deep half"
        assert left == pytest.approx(300.0, rel=0.5)
        assert right == pytest.approx(1500.0, rel=0.5)


class TestVerticalDerivative:
    def test_spectral_dz_matches_analytic_for_pole(self):
        # For T = h/(r^2+h^2)^{3/2}, upward continuation by dz shifts h -> h+dz.
        pixel, h = 50.0, 600.0
        T1 = point_pole_field(128, 128, pixel, [(64, 64, h, 1e9)])
        T2 = point_pole_field(128, 128, pixel, [(64, 64, h + 50.0, 1e9)])
        dTdz_num = (T1 - T2) / 50.0           # -d/dz upward == d/dz downward
        dTdz_spec = vertical_derivative(T1, pixel, pixel)
        # compare in central region away from window taper
        a = dTdz_num[48:80, 48:80].ravel()
        b = dTdz_spec[48:80, 48:80].ravel()
        corr = np.corrcoef(a, b)[0, 1]
        assert corr > 0.97


class TestEulerDeconvolution:
    def test_recovers_isolated_source_depth(self):
        pixel = 50.0
        true_depth = 500.0
        field = point_pole_field(128, 128, pixel, [(64, 64, true_depth, 1e9)])
        g = make_grid(128, 128, pixel_m=pixel)
        g.data = field
        g.mask = np.ones((128, 128), bool)
        sols = euler_deconvolution(g, structural_index=2.0, window_px=12)
        assert len(sols) > 0, "no Euler solutions accepted"
        # weighted median of best solutions near the source
        depths = np.array([s.depth_m for s in sols])
        rel = np.array([s.reliability for s in sols])
        top = depths[np.argsort(rel)[::-1][: max(3, len(sols) // 5)]]
        est = float(np.median(top))
        assert est == pytest.approx(true_depth, rel=0.35), \
            f"Euler estimated {est:.0f} m vs true {true_depth:.0f} m"

    def test_two_sources_distinguished(self):
        pixel = 50.0
        field = point_pole_field(
            160, 160, pixel,
            [(40, 40, 300.0, 1e9), (120, 120, 1200.0, 4e9)])
        g = make_grid(160, 160, pixel_m=pixel)
        g.data = field
        g.mask = np.ones((160, 160), bool)
        sols = euler_deconvolution(g, structural_index=2.0, window_px=12)
        assert sols
        dmap = euler_depth_map(g, sols, radius_px=20)
        d_shallow = np.nanmedian(dmap[20:60, 20:60])
        d_deep = np.nanmedian(dmap[100:140, 100:140])
        assert np.isfinite(d_shallow) and np.isfinite(d_deep)
        assert d_shallow < d_deep

    def test_noise_yields_few_or_no_confident_solutions(self):
        rng = np.random.default_rng(0)
        g = make_grid(96, 96, pixel_m=50.0)
        g.data = rng.normal(size=(96, 96)).astype(np.float32)
        g.mask = np.ones((96, 96), bool)
        sols = euler_deconvolution(g, structural_index=2.0, window_px=12)
        # white noise has no coherent sources; acceptance filters must prune
        dense = len(sols) / (96 * 96 / 36)
        assert dense < 0.5, "Euler accepts too many solutions on pure noise"
