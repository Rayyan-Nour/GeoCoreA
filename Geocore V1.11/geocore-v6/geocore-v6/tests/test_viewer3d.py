"""
Regression test for the real-world black-screen bug: a geographic (lat/lon)
DEM must produce 3D terrain extents in METERS, not degrees. With degree
extents the vertical scale blows up ~100,000x and the camera ends up
inside the mesh.
"""
import json
import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from viewer.export_3d import export_viewer_3d


def _write_geographic_dem(path, H=120, W=160, lat0=40.8, lon0=-112.4,
                          px_deg=1.0 / 3600.0 * 10):   # ~10 m pixels
    """A small DEM in EPSG:4326 like a clipped USGS 3DEP tile."""
    yy, xx = np.mgrid[0:H, 0:W]
    dem = (1500 + 300 * np.sin(yy / 18.0) * np.cos(xx / 23.0)).astype(
        np.float32)
    transform = from_origin(lon0, lat0, px_deg, px_deg)
    with rasterio.open(
            path, "w", driver="GTiff", height=H, width=W, count=1,
            dtype="float32", crs="EPSG:4326", transform=transform) as dst:
        dst.write(dem, 1)
    return dem, transform


def _payload(html_path):
    html = open(html_path).read()
    start = html.index("const GC = ") + len("const GC = ")
    end = html.index(";\nif (typeof THREE", start)
    return json.loads(html[start:end])


class TestGeographicDEM3D:
    def test_extents_are_metric(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        H, W = 120, 160
        px_deg = 1.0 / 3600.0 * 10
        _write_geographic_dem(run / "geocore_dem.tif", H, W, px_deg=px_deg)

        out = export_viewer_3d(run, tmp_path / "v.html")
        gc = _payload(out)

        # Expected metric extents at ~40.8 degrees N
        m_per_deg = 111_320.0
        exp_y = H * px_deg * m_per_deg                       # ~3.7 km
        exp_x = W * px_deg * m_per_deg * math.cos(math.radians(40.8))

        assert gc["extentY"] == pytest.approx(exp_y, rel=0.05)
        assert gc["extentX"] == pytest.approx(exp_x, rel=0.05)
        # Degree extents would be ~0.3-0.4; metric are thousands
        assert gc["extentX"] > 1000

    def test_vertical_scale_is_sane(self, tmp_path):
        """zRange/extentX drives terrain height; must be O(0.1), not O(1e5)."""
        run = tmp_path / "run"
        run.mkdir()
        _write_geographic_dem(run / "geocore_dem.tif")
        gc = _payload(export_viewer_3d(run, tmp_path / "v.html"))
        ratio = (gc["zmax"] - gc["zmin"]) / gc["extentX"]
        assert ratio < 1.0, (
            f"vertical/horizontal ratio {ratio:.1f} - terrain would render "
            f"as a vertical spike (the black-screen bug)")

    def test_projected_dem_unchanged(self, tmp_path):
        """Metric DEMs (UTM) must keep their native extents."""
        run = tmp_path / "run"
        run.mkdir()
        H, W, px = 100, 100, 100.0
        dem = np.random.default_rng(0).normal(1200, 50, (H, W)).astype(
            np.float32)
        transform = from_origin(400000, 4500000, px, px)
        with rasterio.open(
                run / "geocore_dem.tif", "w", driver="GTiff", height=H,
                width=W, count=1, dtype="float32", crs="EPSG:32612",
                transform=transform) as dst:
            dst.write(dem, 1)
        gc = _payload(export_viewer_3d(run, tmp_path / "v.html"))
        assert gc["extentX"] == pytest.approx(W * px, rel=1e-6)
        assert gc["extentY"] == pytest.approx(H * px, rel=1e-6)
