"""Shapefile (.shp) support tests: points, lines, polygons, CRS handling."""
import numpy as np
import pytest
import shapefile

from geocore.vector_io import shapefile_to_raster, shapefile_deposit_points
from geocore.sampling import load_deposits_shp
from tests.synthetic import make_grid

UTM12_WKT = (
    'PROJCS["WGS 84 / UTM zone 12N",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-111],'
    'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]')

WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
    '298.257223563]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]]')


def _grid():
    return make_grid(100, 100, pixel_m=100.0)  # UTM 12N, x 500k-510k


def _world(grid, r, c):
    x = grid.transform.c + (c + 0.5) * grid.transform.a
    y = grid.transform.f + (r + 0.5) * grid.transform.e
    return x, y


class TestPointShp:
    def test_points_become_proximity(self, tmp_path):
        grid = _grid()
        p = tmp_path / "pts"
        w = shapefile.Writer(str(p), shapeType=shapefile.POINT)
        w.field("site", "C")
        for r, c in [(20, 20), (80, 70)]:
            x, y = _world(grid, r, c)
            w.point(x, y); w.record("s")
        w.close()
        (tmp_path / "pts.prj").write_text(UTM12_WKT)

        ras = shapefile_to_raster(tmp_path / "pts.shp", grid)
        assert ras.shape == grid.shape
        assert ras[20, 20] > 0.95            # at a point: ~1
        assert ras[20, 20] > ras[50, 45]     # decays away

    def test_value_field_triggers_idw(self, tmp_path):
        grid = _grid()
        p = tmp_path / "geochem"
        w = shapefile.Writer(str(p), shapeType=shapefile.POINT)
        w.field("cu_ppm", "N", decimal=2)
        rng = np.random.default_rng(0)
        for _ in range(40):
            r, c = rng.integers(0, 100, 2)
            x, y = _world(grid, r, c)
            w.point(x, y); w.record(float(100 + 50 * (r / 100)))
        w.close()
        (tmp_path / "geochem.prj").write_text(UTM12_WKT)

        ras = shapefile_to_raster(tmp_path / "geochem.shp", grid)
        # IDW of cu_ppm: top rows ~100, bottom rows ~150 (row 0 = top = north)
        assert np.nanmean(ras[75:]) > np.nanmean(ras[:25])
        assert ras.max() > 90                # real ppm scale, not 0..1

    def test_wgs84_points_reprojected(self, tmp_path):
        grid = _grid()
        from pyproj import Transformer
        inv = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        x, y = _world(grid, 50, 50)
        lon, lat = inv.transform(x, y)

        p = tmp_path / "ll"
        w = shapefile.Writer(str(p), shapeType=shapefile.POINT)
        w.field("site", "C")
        w.point(lon, lat); w.record("center")
        w.close()
        (tmp_path / "ll.prj").write_text(WGS84_WKT)

        ras = shapefile_to_raster(tmp_path / "ll.shp", grid)
        assert ras[50, 50] > 0.95

    def test_missing_prj_geographic_assumed_wgs84(self, tmp_path):
        grid = _grid()
        from pyproj import Transformer
        inv = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        x, y = _world(grid, 30, 60)
        lon, lat = inv.transform(x, y)
        p = tmp_path / "noprj"
        w = shapefile.Writer(str(p), shapeType=shapefile.POINT)
        w.field("a", "C"); w.point(lon, lat); w.record("x")
        w.close()  # no .prj written

        ras = shapefile_to_raster(tmp_path / "noprj.shp", grid)
        assert ras[30, 60] > 0.9


class TestLinePolygonShp:
    def test_fault_lines(self, tmp_path):
        grid = _grid()
        p = tmp_path / "faults"
        w = shapefile.Writer(str(p), shapeType=shapefile.POLYLINE)
        w.field("name", "C")
        x0, y0 = _world(grid, 10, 10)
        x1, y1 = _world(grid, 90, 90)
        w.line([[(x0, y0), (x1, y1)]]); w.record("F1")
        w.close()
        (tmp_path / "faults.prj").write_text(UTM12_WKT)

        ras = shapefile_to_raster(tmp_path / "faults.shp", grid)
        assert ras[50, 50] > 0.9             # on the diagonal
        assert ras[10, 90] < ras[50, 50]     # far corner decays

    def test_alteration_polygon(self, tmp_path):
        grid = _grid()
        p = tmp_path / "alt"
        w = shapefile.Writer(str(p), shapeType=shapefile.POLYGON)
        w.field("name", "C")
        corners_px = [(30, 30), (30, 70), (70, 70), (70, 30)]
        ring = [_world(grid, r, c) for r, c in corners_px]
        w.poly([ring]); w.record("argillic")
        w.close()
        (tmp_path / "alt.prj").write_text(UTM12_WKT)

        ras = shapefile_to_raster(tmp_path / "alt.shp", grid)
        assert ras[50, 50] == pytest.approx(1.0)   # inside
        assert ras[5, 5] < 0.5                     # outside decays


class TestShpDeposits:
    def test_deposits_filtered_by_commodity(self, tmp_path):
        grid = _grid()
        p = tmp_path / "mrds"
        w = shapefile.Writer(str(p), shapeType=shapefile.POINT)
        w.field("site_name", "C"); w.field("commod1", "C")
        for i, (r, c, comm) in enumerate(
                [(20, 20, "Copper"), (40, 40, "Copper, Gold"),
                 (60, 60, "Gold"), (80, 80, "Silver")]):
            x, y = _world(grid, r, c)
            w.point(x, y); w.record(f"D{i}", comm)
        w.close()
        (tmp_path / "mrds.prj").write_text(UTM12_WKT)

        deps = load_deposits_shp(str(tmp_path / "mrds.shp"), grid, "copper")
        assert len(deps) == 2
        assert {(d.row, d.col) for d in deps} == {(20, 20), (40, 40)}
        # true pixel coordinates preserved (the v4 bug stays dead)
        for d in deps:
            assert grid.mask[d.row, d.col]

    def test_polygon_deposits_rejected(self, tmp_path):
        grid = _grid()
        p = tmp_path / "poly"
        w = shapefile.Writer(str(p), shapeType=shapefile.POLYGON)
        w.field("a", "C")
        ring = [_world(grid, r, c) for r, c in
                [(10, 10), (10, 20), (20, 20), (20, 10)]]
        w.poly([ring]); w.record("x")
        w.close()
        with pytest.raises(RuntimeError, match="POINT"):
            shapefile_deposit_points(tmp_path / "poly.shp", grid)
