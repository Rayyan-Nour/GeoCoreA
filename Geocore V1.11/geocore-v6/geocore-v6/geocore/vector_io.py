"""
Shapefile (.shp) support without a GDAL build.

* Geometry + attributes via pyshp (pure Python)
* CRS from the sidecar .prj (WKT) via pyproj, reprojected to the grid CRS
* Rasterization via rasterio.features (rasterio bundles its own GDAL)

What each geometry type becomes on the analysis grid:

  POINTS    -> proximity raster (distance decay to nearest point), or an
               IDW-interpolated surface if a numeric value field is present
  LINES     -> proximity raster to the line work (faults, contacts, dikes)
  POLYGONS  -> proximity raster to the polygon boundary, with interior = 1
               (alteration footprints, intrusive map units)

Deposits can also be loaded from point shapefiles (MRDS/USMIN exports).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import shapefile  # pyshp
except ImportError:  # pragma: no cover
    shapefile = None

from pyproj import CRS, Transformer
from rasterio import features as rio_features

from .raster_io import Grid
from .features import proximity_raster, idw_raster


def _require_pyshp():
    if shapefile is None:
        raise RuntimeError("Reading .shp needs the 'pyshp' package: "
                           "pip install pyshp")


def _shp_crs(path: Path) -> Optional[CRS]:
    prj = path.with_suffix(".prj")
    if prj.exists():
        try:
            return CRS.from_wkt(prj.read_text().strip())
        except Exception:
            return None
    return None


def _transformer(path: Path, grid: Grid) -> Optional[Transformer]:
    src = _shp_crs(path)
    dst = CRS.from_user_input(grid.crs)
    if src is None:
        return None          # assume already in grid CRS
    if src == dst:
        return None
    return Transformer.from_crs(src, dst, always_xy=True)


def _tx_coords(coords, tr: Optional[Transformer]):
    if tr is None:
        return [(float(x), float(y)) for x, y in coords]
    xs, ys = tr.transform([c[0] for c in coords], [c[1] for c in coords])
    return list(zip(map(float, xs), map(float, ys)))


def _looks_geographic(pts) -> bool:
    """No .prj present but coordinates are clearly lat/lon."""
    return all(abs(x) <= 360 and abs(y) <= 90 for x, y in pts[:200])


def read_shapefile(path: str | Path, grid: Grid):
    """
    Returns (geom_type, shapes, records, fields)
      geom_type in {'point', 'line', 'polygon'}
      shapes: list of coordinate structures in GRID CRS
        point   -> [(x, y), ...]
        line    -> [[(x, y), ...] per part, ...]
        polygon -> [[(x, y), ...] per ring, ...]
    """
    _require_pyshp()
    path = Path(path)
    rd = shapefile.Reader(str(path))
    tr = _transformer(path, grid)

    st = rd.shapeType
    if st in (shapefile.POINT, shapefile.POINTZ, shapefile.POINTM,
              shapefile.MULTIPOINT, shapefile.MULTIPOINTZ,
              shapefile.MULTIPOINTM):
        gtype = "point"
    elif st in (shapefile.POLYLINE, shapefile.POLYLINEZ,
                shapefile.POLYLINEM):
        gtype = "line"
    elif st in (shapefile.POLYGON, shapefile.POLYGONZ, shapefile.POLYGONM):
        gtype = "polygon"
    else:
        raise RuntimeError(f"Unsupported shapefile geometry type {st} "
                           f"in {path.name}")

    raw_pts_probe: List[Tuple[float, float]] = []
    geoms = []
    for sh in rd.shapes():
        pts = [(p[0], p[1]) for p in sh.points]
        raw_pts_probe.extend(pts[:5])
        if gtype == "point":
            geoms.extend(pts)
        else:
            parts = list(sh.parts) + [len(pts)]
            geoms.append([pts[parts[i]:parts[i + 1]]
                          for i in range(len(parts) - 1)])

    # No .prj but lat/lon-looking coordinates and a projected grid:
    # assume WGS84 (the overwhelmingly common case for public data)
    if tr is None and raw_pts_probe and _looks_geographic(raw_pts_probe) \
            and CRS.from_user_input(grid.crs).is_projected:
        tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)

    if gtype == "point":
        shapes = _tx_coords(geoms, tr)
    else:
        shapes = [[_tx_coords(part, tr) for part in g] for g in geoms]

    fields = [f[0] for f in rd.fields[1:]]  # skip DeletionFlag
    records = [list(r) for r in rd.records()]
    rd.close()
    return gtype, shapes, records, fields


# ----------------------------------------------------------------------
# Evidence layer from a shapefile
# ----------------------------------------------------------------------

def _numeric_value_field(fields: List[str], records) -> Optional[int]:
    """Index of a plausible numeric value column (grade/ppm/value), if any."""
    keys = ("value", "ppm", "ppb", "grade", "conc", "pct")
    for i, f in enumerate(fields):
        if any(k in f.lower() for k in keys):
            try:
                float(records[0][i])
                return i
            except (TypeError, ValueError, IndexError):
                continue
    return None


def shapefile_to_raster(path: str | Path, grid: Grid) -> np.ndarray:
    """Rasterize a shapefile onto the analysis grid as an evidence layer."""
    gtype, shapes, records, fields = read_shapefile(path, grid)
    H, W = grid.shape

    if gtype == "point":
        if not shapes:
            raise RuntimeError(f"{Path(path).name}: no points")
        vi = _numeric_value_field(fields, records) if records else None
        if vi is not None:
            pts = []
            for (x, y), rec in zip(shapes, records):
                try:
                    pts.append((x, y, float(rec[vi])))
                except (TypeError, ValueError):
                    continue
            if len(pts) >= 5:
                return idw_raster(pts, grid)
        return proximity_raster(shapes, grid)

    if gtype == "line":
        geojson = [{"type": "MultiLineString", "coordinates": g}
                   for g in shapes if g and any(len(p) >= 2 for p in g)]
        if not geojson:
            raise RuntimeError(f"{Path(path).name}: no line work")
        burned = rio_features.rasterize(
            [(g, 1) for g in geojson], out_shape=(H, W),
            transform=grid.transform, fill=0, all_touched=True
        ).astype(np.uint8)
        return _distance_decay(burned, grid)

    # polygon: proximity to boundary, interior saturated at 1
    geojson = []
    for g in shapes:
        rings = [r for r in g if len(r) >= 3]
        if rings:
            geojson.append({"type": "Polygon",
                            "coordinates": [[*r, r[0]] for r in rings]})
    if not geojson:
        raise RuntimeError(f"{Path(path).name}: no polygons")
    inside = rio_features.rasterize(
        [(g, 1) for g in geojson], out_shape=(H, W),
        transform=grid.transform, fill=0, all_touched=True
    ).astype(np.uint8)
    boundary = rio_features.rasterize(
        [({"type": "MultiLineString", "coordinates": g["coordinates"]}, 1)
         for g in geojson], out_shape=(H, W),
        transform=grid.transform, fill=0, all_touched=True
    ).astype(np.uint8)
    decay = _distance_decay(boundary, grid)
    return np.where(inside == 1, 1.0, decay).astype(np.float32)


def _distance_decay(burned: np.ndarray, grid: Grid,
                    decay_m: float = 2500.0) -> np.ndarray:
    """exp(-d/decay) distance transform from burned cells, in meters."""
    from scipy.ndimage import distance_transform_edt
    dx, dy = grid.pixel_size_m()
    if burned.max() == 0:
        return np.zeros(burned.shape, np.float32)
    dist = distance_transform_edt(burned == 0, sampling=(dy, dx))
    return np.exp(-dist / decay_m).astype(np.float32)


# ----------------------------------------------------------------------
# Deposits from a point shapefile
# ----------------------------------------------------------------------

def shapefile_deposit_points(path: str | Path, grid: Grid
                             ) -> List[Tuple[float, float, str, str]]:
    """
    (x, y, name, commodity_text) in grid CRS from a point shapefile.
    Commodity/name pulled from common MRDS/USMIN field names when present.
    """
    gtype, shapes, records, fields = read_shapefile(path, grid)
    if gtype != "point":
        raise RuntimeError(f"{Path(path).name}: deposits must be a POINT "
                           f"shapefile (got {gtype})")
    lower = [f.lower() for f in fields]

    def col(*names):
        for n in names:
            if n in lower:
                return lower.index(n)
        return None

    ci = col("commod1", "commodity", "commod", "commodities")
    ni = col("site_name", "name", "dep_name", "site")
    out = []
    for (x, y), rec in zip(shapes, records):
        name = str(rec[ni]) if ni is not None else "deposit"
        comm = str(rec[ci]) if ci is not None else ""
        out.append((x, y, name, comm))
    return out
