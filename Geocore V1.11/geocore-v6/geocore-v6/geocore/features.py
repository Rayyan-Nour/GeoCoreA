"""
Feature engineering.

Terrain features from the DEM (TPI, slope, aspect, curvature, hillshade) plus
rasterization of vector/CSV layers (proximity decay or IDW interpolation).

Fixes from v4:
  * Slope/gradient computed in METERS (v4 mixed degrees and meters, so slope
    magnitudes were CRS-dependent and meaningless in geographic CRS).
  * Aspect encoded as (sin, cos) pair - raw arctan2 angle is circular and a
    tree split at +/-pi is geologically meaningless.
  * Polygon rasterization burns the FULL geometry via rasterio.features
    (v4 sampled only the centroid + 20 boundary points, so large geology
    polygons were mostly invisible to the model).
  * Proximity decay distance specified in METERS, not "500 pixels".
  * NaN-aware: features carry validity masks; invalid pixels are median-filled
    only at model time and excluded from training.

v5.1:
  * Terrain derivatives are now computed at a configurable PHYSICAL scale
    (terrain_scale_m) rather than at the native pixel gradient. The DEM is
    low-pass filtered to that scale before differencing so slope/aspect/
    curvature/TPI describe range-scale topography instead of pixel roughness.
    This keeps terrain features at a spatial scale comparable to the
    geophysical layers (often ~1 km) and stops a tree model from winning on
    high-frequency terrain the geophysics doesn't share. Set terrain_scale_m=0
    to recover the old native-resolution behaviour.
"""
from __future__ import annotations

import csv
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .raster_io import Grid


# ----------------------------------------------------------------------
# Terrain features
# ----------------------------------------------------------------------

def nan_uniform_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """Mean filter that ignores NaNs."""
    filled = np.nan_to_num(arr, nan=0.0)
    valid = np.isfinite(arr).astype(np.float32)
    num = ndimage.uniform_filter(filled, size=size)
    den = ndimage.uniform_filter(valid, size=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0] = np.nan
    return out


def topographic_position_index(dem: np.ndarray, window: int = 11) -> np.ndarray:
    """TPI: elevation relative to neighborhood mean. + = ridge, - = valley."""
    return dem - nan_uniform_filter(dem, window)


def terrain_features(dem_grid: Grid,
                     terrain_scale_m: float = 1000.0) -> Dict[str, np.ndarray]:
    """
    Terrain derivatives computed at a physical analysis scale.

    The DEM is low-pass filtered to `terrain_scale_m` before differencing, so
    slope / aspect / curvature describe range-scale topography instead of
    pixel-level roughness - keeping terrain at a spatial scale comparable to
    the geophysical layers (often ~1 km) so the model can't win by memorizing
    high-frequency terrain the geophysics doesn't share.

    The meters->pixels conversion uses the grid's true pixel size, so the
    physical scale is honoured regardless of the analysis resolution (e.g. a
    1200 px cap on a 10k px DEM yields ~90 m pixels, and 1 km lands at ~11 px).

    terrain_scale_m = 0 recovers the old native-resolution behaviour (for A/B).
    """
    dem = dem_grid.data
    px_m, py_m = dem_grid.pixel_size_m()
    pix_m = 0.5 * (px_m + py_m)

    # Physical scale -> odd window in pixels (1 == no smoothing).
    win_px = max(1, int(round(terrain_scale_m / pix_m))) if terrain_scale_m else 1
    if win_px % 2 == 0:
        win_px += 1

    # Scale-matched surface (NaN-aware) for the first/second derivatives.
    dem_s = nan_uniform_filter(dem, win_px) if win_px > 1 else dem

    # Gradients in meters of elevation per meter of distance.
    gy, gx = np.gradient(dem_s, py_m, px_m)
    slope = np.sqrt(gx ** 2 + gy ** 2)                     # rise/run
    aspect = np.arctan2(gy, gx)

    gyy, _ = np.gradient(gy, py_m, px_m)
    _, gxx = np.gradient(gx, py_m, px_m)
    curvature = gxx + gyy

    return {
        # TPI is a residual-from-neighborhood, so it stays on the raw DEM -
        # but now at the same window as everything else.
        "tpi": topographic_position_index(dem, win_px).astype(np.float32),
        "slope": slope.astype(np.float32),
        "aspect_sin": np.sin(aspect).astype(np.float32),
        "aspect_cos": np.cos(aspect).astype(np.float32),
        "curvature": curvature.astype(np.float32),
    }


# ----------------------------------------------------------------------
# Vector / CSV rasterization
# ----------------------------------------------------------------------

def proximity_raster(points_xy: List[Tuple[float, float]], grid: Grid,
                     decay_m: float = 5000.0) -> np.ndarray:
    """
    Distance-decay raster: 1.0 at a feature, linear decay to 0.0 at decay_m.
    Distances computed in meters regardless of CRS.
    """
    H, W = grid.shape
    presence = np.zeros((H, W), dtype=bool)
    for x, y in points_xy:
        c, r = grid.world_to_pixel(x, y)
        if 0 <= r < H and 0 <= c < W:
            presence[r, c] = True
    if not presence.any():
        return np.zeros((H, W), dtype=np.float32)

    px_m, py_m = grid.pixel_size_m()
    dist_px = ndimage.distance_transform_edt(~presence, sampling=(py_m, px_m))
    return np.clip(1.0 - dist_px / decay_m, 0.0, 1.0).astype(np.float32)


def idw_raster(points: List[Tuple[float, float, float]], grid: Grid,
               power: float = 2.0, max_points: int = 2000) -> np.ndarray:
    """
    Inverse-distance-weighted interpolation of point values (geochemistry).
    Vectorized; subsamples if there are huge numbers of points.
    """
    H, W = grid.shape
    if not points:
        return np.zeros((H, W), dtype=np.float32)

    pts = np.array(points, dtype=np.float64)
    if len(pts) > max_points:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]

    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    # pixel centers in world coords
    xs = grid.transform.c + (cols + 0.5) * grid.transform.a
    ys = grid.transform.f + (rows + 0.5) * grid.transform.e

    px_m, py_m = grid.pixel_size_m()
    sx = px_m / abs(grid.transform.a)
    sy = py_m / abs(grid.transform.e)

    num = np.zeros((H, W), dtype=np.float64)
    den = np.zeros((H, W), dtype=np.float64)
    for x, y, v in pts:
        d2 = ((xs - x) * sx) ** 2 + ((ys - y) * sy) ** 2
        w = 1.0 / np.maximum(d2, 1.0) ** (power / 2.0)
        num += w * v
        den += w
    out = num / np.maximum(den, 1e-12)
    return out.astype(np.float32)


def csv_points(path: str, value_field: Optional[str] = None
               ) -> List[Tuple[float, float, float]]:
    """
    Read (x, y, value) points from CSV. Auto-detects coordinate columns
    (lon/lat | longitude/latitude | x/y | east/north) and, if value_field is
    None, the first numeric non-coordinate column (value defaults to 1.0).
    """
    out: List[Tuple[float, float, float]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        lower = {c.lower().strip(): c for c in reader.fieldnames}

        def pick(*names):
            for n in names:
                if n in lower:
                    return lower[n]
            return None

        xcol = pick("longitude", "lon", "long", "x", "east", "easting", "dec_long")
        ycol = pick("latitude", "lat", "y", "north", "northing", "dec_lat")
        if xcol is None or ycol is None:
            return out
        vcol = lower.get(value_field.lower()) if value_field else None

        for row in reader:
            try:
                x = float(row[xcol]); y = float(row[ycol])
            except (TypeError, ValueError):
                continue
            v = 1.0
            if vcol:
                try:
                    v = float(row[vcol])
                except (TypeError, ValueError):
                    continue
            out.append((x, y, v))
    return out


# ----------------------------------------------------------------------
# Feature stack assembly
# ----------------------------------------------------------------------

class FeatureStack:
    """Named stack of aligned (H, W) features with a combined validity mask."""

    def __init__(self, grid: Grid):
        self.grid = grid
        self.names: List[str] = []
        self.layers: List[np.ndarray] = []

    def add(self, name: str, layer: np.ndarray) -> None:
        if layer.shape != self.grid.shape:
            raise ValueError(
                f"layer '{name}' shape {layer.shape} != grid {self.grid.shape}")
        if name in self.names:
            raise ValueError(f"duplicate feature name '{name}'")
        # Reject zero-information layers
        finite = layer[np.isfinite(layer)]
        if finite.size == 0 or float(np.nanstd(layer)) < 1e-12:
            return
        self.names.append(name)
        self.layers.append(layer.astype(np.float32))

    def stack(self) -> np.ndarray:
        return np.stack(self.layers, axis=-1)  # (H, W, F)

    def valid_mask(self) -> np.ndarray:
        """Pixels valid in the DEM. Per-feature NaNs are median-filled."""
        return self.grid.mask.copy()

    def matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (X, valid) where X is (H*W, F) median-filled and valid is the
        flattened DEM mask. Medians computed over valid pixels only.
        """
        S = self.stack()
        H, W, F = S.shape
        X = S.reshape(-1, F)
        for j in range(F):
            col = X[:, j]
            bad = ~np.isfinite(col)
            if bad.any():
                med = np.nanmedian(col)
                if not np.isfinite(med):
                    med = 0.0
                col[bad] = med
        return X, self.valid_mask().reshape(-1)
