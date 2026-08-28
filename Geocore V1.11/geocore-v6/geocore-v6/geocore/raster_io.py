"""
Raster I/O and grid alignment built on rasterio.

Fixes from v4:
  * Nodata is MASKED, never replaced with 0. (In v4, nodata -> 0 silently
    injected fake "zero anomaly" values into magnetics/gravity features.)
  * All feature rasters are reprojected/resampled onto the DEM grid with
    rasterio's warp machinery instead of ad-hoc crop arithmetic.
  * Geographic vs projected CRS handled explicitly; pixel size in meters is
    computed correctly in both cases.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import Affine


@dataclass
class Grid:
    """An aligned analysis grid: data + georeferencing + validity mask."""
    data: np.ndarray            # float32 (H, W)
    mask: np.ndarray            # bool (H, W) - True where data is VALID
    transform: Affine
    crs: object
    nodata: Optional[float] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        H, W = self.data.shape
        x_min, y_max = self.transform * (0, 0)
        x_max, y_min = self.transform * (W, H)
        return x_min, y_min, x_max, y_max

    def pixel_size_m(self) -> Tuple[float, float]:
        """Pixel size in meters, handling geographic CRS correctly."""
        px = abs(self.transform.a)
        py = abs(self.transform.e)
        if self.crs and getattr(self.crs, "is_geographic", False):
            x_min, y_min, x_max, y_max = self.bounds
            lat_c = 0.5 * (y_min + y_max)
            m_per_deg_lat = 111_320.0
            m_per_deg_lon = 111_320.0 * max(math.cos(math.radians(lat_c)), 1e-6)
            return px * m_per_deg_lon, py * m_per_deg_lat
        return px, py

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        col, row = ~self.transform * (x, y)
        return int(col), int(row)


def load_dem(path: str, crop_mode: str = "full",
             max_px: int = 0) -> Grid:
    """
    Load the DEM that defines the analysis grid.

    max_px > 0 caps the longest side via a decimated (overview) read - the
    standard way to run regional prospectivity at a working resolution
    instead of grinding through every native DEM pixel. The transform is
    rescaled so all georeferencing stays exact.
    """
    with rasterio.open(path) as ds:
        H0, W0 = ds.height, ds.width
        if max_px and max(H0, W0) > max_px:
            scale = max(H0, W0) / float(max_px)
            out_h = max(1, int(round(H0 / scale)))
            out_w = max(1, int(round(W0 / scale)))
            data = ds.read(1, out_shape=(out_h, out_w),
                           resampling=Resampling.average).astype(np.float32)
            transform = ds.transform * Affine.scale(W0 / out_w, H0 / out_h)
        else:
            data = ds.read(1).astype(np.float32)
            transform = ds.transform
        nodata = ds.nodata
        crs = ds.crs

    H, W = data.shape
    if crop_mode == "center":
        sy, sx = H // 4, W // 4
        ey, ex = sy + H // 2, sx + W // 2
        data = data[sy:ey, sx:ex]
        transform = transform * Affine.translation(sx, sy)

    mask = np.isfinite(data)
    if nodata is not None:
        mask &= data != nodata
    # Defensive: catch unflagged sentinel values
    mask &= np.abs(data) < 1e30

    clean = np.where(mask, data, np.float32(np.nan))
    return Grid(data=clean, mask=mask, transform=transform, crs=crs, nodata=nodata)


def load_aligned(path: str, target: Grid,
                 resampling: Resampling = Resampling.bilinear) -> Optional[Grid]:
    """
    Load any raster and warp it onto the target grid (same CRS, transform,
    shape). Returns None if it does not overlap the target at all.
    """
    H, W = target.shape
    out = np.full((H, W), np.nan, dtype=np.float32)

    with rasterio.open(path) as src:
        src_data = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        if src_nodata is not None:
            src_data[src_data == src_nodata] = np.nan
        src_data[np.abs(src_data) > 1e30] = np.nan

        reproject(
            source=src_data,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs or target.crs,
            dst_transform=target.transform,
            dst_crs=target.crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    mask = np.isfinite(out)
    if not mask.any():
        return None
    return Grid(data=out, mask=mask, transform=target.transform, crs=target.crs)


def save_geotiff(data: np.ndarray, path: str, like: Grid,
                 nodata: float = -9999.0) -> None:
    """Write a float32 GeoTIFF on the analysis grid, nodata-aware."""
    H, W = data.shape
    out = np.where(np.isfinite(data), data, nodata).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "dtype": "float32",
        "transform": like.transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    if like.crs is not None:
        profile["crs"] = like.crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
