"""
Shared synthetic-world builders for the test suite.

The synthetic world has KNOWN ground truth: deposits are planted where a
hidden function of the evidence layers is high, and magnetic sources are
buried at known depths. If the engine can't recover what we planted, the
tests fail.
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geocore.raster_io import Grid


def make_grid(H=200, W=200, pixel_m=100.0, x0=500_000.0, y0=4_000_000.0,
              crs="EPSG:32612") -> Grid:
    transform = from_origin(x0, y0, pixel_m, pixel_m)
    data = np.zeros((H, W), dtype=np.float32)
    mask = np.ones((H, W), dtype=bool)
    return Grid(data=data, mask=mask, transform=transform,
                crs=rasterio.crs.CRS.from_string(crs))


def write_tif(path, data, grid: Grid, nodata=-9999.0):
    H, W = data.shape
    out = np.where(np.isfinite(data), data, nodata).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=1,
        dtype="float32", transform=grid.transform, crs=grid.crs,
        nodata=nodata,
    ) as dst:
        dst.write(out, 1)


def synthetic_dem(H=200, W=200, seed=0):
    """Smooth random terrain, 500-1500 m elevation."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(H // 8 + 2, W // 8 + 2))
    from scipy import ndimage
    z = ndimage.zoom(base, (H / base.shape[0], W / base.shape[1]), order=3)
    z = ndimage.gaussian_filter(z, 3)[:H, :W]
    z = 1000 + 500 * (z - z.min()) / max(float(np.ptp(z)), 1e-9)
    return z.astype(np.float32)


def field_from_sources_at_depth(H, W, pixel_m, depth_m, seed=0):
    """
    Magnetic-like field whose radially averaged power spectrum decays as
    exp(-2|k|h): white-noise sources upward-continued by depth_m. This makes
    the spectral depth recoverable EXACTLY in expectation.
    """
    rng = np.random.default_rng(seed)
    src = rng.normal(size=(H, W))
    ky = 2 * np.pi * np.fft.fftfreq(H, d=pixel_m)
    kx = 2 * np.pi * np.fft.fftfreq(W, d=pixel_m)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    F = np.fft.fft2(src) * np.exp(-K * depth_m)
    return np.real(np.fft.ifft2(F)).astype(np.float32)


def point_pole_field(H, W, pixel_m, sources):
    """
    Field of monopole-like sources: T = m * h / (r^2 + h^2)^{3/2}.
    A monopole has Euler structural index N=2... actually N=2 corresponds
    to a pole for the field magnitude; we use it consistently with the
    SI passed to the Euler test. `sources` = list of (row, col, depth_m,
    moment).
    """
    ys = (np.arange(H) + 0.5) * pixel_m
    xs = (np.arange(W) + 0.5) * pixel_m
    XX, YY = np.meshgrid(xs, ys)
    T = np.zeros((H, W))
    for r, c, h, m in sources:
        sx, sy = (c + 0.5) * pixel_m, (r + 0.5) * pixel_m
        R2 = (XX - sx) ** 2 + (YY - sy) ** 2
        T += m * h / np.power(R2 + h ** 2, 1.5)
    return T.astype(np.float32)


def plant_deposits(prospectivity_truth, n=60, threshold_q=0.92, seed=0):
    """Plant deposits preferentially where hidden truth is high."""
    rng = np.random.default_rng(seed)
    H, W = prospectivity_truth.shape
    thr = np.quantile(prospectivity_truth, threshold_q)
    rows, cols = np.where(prospectivity_truth >= thr)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return list(zip(rows[pick].tolist(), cols[pick].tolist()))


def deposits_to_csv(path, pixel_deposits, grid: Grid, commodity="copper"):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_name", "latitude", "longitude", "commod1", "dev_stat"])
        for i, (r, c) in enumerate(pixel_deposits):
            x = grid.transform.c + (c + 0.5) * grid.transform.a
            y = grid.transform.f + (r + 0.5) * grid.transform.e
            w.writerow([f"SYN-{i:03d}", y, x,
                        "Copper" if commodity == "copper" else "REE",
                        "Producer" if i % 3 == 0 else "Prospect"])
