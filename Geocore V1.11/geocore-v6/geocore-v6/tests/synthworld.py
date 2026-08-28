"""
synthworld.py - synthetic ground-truth worlds for testing the GeoCore engine.

Builds a fully-known geological "world" as GeoTIFFs + a deposit CSV:

  * DEM: fractal terrain (realistic ridges/valleys).
  * Magnetics: anomalies from buried intrusive stocks at KNOWN locations
    and depths (dipole-ish Gaussian anomalies scaled by depth).
  * Gravity: smooth regional field + weak local highs at stocks.
  * Geochem: Cu points elevated near true deposits.
  * Deposits: placed by a KNOWN rule per scenario.

Scenarios:
  A "signal":    deposits sit exactly on a subset of buried stocks in ALL
                 quadrants. Real, spatially-generalizable geophysical signal.
                 The engine SHOULD recover held-out deposits.
  B "confound":  deposits placed purely on high-slope terrain, no relation
                 to geophysics. Any apparent skill is a terrain artifact.
                 The engine SHOULD flag (holdout should fail).
  C "mixed":     deposits on stocks, but stocks only sampled in mountains
                 (exploration bias). Signal exists but is entangled with
                 terrain. Terrain-matching should partially rescue it.

Everything is written in EPSG:32633 (projected, meters) so pixel sizes are
exact and no geographic-CRS ambiguity enters the test.
"""
from __future__ import annotations

import csv
import numpy as np
from pathlib import Path

import rasterio
from rasterio.transform import from_origin

CRS = "EPSG:32633"
PIX = 100.0            # 100 m pixels
H, W = 600, 600        # 60 x 60 km world
X0, Y0 = 500000.0, 4600000.0   # arbitrary projected origin (top-left)

TRANSFORM = from_origin(X0, Y0, PIX, PIX)


def _write(path, arr):
    profile = dict(driver="GTiff", height=H, width=W, count=1,
                   dtype="float32", transform=TRANSFORM, crs=CRS,
                   nodata=-9999.0)
    out = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def _fractal_surface(rng, beta=2.0, amp=1.0):
    """Power-law (1/f^beta) random surface -> realistic terrain."""
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1e-6
    phase = np.exp(2j * np.pi * rng.random((H, W)))
    spec = phase / k ** (beta / 2.0)
    surf = np.real(np.fft.ifft2(spec))
    surf = (surf - surf.mean()) / surf.std()
    return amp * surf


def _pixel_xy(r, c):
    """Center-of-pixel projected coordinates."""
    x = X0 + (c + 0.5) * PIX
    y = Y0 - (r + 0.5) * PIX
    return x, y


def _mag_anomaly_field(stocks, rng):
    """Sum of Gaussian anomalies; width & amplitude scale with source depth."""
    yy, xx = np.mgrid[0:H, 0:W]
    field = np.zeros((H, W))
    for (r, c, depth_m, strength) in stocks:
        sigma_px = max(2.0, depth_m / PIX * 0.9)   # deeper -> broader
        amp = strength * np.exp(-depth_m / 2500.0)  # deeper -> weaker
        field += amp * np.exp(-(((xx - c) ** 2 + (yy - r) ** 2)
                                / (2 * sigma_px ** 2)))
    field += _fractal_surface(rng, beta=2.5, amp=0.04 * (field.std() + 1e-9))
    return field


def build_world(scenario: str, out_dir: str, seed: int = 7):
    """
    Returns dict with paths + ground truth:
      dem, magnetics, gravity, geochem_csv, deposits_csv,
      stocks [(r,c,depth,strength)], deposit_rcs [(r,c)]
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------- terrain: mountains in the N-half, basin in the S ----------
    base = _fractal_surface(rng, beta=2.2, amp=300.0)
    ramp = np.linspace(1.4, 0.2, H)[:, None]        # N (top) high, S low
    dem = 800.0 + base * ramp + 500.0 * ramp
    _write(out / "dem.tif", dem)

    # slope for placement rules (simple pixel gradient, meters)
    gy, gx = np.gradient(dem, PIX, PIX)
    slope = np.sqrt(gx ** 2 + gy ** 2)

    # ---------- buried stocks: everywhere (all quadrants) ----------
    n_stocks = 44
    stocks = []
    margin = 40
    for _ in range(n_stocks):
        r = int(rng.integers(margin, H - margin))
        c = int(rng.integers(margin, W - margin))
        depth = float(rng.uniform(400, 1500))     # m below surface
        strength = float(rng.uniform(80, 160))    # nT-ish
        stocks.append((r, c, depth, strength))

    mag = _mag_anomaly_field(stocks, rng)
    _write(out / "magnetics.tif", mag)

    # gravity: smooth regional + weak local highs at stocks
    grav_regional = _fractal_surface(rng, beta=3.2, amp=8.0)
    yy, xx = np.mgrid[0:H, 0:W]
    grav_local = np.zeros((H, W))
    for (r, c, depth_m, s) in stocks:
        sig = max(4.0, depth_m / PIX * 1.2)
        grav_local += 1.5 * np.exp(-(((xx - c) ** 2 + (yy - r) ** 2)
                                     / (2 * sig ** 2)))
    _write(out / "gravity.tif", grav_regional + grav_local)

    # ---------- deposits by scenario rule ----------
    if scenario == "signal":
        # deposits on a random subset of stocks, all quadrants
        idx = rng.permutation(n_stocks)[:30]
        deposit_rcs = [(stocks[i][0], stocks[i][1]) for i in idx]
    elif scenario == "confound":
        # deposits purely on steep terrain; geophysics irrelevant
        flat = np.argsort(slope.ravel())[::-1]
        picks, used = [], []
        for f in flat:
            r, c = divmod(int(f), W)
            if margin < r < H - margin and margin < c < W - margin:
                if all((r - ur) ** 2 + (c - uc) ** 2 > 20 ** 2
                       for ur, uc in used):
                    picks.append((r, c)); used.append((r, c))
            if len(picks) >= 30:
                break
        deposit_rcs = picks
    elif scenario == "mixed":
        # deposits on stocks, but ONLY the stocks in the mountainous N half
        north = [s for s in stocks if s[0] < H // 2]
        idx = rng.permutation(len(north))[:min(26, len(north))]
        deposit_rcs = [(north[i][0], north[i][1]) for i in idx]
    else:
        raise ValueError(scenario)

    # jitter deposits by up to 2 px (location noise, like MRDS)
    dep = []
    for (r, c) in deposit_rcs:
        rj = int(np.clip(r + rng.integers(-2, 3), 0, H - 1))
        cj = int(np.clip(c + rng.integers(-2, 3), 0, W - 1))
        dep.append((rj, cj))
    deposit_rcs = dep

    with open(out / "deposits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_name", "latitude", "longitude", "dev_stat",
                    "commod1"])
        # projected grid -> we write projected coords into lon/lat columns;
        # loader passes them through because grid CRS is projected and
        # values exceed geographic ranges.
        for i, (r, c) in enumerate(deposit_rcs):
            x, y = _pixel_xy(r, c)
            w.writerow([f"synth_{i:02d}", f"{y:.1f}", f"{x:.1f}",
                        "producer", "Copper"])

    # ---------- geochem: Cu ppm elevated near deposits ----------
    n_pts = 700
    rows = rng.integers(5, H - 5, n_pts)
    cols = rng.integers(5, W - 5, n_pts)
    with open(out / "geochem_cu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["longitude", "latitude", "cu_ppm"])
        for r, c in zip(rows, cols):
            d2 = min(((r - dr) ** 2 + (c - dc) ** 2)
                     for dr, dc in deposit_rcs)
            halo = 400.0 * np.exp(-d2 / (2 * 15.0 ** 2))
            val = float(rng.lognormal(np.log(20), 0.5) + halo
                        + rng.normal(0, 5))
            x, y = _pixel_xy(int(r), int(c))
            w.writerow([f"{x:.1f}", f"{y:.1f}", f"{max(val, 1.0):.1f}"])

    return {
        "dem": str(out / "dem.tif"),
        "magnetics": str(out / "magnetics.tif"),
        "gravity": str(out / "gravity.tif"),
        "geochem_csv": str(out / "geochem_cu.csv"),
        "deposits_csv": str(out / "deposits.csv"),
        "stocks": stocks,
        "deposit_rcs": deposit_rcs,
    }


if __name__ == "__main__":
    import sys
    sc = sys.argv[1] if len(sys.argv) > 1 else "signal"
    info = build_world(sc, f"/home/claude/geocore_test/world_{sc}")
    print(f"built '{sc}' world:", {k: v for k, v in info.items()
                                   if isinstance(v, str)})
