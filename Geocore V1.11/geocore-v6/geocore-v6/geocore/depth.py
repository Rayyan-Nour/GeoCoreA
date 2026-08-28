"""
Depth-to-magnetic-source estimation.

This is the scientifically defensible version of "predict how deep the ore
is". Surface ML cannot see depth; potential-field geophysics can constrain
the depth of the *magnetic sources* that are often associated with intrusive
mineral systems (e.g. the stock beneath a porphyry).

Two independent, classical estimators:

1. Windowed radially-averaged power spectrum (Spector & Grant, 1970,
   Geophysics 35).  For an ensemble of sources at depth h below the
   observation surface, the radially averaged power spectrum behaves as
       ln P(k) ~ const - 2 h |k|
   so the slope of ln P vs |k| over the low-to-mid wavenumber band gives the
   mean ensemble source depth:  h = -slope / 2.

2. 3-D Euler deconvolution (Reid et al., 1990, Geophysics 55). Solves, in a
   sliding window, the homogeneity equation
       (x-x0) dT/dx + (y-y0) dT/dy + (z-z0) dT/dz = N (B - T)
   for source position (x0, y0, z0) given a structural index N
   (N=2 ~ vertical pipe/stock, appropriate for porphyry systems;
    N=3 ~ sphere/compact body; N=1 ~ dike/sill edge).
   The vertical derivative is computed spectrally (exact for harmonic fields).

Outputs are DEPTH BELOW THE MAGNETIC SENSOR DATUM. For draped airborne
surveys this approximates depth below ground; for fixed-altitude surveys
subtract terrain clearance. The report states this explicitly.

These estimate the depth of magnetic source bodies - a drilling-vector
constraint, NOT an ore-depth measurement. Both estimators are validated in
tests/test_depth.py against synthetic fields with known source depths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .raster_io import Grid


# ----------------------------------------------------------------------
# Shared spectral utilities
# ----------------------------------------------------------------------

def _fill_and_detrend(arr: np.ndarray) -> np.ndarray:
    """Median-fill NaNs and remove the best-fit plane (regional trend)."""
    out = arr.astype(np.float64).copy()
    bad = ~np.isfinite(out)
    if bad.all():
        return np.zeros_like(out)
    if bad.any():
        out[bad] = np.nanmedian(out)
    H, W = out.shape
    yy, xx = np.mgrid[0:H, 0:W]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(H * W)])
    coef, *_ = np.linalg.lstsq(A, out.ravel(), rcond=None)
    return out - (A @ coef).reshape(H, W)


def _wavenumbers(shape: Tuple[int, int], dx_m: float, dy_m: float):
    """Radial wavenumber grid |k| in rad/m."""
    H, W = shape
    ky = 2.0 * np.pi * np.fft.fftfreq(H, d=dy_m)
    kx = 2.0 * np.pi * np.fft.fftfreq(W, d=dx_m)
    KX, KY = np.meshgrid(kx, ky)
    return np.sqrt(KX ** 2 + KY ** 2), KX, KY


def vertical_derivative(field: np.ndarray, dx_m: float, dy_m: float
                        ) -> np.ndarray:
    """dT/dz via the spectral operator (multiplication by |k|)."""
    f = _fill_and_detrend(field)
    K, _, _ = _wavenumbers(f.shape, dx_m, dy_m)
    F = np.fft.fft2(f * _tukey2d(f.shape))
    return np.real(np.fft.ifft2(F * K))


def _tukey2d(shape: Tuple[int, int], alpha: float = 0.2) -> np.ndarray:
    """Separable Tukey window to suppress FFT edge leakage."""
    def tukey(n):
        if n <= 1:
            return np.ones(n)
        t = np.linspace(0, 1, n)
        w = np.ones(n)
        edge = t < alpha / 2
        w[edge] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (t[edge] - alpha / 2)))
        edge = t >= 1 - alpha / 2
        w[edge] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (t[edge] - 1 + alpha / 2)))
        return w
    return np.outer(tukey(shape[0]), tukey(shape[1]))


# ----------------------------------------------------------------------
# 1) Spectral (Spector & Grant) depth
# ----------------------------------------------------------------------

def spectral_depth(field: np.ndarray, dx_m: float, dy_m: float
                   ) -> Tuple[float, float]:
    """
    Ensemble source depth from one window (Spector & Grant, 1970).

    Implementation notes (validated in tests/test_depth.py):

    * A full Hann taper is used before the FFT.  Deep sources have an
      enormous spectral dynamic range (exp(-2|k|h)), so window-leakage
      sidelobes from weaker tapers flatten the high-k spectrum and bias
      depth estimates shallow.
    * Power is radially averaged on the *natural* FFT rings (integer
      multiples of the fundamental wavenumber) rather than coarse linear
      bins, preserving resolution at low k where deep-source signal lives.
    * The fitted band is chosen adaptively: the leakage/noise floor is the
      median log-power of the top-quartile rings, and the fit uses the
      contiguous low-k rings standing at least 2 e-units above that floor.

    Returns (depth_m, r_squared). depth is NaN if the fit is unusable.
    """
    f = _fill_and_detrend(field)
    if float(np.std(f)) < 1e-12:
        return float("nan"), 0.0

    H, W = f.shape
    F = np.fft.fft2(f * np.outer(np.hanning(H), np.hanning(W)))
    P = np.abs(F) ** 2
    K, _, _ = _wavenumbers(f.shape, dx_m, dy_m)

    k_fund = 2.0 * np.pi / (min(H, W) * max(dx_m, dy_m))
    k_max = 0.6 * np.pi / max(dx_m, dy_m)

    ring = np.round(K / k_fund).astype(int)
    n_rings = int(k_max / k_fund)
    k_mid_all, logp_all = [], []
    for n in range(1, n_rings + 1):
        sel = ring == n
        if sel.sum() >= 3:
            pm = float(P[sel].mean())
            if pm > 0:
                k_mid_all.append(n * k_fund)
                logp_all.append(np.log(pm))
    if len(k_mid_all) < 6:
        return float("nan"), 0.0
    k_mid_all = np.array(k_mid_all)
    logp_all = np.array(logp_all)

    # Leakage/noise floor: median log-power of the top-quartile rings
    q = max(3, len(logp_all) // 4)
    noise = float(np.median(logp_all[-q:]))

    # Contiguous low-k rings at least 2 e-units above the floor
    above = logp_all > noise + 2.0
    n_use = 0
    for ok in above:
        if not ok:
            break
        n_use += 1
    if n_use < 4:
        # shallow sources: spectrum may sit above floor across most rings
        idx = above
        if int(idx.sum()) < 4:
            return float("nan"), 0.0
        k_fit = k_mid_all[idx]
        logp_fit = logp_all[idx]
    else:
        k_fit = k_mid_all[:n_use]
        logp_fit = logp_all[:n_use]

    A = np.column_stack([k_fit, np.ones_like(k_fit)])
    coef, *_ = np.linalg.lstsq(A, logp_fit, rcond=None)
    slope = coef[0]
    pred = A @ coef
    ss_res = float(np.sum((logp_fit - pred) ** 2))
    ss_tot = float(np.sum((logp_fit - logp_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    depth = -slope / 2.0
    if depth <= 0:
        return float("nan"), r2
    return float(depth), float(r2)


def spectral_depth_map(grid: Grid, window_px: int = 64,
                       min_r2: float = 0.5) -> np.ndarray:
    """
    Sliding-window spectral depth map (window stride = window/2), bilinearly
    upsampled to the full grid. NaN where the fit is poor or data invalid.
    """
    data = grid.data
    H, W = data.shape
    dx_m, dy_m = grid.pixel_size_m()
    step = max(window_px // 2, 8)

    rows = list(range(0, max(H - window_px, 0) + 1, step)) or [0]
    cols = list(range(0, max(W - window_px, 0) + 1, step)) or [0]
    coarse = np.full((len(rows), len(cols)), np.nan)

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            win = data[r:r + window_px, c:c + window_px]
            if np.isfinite(win).mean() < 0.5:
                continue
            d, r2 = spectral_depth(win, dx_m, dy_m)
            if np.isfinite(d) and r2 >= min_r2:
                coarse[i, j] = d

    return _upsample_nan(coarse, (H, W))


def _upsample_nan(coarse: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Nearest-valid + bilinear upsample of a coarse NaN-bearing grid."""
    from scipy import ndimage
    H, W = shape
    ch, cw = coarse.shape
    if not np.isfinite(coarse).any():
        return np.full(shape, np.nan, dtype=np.float32)
    filled = coarse.copy()
    bad = ~np.isfinite(filled)
    if bad.any():
        idx = ndimage.distance_transform_edt(
            bad, return_distances=False, return_indices=True)
        filled = filled[tuple(idx)]
    zy, zx = H / ch, W / cw
    out = ndimage.zoom(filled, (zy, zx), order=1)
    return out[:H, :W].astype(np.float32)


# ----------------------------------------------------------------------
# 2) Euler deconvolution
# ----------------------------------------------------------------------

@dataclass
class EulerSolution:
    x: float
    y: float
    depth_m: float
    reliability: float   # 1/relative-uncertainty of the depth estimate


def euler_deconvolution(grid: Grid, structural_index: float = 2.0,
                        window_px: int = 10, max_depth_m: float = 10000.0,
                        max_solutions: int = 5000) -> list:
    """
    Sliding-window Euler deconvolution. Returns accepted EulerSolutions in
    world coordinates with depth below the sensor datum.

    Acceptance: depth in (0, max_depth_m), depth uncertainty < 30% of depth,
    solution located within twice the window of its window center.
    """
    data = _fill_and_detrend(grid.data)
    H, W = data.shape
    dx_m, dy_m = grid.pixel_size_m()

    dTdy, dTdx = np.gradient(data, dy_m, dx_m)
    dTdz = vertical_derivative(grid.data, dx_m, dy_m)

    xs = (np.arange(W) + 0.5) * dx_m
    ys = (np.arange(H) + 0.5) * dy_m
    XX, YY = np.meshgrid(xs, ys)

    sols: list = []
    step = max(window_px // 2, 2)
    N = structural_index

    for r in range(0, H - window_px, step):
        for c in range(0, W - window_px, step):
            sl = (slice(r, r + window_px), slice(c, c + window_px))
            gx = dTdx[sl].ravel(); gy = dTdy[sl].ravel(); gz = dTdz[sl].ravel()
            T = data[sl].ravel()
            # Window-centred coordinates: keeps the system well-conditioned
            # and makes the goodness-of-fit statistic reflect field
            # structure rather than the coordinate ramp.
            wc_x = xs[c + window_px // 2]; wc_y = ys[r + window_px // 2]
            x = XX[sl].ravel() - wc_x; y = YY[sl].ravel() - wc_y

            if np.std(T) < 1e-10:
                continue

            # A [x0, y0, z0, b] = rhs ; observation plane z = 0, +z down
            A = np.column_stack([gx, gy, gz, N * np.ones_like(T)])
            rhs = x * gx + y * gy + N * T
            try:
                sol, res, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank < 4:
                continue
            x0, y0, z0 = sol[0], sol[1], sol[2]
            depth = z0  # +z down with observation at z=0

            if not (0 < depth < max_depth_m):
                continue

            # Depth uncertainty from least-squares covariance
            dof = len(T) - 4
            if dof <= 0 or len(res) == 0:
                continue
            sigma2 = res[0] / dof
            try:
                cov = sigma2 * np.linalg.inv(A.T @ A)
            except np.linalg.LinAlgError:
                continue
            dz = float(np.sqrt(max(cov[2, 2], 0)))
            if dz > 0.3 * depth:
                continue

            # Goodness of fit of the homogeneity equation itself. A field
            # generated by a coherent source satisfies Euler's equation
            # almost exactly within the window; incoherent noise does not.
            ss_tot = float(np.sum((rhs - rhs.mean()) ** 2))
            if ss_tot <= 0:
                continue
            fit_r2 = 1.0 - float(res[0]) / ss_tot
            if fit_r2 < 0.95:
                continue

            # Convert offsets back to grid-local coordinates
            x0 = x0 + wc_x; y0 = y0 + wc_y

            # Reject solutions far outside their window
            win_m = window_px * max(dx_m, dy_m)
            if abs(x0 - wc_x) > 2 * win_m or abs(y0 - wc_y) > 2 * win_m:
                continue

            # Map window coords back to world
            wx = grid.transform.c + (x0 / dx_m) * grid.transform.a
            wy = grid.transform.f + (y0 / dy_m) * grid.transform.e
            sols.append(EulerSolution(x=float(wx), y=float(wy),
                                      depth_m=float(depth),
                                      reliability=float(depth / max(dz, 1e-6))))
            if len(sols) >= max_solutions:
                return sols
    return sols


def euler_depth_map(grid: Grid, solutions: list,
                    radius_px: int = 12) -> np.ndarray:
    """Rasterize Euler solutions: reliability-weighted local mean depth."""
    H, W = grid.shape
    num = np.zeros((H, W)); den = np.zeros((H, W))
    for s in solutions:
        c, r = grid.world_to_pixel(s.x, s.y)
        if not (0 <= r < H and 0 <= c < W):
            continue
        r0, r1 = max(0, r - radius_px), min(H, r + radius_px + 1)
        c0, c1 = max(0, c - radius_px), min(W, c + radius_px + 1)
        w = s.reliability
        num[r0:r1, c0:c1] += w * s.depth_m
        den[r0:r1, c0:c1] += w
    out = np.full((H, W), np.nan, dtype=np.float32)
    has = den > 0
    out[has] = (num[has] / den[has]).astype(np.float32)
    return out


def multiscale_euler(grid: Grid, structural_index: float = 2.0,
                     windows: Optional[list] = None,
                     max_depth_m: float = 10000.0,
                     max_solutions_per_window: int = 4000) -> list:
    """
    Euler deconvolution fused across several window sizes.

    A single window resolves only sources whose depth is comparable to the
    window span; multi-scale runs recover shallow and deep sources together,
    and euler_depth_map's reliability weighting fuses the solution cloud.
    On synthetic ground truth this reduces median depth error vs the single
    10-px default (see tests). Windows are clipped to the grid size.
    """
    H, W = grid.shape
    if windows is None:
        windows = [w for w in (10, 16, 24, 32) if w <= min(H, W) // 3]
        if not windows:
            windows = [max(6, min(H, W) // 3)]
    sols: list = []
    for w in windows:
        sols.extend(euler_deconvolution(
            grid, structural_index=structural_index, window_px=w,
            max_depth_m=max_depth_m,
            max_solutions=max_solutions_per_window))
    return sols
