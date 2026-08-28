"""
Deposit loading and training-sample construction.

Fixes from v4:
  * EVERY sample (positive AND negative) keeps its true pixel coordinates.
    v4 assigned *random* coordinates to negatives, which silently destroyed
    the spatial cross-validation (the headline validation metric).
  * Holdout deposits are chosen as a spatially coherent block (a quadrant),
    not arbitrary list order - this is the honest "can it find deposits in
    an area it has never seen" test.
  * Negative buffer is metric (meters), commodity-specific, literature-cited.
  * MRDS commodity matching is explicit and logged.

v6.0:
  * The negative exclusion buffer is now a TRUE metric radius (Euclidean
    distance transform with per-axis sampling). Iterative binary dilation
    grew a Manhattan diamond, leaving diagonal clearance ~22-29% short of
    the requested radius - so background could be sampled inside the
    mineralized halo. Also correct for anisotropic pixels.

v5.1:
  * Negatives are TERRAIN-MATCHED to the positives (build_samples,
    terrain_match=True by default). Uniform background sampling let a tree
    model separate deposits (which sit in the ranges) from background (mostly
    basin) on terrain alone - the slope artifact. Matching the negatives'
    elevation+slope distribution to the positives removes that shortcut and
    forces the model onto the geophysical/geochemical evidence. This makes the
    task harder on purpose: spatial-CV AUC may drop, which is the terrain
    inflation coming out, not a regression.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .raster_io import Grid


@dataclass
class Deposit:
    row: int
    col: int
    x: float
    y: float
    name: str = ""
    dev_status: str = ""


@dataclass
class Samples:
    """Training matrix with provenance for honest spatial validation."""
    X: np.ndarray                 # (N, F)
    y: np.ndarray                 # (N,)  1=deposit 0=background
    rows: np.ndarray              # (N,) pixel row of each sample
    cols: np.ndarray              # (N,) pixel col of each sample
    feature_names: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# MRDS / deposit CSV loading
# ----------------------------------------------------------------------

_COMMODITY_TOKENS = {
    "copper": {"cu", "copper"},
    "ree": {"ree", "rare earth", "rare-earth", "lanthanum", "cerium",
            "neodymium", "yttrium", "monazite", "bastnaesite", "bastnasite"},
}

# MRDS dev_stat values, roughly ordered by confidence
_STATUS_RANK = {
    "producer": 4, "past producer": 4, "past-producer": 4,
    "plant": 3, "mine": 3,
    "prospect": 2, "occurrence": 1, "": 1,
}


def _to_grid_crs(x: float, y: float, grid: Grid):
    """
    Deposit databases (MRDS/USMIN) store lat/lon; analysis grids are usually
    projected. If a coordinate looks geographic and the grid is projected,
    reproject from WGS84. Cached per-grid transformer.
    """
    if not (abs(x) <= 360 and abs(y) <= 90):
        return x, y
    from pyproj import CRS, Transformer
    crs = CRS.from_user_input(grid.crs)
    if not crs.is_projected:
        return x, y
    key = str(grid.crs)
    tr = _TO_GRID_CACHE.get(key)
    if tr is None:
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        _TO_GRID_CACHE[key] = tr
    return tr.transform(x, y)


_TO_GRID_CACHE: dict = {}


def load_deposits_shp(path: str, grid: Grid, commodity: str) -> List[Deposit]:
    """Load deposits from a POINT shapefile (MRDS/USMIN exports)."""
    from .vector_io import shapefile_deposit_points
    tokens = _COMMODITY_TOKENS.get(commodity, {commodity})
    H, W = grid.shape
    out: List[Deposit] = []
    for x, y, name, comm in shapefile_deposit_points(path, grid):
        if comm and not any(t in comm.lower() for t in tokens):
            continue
        c_px, r_px = grid.world_to_pixel(x, y)
        if 0 <= r_px < H and 0 <= c_px < W and grid.mask[r_px, c_px]:
            out.append(Deposit(row=r_px, col=c_px, x=x, y=y,
                               name=name, dev_status=""))
    return _dedupe_pixels(out)


def load_deposits_csv(path: str, grid: Grid, commodity: str,
                      min_status_rank: int = 1) -> List[Deposit]:
    """
    Load deposits from an MRDS-style CSV, keeping rows whose commodity fields
    mention the target commodity and which fall inside the analysis grid.
    """
    tokens = _COMMODITY_TOKENS.get(commodity, {commodity})
    H, W = grid.shape
    x_min, y_min, x_max, y_max = grid.bounds
    out: List[Deposit] = []

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        lower = {c.lower().strip(): c for c in reader.fieldnames}

        def col(*names):
            for n in names:
                if n in lower:
                    return lower[n]
            return None

        xcol = col("longitude", "lon", "dec_long", "x")
        ycol = col("latitude", "lat", "dec_lat", "y")
        if xcol is None or ycol is None:
            raise ValueError(f"No coordinate columns found in {path}")
        namecol = col("site_name", "name", "dep_name")
        statcol = col("dev_stat", "dev_status", "status")
        commodity_cols = [lower[c] for c in
                          ("commod1", "commod2", "commod3", "commodity",
                           "commodities", "code_list") if c in lower]

        for row in reader:
            try:
                x = float(row[xcol]); y = float(row[ycol])
            except (TypeError, ValueError):
                continue
            x, y = _to_grid_crs(x, y, grid)
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue

            if commodity_cols:
                blob = " ".join(str(row.get(c) or "") for c in commodity_cols).lower()
                if not any(t in blob for t in tokens):
                    continue

            status = str(row.get(statcol) or "").lower().strip() if statcol else ""
            if _STATUS_RANK.get(status, 1) < min_status_rank:
                continue

            c_px, r_px = grid.world_to_pixel(x, y)
            if 0 <= r_px < H and 0 <= c_px < W and grid.mask[r_px, c_px]:
                out.append(Deposit(row=r_px, col=c_px, x=x, y=y,
                                   name=str(row.get(namecol) or ""),
                                   dev_status=status))
    return _dedupe_pixels(out)


def _dedupe_pixels(deposits: List[Deposit]) -> List[Deposit]:
    """Multiple MRDS records often share a pixel; keep highest-status one."""
    best = {}
    for d in deposits:
        k = (d.row, d.col)
        if k not in best or _STATUS_RANK.get(d.dev_status, 1) > \
                _STATUS_RANK.get(best[k].dev_status, 1):
            best[k] = d
    return list(best.values())


# ----------------------------------------------------------------------
# Spatially coherent holdout
# ----------------------------------------------------------------------

def spatial_holdout(deposits: Sequence[Deposit], grid_shape: Tuple[int, int],
                    fraction: float = 0.2, seed: int = 42
                    ) -> Tuple[List[Deposit], List[Deposit]]:
    """
    Reserve a spatially coherent group of deposits (one quadrant chosen to be
    closest to `fraction` of the total) as a holdout. Returns (train, holdout).
    """
    if len(deposits) < 10 or fraction <= 0:
        return list(deposits), []

    H, W = grid_shape
    quads = {0: [], 1: [], 2: [], 3: []}
    for d in deposits:
        q = (2 if d.row >= H // 2 else 0) + (1 if d.col >= W // 2 else 0)
        quads[q].append(d)

    target = fraction * len(deposits)
    candidates = [(abs(len(v) - target), q) for q, v in quads.items() if v]
    candidates.sort()
    _, q_hold = candidates[0]
    holdout = quads[q_hold]
    # Never hold out more than half the data
    if len(holdout) > len(deposits) // 2:
        return list(deposits), []
    train = [d for q, v in quads.items() if q != q_hold for d in v]
    return train, holdout


# ----------------------------------------------------------------------
# Negative sampling
# ----------------------------------------------------------------------

def _terrain_match_weights(grid: Grid, feature_matrix: np.ndarray,
                           feature_names: Optional[Sequence[str]],
                           pos_idx: np.ndarray, cand_flat: np.ndarray,
                           n_bins: int = 8) -> Optional[np.ndarray]:
    """
    Per-candidate sampling weights that make the drawn negatives match the
    POSITIVES' joint elevation+slope distribution.

    Each terrain covariate (elevation from the DEM, slope from the feature
    stack) is cut into `n_bins` quantile bins using edges from the candidate
    pool. For joint bin b, the importance weight of a candidate is
    f_pos[b] / f_cand[b]; sampling proportional to it reproduces the positive
    terrain distribution among the negatives. A tiny uniform floor is mixed in
    so every candidate keeps a strictly positive probability (so the caller's
    without-replacement draw can always reach its target count).

    Returns None (-> caller falls back to uniform sampling) when no usable
    matching covariate is available.
    """
    names = list(feature_names or [])

    cov_pos: List[np.ndarray] = []
    cov_cand: List[np.ndarray] = []

    # Elevation (raw DEM).
    dem_flat = grid.data.reshape(-1)
    elev_pos = dem_flat[pos_idx]
    elev_cand = dem_flat[cand_flat]
    if np.isfinite(elev_pos).all() and np.isfinite(elev_cand).all():
        cov_pos.append(elev_pos)
        cov_cand.append(elev_cand)

    # Slope (if it is in the feature stack).
    if "slope" in names:
        si = names.index("slope")
        sp = feature_matrix[pos_idx, si]
        sc = feature_matrix[cand_flat, si]
        if np.isfinite(sp).all() and np.isfinite(sc).all():
            cov_pos.append(sp)
            cov_cand.append(sc)

    if not cov_pos:
        return None

    # Joint quantile bin (mixed-radix index over the covariates).
    pos_bin = np.zeros(len(pos_idx), dtype=np.int64)
    cand_bin = np.zeros(len(cand_flat), dtype=np.int64)
    mult = 1
    for cpos, ccand in zip(cov_pos, cov_cand):
        edges = np.unique(
            np.quantile(ccand, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
        if edges.size == 0:                      # constant covariate -> skip
            continue
        b_pos = np.searchsorted(edges, cpos, side="right")
        b_cand = np.searchsorted(edges, ccand, side="right")
        nb = edges.size + 1
        pos_bin = pos_bin * nb + b_pos
        cand_bin = cand_bin * nb + b_cand
        mult *= nb

    if mult == 1:
        return None

    n_pos = len(pos_idx)
    n_cand = len(cand_flat)
    pos_counts = np.bincount(pos_bin, minlength=mult).astype(np.float64)
    cand_counts = np.bincount(cand_bin, minlength=mult).astype(np.float64)
    f_pos = pos_counts / n_pos
    f_cand = cand_counts / n_cand

    with np.errstate(divide="ignore", invalid="ignore"):
        bin_w = np.where(cand_counts > 0, f_pos / f_cand, 0.0)
    w = bin_w[cand_bin]

    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return None

    # Normalize, then mix in a small uniform floor (~all mass stays matched).
    w = w / s
    w = 0.999 * w + 0.001 / n_cand
    return w / w.sum()


def build_samples(feature_matrix: np.ndarray, valid_flat: np.ndarray,
                  grid: Grid, train_deposits: Sequence[Deposit],
                  all_deposits: Sequence[Deposit],
                  buffer_m: float, negative_ratio: int = 2,
                  seed: int = 42,
                  feature_names: Optional[List[str]] = None,
                  terrain_match: bool = True,
                  match_bins: int = 8) -> Samples:
    """
    Build the training matrix.

    Negatives are drawn outside a metric buffer around ALL known deposits
    (including holdout - we must not label near-holdout pixels as background)
    and their true coordinates are retained for spatial CV.

    Terrain-matched negatives (terrain_match=True, the default)
    -----------------------------------------------------------
    Uniform background sampling makes "is this in the high country?" a clean
    separator between deposits (which sit in the ranges) and background (mostly
    basin), so a tree model learns topography instead of geology - the slope
    artifact. Here negatives are instead drawn to MATCH the elevation+slope
    distribution of the positives (stratified importance sampling over terrain
    quantile bins). With terrain comparable across the two classes, slope can
    no longer discriminate and the model is forced onto the geophysical /
    geochemical evidence.

    This deliberately makes the problem harder: expect spatial-CV AUC to DROP
    relative to uniform sampling - that drop is terrain inflation coming out,
    not a regression. The metric to watch is the held-out deposit test
    (hit rate and where the top targets land). Set terrain_match=False to
    recover the old uniform behaviour for an A/B.
    """
    H, W = grid.shape
    rng = np.random.default_rng(seed)

    pos_idx = np.array([d.row * W + d.col for d in train_deposits], dtype=np.int64)

    # Exclusion buffer as a TRUE metric radius.
    #
    # v5.1 and earlier used ndimage.binary_dilation(iterations=n), whose
    # default connectivity-1 element grows a DIAMOND (Manhattan metric), not
    # a disc: along diagonals the real clearance was only ~1/sqrt(2) of the
    # requested radius (~22-29% short), so "background" pixels could be drawn
    # INSIDE the mineralized halo that negative_buffer_m exists to exclude -
    # label contamination that quietly weakens every downstream metric.
    #
    # A Euclidean distance transform with per-axis metric sampling is exact,
    # and it also handles anisotropic pixels (px_m != py_m, normal in a
    # geographic CRS away from the equator) correctly.
    px_m, py_m = grid.pixel_size_m()

    dep_mask = np.zeros((H, W), dtype=bool)
    for d in all_deposits:
        dep_mask[d.row, d.col] = True
    dist_m = ndimage.distance_transform_edt(~dep_mask,
                                            sampling=(py_m, px_m))

    buf_m = float(buffer_m)
    buffered = dist_m < buf_m
    candidate = (~buffered) & valid_flat.reshape(H, W)
    cand_rc = np.argwhere(candidate)

    n_neg = negative_ratio * len(pos_idx)
    pix_m = max(1.0, 0.5 * (px_m + py_m))
    while len(cand_rc) < max(n_neg, 1) and buf_m > pix_m:
        # Shrink buffer rather than silently mislabeling or dying
        buf_m = max(pix_m, buf_m / 3.0)
        buffered = dist_m < buf_m
        candidate = (~buffered) & valid_flat.reshape(H, W)
        cand_rc = np.argwhere(candidate)
    n_neg = min(n_neg, len(cand_rc))
    if n_neg == 0:
        raise RuntimeError(
            "No valid background area for negative sampling - the deposit "
            "buffer covers the whole AOI. Use a larger AOI or smaller buffer.")

    cand_flat = cand_rc[:, 0] * W + cand_rc[:, 1]

    # Terrain-matched selection (falls back to uniform if unavailable).
    weights = None
    if terrain_match and len(pos_idx) >= 5:
        weights = _terrain_match_weights(
            grid, feature_matrix, feature_names, pos_idx, cand_flat,
            n_bins=match_bins)

    if weights is not None:
        pick = rng.choice(len(cand_rc), size=n_neg, replace=False, p=weights)
    else:
        pick = rng.choice(len(cand_rc), size=n_neg, replace=False)
    neg_rc = cand_rc[pick]
    neg_idx = neg_rc[:, 0] * W + neg_rc[:, 1]

    rows = np.concatenate([[d.row for d in train_deposits], neg_rc[:, 0]])
    cols = np.concatenate([[d.col for d in train_deposits], neg_rc[:, 1]])
    X = np.vstack([feature_matrix[pos_idx], feature_matrix[neg_idx]])
    y = np.concatenate([np.ones(len(pos_idx)), np.zeros(n_neg)])

    return Samples(X=X.astype(np.float32), y=y.astype(np.int8),
                   rows=rows.astype(np.int32), cols=cols.astype(np.int32),
                   feature_names=list(feature_names or []))
