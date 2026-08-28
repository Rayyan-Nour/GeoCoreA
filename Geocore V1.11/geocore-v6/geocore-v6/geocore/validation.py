"""
Validation: spatial block cross-validation done correctly.

In v4, "spatial CV" assigned random coordinates to negative samples, so the
blocks were not spatial for half-plus of the data, and reported AUCs were
optimistically biased. Here every sample carries its true pixel location
(see sampling.Samples) and folds are contiguous geographic blocks
(Valavi et al., 2019, Methods in Ecology and Evolution).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from .sampling import Deposit, Samples


@dataclass
class CVReport:
    fold_aucs: List[float] = field(default_factory=list)
    fold_aps: List[float] = field(default_factory=list)
    n_folds_used: int = 0

    @property
    def auc_mean(self) -> float:
        return float(np.mean(self.fold_aucs)) if self.fold_aucs else float("nan")

    @property
    def auc_std(self) -> float:
        return float(np.std(self.fold_aucs)) if self.fold_aucs else float("nan")

    @property
    def ap_mean(self) -> float:
        return float(np.mean(self.fold_aps)) if self.fold_aps else float("nan")


def assign_blocks(rows: np.ndarray, cols: np.ndarray,
                  grid_shape, k: int = 5, seed: int = 42) -> np.ndarray:
    """
    Assign each sample to one of k folds by tiling the map into an n x n
    checkerboard of blocks (n*n >= 2k) and distributing whole blocks to folds.
    Spatial structure is preserved: nearby samples share a fold.
    """
    H, W = grid_shape
    n = int(np.ceil(np.sqrt(2 * k)))
    br = np.minimum((rows / max(H, 1) * n).astype(int), n - 1)
    bc = np.minimum((cols / max(W, 1) * n).astype(int), n - 1)
    block_id = br * n + bc

    rng = np.random.default_rng(seed)
    blocks = np.unique(block_id)
    rng.shuffle(blocks)
    fold_of_block = {b: i % k for i, b in enumerate(blocks)}
    return np.array([fold_of_block[b] for b in block_id], dtype=np.int32)


def spatial_cv(model, samples: Samples, grid_shape, k: int = 5,
               seed: int = 42) -> CVReport:
    """Spatial block CV. Skips folds lacking both classes (reported honestly)."""
    folds = assign_blocks(samples.rows, samples.cols, grid_shape, k, seed)
    report = CVReport()
    for f in range(k):
        test = folds == f
        train = ~test
        if len(np.unique(samples.y[train])) < 2 or \
           len(np.unique(samples.y[test])) < 2 or test.sum() < 5:
            continue
        m = clone(model)
        m.fit(samples.X[train], samples.y[train])
        p = m.predict_proba(samples.X[test])[:, 1]
        report.fold_aucs.append(float(roc_auc_score(samples.y[test], p)))
        report.fold_aps.append(float(average_precision_score(samples.y[test], p)))
        report.n_folds_used += 1
    return report


def holdout_evaluation(prob_map: np.ndarray, holdout: Sequence[Deposit],
                       valid_mask: np.ndarray, seed: int = 42,
                       n_background: int = 5000,
                       dem: np.ndarray = None,
                       slope: np.ndarray = None) -> Dict:
    """
    The honest demo metric: how do held-out deposits score versus random
    background? Reports hit rates AND a background-contrast AUC, because
    "mean probability 0.6" means nothing without knowing the background.
    """
    if not holdout:
        return {}
    H, W = prob_map.shape
    scores = np.array([prob_map[d.row, d.col] for d in holdout
                       if 0 <= d.row < H and 0 <= d.col < W], dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return {}

    rng = np.random.default_rng(seed)
    vr, vc = np.where(valid_mask & np.isfinite(prob_map))
    if vr.size == 0:
        return {}
    pick = rng.choice(vr.size, size=min(n_background, vr.size), replace=False)
    bg = prob_map[vr[pick], vc[pick]].astype(np.float64)

    y = np.concatenate([np.ones_like(scores), np.zeros_like(bg)])
    p = np.concatenate([scores, bg])
    out = {
        "count": int(scores.size),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "hit_rate_050": float((scores >= 0.50).mean()),
        "hit_rate_070": float((scores >= 0.70).mean()),
        "background_mean": float(bg.mean()),
        "contrast_auc": float(roc_auc_score(y, p)),
    }

    # Terrain-matched contrast: background drawn to match the holdout
    # deposits' elevation+slope distribution, so the metric cannot be
    # inflated by a terrain artifact (mirrors terrain-matched training).
    if dem is not None or slope is not None:
        H2, W2 = prob_map.shape
        dep_idx = np.array([d.row * W2 + d.col for d in holdout
                            if 0 <= d.row < H2 and 0 <= d.col < W2],
                           dtype=np.int64)
        cand_idx = (vr * W2 + vc).astype(np.int64)
        dem_flat = dem.reshape(-1) if dem is not None else None
        slope_flat = slope.reshape(-1) if slope is not None else None
        pick_m = _matched_background_pick(
            dem_flat, slope_flat, dep_idx, cand_idx,
            min(n_background, cand_idx.size), rng)
        if pick_m is not None:
            bg_m = prob_map.reshape(-1)[cand_idx[pick_m]].astype(np.float64)
            y2 = np.concatenate([np.ones_like(scores), np.zeros_like(bg_m)])
            p2 = np.concatenate([scores, bg_m])
            out["background_mean_matched"] = float(bg_m.mean())
            out["contrast_auc_matched"] = float(roc_auc_score(y2, p2))
    return out


def _matched_background_pick(dem_flat, slope_flat, dep_idx, cand_idx,
                             n_pick, rng, n_bins=8):
    """Sample background candidates whose (elevation, slope) distribution
    matches the holdout deposits' - removes the terrain channel from the
    contrast metric, mirroring terrain-matched training negatives."""
    covs = []
    for arr in (dem_flat, slope_flat):
        if arr is None:
            continue
        dpv = arr[dep_idx]; cnv = arr[cand_idx]
        if np.isfinite(dpv).all() and np.isfinite(cnv).all():
            covs.append((dpv, cnv))
    if not covs:
        return None
    dep_bin = np.zeros(len(dep_idx), dtype=np.int64)
    cand_bin = np.zeros(len(cand_idx), dtype=np.int64)
    mult = 1
    for dpv, cnv in covs:
        edges = np.unique(np.quantile(cnv, np.linspace(0, 1, n_bins + 1)[1:-1]))
        if edges.size == 0:
            continue
        dep_bin = dep_bin * (edges.size + 1) + np.searchsorted(edges, dpv, side="right")
        cand_bin = cand_bin * (edges.size + 1) + np.searchsorted(edges, cnv, side="right")
        mult *= (edges.size + 1)
    if mult == 1:
        return None
    dep_f = np.bincount(dep_bin, minlength=mult).astype(float) / len(dep_idx)
    cand_c = np.bincount(cand_bin, minlength=mult).astype(float)
    cand_f = cand_c / len(cand_idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        bw = np.where(cand_c > 0, dep_f / cand_f, 0.0)
    w = bw[cand_bin]
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    w = 0.999 * (w / s) + 0.001 / len(cand_idx)
    w = w / w.sum()
    return rng.choice(len(cand_idx), size=min(n_pick, len(cand_idx)),
                      replace=False, p=w)


def synthesize_verdict(cv, holdout: Dict) -> Dict:
    """One honest, headline verdict from all validation evidence.

    VALIDATED  - real, spatially generalizable signal
    NO_SIGNAL  - honest metrics show no usable model
    ARTIFACT   - looks skilled in-region but fails out-of-region
    WEAK       - inconclusive; treat as exploratory
    """
    reasons = []
    cv_ok = cv.n_folds_used > 0 and np.isfinite(cv.auc_mean)
    m_auc = holdout.get("contrast_auc_matched") if holdout else None
    u_auc = holdout.get("contrast_auc") if holdout else None
    hit = holdout.get("hit_rate_050") if holdout else None

    if cv_ok and cv.auc_mean < 0.60:
        reasons.append(f"spatial-CV AUC {cv.auc_mean:.2f} is at/near chance")
        verdict = "NO_SIGNAL"
    elif holdout:
        key = m_auc if m_auc is not None else u_auc
        decision_grade = (key is not None and key >= 0.70
                          and (hit or 0) >= 0.50)
        if decision_grade and cv_ok and cv.auc_mean >= 0.70:
            reasons.append(
                f"held-out deposits recovered at decision grade (matched "
                f"contrast {key:.2f}, hit@0.50 {100*(hit or 0):.0f}%)")
            verdict = "VALIDATED"
        elif cv_ok and cv.auc_mean >= 0.70:
            # In-region skill that does NOT carry to held-out ground at
            # decision grade. For exploration capital this must be flagged,
            # not shrugged at: conservatism is the defensible default.
            reasons.append(
                f"in-region skill (CV {cv.auc_mean:.2f}) does not carry to "
                f"held-out ground at decision grade (matched contrast "
                f"{'-' if key is None else f'{key:.2f}'}, hit@0.50 "
                f"{100*(hit or 0):.0f}%)")
            verdict = "ARTIFACT"
        else:
            reasons.append("evidence mixed across CV and holdout")
            verdict = "WEAK"
    else:
        reasons.append("no spatial holdout available")
        verdict = "WEAK" if (cv_ok and cv.auc_mean >= 0.70) else "NO_SIGNAL"
    if m_auc is not None and u_auc is not None and (u_auc - m_auc) > 0.15:
        reasons.append(
            f"uniform-background contrast ({u_auc:.2f}) exceeds terrain-"
            f"matched contrast ({m_auc:.2f}) - a terrain artifact was "
            f"inflating the naive metric")
    return {"verdict": verdict, "reasons": reasons}
