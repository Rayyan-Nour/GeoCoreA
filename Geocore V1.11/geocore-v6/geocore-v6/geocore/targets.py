"""
Predicted-target extraction.

Turns the continuous probability map into a ranked, reviewable list of
discrete targets — the thing an exploration manager actually wants:

* Local probability maxima above the high-prospectivity threshold,
  separated by a minimum distance so one anomaly isn't listed ten times
* For each target: coordinates (grid + lat/lon), probability, model
  uncertainty, target class, spectral and Euler source depths
* WHY it was predicted: for the model's most important features, the
  percentile of that feature's value at the target versus the whole map -
  e.g. "magnetics: 94th percentile" is an honest, checkable explanation
* Grade context comes from the commodity's published grade-tonnage model
  (population statistics, never a per-pixel grade claim)

Targets are written to geocore_targets.csv and embedded in the viewers
and reports.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pyproj import Transformer

from .raster_io import Grid


def extract_targets(prob: np.ndarray,
                    unc: np.ndarray,
                    classes: np.ndarray,
                    grid: Grid,
                    X_flat: Optional[np.ndarray] = None,
                    valid_flat: Optional[np.ndarray] = None,
                    feature_names: Optional[Sequence[str]] = None,
                    importances: Optional[List[Tuple[str, float]]] = None,
                    depth_spectral: Optional[np.ndarray] = None,
                    depth_euler: Optional[np.ndarray] = None,
                    threshold: float = 0.7,
                    min_separation_px: int = 12,
                    max_targets: int = 25,
                    n_why_features: int = 5,
                    deposits: Optional[Sequence] = None) -> List[Dict]:
    """Ranked target list from the probability map."""
    H, W = prob.shape
    p = np.where(np.isfinite(prob), prob, -1.0)

    # Local maxima via maximum filter
    from scipy.ndimage import maximum_filter
    footprint = max(3, 2 * min_separation_px + 1)
    local_max = (p == maximum_filter(p, size=footprint)) & (p >= threshold)
    rows, cols = np.where(local_max)
    if len(rows) == 0:
        return []
    order = np.argsort(p[rows, cols])[::-1]
    rows, cols = rows[order], cols[order]

    # Greedy min-distance suppression (maximum_filter ties can cluster)
    kept: List[Tuple[int, int]] = []
    for r, c in zip(rows, cols):
        if all((r - kr) ** 2 + (c - kc) ** 2 >= min_separation_px ** 2
               for kr, kc in kept):
            kept.append((int(r), int(c)))
        if len(kept) >= max_targets:
            break

    # lat/lon transformer
    try:
        to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    except Exception:
        to_ll = None

    # "Why predicted": percentile of each important feature at the target
    why_features: List[Tuple[int, str]] = []
    sorted_vals: Dict[int, np.ndarray] = {}
    if (X_flat is not None and valid_flat is not None
            and feature_names and importances):
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        for fname, _ in importances[:n_why_features]:
            if fname in name_to_idx:
                fi = name_to_idx[fname]
                why_features.append((fi, fname))
                v = X_flat[valid_flat, fi]
                sorted_vals[fi] = np.sort(v[np.isfinite(v)])

    def pct(fi: int, value: float) -> Optional[int]:
        sv = sorted_vals.get(fi)
        if sv is None or len(sv) == 0 or not np.isfinite(value):
            return None
        return int(round(100.0 * np.searchsorted(sv, value) / len(sv)))

    targets: List[Dict] = []
    for rank, (r, c) in enumerate(kept, start=1):
        x = grid.transform.c + (c + 0.5) * grid.transform.a
        y = grid.transform.f + (r + 0.5) * grid.transform.e
        lon, lat = (to_ll.transform(x, y) if to_ll else (x, y))

        why = []
        if why_features:
            flat_idx = r * W + c
            for fi, fname in why_features:
                pc = pct(fi, float(X_flat[flat_idx, fi]))
                if pc is not None:
                    why.append({"feature": fname, "percentile": pc})

        def _at(arr):
            if arr is None:
                return None
            v = arr[r, c]
            return float(v) if np.isfinite(v) else None

        # nearest known deposit (metric distance)
        near_km = near_name = None
        if deposits:
            px_m, py_m = grid.pixel_size_m()
            bd = None
            for d in deposits:
                dist = (((r - d.row) * py_m) ** 2
                        + ((c - d.col) * px_m) ** 2) ** 0.5
                if bd is None or dist < bd:
                    bd = dist
                    near_name = d.name or "deposit"
            if bd is not None:
                near_km = round(bd / 1000.0, 2)

        # spectral vs Euler corroboration
        _ds = _at(depth_spectral)
        _de = _at(depth_euler)
        if _ds is not None and _de is not None:
            rel = abs(_ds - _de) / max((_ds + _de) / 2.0, 1e-6)
            depth_status = "corroborated" if rel <= 0.35 else "discrepant"
        elif _ds is not None or _de is not None:
            depth_status = "single-method"
        else:
            depth_status = None

        targets.append({
            "rank": rank, "row": r, "col": c,
            "nearest_deposit_km": near_km,
            "nearest_deposit_name": near_name,
            "depth_status": depth_status,
            "x": float(x), "y": float(y),
            "lat": float(lat), "lon": float(lon),
            "probability": float(prob[r, c]),
            "uncertainty": _at(unc),
            "target_class": int(classes[r, c])
                            if np.isfinite(classes[r, c]) else None,
            "depth_spectral_m": _at(depth_spectral),
            "depth_euler_m": _at(depth_euler),
            "why": why,
        })
    return targets


def write_targets_csv(path: str | Path, targets: List[Dict],
                      grade_statement: str = "") -> Path:
    """geocore_targets.csv - opens cleanly in Excel."""
    path = Path(path)
    why_cols = max((len(t["why"]) for t in targets), default=0)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["rank", "latitude", "longitude", "probability",
                  "uncertainty", "target_class", "depth_spectral_m",
                  "depth_euler_m", "depth_status",
                  "nearest_deposit_km", "nearest_deposit_name"]
        for i in range(why_cols):
            header += [f"driver_{i+1}", f"driver_{i+1}_percentile"]
        w.writerow(header)
        for t in targets:
            row = [t["rank"], f"{t['lat']:.6f}", f"{t['lon']:.6f}",
                   f"{t['probability']:.3f}",
                   "" if t["uncertainty"] is None
                   else f"{t['uncertainty']:.3f}",
                   t["target_class"] if t["target_class"] is not None else "",
                   "" if t["depth_spectral_m"] is None
                   else f"{t['depth_spectral_m']:.0f}",
                   "" if t["depth_euler_m"] is None
                   else f"{t['depth_euler_m']:.0f}",
                   t.get("depth_status") or "",
                   "" if t.get("nearest_deposit_km") is None
                   else f"{t['nearest_deposit_km']:.2f}",
                   t.get("nearest_deposit_name") or ""]
            for i in range(why_cols):
                if i < len(t["why"]):
                    row += [t["why"][i]["feature"],
                            t["why"][i]["percentile"]]
                else:
                    row += ["", ""]
            w.writerow(row)
        if grade_statement:
            w.writerow([])
            w.writerow(["GRADE CONTEXT", grade_statement])
            w.writerow(["NOTE", "Probabilities rank ground for follow-up; "
                                "they are not discovery guarantees. Depths "
                                "are magnetic source depths below sensor "
                                "datum, not ore depths. Only drilling "
                                "measures grade."])
    return path
