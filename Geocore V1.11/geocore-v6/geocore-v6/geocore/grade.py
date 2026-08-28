"""
Grade-tonnage CONTEXT (deliberately not a "grade map").

v4 painted a per-pixel "grade" raster by pushing the prospectivity
probability through the inverse CDF of the USGS grade distribution. That is
scientifically indefensible: prospectivity probability is not a grade
percentile, and a map of Cu% values reads as assay data no matter how many
disclaimers surround it. A competent reviewer at any major will reject it -
and under securities rules (e.g. NI 43-101 / JORC), implying grade without
drilling is the kind of thing that ends companies.

What IS defensible, and what this module provides:
  * The published grade-tonnage distribution for the deposit type
    (Singer et al., 2008 for porphyry Cu), as percentile statistics:
    "IF a porphyry deposit exists here, deposits of this type historically
    grade P10=0.25%, P50=0.44%, P90=0.80% Cu."
  * An optional categorical CONFIDENCE raster (high/medium/low prospectivity
    classes) for map display - classes, not fake assays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

from .config import CommoditySpec


@dataclass
class GradeContext:
    unit: str
    reference: str
    percentiles: Dict[str, float]   # e.g. {"P10": 0.25, "P50": 0.44, "P90": 0.80}
    statement: str


def grade_context(spec: CommoditySpec) -> GradeContext:
    if spec.grade_log10_mean is not None:
        mu, sd = spec.grade_log10_mean, spec.grade_log10_std
        pct = {f"P{p}": float(10 ** norm.ppf(p / 100, loc=mu, scale=sd))
               for p in (10, 25, 50, 75, 90)}
        statement = (
            f"IF a {spec.deposit_type} deposit exists in a highlighted zone, "
            f"deposits of this type historically grade "
            f"P10={pct['P10']:.2f}, P50={pct['P50']:.2f}, "
            f"P90={pct['P90']:.2f} {spec.unit_str()}. "
            "These are population statistics from the cited reference, not "
            "predictions for this site. Grade can only be established by "
            "drilling and assay."
        )
    elif spec.grade_range is not None:
        lo, hi = spec.grade_range
        pct = {"min": float(lo), "max": float(hi),
               "typical": float((lo + hi) / 4)}
        statement = (
            f"Economic {spec.deposit_type} deposits historically range "
            f"~{lo}-{hi} {spec.unit_str()} ({spec.grade_reference}). "
            "Population reference only; site grade requires drilling."
        )
    else:
        pct = {}
        statement = "No published grade-tonnage model configured."
    return GradeContext(unit=spec.grade_unit, reference=spec.grade_reference,
                        percentiles=pct, statement=statement)


def prospectivity_classes(prob_map: np.ndarray,
                          thresholds=(0.5, 0.7)) -> np.ndarray:
    """
    Categorical display raster: 0=background, 1=moderate, 2=high.
    NaN preserved. Thresholds are on calibrated probability.
    """
    out = np.full(prob_map.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(prob_map)
    out[finite] = 0.0
    out[finite & (prob_map >= thresholds[0])] = 1.0
    out[finite & (prob_map >= thresholds[1])] = 2.0
    return out


# Convenience so f-strings above stay clean
def _unit_str(self: CommoditySpec) -> str:
    return self.grade_unit


CommoditySpec.unit_str = _unit_str
