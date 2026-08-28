"""
Configuration for the GeoCore pipeline.

Replaces the v4 environment-variable soup and hardcoded Windows paths with a
single validated, serializable config object. Every run can be reproduced from
its saved config JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CommoditySpec:
    """Literature-grounded parameters for a commodity / deposit model."""
    key: str
    label: str
    deposit_type: str
    min_deposits_for_training: int
    negative_buffer_m: float          # exclusion buffer around known deposits
    # Grade-tonnage context (NOT predictions). For copper: Singer et al. 2008
    # lognormal in log10(Cu %). For REE: Verplanck et al. 2014 ranges.
    grade_log10_mean: Optional[float] = None
    grade_log10_std: Optional[float] = None
    grade_range: Optional[tuple] = None
    grade_unit: str = "%"
    grade_reference: str = ""


COMMODITIES = {
    "copper": CommoditySpec(
        key="copper",
        label="Copper (Cu)",
        deposit_type="Porphyry Copper",
        min_deposits_for_training=20,
        # Porphyry systems are 1-2 km scale (Sillitoe 2010). 1500 m keeps
        # negatives outside the mineralized halo.
        negative_buffer_m=1500.0,
        grade_log10_mean=-0.357,
        grade_log10_std=0.227,
        grade_unit="% Cu",
        grade_reference="Singer, Berger & Moring (2008) USGS OFR 2008-1155",
    ),
    "ree": CommoditySpec(
        key="ree",
        label="Rare Earth Elements (REE)",
        deposit_type="Carbonatite-hosted REE",
        min_deposits_for_training=5,
        negative_buffer_m=5000.0,
        grade_range=(0.5, 10.0),
        grade_unit="% TREO",
        grade_reference="Verplanck et al. (2014); Woolley & Kjarsgaard (2008)",
    ),
}


@dataclass
class PipelineConfig:
    """Everything needed to run (and re-run) an analysis."""

    # --- inputs ---
    dem_path: str = ""
    deposit_csv: str = ""                  # MRDS-style CSV (or USMIN export)
    feature_rasters: dict = field(default_factory=dict)   # name -> path (.tif)
    feature_vectors: dict = field(default_factory=dict)   # name -> path (.shp/.csv)
    magnetics_layer: str = "magnetics"     # feature name used for depth estimation

    # --- analysis ---
    commodity: str = "copper"
    crop_mode: str = "full"                # 'full' or 'center'
    analysis_max_px: int = 1200            # cap longest grid side (0 = native)
    random_seed: int = 42
    n_trees: int = 300
    negative_ratio: int = 2                # negatives per positive (Dong et al. 2024)
    spatial_blocks: int = 5                # k for spatial block CV
    blend_system_prior: float = 0.0        # OFF by default. If >0, the blend is
                                           # disclosed in every report/output.
    holdout_fraction: float = 0.2          # spatially-coherent deposit holdout
    min_valid_fraction: float = 0.05       # raster must cover >=5% of AOI

    # --- depth estimation ---
    estimate_depth: bool = True
    depth_window_px: int = 64              # spectral window (power of 2)
    euler_structural_index: float = 2.0    # SI=2 ~ vertical pipe (porphyry stock)

    # --- outputs ---
    results_dir: str = "results_v5"

    def spec(self) -> CommoditySpec:
        if self.commodity not in COMMODITIES:
            raise ValueError(
                f"Unknown commodity '{self.commodity}'. "
                f"Supported: {sorted(COMMODITIES)}"
            )
        return COMMODITIES[self.commodity]

    def validate(self) -> list:
        """Return a list of human-readable problems (empty == valid)."""
        problems = []
        if not self.dem_path:
            problems.append("dem_path is required")
        elif not Path(self.dem_path).exists():
            problems.append(f"DEM not found: {self.dem_path}")
        if self.commodity not in COMMODITIES:
            problems.append(f"unknown commodity '{self.commodity}'")
        if self.deposit_csv and not Path(self.deposit_csv).exists():
            problems.append(f"deposit CSV not found: {self.deposit_csv}")
        for name, p in {**self.feature_rasters, **self.feature_vectors}.items():
            if p and not Path(p).exists():
                problems.append(f"feature '{name}' not found: {p}")
        if not (0.0 <= self.blend_system_prior <= 0.5):
            problems.append("blend_system_prior must be in [0, 0.5]")
        if self.depth_window_px & (self.depth_window_px - 1):
            problems.append("depth_window_px must be a power of two")
        return problems

    # --- persistence ---
    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
