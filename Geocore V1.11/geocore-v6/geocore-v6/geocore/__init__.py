"""
GeoCore Analytics v5 - Mineral Prospectivity Engine
====================================================

A tested, literature-grounded engine for mineral prospectivity mapping:

- Random Forest prospectivity modeling with *correct* spatial cross-validation
- Uncertainty quantification (tree-ensemble dispersion)
- Depth-to-magnetic-source estimation (spectral + Euler deconvolution)
- Honest grade-tonnage context (USGS models, never presented as assay)
- Project database (SQLite), reproducible configs, full provenance

Methodological references:
  Carranza & Laborte (2015) Ore Geology Reviews 71
  Zuo & Wang (2020) Natural Resources Research 29
  Singer, Berger & Moring (2008) USGS OFR 2008-1155
  Spector & Grant (1970) Geophysics 35(2) - spectral depth estimation
  Reid et al. (1990) Geophysics 55(1) - Euler deconvolution
  Valavi et al. (2019) Methods Ecol Evol - spatial block CV
  Sillitoe (2010) Economic Geology 105 - porphyry systems
"""

__version__ = "6.0.0"

from .config import PipelineConfig, CommoditySpec, COMMODITIES
from .pipeline import run_pipeline, PipelineResult

__all__ = [
    "PipelineConfig",
    "CommoditySpec",
    "COMMODITIES",
    "run_pipeline",
    "PipelineResult",
    "__version__",
]
