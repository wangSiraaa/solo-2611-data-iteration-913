"""Independent elevation catalog and opened-outlet storage plan verifier.

Synthetic discrete model only -- not for field hydraulic decisions.
"""

from .catalog import Catalog
from .model import (
    BatchError,
    BatchResult,
    Cell,
    Outlet,
    Slice,
    SourceRef,
    TileSnapshot,
    WaterResult,
    canonical_json,
    sha256_hex,
)
from .publisher import PublishError, PublishResult, Publisher, VerifyResult
from .solver import BottleneckCache, SolveResult, SolverStats, enumerate_subsets, solve
from .xregion import (
    Curator,
    RecheckResult,
    RegionProduct,
    SchemeResult,
    Seam,
    build_product,
    full_connected,
    full_levels,
    full_volumes,
    issue_certificate,
    recheck_certificate,
)

__all__ = [
    "BatchError",
    "BatchResult",
    "BottleneckCache",
    "Catalog",
    "Cell",
    "Curator",
    "Outlet",
    "PublishError",
    "PublishResult",
    "Publisher",
    "RecheckResult",
    "RegionProduct",
    "SchemeResult",
    "Seam",
    "Slice",
    "SolveResult",
    "SolverStats",
    "SourceRef",
    "TileSnapshot",
    "VerifyResult",
    "WaterResult",
    "build_product",
    "canonical_json",
    "enumerate_subsets",
    "full_connected",
    "full_levels",
    "full_volumes",
    "issue_certificate",
    "recheck_certificate",
    "sha256_hex",
    "solve",
]
