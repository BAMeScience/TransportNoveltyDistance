"""
High-level package exports for MatSciNovelty.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="`torch_geometric.distributed` has been deprecated",
    category=DeprecationWarning,
)

from .gcn import (  # noqa: E402
    CGCNNEncoder,
    EGNNLayer,
    EquivariantCrystalGCN,
    SchNetEncoder,
    info_nce_loss,
    train_contrastive_model,
    validate,
)
from .utils import (  # noqa: E402
    StructureDataset,
    augment,
    canonicalize_structure,
    coverage_score,
    load_structures_from_json_column,
    load_wyckoff_structures,
    novelty_score,
    perturb_structures,
    perturb_structures_corrupt,
    perturb_structures_gaussian,
    read_csv,
    read_structure_from_csv,
    structure_to_graph,
)
from .wasserstein_novelty import OTNoveltyScorer  # noqa: E402

__all__ = [
    "CGCNNEncoder",
    "EGNNLayer",
    "EquivariantCrystalGCN",
    "SchNetEncoder",
    "canonicalize_structure",
    "info_nce_loss",
    "train_contrastive_model",
    "validate",
    "StructureDataset",
    "augment",
    "coverage_score",
    "load_structures_from_json_column",
    "load_wyckoff_structures",
    "novelty_score",
    "perturb_structures",
    "perturb_structures_corrupt",
    "perturb_structures_gaussian",
    "read_csv",
    "read_structure_from_csv",
    "structure_to_graph",
    "OTNoveltyScorer",
]
