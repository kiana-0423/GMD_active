"""Region extraction (cluster, slab, graph, block), H capping and embedding preparation.

All extractors take a :class:`~gmd_active_learning.contracts.Frame` plus stable hotspot atom IDs
and return :class:`~gmd_active_learning.contracts.ExtractedRegion` objects carrying parent atom IDs
and integer image shifts. PHAL names map as follows: ``extract_cluster_region`` ->
:func:`extract_cluster`, ``extract_slab_patch`` -> :func:`extract_slab`, ``extract_graph_region``
-> :func:`extract_graph`; ``add_h_caps`` now takes ``(region, frame, config)``; ``build_embedding``
returns a new region instead of mutating its argument.
"""

from gmd_active_learning.extraction.block import (
    anomalous_blocks,
    assign_atoms_to_spatial_blocks,
    block_grid_shape,
    extract_block_region,
    extract_block_regions,
    invert_block_mapping,
    merge_adjacent_blocks,
)
from gmd_active_learning.extraction.cluster import extract_cluster
from gmd_active_learning.extraction.config import (
    BlockConfig,
    CappingConfig,
    EmbeddingConfig,
    ExtractionConfig,
    GraphConfig,
    PeriodicGuardConfig,
    SlabConfig,
)
from gmd_active_learning.extraction.cooldown import BlockCooldownTracker
from gmd_active_learning.extraction.dispatch import extract_and_prepare, extract_regions, load_sections, prepare_region
from gmd_active_learning.extraction.electronic import resolve_region_electronic_state
from gmd_active_learning.extraction.embedding import EMBEDDING_CAPABILITIES, build_embedding, require_embedding_backend
from gmd_active_learning.extraction.graph import extract_graph
from gmd_active_learning.extraction.h_capping import add_h_caps, assess_capping, check_labelable
from gmd_active_learning.extraction.regions import assign_radial_regions, radial_roles, region_label_vector
from gmd_active_learning.extraction.slab import extract_slab

__all__ = [
    "EMBEDDING_CAPABILITIES",
    "BlockConfig",
    "BlockCooldownTracker",
    "CappingConfig",
    "EmbeddingConfig",
    "ExtractionConfig",
    "GraphConfig",
    "PeriodicGuardConfig",
    "SlabConfig",
    "add_h_caps",
    "anomalous_blocks",
    "assess_capping",
    "assign_atoms_to_spatial_blocks",
    "assign_radial_regions",
    "block_grid_shape",
    "build_embedding",
    "check_labelable",
    "extract_and_prepare",
    "extract_block_region",
    "extract_block_regions",
    "extract_cluster",
    "extract_graph",
    "extract_regions",
    "extract_slab",
    "invert_block_mapping",
    "load_sections",
    "merge_adjacent_blocks",
    "prepare_region",
    "radial_roles",
    "region_label_vector",
    "require_embedding_backend",
    "resolve_region_electronic_state",
]
