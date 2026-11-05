"""Mode dispatch and the extraction -> capping -> embedding preparation pipeline.

PHAL's labeling path ran block extraction for ``mode: block`` and cluster extraction for anything
else (finding F-05). Here every configured mode is honoured by every caller that goes through
:func:`extract_regions`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from gmd_active_learning.contracts import ExtractedRegion, Frame
from gmd_active_learning.core.exceptions import ConfigurationError
from gmd_active_learning.extraction.block import extract_block_regions
from gmd_active_learning.extraction.cluster import extract_cluster
from gmd_active_learning.extraction.config import CappingConfig, EmbeddingConfig, ExtractionConfig
from gmd_active_learning.extraction.cooldown import BlockCooldownTracker
from gmd_active_learning.extraction.embedding import build_embedding
from gmd_active_learning.extraction.graph import extract_graph
from gmd_active_learning.extraction.h_capping import add_h_caps
from gmd_active_learning.extraction.slab import extract_slab

LOGGER = logging.getLogger(__name__)

_HOTSPOT_MODES = {"cluster": extract_cluster, "slab": extract_slab, "graph": extract_graph}


def extract_regions(
    frame: Frame,
    hotspot_groups: Sequence[Sequence[int]],
    config: ExtractionConfig | None = None,
    *,
    event_id: str | None = None,
    cooldown: BlockCooldownTracker | None = None,
) -> list[ExtractedRegion]:
    """Regions for one frame. ``hotspot_groups`` are lists of stable atom IDs (one region per group
    for cluster/slab/graph; block mode uses the union of all groups as its anomalous atoms)."""

    config = config or ExtractionConfig()
    if not config.enabled:
        LOGGER.info("extraction disabled by config; no regions for step %s", frame.step)
        return []
    groups = [list(map(int, group)) for group in hotspot_groups if len(group)]
    if not groups:
        return []
    if config.mode == "block":
        ids = sorted({atom_id for group in groups for atom_id in group})
        return extract_block_regions(frame, ids, config, cooldown=cooldown, event_id=event_id)
    try:
        extractor = _HOTSPOT_MODES[config.mode]
    except KeyError as exc:  # pragma: no cover - pydantic restricts the mode
        raise ConfigurationError(f"unsupported extraction mode {config.mode!r}") from exc
    regions = []
    for index, group in enumerate(groups):
        region = extractor(frame, group, config, event_id=event_id)
        region.metadata["hotspot_group_index"] = index
        regions.append(region)
    return regions


def prepare_region(
    region: ExtractedRegion,
    frame: Frame,
    *,
    capping: CappingConfig | None = None,
    embedding: EmbeddingConfig | None = None,
) -> ExtractedRegion:
    """Capping (recorded as skipped when disabled) followed by embedding data (disabled by default)."""

    capped = add_h_caps(region, frame, capping or CappingConfig())
    return build_embedding(capped, frame, embedding or EmbeddingConfig())


def extract_and_prepare(
    frame: Frame,
    hotspot_groups: Sequence[Sequence[int]],
    *,
    extraction: ExtractionConfig | None = None,
    capping: CappingConfig | None = None,
    embedding: EmbeddingConfig | None = None,
    event_id: str | None = None,
    cooldown: BlockCooldownTracker | None = None,
) -> list[ExtractedRegion]:
    regions = extract_regions(frame, hotspot_groups, extraction, event_id=event_id, cooldown=cooldown)
    return [prepare_region(region, frame, capping=capping, embedding=embedding) for region in regions]


def load_sections(config: dict[str, Any]) -> tuple[ExtractionConfig, CappingConfig, EmbeddingConfig]:
    """Validate the three sections from a merged configuration mapping."""

    return (
        ExtractionConfig.from_mapping(config.get("extraction")),
        CappingConfig.from_mapping(config.get("capping")),
        EmbeddingConfig.from_mapping(config.get("embedding")),
    )


__all__ = ["extract_and_prepare", "extract_regions", "load_sections", "prepare_region"]
