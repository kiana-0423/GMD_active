"""Mode dispatch and the extraction -> capping -> embedding preparation pipeline.

PHAL's labeling path ran block extraction for ``mode: block`` and cluster extraction for anything
else (finding F-05). Here every configured mode is honoured by every caller that goes through
:func:`extract_regions`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from gmd_active_learning.contracts import EventRecord, ExtractedRegion, Frame
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


HotspotGrouper = Callable[[Frame, list[int]], list[list[int]]]


class EventRegionExtractor:
    """Workflow ``RegionExtractor``: event -> prepared regions scored with the event's max score.

    Regions are cut from the event's trigger frame around ``event.hotspot_atom_ids``. ``grouper``
    splits those IDs into hotspot groups (one region per group in cluster/slab/graph mode); the
    default keeps them as one group. Block cooldown state is loaded from and saved to
    ``extraction.block.cooldown_state_path`` when it is set (relative paths resolve against
    ``state_dir``), so cooldowns survive restarts.
    """

    def __init__(
        self,
        extraction: ExtractionConfig | None = None,
        capping: CappingConfig | None = None,
        embedding: EmbeddingConfig | None = None,
        *,
        grouper: HotspotGrouper | None = None,
        state_dir: str | Path | None = None,
    ) -> None:
        self.extraction = extraction or ExtractionConfig()
        self.capping = capping or CappingConfig()
        self.embedding = embedding or EmbeddingConfig()
        self.grouper = grouper
        self.cooldown: BlockCooldownTracker | None = None
        self.cooldown_path: Path | None = None
        block = self.extraction.block
        if self.extraction.mode == "block" and block.cooldown_steps > 0:
            if block.cooldown_state_path:
                path = Path(block.cooldown_state_path)
                self.cooldown_path = path if path.is_absolute() or state_dir is None else Path(state_dir) / path
                self.cooldown = BlockCooldownTracker.load(self.cooldown_path, block.cooldown_steps)
            else:
                self.cooldown = BlockCooldownTracker(block.cooldown_steps)

    def extract(self, event: EventRecord) -> list[tuple[ExtractedRegion, float]]:
        frame = event.trigger_frame
        ids = [int(atom_id) for atom_id in event.hotspot_atom_ids]
        groups = self.grouper(frame, ids) if self.grouper is not None else [ids]
        regions = extract_and_prepare(
            frame,
            groups,
            extraction=self.extraction,
            capping=self.capping,
            embedding=self.embedding,
            event_id=event.event_id,
            cooldown=self.cooldown,
        )
        if self.cooldown is not None and self.cooldown_path is not None:
            self.cooldown.save(self.cooldown_path)
        return [(region, float(event.max_score)) for region in regions]


__all__ = ["EventRegionExtractor", "HotspotGrouper", "extract_and_prepare", "extract_regions", "load_sections", "prepare_region"]
