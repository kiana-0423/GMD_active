"""Feature switches, mode dispatch, package surface and default config files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from ase.build import bulk

import gmd_active_learning.extraction as extraction
from gmd_active_learning.contracts import RegionRole
from gmd_active_learning.datasets.config import DatasetsConfig
from gmd_active_learning.extraction import (
    CappingConfig,
    EmbeddingConfig,
    ExtractionConfig,
    extract_and_prepare,
    extract_regions,
    load_sections,
)

from extraction.test_extraction_modes import chain_frame, frame_of

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "sections"


@pytest.mark.parametrize("mode", ["cluster", "slab", "graph", "block"])
def test_each_mode_is_selectable(mode: str) -> None:
    frame = frame_of(bulk("Cu", "fcc", a=3.61, cubic=True).repeat(5))
    config = ExtractionConfig(mode=mode, extract_radius=4.0, core_radius=2.0, slab={"lx": 8.0, "ly": 8.0, "z_margin": 3.0}, block={"size": [6.0, 6.0, 6.0], "halo": 2.0, "buffer": {"inner": 1.5, "outer": 2.0}})
    regions = extract_regions(frame, [[0], [40]], config, event_id="evt")
    assert regions and all(region.mode == mode for region in regions)
    assert all(region.event_id == "evt" for region in regions)
    if mode != "block":
        assert len(regions) == 2


def test_extraction_disabled_gives_no_regions() -> None:
    frame = chain_frame()
    assert extract_regions(frame, [[3]], ExtractionConfig(enabled=False)) == []
    assert extract_regions(frame, [], ExtractionConfig()) == []


def test_capping_and_embedding_off_versus_on() -> None:
    frame = chain_frame()
    config = ExtractionConfig(extract_radius=2.0, core_radius=0.5, boundary_thickness=1.5)
    off = extract_and_prepare(frame, [[3]], extraction=config, capping=CappingConfig(enabled=False), embedding=EmbeddingConfig(enabled=False))[0]
    assert RegionRole.H_CAP not in off.roles
    assert off.metadata["h_capping_skipped_reason"] == "disabled_by_config"
    assert off.metadata["embedding"]["status"] == "disabled"
    on = extract_and_prepare(frame, [[3]], extraction=config, capping=CappingConfig(), embedding=EmbeddingConfig(enabled=True, default_charge=0.0))[0]
    assert RegionRole.H_CAP in on.roles
    assert on.metadata["embedding"]["status"] == "ok"


def test_default_yaml_sections_match_the_models() -> None:
    sections = {}
    for name in ("extraction", "capping", "embedding", "datasets"):
        sections.update(yaml.safe_load((CONFIGS / f"{name}.yaml").read_text()))
    extraction_cfg, capping_cfg, embedding_cfg = load_sections(sections)
    assert extraction_cfg == ExtractionConfig()
    assert capping_cfg == CappingConfig()
    assert embedding_cfg == EmbeddingConfig()
    assert DatasetsConfig.from_mapping(sections["datasets"]) == DatasetsConfig()
    with pytest.raises(ValueError):
        ExtractionConfig.from_mapping({"unknown_key": 1})


def test_package_surface() -> None:
    for name in (
        "extract_cluster", "extract_slab", "extract_graph", "extract_block_regions", "extract_block_region",
        "add_h_caps", "build_embedding", "assign_radial_regions", "region_label_vector", "BlockCooldownTracker",
        "assign_atoms_to_spatial_blocks", "merge_adjacent_blocks", "invert_block_mapping",
    ):
        assert callable(getattr(extraction, name))


def test_event_region_extractor_satisfies_the_workflow_protocol(tmp_path) -> None:
    from gmd_active_learning.contracts import EventRecord
    from gmd_active_learning.extraction import EventRegionExtractor

    frame = frame_of(bulk("Cu", "fcc", a=3.61, cubic=True).repeat(6), step=40)
    event = EventRecord(event_id="evt-7", run_id="run", trigger_step=40, trigger_steps=[40], pre_frames=[], trigger_frame=frame,
                        post_frames=[], post_frames_requested=0, complete=True, hotspot_atom_ids=[0], trigger_reasons=["force"], max_score=7.5)
    try:
        from gmd_active_learning.workflow.protocols import RegionExtractor
    except ImportError:  # pragma: no cover - workflow area not present
        RegionExtractor = None
    config = ExtractionConfig(mode="block", block={"size": [5.4, 5.4, 5.4], "cooldown_steps": 100, "cooldown_state_path": "cooldown.json"})
    extractor = EventRegionExtractor(config, state_dir=tmp_path)
    if RegionExtractor is not None:
        assert isinstance(extractor, RegionExtractor)
    regions = extractor.extract(event)
    assert len(regions) == 1 and regions[0][1] == 7.5 and regions[0][0].event_id == "evt-7"
    assert regions[0][0].metadata["embedding"]["status"] == "disabled" and "h_capping" in regions[0][0].metadata
    assert EventRegionExtractor(config, state_dir=tmp_path).extract(event) == []  # cooldown persisted
