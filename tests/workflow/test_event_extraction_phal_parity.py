"""AL-001: event -> regions through the production path reproduces PHAL for all 9 golden events.

The scenarios are replayed through GMD's staged monitor and event buffer with PHAL's scoring
conventions (``tests/monitors/phal_replay``), so the events here are GMD ``EventRecord`` objects.
Each event goes through ``EventRegionExtractor`` with ``workflow.assembly.hotspot_grouper`` (what
``build_controller`` wires) in every extraction mode, and must give PHAL's
``extract_regions_for_result`` output: region count and order, parent atom order, positions
(fixture precision) and the parent step.

PHAL extracted from the FIRST trigger's result: ``detect_hotspots(result.atom_scores >=
hotspot.threshold)`` clustered with ``merge_radius`` and sorted by max score. In the fixtures
``hotspot.threshold == ood_score.label_threshold`` (6.0), so those atoms are GMD's first-trigger
``hotspot_atom_ids``. Atoms flagged by later merged triggers (ethane evt-50/110, polyethylene
evt-50/90) are in ``event.hotspot_atom_ids`` but were flagged in later frames and are not used.
"""

from __future__ import annotations

import numpy as np
import pytest

from extraction.phal_fixtures import FIXTURE_POSITION_ATOL, MODES, extraction_config, region_records
from gmd_active_learning.contracts import EventRecord
from gmd_active_learning.events import events_config_from_phal
from gmd_active_learning.extraction.config import CappingConfig
from gmd_active_learning.extraction.dispatch import EventRegionExtractor
from gmd_active_learning.monitors import LocalLJEstimator, OnlineMonitorDriver, StagedOODMonitor, monitor_config_from_phal
from gmd_active_learning.workflow.assembly import hotspot_grouper
from monitors.phal_replay import SCENARIOS, NegatedForceLJ, load_scenario, primary_model_forces

_EVENTS: dict[str, list[EventRecord]] = {}


def _gmd_events(name: str) -> list[EventRecord]:
    if name not in _EVENTS:
        scenario = load_scenario(name)
        config = monitor_config_from_phal(scenario.config, legacy_scoring=True)
        monitor = StagedOODMonitor(
            config,
            reference_forces=primary_model_forces,
            lj_estimator=NegatedForceLJ(LocalLJEstimator(cutoff=config.neighbors.lj_cutoff)),
        )
        driver = OnlineMonitorDriver(config, events_config_from_phal(scenario.config),
                                     {"merge_radius": scenario.config["hotspot"]["merge_radius"]}, monitor=monitor)
        _EVENTS[name] = list(driver.run(scenario.frames).events)
        assert len(_EVENTS[name]) == len(scenario.events)
    return _EVENTS[name]


CASES = [(name, index, reference["event_id"]) for name in SCENARIOS for index, reference in enumerate(load_scenario(name).events)]


def test_all_nine_golden_events_are_covered() -> None:
    assert len(CASES) == 9


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("scenario", "index", "phal_event_id"), CASES)
def test_event_extractor_reproduces_phal_regions(scenario: str, index: int, phal_event_id: str, mode: str) -> None:
    event = _gmd_events(scenario)[index]
    records = region_records(scenario, phal_event_id, mode)
    assert isinstance(records, list), records  # every golden event/mode extracted without error in PHAL
    merge_radius = load_scenario(scenario).config["hotspot"]["merge_radius"]
    extractor = EventRegionExtractor(  # fresh per event: PHAL's capture had no block cooldown across events
        extraction_config(scenario, mode, periodic_guard={"policy": "flag"}),
        CappingConfig(enabled=False),
        grouper=hotspot_grouper(merge_radius),
    )
    regions = [region for region, _score in extractor.extract(event)]
    assert len(regions) == len(records), f"{scenario} {phal_event_id} {mode}"
    for region, record in zip(regions, records):
        raw = record["raw_region"]
        assert region.parent_atom_ids.tolist() == raw["original_indices"], f"{scenario} {phal_event_id} {mode}"
        np.testing.assert_allclose(region.positions, np.asarray(raw["atoms"]["positions"]), atol=FIXTURE_POSITION_ATOL, rtol=0)
        assert region.event_id == event.event_id
        assert region.parent_step == raw["metadata"]["original_frame_id"] == event.trigger_step
        assert region.mode == mode


def test_later_trigger_atoms_are_recorded_but_not_extracted() -> None:
    """ethane evt-50 merged triggers at 50/60/70: the event keeps all 7 atoms, the region uses step 50's."""

    event = _gmd_events("ethane_box")[0]
    assert event.trigger_steps == [50, 60, 70]
    assert event.hotspot_atom_ids == [6, 24, 40, 104, 109, 112, 164]
    ids, scores = EventRegionExtractor.trigger_hotspots(event)
    assert ids == [164] and scores is not None and scores.shape == (event.trigger_frame.n_atoms,)


def test_grouper_orders_regions_by_max_score_like_phal() -> None:
    """pt111 evt-30: hotspot atoms 3 and {64, 65} form two groups; PHAL put the higher-scoring CO first."""

    event = _gmd_events("pt111_co_slab")[0]
    ids, scores = EventRegionExtractor.trigger_hotspots(event)
    grouper = hotspot_grouper(load_scenario("pt111_co_slab").config["hotspot"]["merge_radius"])
    assert grouper(event.trigger_frame, ids) == [[3], [64, 65]]  # no scores: first appearance
    assert grouper(event.trigger_frame, ids, scores) == [[64, 65], [3]]


def test_events_without_trigger_metadata_fall_back_to_event_hotspots() -> None:
    """event.json files written before trigger_hotspot_atom_ids existed still extract."""

    event = _gmd_events("ethane_box")[1]
    legacy = EventRecord(**{**{slot: getattr(event, slot) for slot in EventRecord.__slots__},
                            "metadata": {"atom_scores": event.metadata["atom_scores"]}})
    ids, scores = EventRegionExtractor.trigger_hotspots(legacy)
    assert ids == event.hotspot_atom_ids
    np.testing.assert_array_equal(scores, np.asarray(event.metadata["atom_scores"]))
