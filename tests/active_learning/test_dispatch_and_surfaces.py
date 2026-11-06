"""Extraction dispatch from an event (ledger AL-001) and light package surfaces (AL-007, BE-004).

AL-001: PHAL's extract_regions_for_result grouped the trigger frame's hotspot atoms (merge radius
8 A) and dispatched to the configured mode. GMD's path is event.hotspot_atom_ids ->
workflow.assembly.hotspot_grouper (minimum-image clustering) -> extraction.dispatch.extract_regions.
For the golden events whose hotspot atoms all sit inside PHAL's regions, every mode yields the same
number of regions with identical parent-atom order and positions (5e-8 A: fixture precision).
Known divergence (documented, not asserted): when a merged hotspot spans more atoms than PHAL's
region kept (ethane evt-50/110, polyethylene, pt111 evt-30 non-block), PHAL centred the region on
part of the hotspot while GMD centres it on the whole group, so the atom sets differ.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from extraction.phal_fixtures import FIXTURE_POSITION_ATOL, MODES, ROOT, extraction_config, frame_at, phal_config, region_records
from gmd_active_learning.extraction.config import ExtractionConfig
from gmd_active_learning.extraction.dispatch import extract_regions
from gmd_active_learning.workflow.assembly import hotspot_grouper

REPO = Path(__file__).resolve().parents[2]
PARITY_EVENTS = [("ethane_box", "evt-80-00000001"), ("ethane_box", "evt-140-00000003"), ("ethane_box", "evt-190-00000004"),
                 ("pt111_co_slab", "evt-55-00000001")]


def _event(scenario: str, event_id: str) -> dict:
    return next(event for event in json.loads((ROOT / scenario / "events.json").read_text()) if event["event_id"] == event_id)


@pytest.mark.parametrize(("scenario", "event_id"), PARITY_EVENTS)
@pytest.mark.parametrize("mode", MODES)
def test_event_dispatch_matches_phal_regions(scenario: str, event_id: str, mode: str) -> None:
    event = _event(scenario, event_id)
    records = region_records(scenario, event_id, mode)
    frame = frame_at(scenario, int(records[0]["raw_region"]["metadata"]["original_frame_id"]))
    groups = hotspot_grouper(phal_config(scenario)["hotspot"]["merge_radius"])(frame, event["hotspot_atoms"])
    regions = extract_regions(frame, groups, extraction_config(scenario, mode, periodic_guard={"policy": "flag"}), event_id=event_id)
    assert len(regions) == len(records)
    for region, record in zip(regions, records):
        raw = record["raw_region"]
        assert region.parent_atom_ids.tolist() == raw["original_indices"]
        np.testing.assert_allclose(region.positions, np.asarray(raw["atoms"]["positions"]), atol=FIXTURE_POSITION_ATOL, rtol=0)
        assert region.event_id == event_id == raw["metadata"]["event_id"]
        assert region.parent_step == raw["metadata"]["original_frame_id"]
        assert region.mode == mode


def test_unknown_mode_raises_value_error() -> None:
    with pytest.raises(ValueError, match="mode"):
        ExtractionConfig.model_validate({"mode": "sphere"})
    assert issubclass(ValidationError, ValueError)


def test_package_imports_stay_light() -> None:
    """AL-007/BE-004: cp2k exposes its parse API without SCF tooling; adapters/active_learning/workflow
    import no backend eagerly; every new package imports."""

    code = """
import importlib, sys
import gmd_active_learning.adapters.dft.cp2k as cp2k
assert callable(cp2k.parse_cp2k_output_file) and callable(cp2k.validation_reasons) and cp2k.CP2KOutput
assert not [m for m in sys.modules if m.startswith('gmd_active_learning.adapters.dft.scf')], 'cp2k pulled in SCF tooling'
assert 'torch' not in sys.modules, 'cp2k pulled in torch'
import gmd_active_learning.adapters, gmd_active_learning.active_learning, gmd_active_learning.active_learning.candidates
import gmd_active_learning.workflow.controller, gmd_active_learning.workflow.store, gmd_active_learning.retraining.registry
import gmd_active_learning.retraining.triggers, gmd_active_learning.adapters.gmd.engine, gmd_active_learning.adapters.registry
bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'gmd_sgt') or m.startswith('gmd_active_learning.adapters.sgt')]
assert not bad, bad
from gmd_active_learning.adapters.registry import BackendRole, BackendRegistry
assert 'gmd_active_learning.adapters.sgt.backend' not in sys.modules  # mlip:gmd_sgt registered lazily, imported on create
print('ok')
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert completed.returncode == 0 and completed.stdout.strip() == "ok", completed.stdout + completed.stderr
