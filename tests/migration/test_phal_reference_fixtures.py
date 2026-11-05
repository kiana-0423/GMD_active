"""Integrity checks for the PHAL golden outputs in tests/fixtures/phal_reference/.

The fixtures are the parity oracle for the migrated monitoring, event, extraction,
mask and dataset code. These tests make sure the oracle itself is intact and
internally consistent; the re-capture test (run only while ref/lammps_monitor
exists) proves the fixtures are exactly what PHAL produces.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "phal_reference"
REFERENCE_TREE = ROOT / "ref" / "lammps_monitor"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
SCENARIOS = MANIFEST["scenarios"]


def test_every_fixture_file_matches_the_manifest() -> None:
    on_disk = {
        path.relative_to(FIXTURES).as_posix()
        for path in FIXTURES.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert on_disk == set(MANIFEST["files"])
    for relative, digest in MANIFEST["files"].items():
        assert hashlib.sha256((FIXTURES / relative).read_bytes()).hexdigest() == digest, relative


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_fixtures_are_internally_consistent(scenario: str) -> None:
    root = FIXTURES / scenario
    frames = read(root / "input_frames.extxyz", index=":")
    results = json.loads((root / "monitor_results.json").read_text(encoding="utf-8"))
    events = json.loads((root / "events.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    committee_size = len(config["allegro"]["model_paths"])

    assert len(results) == len(frames)
    with np.load(root / "monitor_arrays.npz") as arrays:
        for index, (atoms, record) in enumerate(zip(frames, results)):
            assert atoms.info["step"] == record["step"]
            # ASE's extxyz reader moves the "forces" column onto a SinglePointCalculator.
            assert atoms.get_forces().shape == (len(atoms), 3)
            for member in range(committee_size):
                assert atoms.arrays[f"committee_forces_{member}"].shape == (len(atoms), 3)
            assert arrays[f"frame{index:03d}/atom_scores"].shape == (len(atoms),)
            for metric in record["evaluated_metrics"]:
                assert arrays[f"frame{index:03d}/metric/{metric}"].shape == (len(atoms),)
            assert all(0 <= atom < len(atoms) for atom in record["hotspot_indices"])

    steps = {record["step"] for record in results}
    triggered = {record["step"] for record in results if record["triggered"]}
    assert events, "every scenario is built to produce at least one event"
    for event in events:
        assert event["step"] in triggered
        assert set(event["pre_frame_steps"]) | set(event["post_frame_steps"]) <= steps

    for event in events:
        for mode in ("cluster", "slab", "graph", "block"):
            payload = json.loads((root / "regions" / event["event_id"] / f"{mode}.json").read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                assert set(payload) == {"error"}
                continue
            for record in payload:
                raw = record["raw_region"]
                n_raw = len(raw["atoms"]["symbols"])
                assert len(raw["original_indices"]) == n_raw
                assert all(0 <= index < len(frames[0]) for index in raw["original_indices"])
                if record["capping_error"] is not None:
                    continue
                capped = record["capped_region"]
                n_capped = len(capped["atoms"]["symbols"])
                assert n_capped == n_raw + len(capped["h_cap_indices"])
                assert len(record["mask"]) == n_capped
                assert len(record["region_labels"]) == len(record["region_codes"])


def test_known_phal_defects_are_pinned() -> None:
    """Recorded defects must stay visible until the migrated code fixes them deliberately."""

    block_records = [
        record
        for scenario in SCENARIOS
        for path in sorted((FIXTURES / scenario / "regions").glob("*/block.json"))
        for record in json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict) and record.get("capped_region") and record["capped_region"]["h_cap_indices"]
    ]
    assert block_records, "fixtures must contain capped block regions"
    for record in block_records:
        # PHAL's add_h_caps does not extend metadata["atom_role"], which generate_region_labels
        # prefers for block regions, so region codes are shorter than the capped atom list.
        assert len(record["region_labels"]) < len(record["capped_region"]["atoms"]["symbols"])
        assert record["dataset_error"]["type"] == "ValueError"


@pytest.mark.skipif(not REFERENCE_TREE.is_dir(), reason="PHAL reference tree has been removed")
def test_recapture_reproduces_the_fixtures(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "migration_capture", ROOT / "tools" / "migration" / "capture_phal_reference.py"
    )
    assert spec is not None and spec.loader is not None
    capture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = capture
    spec.loader.exec_module(capture)
    manifest_path = capture.capture_all(tmp_path, REFERENCE_TREE, list(SCENARIOS))
    fresh = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert fresh["phal_commit"] == MANIFEST["phal_commit"]
    assert fresh["files"] == MANIFEST["files"]
