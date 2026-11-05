"""The recorded real-GMD fixture (tests/fixtures/gmd/, written by the patched GMD Torch build through
the adapter; regenerate with tools/fixtures/make_gmd_frame_fixture.py) read into the Frame contract
(ledger FIX-001, BE-009, MD-003 species order, TST-005 offline half)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from gmd_active_learning.adapters.gmd.engine import GMDConfig, GMDEngine, iter_frames, structure_species
from gmd_active_learning.contracts import UNITS, ModelArtifact
from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.workflow.protocols import SegmentRequest

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "gmd"
EXPORT = REPO / "tests" / "fixtures" / "sgt" / "member_11" / "model.pt"


def _raw_frames(text: str) -> list[dict]:
    """Independent minimal parser of GMD's frame lines (no ASE)."""

    lines, frames, cursor = text.splitlines(), [], 0
    while cursor < len(lines):
        n = int(lines[cursor])
        header = lines[cursor + 1]
        rows = [line.split() for line in lines[cursor + 2 : cursor + 2 + n]]
        info = dict(item.split("=", 1) for item in header.replace('"', "").split() if "=" in item and not item.startswith("Lattice"))
        frames.append({"step": int(info["step"]), "model_sha256": info["model_sha256"], "energy": float(info["energy"]),
                       "ids": [int(row[0]) for row in rows], "symbols": [row[1] for row in rows],
                       "positions": np.array([[float(x) for x in row[2:5]] for row in rows]),
                       "forces": np.array([[float(x) for x in row[5:8]] for row in rows]), "header": header})
        cursor += 2 + n
    return frames


def test_manifest_hashes_match_the_fixture_files() -> None:
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest, name
    assert manifest["model_export_sha256"] == file_sha256(EXPORT)


def test_real_gmd_frames_read_into_the_frame_contract() -> None:
    raw = _raw_frames((FIXTURE / "frames.extxyz").read_text())
    frames = list(iter_frames(FIXTURE / "frames.extxyz", run_id="fixture-run"))
    assert [frame.step for frame in frames] == [item["step"] for item in raw] == [0, 2, 4, 6]
    expected_species = structure_species(FIXTURE / "water2.xyz")
    assert expected_species == [8, 1, 1, 1, 8, 1]  # second molecule listed H-O-H: order must survive
    for frame, item in zip(frames, raw):
        assert frame.atom_ids.tolist() == item["ids"] == list(range(6))  # stable IDs = GMD tags
        assert frame.numbers.tolist() == expected_species and frame.symbols == item["symbols"]
        np.testing.assert_array_equal(frame.positions, item["positions"])
        np.testing.assert_array_equal(frame.forces, item["forces"])
        assert frame.forces.shape == (6, 3) and np.all(np.isfinite(frame.forces)) and np.abs(frame.forces).max() > 0
        assert frame.energy == item["energy"] and frame.pbc == (True, True, True)
        np.testing.assert_array_equal(frame.cell, np.eye(3) * 7.5)
        assert frame.model_sha256 == item["model_sha256"] == file_sha256(EXPORT)
        assert frame.time_fs == 0.25 * frame.step
        units = dict(pair.split(":", 1) for pair in item["header"].split('units="')[1].split('"')[0].split())
        assert (units["length"], units["energy"], units["force"], units["time"]) == (UNITS["length"], UNITS["energy"], "eV/Angstrom", UNITS["time"])


def test_completed_segment_and_read_frames_round_trip(tmp_path: Path, fake_gmd: Path) -> None:
    work = tmp_path / "segment"
    shutil.copytree(FIXTURE, work)
    (work / "segment.gmdchk").write_text("checkpoint placeholder")  # the binary checkpoint is not stored
    engine = GMDEngine(GMDConfig(executable=str(fake_gmd), initial_structure=str(FIXTURE / "water2.xyz")))
    model = ModelArtifact("m11", str(EXPORT), file_sha256(EXPORT), str(EXPORT), file_sha256(EXPORT), None, None, (1, 8), 4.0, dict(UNITS), {}, "x", "gmd_sgt", "")
    request = SegmentRequest("fixture-run", 0, work, model, 6, None, None)
    result = engine.completed_segment(request)
    assert result is not None and result.status == "completed" and result.end_step == 6
    frames = list(engine.read_frames(result, request))
    reference = list(iter_frames(FIXTURE / "frames.extxyz", run_id="fixture-run"))
    assert len(frames) == 4 and all(frame.model_version == "m11" for frame in frames)
    for ours, theirs in zip(frames, reference):
        np.testing.assert_array_equal(ours.positions, theirs.positions)
        np.testing.assert_array_equal(ours.forces, theirs.forces)


def test_input_writer_reproduces_the_recorded_run_in(tmp_path: Path, fake_gmd: Path) -> None:
    """MD-003: same structure, species order, timestep, output cadence and run length as the GMD run."""

    engine = GMDEngine(GMDConfig(executable=str(fake_gmd), initial_structure=str(FIXTURE / "water2.xyz"), frame_interval=2,
                                 directives={"time_step": 0.25, "velocity": 50.0, "velocity_init": "random", "velocity_seed": 7}))
    model = ModelArtifact("m11", str(EXPORT), file_sha256(EXPORT), str(EXPORT), file_sha256(EXPORT), None, None, (1, 8), 4.0, dict(UNITS), {}, "x", "gmd_sgt", "")
    engine.prepare(SegmentRequest("fixture-run", 0, tmp_path / "seg", model, 6, None, None))
    written = (tmp_path / "seg" / "run.in").read_text().replace(str(EXPORT), "model.pt")
    assert written == (FIXTURE / "run.in").read_text()
    assert (tmp_path / "seg" / "in.xyz").read_text() == (FIXTURE / "water2.xyz").read_text()
