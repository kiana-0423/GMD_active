"""Real GMD (Torch build) write -> run -> read round trip with a mixed-species system (ledger MD-003,
MD-005, TST-005). Skipped only when the GMD binary is unavailable."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.adapters.gmd.engine import GMDConfig, GMDEngine, iter_frames, structure_species
from gmd_active_learning.contracts import UNITS, FrameAction, ModelArtifact
from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.core.exceptions import UnsupportedCombinationError
from gmd_active_learning.workflow.protocols import SegmentRequest

pytestmark = pytest.mark.integration
REPO = Path(__file__).resolve().parents[2]
GMD_BIN = Path(os.environ.get("GMD_EXECUTABLE", REPO.parent / "GMD" / "build-torch" / "gmd"))
FIXTURE = REPO / "tests" / "fixtures" / "gmd"
EXPORT = REPO / "tests" / "fixtures" / "sgt" / "member_11" / "model.pt"


def _engine() -> GMDEngine:
    if not GMD_BIN.is_file():
        pytest.skip(f"GMD binary not found at {GMD_BIN} (build GMD with -DGMD_ENABLE_TORCH=ON)")
    return GMDEngine(GMDConfig(executable=str(GMD_BIN), initial_structure=str(FIXTURE / "water2.xyz"), frame_interval=2,
                               directives={"time_step": 0.25, "velocity": 50.0, "velocity_init": "random", "velocity_seed": 7}))


def _model(species: tuple[int, ...]) -> ModelArtifact:
    return ModelArtifact("m11", str(EXPORT), file_sha256(EXPORT), str(EXPORT), file_sha256(EXPORT), None, None, species, 4.0, dict(UNITS), {}, "x", "gmd_sgt", "")


def test_species_order_survives_a_real_gmd_run_and_matches_the_recorded_fixture(tmp_path: Path) -> None:
    engine = _engine()
    frames: list = []
    result = engine.run_segment(SegmentRequest("fixture-run", 0, tmp_path / "seg", _model((1, 8)), 6, None, None),
                                lambda frame: frames.append(frame) or FrameAction.CONTINUE)
    assert result.status == "completed" and result.end_step == 6
    species = structure_species(FIXTURE / "water2.xyz")
    assert all(frame.numbers.tolist() == species for frame in frames)  # O H H H O H, not sorted or remapped
    assert all(frame.energy is not None and np.isfinite(frame.energy) for frame in frames)
    recorded = list(iter_frames(FIXTURE / "frames.extxyz", run_id="fixture-run"))
    assert [f.step for f in frames] == [f.step for f in recorded]
    for ours, theirs in zip(frames, recorded):  # deterministic (seeded velocities, serial TorchScript)
        np.testing.assert_allclose(ours.positions, theirs.positions, atol=1e-8, rtol=0)
        np.testing.assert_allclose(ours.forces, theirs.forces, atol=1e-6, rtol=0)


def test_a_model_without_the_systems_species_is_refused_before_gmd_starts(tmp_path: Path) -> None:
    engine = _engine()
    with pytest.raises(UnsupportedCombinationError, match=r"species \[8\]"):
        engine.run_segment(SegmentRequest("r", 0, tmp_path / "seg", _model((1,)), 6, None, None), lambda frame: FrameAction.CONTINUE)
    assert not (tmp_path / "seg" / "gmd.stdout").exists()
