"""Additive contract extensions (ledger CORE-001..004, CORE-006).

* CORE-001: Frame keeps ASE constraints (FixAtoms...), extra per-atom arrays and info through
  pickle, ASE and extxyz round trips (port of PHAL tests/unit/test_models_serialization.py), and
  converts losslessly to/from the legacy ``core.data_types.MDFrame`` for the shared fields.
* CORE-002: MonitorResult carries per-metric per-atom scores next to the raw metric values.
* CORE-003: EventRecord carries pre_frames_requested; nested frames round-trip through pickle.
* CORE-004: the FrameSource contract (see tests/adapters/gmd/test_engine.py for the GMD source).
* CORE-006: ``import gmd_active_learning`` imports no adapter, torch or GMD_SGT.
"""

from __future__ import annotations

import io
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian
from ase.io import read, write

from gmd_active_learning.contracts import EventRecord, Frame, MetricValues, MonitorResult, Stage
from gmd_active_learning.core.data_types import MDFrame
from gmd_active_learning.core.exceptions import DataContractError

REPO = Path(__file__).resolve().parents[2]


def _atoms() -> Atoms:
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], cell=np.eye(3) * 9.0, pbc=(True, True, False))
    atoms.arrays["atom_id"] = np.array([11, 4, 7])
    atoms.arrays["forces"] = np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
    atoms.arrays["force_weight"] = np.array([1.0, 0.5, 0.0])
    atoms.arrays["region_code"] = np.array([0, 1, 4])
    atoms.set_constraint([FixAtoms(indices=[0]), FixCartesian(2, mask=(True, False, True))])
    atoms.info.update({"step": 30, "energy": -14.2, "temperature": 300.0, "label": "burst"})
    return atoms


def _assert_same(frame: Frame, other: Frame) -> None:
    assert other.step == frame.step and other.atom_ids.tolist() == frame.atom_ids.tolist()
    np.testing.assert_array_equal(other.positions, frame.positions)
    np.testing.assert_array_equal(other.forces, frame.forces)
    assert other.energy == frame.energy and other.pbc == frame.pbc
    assert set(other.arrays) == set(frame.arrays)
    for name in frame.arrays:
        np.testing.assert_array_equal(other.arrays[name], frame.arrays[name])
    assert other.constraints == frame.constraints
    assert other.info == frame.info


def test_frame_keeps_constraints_extra_arrays_and_info_through_ase_and_pickle() -> None:
    frame = Frame.from_atoms(_atoms(), run_id="run")
    assert [item["name"] for item in frame.constraints] == ["FixAtoms", "FixCartesian"]
    assert set(frame.arrays) == {"force_weight", "region_code"} and frame.info == {"temperature": 300.0, "label": "burst"}
    _assert_same(frame, pickle.loads(pickle.dumps(frame)))
    atoms = frame.to_atoms()
    assert isinstance(atoms.constraints[0], FixAtoms) and atoms.constraints[0].get_indices().tolist() == [0]
    assert atoms.info["label"] == "burst" and atoms.arrays["force_weight"].tolist() == [1.0, 0.5, 0.0]
    _assert_same(frame, Frame.from_atoms(atoms, run_id="run"))


def test_frame_extxyz_round_trip_keeps_arrays_and_info() -> None:
    """(Constraints survive pickle and ASE objects; extxyz constraint support is ASE's own.)"""

    frame = Frame.from_atoms(_atoms(), run_id="run")
    buffer = io.StringIO()
    write(buffer, frame.to_atoms(), format="extxyz")
    back = Frame.from_atoms(read(io.StringIO(buffer.getvalue()), format="extxyz"), run_id="run")
    assert back.energy == frame.energy and back.info["label"] == "burst"
    np.testing.assert_array_equal(back.arrays["force_weight"], frame.arrays["force_weight"])
    np.testing.assert_array_equal(back.arrays["region_code"], frame.arrays["region_code"])


def test_frame_rejects_bad_extensions() -> None:
    with pytest.raises(DataContractError, match="per-atom array"):
        Frame.from_atoms(_atoms(), run_id="run", arrays={"w": np.ones(2)})
    with pytest.raises(DataContractError, match="todict"):
        Frame.from_atoms(_atoms(), run_id="run", constraints=[{"kwargs": {}}])

    class Opaque:
        def get_removed_dof(self, atoms):  # noqa: ANN001, ANN201
            return 0

    atoms = _atoms()
    atoms.constraints = [Opaque()]
    with pytest.raises(DataContractError, match="cannot be serialised"):
        Frame.from_atoms(atoms, run_id="run")


def test_md_frame_converters_are_lossless_for_shared_fields() -> None:
    frame = Frame.from_atoms(_atoms(), run_id="run-7", committee_forces=np.zeros((2, 3, 3)) + 0.01)
    md = frame.to_md_frame()
    assert isinstance(md, MDFrame) and md.symbols == ["O", "H", "H"] and md.ensemble_forces.shape == (2, 3, 3)
    back = Frame.from_md_frame(md)
    assert back.run_id == "run-7" and back.atom_ids.tolist() == [11, 4, 7] and back.step == 30
    np.testing.assert_array_equal(back.positions, frame.positions)
    np.testing.assert_array_equal(back.forces, frame.forces)
    np.testing.assert_array_equal(back.committee_forces, frame.committee_forces)
    np.testing.assert_array_equal(back.cell, frame.cell)
    assert back.pbc == frame.pbc and back.energy == frame.energy
    legacy = MDFrame(step=3, positions=np.zeros((2, 3)) + [[0, 0, 0], [0, 0, 1.0]], symbols=["H", "H"], forces=np.zeros((2, 3)))
    assert Frame.from_md_frame(legacy, run_id="old").atom_ids.tolist() == [0, 1]
    with pytest.raises(DataContractError):
        Frame(run_id="r", step=0, atom_ids=[0], numbers=[1], positions=[[0, 0, 0]], cell=None, pbc=(False,) * 3).to_md_frame()


def test_monitor_result_carries_per_metric_scores() -> None:
    values = MetricValues.measured("force", np.array([0.1, 3.0, 0.2]))
    result = MonitorResult("r", 1, Stage.FULL, np.array([0.5, 7.0, 0.1]), {"force": values}, 7.0, True, [5], ["force_large"],
                           metric_scores={"force": np.array([0.5, 7.0, 0.1]), "committee": [np.nan, np.nan, np.nan]})
    payload = result.to_dict()
    assert payload["metric_scores"]["force"] == [0.5, 7.0, 0.1] and payload["metric_scores"]["committee"] == [None, None, None]
    assert pickle.loads(pickle.dumps(result)).metric_scores["force"].tolist() == [0.5, 7.0, 0.1]
    with pytest.raises(DataContractError, match="metric score"):
        MonitorResult("r", 1, Stage.FULL, np.zeros(3), {}, 0.0, False, [], [], metric_scores={"x": np.zeros(2)})
    assert MonitorResult("r", 1, Stage.SKIPPED, np.zeros(3), {}, 0.0, False, [], []).metric_scores == {}  # backward compatible


def test_event_record_pre_frames_requested_and_nested_pickle() -> None:
    frames = [Frame.from_atoms(_atoms(), run_id="run", step=step) for step in (10, 20, 30)]
    event = EventRecord("evt", "run", 20, [20], frames[:1], frames[1], frames[2:], 2, False, [4], ["force"], 7.0,
                        truncation_reason="md_stopped", pre_frames_requested=3)
    assert event.summary()["pre_frames_requested"] == 3 and event.summary()["post_frames_requested"] == 2
    back = pickle.loads(pickle.dumps(event))
    assert back.pre_frames_requested == 3 and [f.step for f in back.frames] == [10, 20, 30]
    assert back.trigger_frame.constraints == frames[1].constraints
    np.testing.assert_array_equal(back.post_frames[0].arrays["force_weight"], frames[2].arrays["force_weight"])
    legacy = EventRecord("e2", "run", 1, [1], [], frames[0], [], 0, True, [], [], 0.0)
    assert legacy.pre_frames_requested is None  # old call sites keep working


def test_importing_the_package_loads_no_adapter_torch_or_gmd_sgt() -> None:
    code = ("import sys, gmd_active_learning, gmd_active_learning.contracts, gmd_active_learning.workflow.controller;"
            "bad=[m for m in sys.modules if m.split('.')[0] in ('torch','gmd_sgt','hotspot_al') or m.startswith('gmd_active_learning.adapters')];"
            "print(bad); raise SystemExit(1 if bad else 0)")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert completed.returncode == 0, completed.stdout + completed.stderr
