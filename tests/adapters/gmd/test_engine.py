"""GMD MD adapter: input writing, command/launcher, species checks, frame tail, segments and the
pull-style frame source (ledger MD-001, MD-003, MD-004, MD-005, BE-009, CORE-004, MON-014).

Ported scenarios of PHAL tests/unit/test_lammps_controller.py and test_lammps_input_full.py,
re-expressed for GMD's frame/status files (fake GMD executable, tests/adapters/gmd/fake_gmd.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from gmd_active_learning.adapters.gmd.engine import FrameTail, GMDConfig, GMDEngine, GMDFrameSource, check_model_species, structure_species
from gmd_active_learning.contracts import UNITS, FrameAction, ModelArtifact
from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.core.exceptions import AdapterError, CapabilityUnavailableError, ConfigurationError, UnsupportedCombinationError
from gmd_active_learning.workflow.protocols import SegmentRequest


def _model(tmp_path: Path, species=(1, 8), name: str = "m1") -> ModelArtifact:
    export = tmp_path / f"{name}.pt"
    export.write_bytes(f"torchscript {name}".encode())
    return ModelArtifact(name, str(export), file_sha256(export), str(export), file_sha256(export), None, None, tuple(species), 4.0,
                         dict(UNITS), {}, "fresh", "gmd_sgt", "now")


def _engine(executable: Path, structure: Path, **overrides) -> GMDEngine:  # noqa: ANN003
    return GMDEngine(GMDConfig(**{"executable": str(executable), "initial_structure": str(structure), "frame_interval": 2,
                                  "reader_poll_s": 0.01, **overrides}))


def _request(tmp_path: Path, model: ModelArtifact, steps: int = 6, name: str = "seg0", restart=None, previous=None, index: int = 0) -> SegmentRequest:  # noqa: ANN001
    return SegmentRequest("run-x", index, tmp_path / name, model, steps, restart, previous)


# ----------------------------------------------------------------------------- inputs (MD-003)
def test_run_in_for_a_first_segment_and_a_restart_with_model_change(tmp_path: Path, fake_gmd: Path, water: Path) -> None:
    engine = _engine(fake_gmd, water, directives={"time_step": 0.5, "velocity": 300.0, "velocity_init": "random", "velocity_seed": 3},
                     checkpoint_every=50)
    model = _model(tmp_path)
    argv = engine.prepare(_request(tmp_path, model, steps=100))
    work = tmp_path / "seg0"
    assert argv == [str(fake_gmd), str(work / "in.xyz"), str(work / "run.in")]
    assert (work / "in.xyz").read_text() == water.read_text()  # same structure and species order as the input
    assert (work / "run.in").read_text().splitlines() == [
        "run 100", "force_field ml", f"model_path {model.export_path}", "run_id run-x", "frame_file frames.extxyz", "frame_interval 2",
        "control_poll_interval 1", "stop_file STOP", "checkpoint_request_file CHECKPOINT", "status_file status.json",
        "checkpoint_file segment.gmdchk", "write_checkpoint_every 50", "time_step 0.5", "velocity 300.0", "velocity_init random", "velocity_seed 3",
    ]
    checkpoint = tmp_path / "prev.gmdchk"
    checkpoint.write_text("{}")
    engine.prepare(_request(tmp_path, _model(tmp_path, name="m2"), steps=10, name="seg1", restart=checkpoint, previous=model, index=1))
    restart = (tmp_path / "seg1" / "run.in").read_text().splitlines()
    assert f"restart_from {checkpoint.resolve()}" in restart and "model_change allow" in restart
    assert "velocity_init input" in restart and not any(line.startswith("velocity_seed") for line in restart)


def test_controller_owned_directives_and_bad_time_step_are_rejected(fake_gmd: Path, water: Path) -> None:
    with pytest.raises(ConfigurationError, match="controller-owned"):
        _engine(fake_gmd, water, directives={"frame_file": "x"})
    for bad in (0, -1.0, "fast", True):
        with pytest.raises(ValidationError, match="time_step"):
            GMDConfig(executable=str(fake_gmd), initial_structure=str(water), directives={"time_step": bad})


def test_missing_executable_is_a_capability_error(tmp_path: Path, water: Path) -> None:
    with pytest.raises(CapabilityUnavailableError):
        _engine(tmp_path / "nope" / "gmd", water)
    with pytest.raises(CapabilityUnavailableError):
        _engine(Path("gmd-binary-that-is-not-on-path"), water)


def test_bare_executable_name_is_looked_up_on_path(tmp_path: Path, fake_gmd: Path, water: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", f"{fake_gmd.parent}{os.pathsep}{os.environ['PATH']}")
    assert _engine(Path("gmd"), water).executable == fake_gmd


# ----------------------------------------------------------------------------- command (MD-004)
@pytest.mark.parametrize(
    ("launcher", "prefix"),
    [
        ([], []),
        (["taskset", "-c", "0"], ["taskset", "-c", "0"]),
        (["srun", "-n", "1", "--cpu-bind=cores"], ["srun", "-n", "1", "--cpu-bind=cores"]),
        (["mpiexec", "-n", "1"], ["mpiexec", "-n", "1"]),
        (["numactl", "--cpunodebind=0"], ["numactl", "--cpunodebind=0"]),
    ],
)
def test_command_reference_table(tmp_path: Path, fake_gmd: Path, water: Path, launcher: list[str], prefix: list[str]) -> None:
    engine = _engine(fake_gmd, water, launcher=launcher)
    assert engine.command("in.xyz", "run.in") == [*prefix, str(fake_gmd), "in.xyz", "run.in"]


@pytest.mark.parametrize("launcher", [["mpiexec", "-n", "4"], ["srun", "--ntasks=2"], ["mpirun", "-np", "8"]])
def test_multi_rank_launcher_is_refused_because_ml_runs_are_serial(fake_gmd: Path, water: Path, launcher: list[str]) -> None:
    with pytest.raises(ValidationError, match="serial"):
        GMDConfig(executable=str(fake_gmd), initial_structure=str(water), launcher=launcher)


def test_prepare_returns_argv_without_executing(tmp_path: Path, fake_gmd: Path, water: Path) -> None:
    marker = tmp_path / "ran"
    engine = _engine(fake_gmd, water, env={"FAKE_GMD_MARKER": str(marker)})
    argv = engine.prepare(_request(tmp_path, _model(tmp_path)))
    assert argv[-1].endswith("run.in") and not marker.exists() and not (tmp_path / "seg0" / "frames.extxyz").exists()


# ----------------------------------------------------------------------------- species (MD-005)
def test_species_are_checked_before_launch(tmp_path: Path, fake_gmd: Path, water: Path) -> None:
    marker = tmp_path / "ran"
    engine = _engine(fake_gmd, water, env={"FAKE_GMD_MARKER": str(marker)})
    with pytest.raises(UnsupportedCombinationError, match=r"species \[8\]"):
        engine.run_segment(_request(tmp_path, _model(tmp_path, species=(1,))), lambda frame: FrameAction.CONTINUE)
    assert not marker.exists() and not (tmp_path / "seg0" / "gmd.stdout").exists()  # GMD never started
    with pytest.raises(UnsupportedCombinationError, match="declares no species"):
        check_model_species(water, ())
    disabled = _engine(fake_gmd, water, check_species=False)
    assert disabled.prepare(_request(tmp_path, _model(tmp_path, species=(1,)), name="unchecked"))


def test_structure_species_order_and_unknown_elements(tmp_path: Path) -> None:
    mixed = tmp_path / "mixed.xyz"
    mixed.write_text("4\n9 9 9\nH 0 0 0\nO 1 0 0\nC 2 0 0\nH 3 0 0\n")
    assert structure_species(mixed) == [1, 8, 6, 1]
    bad = tmp_path / "bad.xyz"
    bad.write_text("2\n9 9 9\nH 0 0 0\nXx 1 0 0\n")
    with pytest.raises(ConfigurationError, match="Z=0"):
        structure_species(bad)
    short = tmp_path / "short.xyz"
    short.write_text("3\n9 9 9\nH 0 0 0\n")
    with pytest.raises(ConfigurationError, match="declares 3 atoms"):
        structure_species(short)


# ----------------------------------------------------------------------------- frame tail (MD-001)
def _frame_text(step: int, n: int = 2) -> str:
    lines = [str(n), f'Lattice="5 0 0 0 5 0 0 0 5" Properties=id:I:1:species:S:1:pos:R:3:forces:R:3 pbc="T T T" step={step} format=gmd-frame-v1']
    lines += [f"{i} H {i + 0.1 * step:.6f} 0.0 0.0 0.0 0.0 {0.01 * step:.6f}" for i in range(n)]
    return "\n".join(lines) + "\n"


def test_tail_returns_each_frame_exactly_once_across_partial_writes(tmp_path: Path) -> None:
    path = tmp_path / "frames.extxyz"
    text = "".join(_frame_text(step) for step in range(0, 50, 5))
    tail = FrameTail(path, "run")
    assert tail.read_new() == []  # no file yet
    seen = []
    with path.open("w") as handle:
        for k in range(0, len(text), 37):  # arbitrary byte splits, including mid-line and mid-number
            handle.write(text[k : k + 37])
            handle.flush()
            seen += [frame.step for frame in tail.read_new()]
    seen += [frame.step for frame in tail.read_new()]
    assert seen == list(range(0, 50, 5))
    assert tail.read_new() == []  # nothing new: only appended bytes are read
    with path.open("a") as handle:
        handle.write(_frame_text(50))
    assert [frame.step for frame in tail.read_new()] == [50]


def test_tail_resynchronises_after_truncation_without_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "frames.extxyz"
    path.write_text(_frame_text(0) + _frame_text(10))
    tail = FrameTail(path, "run")
    assert [frame.step for frame in tail.read_new()] == [0, 10]
    path.write_text(_frame_text(0))  # truncated/rewritten by another process
    assert tail.read_new() == [] and tail.resyncs == 1
    with path.open("a") as handle:
        handle.write(_frame_text(10) + _frame_text(20))
    assert [frame.step for frame in tail.read_new()] == [20]


def test_tail_rejects_a_corrupt_header(tmp_path: Path) -> None:
    path = tmp_path / "frames.extxyz"
    path.write_text("not-a-count\nx\n")
    with pytest.raises(AdapterError, match="atom count"):
        FrameTail(path, "run").read_new()


# ----------------------------------------------------------------------------- segments
def test_segment_delivers_every_frame_once_with_partial_writes(tmp_path: Path, fake_gmd: Path, water: Path) -> None:
    engine = _engine(fake_gmd, water, env={"FAKE_GMD_CHUNK": "23", "FAKE_GMD_DELAY": "0.01"})
    model = _model(tmp_path)
    steps: list[int] = []
    result = engine.run_segment(_request(tmp_path, model, steps=10), lambda frame: steps.append(frame.step) or FrameAction.CONTINUE)
    assert result.status == "completed" and result.end_step == 10
    assert steps == [0, 2, 4, 6, 8, 10]
    replay = list(engine.read_frames(result, _request(tmp_path, model, steps=10)))
    assert [frame.step for frame in replay] == steps and all(frame.model_version == model.model_id for frame in replay)


def test_segment_survives_frame_file_truncation(tmp_path: Path, fake_gmd: Path, water: Path, caplog: pytest.LogCaptureFixture) -> None:
    engine = _engine(fake_gmd, water, env={"FAKE_GMD_TRUNCATE_AFTER": "3", "FAKE_GMD_DELAY": "0.1"})
    steps: list[int] = []
    with caplog.at_level("WARNING"):
        result = engine.run_segment(_request(tmp_path, _model(tmp_path), steps=10), lambda frame: steps.append(frame.step) or FrameAction.CONTINUE)
    assert result.status == "completed" and steps == [0, 2, 4, 6, 8, 10]
    assert any("shrank" in record.message for record in caplog.records)  # the reader really saw the truncation


def test_stop_action_stops_the_segment_cooperatively(tmp_path: Path, fake_gmd: Path, water: Path) -> None:
    engine = _engine(fake_gmd, water, frame_interval=1, env={"FAKE_GMD_DELAY": "0.02"})
    seen: list[int] = []

    def on_frame(frame):  # noqa: ANN001, ANN202
        seen.append(frame.step)
        return FrameAction.STOP if frame.step == 3 else FrameAction.CONTINUE

    result = engine.run_segment(_request(tmp_path, _model(tmp_path), steps=1000), on_frame)
    assert result.status == "stopped" and result.end_step < 1000
    assert result.detail["decisions"][0] == {"action": "stop", "step": 3} and result.detail["stop_latency_steps"] >= 0
    assert seen == sorted(set(seen))


def test_non_zero_exit_is_a_failed_segment(tmp_path: Path, fake_gmd: Path, water: Path) -> None:
    engine = _engine(fake_gmd, water, env={"FAKE_GMD_EXIT": "3"})
    result = engine.run_segment(_request(tmp_path, _model(tmp_path)), lambda frame: FrameAction.CONTINUE)
    assert result.status == "failed" and result.detail["returncode"] == 3
    assert "forces exploded" in result.detail["stderr"]


def test_frames_from_a_different_model_are_refused(tmp_path: Path, fake_gmd: Path, water: Path) -> None:
    model = _model(tmp_path)
    impostor = ModelArtifact(**{**model.to_dict(), "species": tuple(model.species), "export_sha256": "0" * 64})
    with pytest.raises(AdapterError, match="expected"):
        _engine(fake_gmd, water).run_segment(_request(tmp_path, impostor), lambda frame: FrameAction.CONTINUE)


# ----------------------------------------------------------------------------- frame source (CORE-004, MON-014)
def test_frame_source_end_of_stream_timeout_and_exit_code(tmp_path: Path, fake_gmd: Path, water: Path) -> None:
    from gmd_active_learning.monitors.online import POLL_TIMEOUT, FrameSource

    source = GMDFrameSource(_engine(fake_gmd, water, env={"FAKE_GMD_DELAY": "0.05"}), _request(tmp_path, _model(tmp_path), steps=6))
    assert isinstance(source, FrameSource)
    with source:
        items = []
        while True:
            item = source.next_frame(timeout=0.01)
            if item is None:
                break
            items.append(item)
        assert POLL_TIMEOUT in items  # nothing yet is distinct from end of stream
        assert [item.step for item in items if item is not POLL_TIMEOUT] == [0, 2, 4, 6]
        assert source.next_frame(timeout=0.01) is None  # end of stream is sticky
    assert source.logs_closed
    failing = GMDFrameSource(_engine(fake_gmd, water, env={"FAKE_GMD_EXIT": "2"}), _request(tmp_path, _model(tmp_path), name="bad"))
    with failing, pytest.raises(AdapterError, match="exited with code 2"):
        while failing.next_frame(timeout=1.0) is not None:
            pass


def test_frame_source_close_terminates_and_closes_logs(tmp_path: Path, fake_gmd: Path, water: Path) -> None:
    source = GMDFrameSource(_engine(fake_gmd, water, frame_interval=1, env={"FAKE_GMD_DELAY": "0.05"}),
                            _request(tmp_path, _model(tmp_path), steps=100000)).start()
    first = source.next_frame(timeout=5.0)
    assert first is not None and first.step == 0
    source.close(timeout=5.0)
    assert source.process is not None and source.process.poll() is not None and source.logs_closed


def test_online_monitor_driver_runs_unchanged_on_the_gmd_stream(tmp_path: Path, fake_gmd: Path, water: Path) -> None:
    """MON-014: the GMD frame stream feeds OnlineMonitorDriver.run directly; STOP reaches GMD."""

    from gmd_active_learning.events.config import EventsConfig
    from gmd_active_learning.hotspots.config import HotspotsConfig
    from gmd_active_learning.monitors.config import MonitorConfig
    from gmd_active_learning.monitors.online import OnlineMonitorDriver

    engine = _engine(fake_gmd, water, frame_interval=1)
    with GMDFrameSource(engine, _request(tmp_path, _model(tmp_path), steps=8)) as source:
        summary = OnlineMonitorDriver(MonitorConfig(), EventsConfig(), HotspotsConfig()).run(source)
    assert summary.stop_reason == "end_of_stream"
    assert [result.step for result in summary.results] == list(range(9))
    assert all(np.all(np.isfinite(result.atom_scores)) for result in summary.results)
