# Ported from PHAL (tests.scf.test_scf_history) at 96263b6; MIT License, see NOTICE.
"""Offline tests for conventional CP2K SCF trajectory recording."""

from __future__ import annotations

import json
from pathlib import Path

from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history, record_scf_history_text
from tests.scf.fakes.fake_cp2k import fake_cp2k_scf_output

CP2K_INPUT = """&FORCE_EVAL
  &DFT
    &SCF
      MAX_SCF 100
      EPS_SCF 1.0E-6
      SCF_GUESS ATOMIC
    &END SCF
  &END DFT
&END FORCE_EVAL
"""


def test_parse_fake_scf_history_collects_required_fields() -> None:
    history = parse_scf_history(fake_cp2k_scf_output(), input_text=CP2K_INPUT)

    assert history.cp2k_version == "CP2K version TEST-2026.1"
    assert history.converged
    assert history.eps_scf == 1.0e-6
    assert history.max_scf == 100
    assert history.initial_guess == "ATOMIC"
    assert [iteration.step for iteration in history.iterations] == [1, 2, 3]
    assert history.iterations[-1].residual == 8.0e-7
    assert history.iterations[-1].energy == -10.26


def test_parse_aligned_cp2k_iteration_table() -> None:
    header = (
        f"{'Step':>5} {'Update method':<20} {'Time':>8} "
        f"{'Convergence':>15} {'Total energy':>20} {'Change':>12}"
    )
    rows = [
        f"{1:5d} {'P_Mix/Diag.':<20} {0.1:8.3f} {1.0e-2:15.8E} {-10.0:20.10f} {-1.0e-1:12.3E}",
        f"{2:5d} {'P_Mix/Diag.':<20} {0.1:8.3f} {5.0e-7:15.8E} {-10.2:20.10f} {-2.0e-1:12.3E}",
    ]
    output = "\n".join([header, "-" * len(header), *rows, " SCF run converged in 2 steps"])

    history = parse_scf_history(output, input_text=CP2K_INPUT)

    assert [iteration.step for iteration in history.iterations] == [1, 2]
    assert history.iterations[0].residual == 1.0e-2
    assert history.iterations[1].energy == -10.2
    assert history.converged


def test_recording_writes_task_scoped_scf_history_json(tmp_path: Path) -> None:
    input_path = tmp_path / "job.inp"
    input_path.write_text(CP2K_INPUT, encoding="utf-8")
    config = {
        "cp2k": {
            "scf_recording": {
                "enabled": True,
                "output_dir": str(tmp_path / "scf"),
            }
        }
    }

    written = record_scf_history_text(
        fake_cp2k_scf_output(),
        input_path=input_path,
        config=config,
        task_id="task-001",
    )

    assert written == tmp_path / "scf" / "task-001" / "scf_history.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["converged"] is True
    assert payload["iterations"][0] == {"step": 1, "residual": 0.01, "energy": -10.0}
    assert payload["eps_scf"] == 1.0e-6
    assert payload["max_scf"] == 100
    assert payload["initial_guess"] == "ATOMIC"


def test_recording_disabled_preserves_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "job.inp"
    input_path.write_text(CP2K_INPUT, encoding="utf-8")

    written = record_scf_history_text(
        fake_cp2k_scf_output(),
        input_path=input_path,
        config={"cp2k": {"scf_recording": {"enabled": False, "output_dir": str(tmp_path / "scf")}}},
    )

    assert written is None
    assert not (tmp_path / "scf").exists()


def test_unconverged_history_is_recorded_as_unconverged() -> None:
    history = parse_scf_history(fake_cp2k_scf_output(converged=False), input_text=CP2K_INPUT)

    assert not history.converged
    assert len(history.iterations) == 3
