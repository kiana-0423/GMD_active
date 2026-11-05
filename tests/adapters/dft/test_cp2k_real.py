"""Real CP2K labeling through LabelingService + LocalScheduler (marker ``cp2k``).

Skipped only when no CP2K binary is found (reason given). Small settings keep each run to seconds.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import units

from gmd_active_learning.adapters.dft.cp2k.backend import CP2KBackend
from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig
from gmd_active_learning.adapters.dft.cp2k.parse import parse_cp2k_output_file
from gmd_active_learning.adapters.schedulers.local import LocalScheduler
from gmd_active_learning.contracts.labels import ExecutionStatus, ValidationStatus
from gmd_active_learning.labeling.service import LabelingService

pytestmark = pytest.mark.cp2k

FAST = {"cutoff": 150, "rel_cutoff": 40, "max_scf": 30, "scf_eps": 1.0e-5, "environment": {"OMP_NUM_THREADS": "1"}}


def _service(tmp_path: Path, cp2k_bin: str, backend_cls=CP2KBackend, **overrides) -> LabelingService:
    config = CP2KConfig.model_validate({**FAST, "executable": cp2k_bin, **overrides})
    return LabelingService(backend_cls(config), LocalScheduler(), task_root=tmp_path / "tasks")


def _label(service: LabelingService, region, timeout: float = 600.0):
    task = service.submit(service.prepare(region))
    deadline = time.monotonic() + timeout
    while task.metadata["stage"] != "done":
        assert time.monotonic() < deadline, task.metadata["attempts"]
        time.sleep(0.2)
        task = service.poll(task)
    return task, service.collect(task)


def _raw_forces(task, attempt: int = -1) -> np.ndarray:
    record = task.metadata["attempts"][attempt]
    text = (Path(task.work_dir) / record["dir"] / record["output"]).read_text()
    rows = [line.split()[2:5] for line in text.splitlines() if line.strip().startswith("FORCES|") and line.split()[1].isdigit()]
    return np.asarray(rows, dtype=float)


def test_h2_molecule_label(tmp_path: Path, cp2k_bin: str, load_region) -> None:
    task, result = _label(_service(tmp_path, cp2k_bin), load_region("h2_cluster"))
    assert task.execution is ExecutionStatus.COMPLETED and result.validation is ValidationStatus.VALID
    assert result.forces.shape == (2, 3) and np.all(np.isfinite(result.forces))
    np.testing.assert_allclose(result.forces, _raw_forces(task) * units.Hartree / units.Bohr, rtol=1e-12)
    assert result.forces[0, 0] == pytest.approx(-result.forces[1, 0], rel=0.2)  # nearly balanced pair force
    assert -32.5 < result.energy < -30.5  # eV (about -1.16 Ha)
    assert result.engine_version == "2026.2" and result.scf_steps > 0
    assert task.metadata["attempts"][0]["exit_code"] == 0


def test_h_capped_cluster_relaxes_caps_then_labels(tmp_path: Path, cp2k_bin: str, load_region) -> None:
    region = load_region("ch3_capped")
    task, result = _label(_service(tmp_path, cp2k_bin), region)
    assert [a["stage"] for a in task.metadata["attempts"]] == ["hopt", "sp"]
    assert result.validation is ValidationStatus.VALID and result.metadata["caps_relaxed"]
    positions = np.asarray(result.metadata["positions"])
    np.testing.assert_allclose(positions[:4], region.positions[:4], atol=1e-6)
    assert 1e-4 < np.linalg.norm(positions[4] - region.positions[4]) < 0.2
    sp_input = (Path(task.work_dir) / task.metadata["attempts"][1]["dir"] / "sp.inp").read_text()
    assert "PERIODIC NONE" in sp_input and "POISSON_SOLVER WAVELET" in sp_input
    assert result.forces.shape == (5, 3) and np.all(np.isfinite(result.forces))


def test_periodic_region_label_with_stress(tmp_path: Path, cp2k_bin: str, load_region) -> None:
    task, result = _label(_service(tmp_path, cp2k_bin), load_region("h2_periodic"))
    assert result.validation is ValidationStatus.VALID
    assert result.stress is not None and np.all(np.isfinite(result.stress))
    out = Path(task.work_dir) / task.metadata["attempts"][0]["dir"] / "sp.out"
    raw = np.asarray(parse_cp2k_output_file(out).stress_raw)
    np.testing.assert_allclose(result.stress, -raw * units.bar, rtol=1e-12)
    assert task.metadata["periodic"] == "XYZ"


def test_slab_region_label_keeps_partial_periodicity(tmp_path: Path, cp2k_bin: str, load_region) -> None:
    task, result = _label(_service(tmp_path, cp2k_bin), load_region("h2_slab"))
    assert result.validation is ValidationStatus.VALID and result.stress is None
    sp_input = (Path(task.work_dir) / task.metadata["attempts"][0]["dir"] / "sp.inp").read_text()
    assert sp_input.count("PERIODIC XY\n") == 2 and "POISSON_SOLVER MT" in sp_input


def test_unconverged_run_is_invalid_with_reason(tmp_path: Path, cp2k_bin: str, load_region) -> None:
    task, result = _label(_service(tmp_path, cp2k_bin, max_scf=2, retry={"enabled": False}), load_region("h2_cluster"))
    assert task.execution is ExecutionStatus.FAILED  # CP2K 2026.2 aborts (exit 1) on SCF non-convergence
    assert result.validation is ValidationStatus.INVALID and not result.converged
    assert any("SCF did not converge" in reason for reason in result.validation_reasons)
    assert task.metadata["attempts"][0]["failure_class"] == "scf_not_converged"


def test_retry_ladder_recovers_the_unconverged_run(tmp_path: Path, cp2k_bin: str, load_region) -> None:
    ladder = [{"use_ot": False, "mixing_method": "BROYDEN_MIXING", "mixing_nbuffer": 4}, {"max_scf_multiplier": 15}]
    service = _service(tmp_path, cp2k_bin, max_scf=2, retry={"ladders": {"scf_not_converged": ladder}})
    task, result = _label(service, load_region("h2_cluster"))
    assert result.validation is ValidationStatus.VALID
    attempts = task.metadata["attempts"]
    assert [a["failure_class"] for a in attempts] == ["scf_not_converged", "scf_not_converged", None]
    inputs = [(Path(task.work_dir) / a["dir"] / a["input"]).read_text() for a in attempts]
    assert "NBUFFER 4" in inputs[1] and "MAX_SCF 2\n" in inputs[1]
    assert "MAX_SCF 30" in inputs[2] and "&OT" in inputs[2]  # base 2 x 15, not compounded


class _FailingMLBackend(CP2KBackend):
    """Stands in for the ML-SCF variant: its decoration makes CP2K's SCF fail (MAX_SCF 1)."""

    ml_scf_enabled = True

    def variant_id(self) -> str:
        return "cp2k_ml_scf_test"

    def decorate_input(self, cp2k_input, work_dir, *, region):  # noqa: ANN001
        return cp2k_input.text.replace(f"MAX_SCF {self.config.max_scf}", "MAX_SCF 1")

    def native_backend(self) -> CP2KBackend:
        return CP2KBackend(self.config)


def test_failed_ml_scf_run_is_resubmitted_natively(tmp_path: Path, cp2k_bin: str, load_region) -> None:
    task, result = _label(_service(tmp_path, cp2k_bin, backend_cls=_FailingMLBackend), load_region("h2_cluster"))
    assert result.validation is ValidationStatus.VALID and result.metadata["ml_fallback_used"]
    first, second = task.metadata["attempts"]
    assert not first["native"] and second["native"] and second["trigger_class"] == "ml_scf_failed"
    assert second["exit_code"] == 0


def test_restart_safe_idempotent_labeling(tmp_path: Path, cp2k_bin: str, load_region) -> None:
    region = load_region("h2_cluster")
    task, first = _label(_service(tmp_path, cp2k_bin), region)
    service = _service(tmp_path, cp2k_bin)  # a new controller
    again = service.submit(service.prepare(replace(region, metadata={"note": "re-extracted"})))
    assert again.task_id == task.task_id and len(again.metadata["attempts"]) == 1
    second = service.collect(again)
    np.testing.assert_array_equal(second.forces, first.forces)
    assert json.loads((Path(task.work_dir) / "task.json").read_text())["validation"] == "valid"
