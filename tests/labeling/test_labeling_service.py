"""LabelingService with a scripted fake CP2K and the real local scheduler."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from ase.units import Bohr, Hartree

from gmd_active_learning.adapters.dft.cp2k.backend import CP2KBackend
from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig
from gmd_active_learning.adapters.registry import BackendRegistry
from gmd_active_learning.adapters.schedulers.base import SchedulerBackend, SubmissionRejected
from gmd_active_learning.adapters.schedulers.config import SchedulerConfig
from gmd_active_learning.adapters.schedulers.local import LocalScheduler
from gmd_active_learning.contracts.labels import ExecutionStatus, ValidationStatus
from gmd_active_learning.contracts.regions import ExtractedRegion
from gmd_active_learning.core.exceptions import LabelingError, SchedulerError, UnsupportedCombinationError
from gmd_active_learning.labeling.service import LabelingService
from gmd_active_learning.labeling.task_store import load_task

REGIONS = Path(__file__).resolve().parents[1] / "fixtures" / "structures" / "regions"
HA_BOHR = Hartree / Bohr


def region(name: str) -> ExtractedRegion:
    return ExtractedRegion.from_dict(json.loads((REGIONS / f"{name}.json").read_text()))


def make_service(tmp_path: Path, fake_cp2k, scheduler: SchedulerBackend | None = None, **settings) -> LabelingService:
    config = CP2KConfig.model_validate({"executable": str(fake_cp2k.executable), **settings})
    return LabelingService(CP2KBackend(config), scheduler or LocalScheduler(), task_root=tmp_path / "tasks")


def run_to_end(service: LabelingService, task, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while task.metadata.get("stage") != "done":
        if time.monotonic() > deadline:
            raise AssertionError(f"task did not finish: {task.execution} {task.metadata.get('attempts')}")
        time.sleep(0.05)
        task = service.poll(task)
    return task


def attempt_inputs(task) -> list[str]:
    return [(Path(task.work_dir) / a["dir"] / a["input"]).read_text() for a in task.metadata["attempts"]]


# -- prepare / submit idempotency --------------------------------------------------------------------
def test_prepare_is_idempotent_and_ids_depend_on_region_and_settings(tmp_path: Path, fake_cp2k) -> None:
    service = make_service(tmp_path, fake_cp2k)
    first = service.prepare(region("h2_cluster"))
    again = service.prepare(region("h2_cluster"))
    assert again.task_id == first.task_id and again.work_dir == first.work_dir
    assert first.task_id.startswith("label-") and first.execution is ExecutionStatus.PREPARED
    assert len(list((tmp_path / "tasks").iterdir())) == 1
    other_settings = make_service(tmp_path, fake_cp2k, cutoff=500.0).prepare(region("h2_cluster"))
    assert other_settings.task_id != first.task_id
    moved = region("h2_cluster")
    moved.positions[1, 0] += 0.01
    assert service.prepare(moved).task_id != first.task_id
    # Execution settings do not change the identity of the calculation.
    assert make_service(tmp_path, fake_cp2k, executable="elsewhere/cp2k").task_id_for(region("h2_cluster")) == first.task_id


def test_submit_is_idempotent(tmp_path: Path, fake_cp2k) -> None:
    service = make_service(tmp_path, fake_cp2k)
    task = service.submit(service.prepare(region("h2_cluster")))
    assert task.job is not None and task.execution in {ExecutionStatus.SUBMITTED, ExecutionStatus.RUNNING}
    again = service.submit(task)
    assert again.job.job_id == task.job.job_id
    assert len(task.metadata["attempts"]) == len(again.metadata["attempts"]) == 1
    run_to_end(service, again)
    assert len(fake_cp2k.calls()) == 1
    assert service.submit(task.task_id).metadata["attempts"] == load_task(task.work_dir).metadata["attempts"]


# -- success paths -----------------------------------------------------------------------------------
def test_single_point_label_is_valid_and_converted(tmp_path: Path, fake_cp2k) -> None:
    service = make_service(tmp_path, fake_cp2k)
    task = run_to_end(service, service.submit(service.prepare(region("h2_cluster"))))
    assert task.execution is ExecutionStatus.COMPLETED and task.validation is ValidationStatus.VALID
    persisted = json.loads((Path(task.work_dir) / "task.json").read_text())
    assert persisted["job"]["job_id"] == task.job.job_id and persisted["job"]["exit_code"] == 0
    assert persisted["metadata"]["attempts"][0]["state"] == "completed"
    result = service.collect(task)
    assert result.validation is ValidationStatus.VALID and result.converged
    assert result.energy == pytest.approx(-1.1 * Hartree)
    assert np.allclose(result.forces, np.array([[0.01, -0.02, 0.005], [0.02, -0.04, 0.005]]) * HA_BOHR)
    assert result.engine_version == "2026.2" and result.output_sha256
    assert result.metadata["caps_relaxed"] is False
    assert result.metadata["electronic_state"]["multiplicity"] == 1


def test_capped_region_runs_cap_relaxation_then_single_point(tmp_path: Path, fake_cp2k) -> None:
    service = make_service(tmp_path, fake_cp2k)
    original = region("ch3_capped")
    task = service.prepare(original)
    assert task.metadata["stages"] == ["hopt", "sp"]
    task = run_to_end(service, service.submit(task))
    stages = [a["stage"] for a in task.metadata["attempts"]]
    kinds = [a["kind"] for a in task.metadata["attempts"]]
    assert stages == ["hopt", "sp"] and kinds == ["initial", "stage"]
    hopt_input, sp_input = attempt_inputs(task)
    assert "RUN_TYPE GEO_OPT" in hopt_input and "LIST 1 2 3 4" in hopt_input
    assert "RUN_TYPE ENERGY_FORCE" in sp_input
    result = service.collect(task)
    assert result.validation is ValidationStatus.VALID and result.metadata["caps_relaxed"] is True
    positions = np.asarray(result.metadata["positions"])
    assert np.allclose(positions[:4], original.positions[:4])  # fixed atoms untouched
    assert positions[4, 0] == pytest.approx(original.positions[4, 0] + 0.01, abs=1e-8)  # relaxed cap
    labeled = json.loads((Path(task.work_dir) / "region_labeled.json").read_text())
    assert labeled["metadata"]["h_caps_relaxed"]["max_cap_displacement"] == pytest.approx(0.01, abs=1e-8)


def test_cap_relaxation_can_be_disabled(tmp_path: Path, fake_cp2k) -> None:
    service = make_service(tmp_path, fake_cp2k, h_only_opt={"enabled": False})
    task = run_to_end(service, service.submit(service.prepare(region("ch3_capped"))))
    assert [a["stage"] for a in task.metadata["attempts"]] == ["sp"]
    assert service.collect(task).metadata["caps_relaxed"] is False


# -- retry ladder ------------------------------------------------------------------------------------
def test_scf_retry_ladder_nbuffer_then_max_scf_without_compounding(tmp_path: Path, fake_cp2k) -> None:
    fake_cp2k.configure(min_max_scf=25)
    service = make_service(tmp_path, fake_cp2k, max_scf=10)
    task = run_to_end(service, service.submit(service.prepare(region("h2_cluster"))))
    assert task.validation is ValidationStatus.VALID
    attempts = task.metadata["attempts"]
    assert [a["kind"] for a in attempts] == ["initial", "retry", "retry"]
    assert [a["failure_class"] for a in attempts] == ["scf_not_converged", "scf_not_converged", None]
    first, second, third = attempt_inputs(task)
    assert "MAX_SCF 10" in first and "&OT" in first
    assert "NBUFFER 4" in second and "BROYDEN_MIXING" in second and "&OT" not in second and "MAX_SCF 10" in second
    assert "MAX_SCF 30" in third and "&OT" in third and "NBUFFER" not in third  # base x3, not 10 x 2 x 3
    assert task.attempt == 3


def test_retry_budget_and_cap_are_bounded(tmp_path: Path, fake_cp2k) -> None:
    fake_cp2k.configure(min_max_scf=10_000)
    service = make_service(tmp_path, fake_cp2k, max_scf=400, retry={"max_scf_cap": 1000})
    task = run_to_end(service, service.submit(service.prepare(region("h2_cluster"))))
    assert task.execution is ExecutionStatus.FAILED and task.validation is ValidationStatus.INVALID
    assert len(task.metadata["attempts"]) == 3  # ladder of two SCF steps
    assert "MAX_SCF 1000" in attempt_inputs(task)[-1]  # 400 x 3 capped
    result = service.collect(task)
    assert result.validation is ValidationStatus.INVALID
    assert any("SCF did not converge" in reason for reason in result.validation_reasons)
    assert any("no further retry step" in reason for reason in result.validation_reasons)


def test_retries_disabled_gives_invalid_after_one_attempt(tmp_path: Path, fake_cp2k) -> None:
    fake_cp2k.configure(["scf_fail"])
    service = make_service(tmp_path, fake_cp2k, retry={"enabled": False})
    task = run_to_end(service, service.submit(service.prepare(region("h2_cluster"))))
    assert len(task.metadata["attempts"]) == 1 and task.validation is ValidationStatus.INVALID


def test_deterministic_abort_is_not_retried(tmp_path: Path, fake_cp2k) -> None:
    fake_cp2k.configure(["abort"])
    service = make_service(tmp_path, fake_cp2k)
    task = run_to_end(service, service.submit(service.prepare(region("h2_cluster"))))
    assert len(task.metadata["attempts"]) == 1
    assert task.metadata["attempts"][0]["failure_class"] == "invalid_result"
    assert any("Unknown basis set FAKE" in reason for reason in service.collect(task).validation_reasons)


def test_crash_is_retried_once_with_the_same_settings(tmp_path: Path, fake_cp2k) -> None:
    fake_cp2k.configure(["crash", "ok"])
    service = make_service(tmp_path, fake_cp2k)
    task = run_to_end(service, service.submit(service.prepare(region("h2_cluster"))))
    assert task.validation is ValidationStatus.VALID
    assert [a["failure_class"] for a in task.metadata["attempts"]] == ["execution_failed", None]
    first, second = attempt_inputs(task)
    assert first == second


@pytest.mark.parametrize("action, reason", [("bad_count", "force block has 1 rows"), ("truncated", "no force block")])
def test_completed_but_invalid_output_is_invalid_not_failed(tmp_path: Path, fake_cp2k, action: str, reason: str) -> None:
    fake_cp2k.configure([action])
    service = make_service(tmp_path, fake_cp2k, retry={"ladders": {"execution_failed": []}})
    task = run_to_end(service, service.submit(service.prepare(region("h2_cluster"))))
    assert task.execution is ExecutionStatus.COMPLETED  # the process exited 0 ...
    assert task.validation is ValidationStatus.INVALID  # ... but the label is not valid
    assert any(reason in r for r in service.collect(task).validation_reasons)


def test_walltime_kill_is_classified_and_retried(tmp_path: Path, fake_cp2k) -> None:
    fake_cp2k.configure(["sleep", "ok"])
    service = make_service(tmp_path, fake_cp2k, resources={"walltime": 1})
    task = run_to_end(service, service.submit(service.prepare(region("h2_cluster"))), timeout=60)
    assert task.metadata["attempts"][0]["failure_class"] == "walltime"
    assert task.validation is ValidationStatus.VALID


# -- ML-SCF native fallback (F-25) -------------------------------------------------------------------
class FakeMLBackend(CP2KBackend):
    ml_scf_enabled = True

    def variant_id(self) -> str:
        return "cp2k_ml_scf_fake"

    def decorate_input(self, cp2k_input, work_dir, *, region):  # noqa: ANN001
        return cp2k_input.text + "! ML_SCF_FAIL\n"

    def native_backend(self) -> CP2KBackend:
        return CP2KBackend(self.config)


def test_failed_ml_scf_attempt_is_resubmitted_natively(tmp_path: Path, fake_cp2k) -> None:
    config = CP2KConfig(executable=str(fake_cp2k.executable))
    service = LabelingService(FakeMLBackend(config), LocalScheduler(), task_root=tmp_path / "tasks")
    task = service.submit(service.prepare(region("h2_cluster")))
    assert task.metadata["native"] is False
    task = run_to_end(service, task)
    assert task.validation is ValidationStatus.VALID
    first, second = task.metadata["attempts"]
    assert first["native"] is False and first["failure_class"] == "scf_not_converged"
    assert second["native"] is True and second["kind"] == "retry" and second["trigger_class"] == "ml_scf_failed"
    assert second["job_id"] is not None  # actually submitted, not merely converted
    assert "ML_SCF_FAIL" not in attempt_inputs(task)[1]
    assert service.collect(task).metadata["ml_fallback_used"] is True


# -- unknown state, restart, submission failures, cancel ------------------------------------------------
class UnknownScheduler(LocalScheduler):
    def poll(self, job):  # noqa: ANN001
        return self._set(job, ExecutionStatus.UNKNOWN, raw_state="NOT_FOUND")

    def reconcile(self, job):  # noqa: ANN001
        return ExecutionStatus.UNKNOWN


def test_unknown_state_never_resubmits(tmp_path: Path, fake_cp2k) -> None:
    service = make_service(tmp_path, fake_cp2k, scheduler=UnknownScheduler())
    task = service.submit(service.prepare(region("h2_cluster")))
    for _ in range(3):
        task = service.poll(task)
    assert task.execution is ExecutionStatus.UNKNOWN
    assert task.metadata["unknown_polls"] == 3
    assert len(task.metadata["attempts"]) == 1
    time.sleep(1.0)
    assert len(fake_cp2k.calls()) == 1
    # The workflow decides after reconciliation; only then does the retry ladder apply.
    task = service.resolve_unknown(task, reason="job not in scheduler history and no exit code")
    assert task.metadata["attempts"][0]["failure_class"] == "execution_failed"
    assert len(task.metadata["attempts"]) == 2


def test_a_restarted_controller_resumes_from_task_json(tmp_path: Path, fake_cp2k) -> None:
    fake_cp2k.configure(["sleep"])
    first = make_service(tmp_path, fake_cp2k)
    task = first.submit(first.prepare(region("h2_cluster")))
    second = make_service(tmp_path, fake_cp2k, scheduler=LocalScheduler(SchedulerConfig()))  # fresh, no Popen handles
    resumed = second.poll(task.task_id)
    assert resumed.job.job_id == task.job.job_id and resumed.execution is ExecutionStatus.RUNNING
    cancelled = second.cancel(resumed)
    assert cancelled.execution is ExecutionStatus.CANCELLED and cancelled.validation is ValidationStatus.INVALID
    assert len(cancelled.metadata["attempts"]) == 1  # cancellation is never retried


class TimeoutScheduler(LocalScheduler):
    def submit(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise SchedulerError("sbatch timed out after 60 s")


class RejectingScheduler(LocalScheduler):
    def submit(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise SubmissionRejected("sbatch: invalid partition")


def test_ambiguous_submission_failure_marks_unknown_and_never_duplicates(tmp_path: Path, fake_cp2k) -> None:
    service = make_service(tmp_path, fake_cp2k, scheduler=TimeoutScheduler())
    task = service.submit(service.prepare(region("h2_cluster")))
    assert task.execution is ExecutionStatus.UNKNOWN and task.job is None
    retried = make_service(tmp_path, fake_cp2k).submit(task.task_id)
    assert len(retried.metadata["attempts"]) == 1 and retried.job is None  # no second submission
    assert make_service(tmp_path, fake_cp2k).reconcile(task.task_id).execution is ExecutionStatus.UNKNOWN


def test_rejected_submission_leaves_task_resubmittable(tmp_path: Path, fake_cp2k) -> None:
    rejecting = make_service(tmp_path, fake_cp2k, scheduler=RejectingScheduler())
    task = rejecting.prepare(region("h2_cluster"))
    with pytest.raises(SubmissionRejected):
        rejecting.submit(task)
    assert load_task(task.work_dir).execution is ExecutionStatus.PREPARED
    service = make_service(tmp_path, fake_cp2k)
    done = run_to_end(service, service.submit(task))
    assert done.validation is ValidationStatus.VALID


def test_collect_before_completion_raises(tmp_path: Path, fake_cp2k) -> None:
    service = make_service(tmp_path, fake_cp2k)
    with pytest.raises(LabelingError, match="has not finished"):
        service.collect(service.prepare(region("h2_cluster")))


def test_enabled_embedding_is_refused_before_a_task_exists(tmp_path: Path, fake_cp2k) -> None:
    config = CP2KConfig(executable=str(fake_cp2k.executable))
    service = LabelingService(CP2KBackend(config, embedding_enabled=True), LocalScheduler(), task_root=tmp_path / "tasks")
    with pytest.raises(UnsupportedCombinationError, match="point-charge embedding"):
        service.prepare(region("h2_cluster"))
    assert not list((tmp_path / "tasks").rglob("task.json"))


def test_service_from_config_uses_the_registry(tmp_path: Path, fake_cp2k) -> None:
    config = {
        "backends": {"dft": {"engine": "cp2k"}, "scheduler": {"engine": "local"}, "discover_entry_points": False},
        "cp2k": {"executable": str(fake_cp2k.executable), "task_dir": "labels"},
        "scheduler": {"command_timeout_seconds": 30},
    }
    service = LabelingService.from_config(config, base_dir=tmp_path, registry=BackendRegistry())
    assert isinstance(service.scheduler, LocalScheduler)
    assert service.task_root == (tmp_path / "labels").resolve()
    task = run_to_end(service, service.submit(service.prepare(region("h2_cluster"))))
    assert task.validation is ValidationStatus.VALID
