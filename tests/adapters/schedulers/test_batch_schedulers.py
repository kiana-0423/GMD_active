"""Slurm and PBS contract tests against fake sbatch/squeue/sacct/scancel/qsub/qstat/qdel on PATH."""

from __future__ import annotations

from pathlib import Path

import pytest

from gmd_active_learning.adapters.schedulers.base import EXIT_CODE_FILE, read_exit_code
from gmd_active_learning.adapters.schedulers.config import SchedulerConfig, SchedulerResources
from gmd_active_learning.adapters.schedulers.pbs import PBSScheduler, parse_qstat_full, pbs_state
from gmd_active_learning.adapters.schedulers.slurm import SLURM_STATES, SlurmScheduler, slurm_state
from gmd_active_learning.contracts.labels import ExecutionStatus
from gmd_active_learning.core.exceptions import SchedulerError

RESOURCES = SchedulerResources(nodes=1, ntasks=4, cpus_per_task=2, walltime="00:30:00", memory="8G", queue="debug", account="grp")


def _slurm(**overrides: object) -> SlurmScheduler:
    return SlurmScheduler(SchedulerConfig.model_validate({"command_timeout_seconds": 20.0, **overrides}))


def _pbs(**overrides: object) -> PBSScheduler:
    return PBSScheduler(SchedulerConfig.model_validate({"command_timeout_seconds": 20.0, **overrides}))


# -- submission ------------------------------------------------------------------------------------
def test_slurm_submit_renders_typed_resources_and_parses_job_id(fake_scheduler, tmp_path: Path) -> None:
    job = _slurm().submit(tmp_path / "w", ["cp2k.psmp", "-i", "x.inp"], RESOURCES, environment={"OMP_NUM_THREADS": "2"}, job_name="lbl")
    assert job.job_id == "42" and job.state is ExecutionStatus.SUBMITTED and job.backend == "slurm"
    script = fake_scheduler.submitted_script("42")
    lines = script.splitlines()
    for directive in ("--ntasks=4", "--cpus-per-task=2", "--time=00:30:00", "--mem=8G", "--partition=debug", "--account=grp", "--job-name=lbl"):
        assert f"#SBATCH {directive}" in lines
    assert not any(line.startswith("#PBS") for line in lines)
    assert lines.index("set -u") > max(i for i, line in enumerate(lines) if line.startswith("#SBATCH"))
    assert "export OMP_NUM_THREADS=2" in lines
    assert fake_scheduler.calls("sbatch")[0]["cwd"] == str((tmp_path / "w").resolve())


def test_pbs_submit_renders_pbs_resources_not_slurm_directives(fake_scheduler, tmp_path: Path) -> None:
    job = _pbs().submit(tmp_path / "w", ["true"], RESOURCES.merged(SchedulerResources(extra_directives=["-W group_list=grp"])))
    assert job.job_id == "42.fakepbs"
    lines = fake_scheduler.submitted_script(job.job_id).splitlines()
    assert "#PBS -l select=1:ncpus=8:mpiprocs=4:ompthreads=2:mem=8gb" in lines
    assert "#PBS -l walltime=00:30:00" in lines
    assert "#PBS -q debug" in lines and "#PBS -A grp" in lines and "#PBS -W group_list=grp" in lines
    assert not any("SBATCH" in line for line in lines)


@pytest.mark.parametrize("scheduler, foreign", [(_slurm, "#PBS -l walltime=1:00:00"), (_pbs, "#SBATCH --time=10")])
def test_foreign_directives_are_rejected(fake_scheduler, tmp_path: Path, scheduler, foreign: str) -> None:
    with pytest.raises(SchedulerError, match="another scheduler"):
        scheduler().submit(tmp_path / "w", ["true"], SchedulerResources(extra_directives=[foreign]))
    assert fake_scheduler.calls("sbatch") == [] and fake_scheduler.calls("qsub") == []


@pytest.mark.parametrize("scheduler, tool", [(_slurm, "sbatch"), (_pbs, "qsub")])
def test_failed_submit_raises_scheduler_error_with_stderr(fake_scheduler, tmp_path: Path, scheduler, tool: str) -> None:
    fake_scheduler.behave(**{tool: "fail"})
    with pytest.raises(SchedulerError, match="simulated failure"):
        scheduler().submit(tmp_path / "w", ["true"])


@pytest.mark.parametrize("scheduler, tool", [(_slurm, "sbatch"), (_pbs, "qsub")])
def test_silent_submit_raises_instead_of_inventing_an_id(fake_scheduler, tmp_path: Path, scheduler, tool: str) -> None:
    fake_scheduler.behave(**{tool: "silent"})
    with pytest.raises(SchedulerError, match="did not report a job id"):
        scheduler().submit(tmp_path / "w", ["true"])


@pytest.mark.parametrize("scheduler, tool", [(_slurm, "sbatch"), (_pbs, "qsub")])
def test_hanging_submit_times_out(fake_scheduler, tmp_path: Path, scheduler, tool: str) -> None:
    fake_scheduler.behave(**{tool: "hang"})
    with pytest.raises(SchedulerError, match="timed out"):
        scheduler(command_timeout_seconds=0.5).submit(tmp_path / "w", ["true"])


def test_missing_submit_binary_raises_scheduler_error(tmp_path: Path) -> None:
    scheduler = _slurm(slurm={"sbatch": str(tmp_path / "no-such-sbatch")})
    with pytest.raises(SchedulerError, match="not found"):
        scheduler.submit(tmp_path / "w", ["true"])
    assert scheduler.check_runtime().available is False


def test_submit_refuses_a_work_dir_with_a_finished_job(fake_scheduler, tmp_path: Path) -> None:
    work = tmp_path / "w"
    work.mkdir()
    (work / EXIT_CODE_FILE).write_text("0\n")
    with pytest.raises(SchedulerError, match="already holds a finished job"):
        _slurm().submit(work, ["true"])


# -- polling: Slurm --------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [("RUNNING", ExecutionStatus.RUNNING), ("PENDING", ExecutionStatus.QUEUED)])
def test_slurm_live_squeue_state(fake_scheduler, tmp_path: Path, raw: str, expected: ExecutionStatus) -> None:
    scheduler = _slurm()
    job = scheduler.submit(tmp_path / "w", ["true"])
    fake_scheduler.set_live(job.job_id, raw)
    assert scheduler.poll(job) is expected
    assert job.raw_state == raw
    assert fake_scheduler.calls("sacct") == []


@pytest.mark.parametrize(
    "history, expected, exit_code",
    [
        ("COMPLETED|0:0", ExecutionStatus.COMPLETED, 0),
        ("FAILED|3:0", ExecutionStatus.FAILED, 3),
        ("OUT_OF_MEMORY|0:125", ExecutionStatus.FAILED, 253),
        ("TIMEOUT|0:15", ExecutionStatus.FAILED, 143),
        ("COMPLETED+|0:0", ExecutionStatus.COMPLETED, 0),
        ("CANCELLED by 1234|0:15", ExecutionStatus.CANCELLED, 143),
        ("BOOT_FAIL|0:0", ExecutionStatus.FAILED, 0),
        ("DEADLINE|0:0", ExecutionStatus.FAILED, 0),
    ],
)
def test_slurm_falls_back_to_sacct_with_exit_code(
    fake_scheduler, tmp_path: Path, history: str, expected: ExecutionStatus, exit_code: int
) -> None:
    scheduler = _slurm()
    job = scheduler.submit(tmp_path / "w", ["true"])
    fake_scheduler.set_history(job.job_id, history)
    assert scheduler.poll(job) is expected
    assert job.exit_code == exit_code
    if history.startswith(("TIMEOUT", "DEADLINE")):
        assert job.metadata["walltime_exceeded"] is True


def test_slurm_state_table_has_explicit_entries_for_previously_unmapped_states() -> None:
    for raw in ("BOOT_FAIL", "DEADLINE", "REQUEUED", "SUSPENDED"):
        assert raw in SLURM_STATES
    assert slurm_state("REQUEUED") is ExecutionStatus.QUEUED
    assert slurm_state("SUSPENDED") is ExecutionStatus.RUNNING
    assert slurm_state("SOMETHING_NEW") is ExecutionStatus.UNKNOWN


def test_slurm_unknown_when_neither_query_knows_the_job(fake_scheduler, tmp_path: Path) -> None:
    scheduler = _slurm()
    job = scheduler.submit(tmp_path / "w", ["true"])
    assert scheduler.poll(job) is ExecutionStatus.UNKNOWN
    assert "poll_error" in job.metadata  # squeue's "Invalid job id" is recorded


def test_slurm_query_timeout_gives_unknown_not_an_exception(fake_scheduler, tmp_path: Path) -> None:
    scheduler = _slurm(command_timeout_seconds=0.5)
    job = scheduler.submit(tmp_path / "w", ["true"])
    fake_scheduler.behave(squeue="hang", sacct="hang")
    assert scheduler.poll(job) is ExecutionStatus.UNKNOWN
    assert "timed out" in job.metadata["poll_error"]


def test_slurm_reconcile_uses_exit_code_file_when_scheduler_forgot(fake_scheduler, tmp_path: Path) -> None:
    fake_scheduler.behave(run_script=True)
    scheduler = _slurm()
    job = scheduler.submit(tmp_path / "w", ["sh", "-c", "exit 7"])
    assert read_exit_code(job.work_dir) == 7  # the real job wrapper ran and recorded it
    assert scheduler.poll(job) is ExecutionStatus.UNKNOWN
    assert scheduler.reconcile(job) is ExecutionStatus.FAILED
    assert job.exit_code == 7


def test_slurm_reconcile_stays_unknown_without_evidence(fake_scheduler, tmp_path: Path) -> None:
    scheduler = _slurm()
    job = scheduler.submit(tmp_path / "w", ["true"])
    assert scheduler.reconcile(job) is ExecutionStatus.UNKNOWN
    assert len(fake_scheduler.calls("sbatch")) == 1  # never resubmits


def test_slurm_cancel(fake_scheduler, tmp_path: Path) -> None:
    scheduler = _slurm()
    job = scheduler.submit(tmp_path / "w", ["sleep", "100"])
    fake_scheduler.set_live(job.job_id, "RUNNING")
    assert scheduler.cancel(job) is ExecutionStatus.CANCELLED
    assert fake_scheduler.calls("scancel")[0]["args"] == [job.job_id]
    assert scheduler.poll(job) is ExecutionStatus.CANCELLED


def test_slurm_cancel_failure_on_a_live_job_raises(fake_scheduler, tmp_path: Path) -> None:
    scheduler = _slurm()
    job = scheduler.submit(tmp_path / "w", ["sleep", "100"])
    fake_scheduler.set_live(job.job_id, "RUNNING")
    fake_scheduler.behave(scancel="fail")
    with pytest.raises(SchedulerError, match="simulated failure"):
        scheduler.cancel(job)


def test_cancel_of_finished_job_is_a_no_op(fake_scheduler, tmp_path: Path) -> None:
    scheduler = _slurm()
    job = scheduler.submit(tmp_path / "w", ["true"])
    fake_scheduler.set_history(job.job_id, "COMPLETED|0:0")
    assert scheduler.cancel(job) is ExecutionStatus.COMPLETED
    assert fake_scheduler.calls("scancel") == []


# -- polling: PBS ----------------------------------------------------------------------------------
def test_pbs_live_state(fake_scheduler, tmp_path: Path) -> None:
    scheduler = _pbs()
    job = scheduler.submit(tmp_path / "w", ["true"])
    fake_scheduler.set_live(job.job_id, f"Job Id: {job.job_id}\n    job_state = R\n")
    assert scheduler.poll(job) is ExecutionStatus.RUNNING
    fake_scheduler.set_live(job.job_id, f"Job Id: {job.job_id}\n    job_state = Q\n")
    assert scheduler.poll(job) is ExecutionStatus.QUEUED


@pytest.mark.parametrize(
    "state, exit_status, expected",
    [
        ("F", "0", ExecutionStatus.COMPLETED),
        ("F", "17", ExecutionStatus.FAILED),
        ("F", "-29", ExecutionStatus.FAILED),
        ("C", "0", ExecutionStatus.COMPLETED),
        ("C", "1", ExecutionStatus.FAILED),  # PHAL reported COMPLETED here (F-26)
        ("F", None, ExecutionStatus.FAILED),
        ("C", None, ExecutionStatus.UNKNOWN),
    ],
)
def test_pbs_history_finished_states_check_exit_status(
    fake_scheduler, tmp_path: Path, state: str, exit_status: str | None, expected: ExecutionStatus
) -> None:
    scheduler = _pbs()
    job = scheduler.submit(tmp_path / "w", ["true"])
    body = f"Job Id: {job.job_id}\n    job_state = {state}\n"
    if exit_status is not None:
        body += f"    Exit_status = {exit_status}\n"
    fake_scheduler.set_history(job.job_id, body)
    assert scheduler.poll(job) is expected
    assert [call["args"][:2] for call in fake_scheduler.calls("qstat")] == [["-f", job.job_id], ["-x", "-f"]]
    if exit_status is not None:
        assert job.exit_code == int(exit_status)
    if exit_status == "-29":
        assert job.metadata["walltime_exceeded"] is True


def test_pbs_history_query_can_be_disabled(fake_scheduler, tmp_path: Path) -> None:
    scheduler = _pbs(pbs={"history_query_enabled": False})
    job = scheduler.submit(tmp_path / "w", ["true"])
    fake_scheduler.set_history(job.job_id, "job_state = F\nExit_status = 0\n")
    assert scheduler.poll(job) is ExecutionStatus.UNKNOWN
    assert len(fake_scheduler.calls("qstat")) == 1


def test_pbs_reconcile_from_exit_code_file(fake_scheduler, tmp_path: Path) -> None:
    fake_scheduler.behave(run_script=True)
    scheduler = _pbs()
    job = scheduler.submit(tmp_path / "w", ["sh", "-c", "echo hi"])
    assert scheduler.poll(job) is ExecutionStatus.UNKNOWN
    assert scheduler.reconcile(job) is ExecutionStatus.COMPLETED
    assert (Path(job.work_dir) / "job.stdout").read_text() == "hi\n"


def test_pbs_cancel_marks_cancelled_even_with_signal_exit_status(fake_scheduler, tmp_path: Path) -> None:
    scheduler = _pbs()
    job = scheduler.submit(tmp_path / "w", ["sleep", "100"])
    fake_scheduler.set_live(job.job_id, "job_state = R\n")
    assert scheduler.cancel(job) is ExecutionStatus.CANCELLED
    assert scheduler.poll(job) is ExecutionStatus.CANCELLED  # history says F / 271
    assert job.exit_code == 271


def test_qstat_continuation_lines_are_joined() -> None:
    fields = parse_qstat_full("Job Id: 1.x\n    Variable_List = A=1,\n\tB=2\n    job_state = F\n    Exit_status = 0\n")
    assert fields["variable_list"] == "A=1,B=2"
    assert pbs_state(fields)[0] is ExecutionStatus.COMPLETED


def test_torque_flavor_renders_nodes_ppn(fake_scheduler, tmp_path: Path) -> None:
    job = _pbs(pbs={"flavor": "torque"}).submit(tmp_path / "w", ["true"], SchedulerResources(nodes=2, ntasks_per_node=4, memory="2G"))
    lines = fake_scheduler.submitted_script(job.job_id).splitlines()
    assert "#PBS -l nodes=2:ppn=4" in lines and "#PBS -l mem=2gb" in lines
