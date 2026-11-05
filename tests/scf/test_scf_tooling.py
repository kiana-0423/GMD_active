"""The tools/scf scripts start, and the SCF tooling can drive the GMD scheduler adapters."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gmd_active_learning.adapters.dft.scf.execution import (
    ExecutionRequest,
    JobState,
    LocalSchedulerBackend,
    resolve_scheduler,
)
from gmd_active_learning.core.exceptions import ConfigurationError

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = sorted(path.stem for path in (REPO / "tools/scf/scripts").glob("*.py") if path.stem != "__init__")


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_script_answers_help(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", f"tools.scf.scripts.{script}", "--help"],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage" in completed.stdout.lower()


def _wait(scheduler, job, timeout: float = 30.0) -> JobState:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = scheduler.poll(job)
        if state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
            return state
        time.sleep(0.05)
    return JobState.UNKNOWN


def request(tmp_path: Path) -> ExecutionRequest:
    return ExecutionRequest.from_command(
        ["/bin/sh", "-c", 'echo "$SCF_MARKER"'],
        work_dir=tmp_path / "job",
        stdout_path="job.out",
        stderr_path="job.err",
        environment={"SCF_MARKER": "hello"},
    )


def test_the_local_backend_runs_a_request(tmp_path: Path) -> None:
    scheduler = LocalSchedulerBackend()

    assert _wait(scheduler, scheduler.submit(request(tmp_path))) is JobState.COMPLETED
    assert (tmp_path / "job" / "job.out").read_text().strip() == "hello"


def test_the_gmd_local_scheduler_is_driven_through_the_bridge(tmp_path: Path) -> None:
    try:
        scheduler = resolve_scheduler("gmd-local")
    except Exception as exc:  # the scheduler area may not be installed
        pytest.skip(f"GMD scheduler adapters unavailable: {exc}")

    job = scheduler.submit(request(tmp_path))

    assert _wait(scheduler, job) is JobState.COMPLETED
    assert (tmp_path / "job" / "job.out").read_text().strip() == "hello"


def test_an_unknown_scheduler_is_refused_not_run_locally() -> None:
    with pytest.raises(ConfigurationError, match="unknown scheduler"):
        resolve_scheduler("magic")
