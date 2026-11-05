"""The real closed loop end to end: GMD -> monitoring -> extraction -> CP2K -> dataset ->
GMD_SGT fine-tuning -> export validation -> GMD smoke test -> gated promotion -> resumed GMD.

Runs ``examples/real_cycle/run_real_cycle.py`` (about 3 minutes on a laptop). Skipped, with the
reason, when GMD (Torch build), GMD_SGT or CP2K is unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.cp2k, pytest.mark.slow]
REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
GMD_BIN = Path(os.environ.get("GMD_EXECUTABLE", WORKSPACE / "GMD" / "build-torch" / "gmd"))
SGT = Path(os.environ.get("GMD_SGT_PATH", WORKSPACE / "GMD_SGT"))
CP2K = os.environ.get("CP2K_EXECUTABLE", "cp2k.psmp")
CP2K_DATA = os.environ.get("CP2K_DATA_DIR", "/opt/homebrew/share/cp2k/data")


def _require() -> None:
    if not GMD_BIN.is_file():
        pytest.skip(f"GMD Torch build not found at {GMD_BIN}")
    if not (SGT / "gmd_sgt").is_dir():
        pytest.skip(f"GMD_SGT not found at {SGT}")
    if shutil.which(CP2K) is None:
        pytest.skip(f"CP2K executable {CP2K} not on PATH")
    if not Path(CP2K_DATA).is_dir():
        pytest.skip(f"CP2K data directory {CP2K_DATA} not found (set CP2K_DATA_DIR)")


def _run(work: Path, *extra: str) -> dict:
    command = [sys.executable, "-W", "ignore", str(REPO / "examples/real_cycle/run_real_cycle.py"), "--work-dir", str(work),
               "--gmd", str(GMD_BIN), "--gmd-sgt", str(SGT), "--cp2k", CP2K, "--cp2k-data", CP2K_DATA, "--n-seed", "6", *extra]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=3600)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    return json.loads((work / "report.json").read_text())


def test_real_closed_loop_promotes_and_resumes_md_with_the_new_model(tmp_path: Path) -> None:
    _require()
    work = tmp_path / "cycle"
    report = _run(work, "--fresh")
    assert report["stage"] == "done"
    assert report["samples"] >= 1 and report["tasks"]["ingested"] == report["samples"]
    promotions = [p for p in report["promotions"] if p["action"] == "promote" and p["previous_model_id"]]
    assert promotions, report["promotions"]
    segments = report["segments"]
    assert len(segments) == 2 and segments[1]["model_id"] == promotions[0]["model_id"]
    status = json.loads((work / "run" / "md" / "segment_00001" / "status.json").read_text())
    assert status["model_changed_from"] and status["start_step"] == segments[0]["end_step"]

    # Re-running the finished run is a no-op: no new jobs, samples or models (idempotent recovery).
    tasks_before = sorted(p.name for p in (work / "cp2k_tasks").iterdir())
    again = _run(work)
    assert again["samples"] == report["samples"] and len(again["models"]) == len(report["models"])
    assert sorted(p.name for p in (work / "cp2k_tasks").iterdir()) == tasks_before
