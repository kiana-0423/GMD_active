"""Fake Slurm/PBS command-line tools placed on PATH for scheduler contract tests.

Each fake (sbatch, squeue, sacct, scancel, qsub, qstat, qdel) is a tiny shell shim that runs
``fake_sched.py`` with the tool name. State lives in a directory the test controls:

- ``behavior.json``: ``{"<tool>": "ok" | "fail" | "silent" | "hang"}`` and ``"run_script": bool``
  (sbatch/qsub execute the submitted script synchronously, so the real exit-code file appears);
- ``jobs/<id>.live``: text the live query prints (squeue ``%T`` line or ``qstat -f`` body);
- ``jobs/<id>.history``: text of the history query (``sacct -P`` line or ``qstat -x -f`` body);
- ``calls.log``: one JSON line per invocation (argv + cwd).
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

FAKE_SCHED = r'''
import json, os, subprocess, sys, time
from pathlib import Path

state = Path(os.environ["FAKE_SCHED_DIR"])
tool, args = sys.argv[1], sys.argv[2:]
with (state / "calls.log").open("a") as log:
    log.write(json.dumps({"tool": tool, "args": args, "cwd": os.getcwd()}) + "\n")
behavior = json.loads((state / "behavior.json").read_text()) if (state / "behavior.json").exists() else {}
mode = behavior.get(tool, "ok")
if mode == "hang":
    time.sleep(30)
if mode == "fail":
    sys.stderr.write(f"{tool}: simulated failure\n")
    sys.exit(1)
jobs = state / "jobs"
jobs.mkdir(exist_ok=True)

def next_id():
    counter = state / "counter"
    value = int(counter.read_text()) + 1 if counter.exists() else 42
    counter.write_text(str(value))
    return str(value)

def job_arg():
    return [a for a in args if not a.startswith("-") and a not in {"%T"}][-1]

if tool in {"sbatch", "qsub"}:
    script = Path(os.getcwd()) / args[-1]
    job = next_id()
    (jobs / f"{job}.script").write_text(script.read_text())
    if behavior.get("run_script"):
        subprocess.run(["/bin/bash", str(script)], cwd=os.getcwd())
    if mode == "silent":
        sys.exit(0)
    print(f"Submitted batch job {job}" if tool == "sbatch" else f"{job}.fakepbs")
    sys.exit(0)

if tool == "squeue":
    job = args[args.index("-j") + 1]
    live = jobs / f"{job}.live"
    if not live.exists():
        sys.stderr.write("slurm_load_jobs error: Invalid job id specified\n")
        sys.exit(1)
    print(live.read_text().strip())
    sys.exit(0)

if tool == "sacct":
    job = args[args.index("-j") + 1]
    history = jobs / f"{job}.history"
    if history.exists():
        print(history.read_text().strip())
    sys.exit(0)

if tool == "qstat":
    job = args[-1].split(".")[0]
    if "-x" in args:
        history = jobs / f"{job}.history"
        if history.exists():
            print(history.read_text())
            sys.exit(0)
        sys.stderr.write(f"qstat: Unknown Job Id {args[-1]}\n")
        sys.exit(153)
    live = jobs / f"{job}.live"
    if not live.exists():
        sys.stderr.write(f"qstat: Unknown Job Id {args[-1]}\n")
        sys.exit(153)
    print(live.read_text())
    sys.exit(0)

if tool in {"scancel", "qdel"}:
    job = args[-1].split(".")[0]
    (jobs / f"{job}.live").unlink(missing_ok=True)
    (jobs / f"{job}.cancelled").write_text("1")
    if tool == "scancel":
        (jobs / f"{job}.history").write_text("CANCELLED by 1000|0:15")
    else:
        (jobs / f"{job}.history").write_text(f"Job Id: {job}.fakepbs\n    job_state = F\n    Exit_status = 271\n")
    sys.exit(0)
sys.exit(2)
'''

TOOLS = ("sbatch", "squeue", "sacct", "scancel", "qsub", "qstat", "qdel")


class FakeScheduler:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = root / "state"
        self.bin = root / "bin"
        self.state.mkdir(parents=True)
        self.bin.mkdir()
        script = self.bin / "fake_sched.py"
        script.write_text(FAKE_SCHED, encoding="utf-8")
        for tool in TOOLS:
            shim = self.bin / tool
            shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" {tool} "$@"\n', encoding="utf-8")
            shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    def behave(self, **behavior: Any) -> None:
        (self.state / "behavior.json").write_text(json.dumps(behavior), encoding="utf-8")

    def set_live(self, job_id: str, text: str | None) -> None:
        path = self.state / "jobs" / f"{job_id.split('.')[0]}.live"
        path.parent.mkdir(exist_ok=True)
        if text is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(text, encoding="utf-8")

    def set_history(self, job_id: str, text: str | None) -> None:
        path = self.state / "jobs" / f"{job_id.split('.')[0]}.history"
        path.parent.mkdir(exist_ok=True)
        if text is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(text, encoding="utf-8")

    def calls(self, tool: str | None = None) -> list[dict[str, Any]]:
        log = self.state / "calls.log"
        if not log.exists():
            return []
        entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        return [entry for entry in entries if tool is None or entry["tool"] == tool]

    def submitted_script(self, job_id: str) -> str:
        return (self.state / "jobs" / f"{job_id.split('.')[0]}.script").read_text(encoding="utf-8")


@pytest.fixture()
def fake_scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeScheduler:
    fake = FakeScheduler(tmp_path / "fake_sched")
    monkeypatch.setenv("FAKE_SCHED_DIR", str(fake.state))
    monkeypatch.setenv("PATH", f"{fake.bin}{os.pathsep}{os.environ.get('PATH', '')}")
    return fake
