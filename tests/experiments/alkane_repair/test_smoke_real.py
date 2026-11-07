"""The whole repair-benchmark chain on real CP2K (smoke-only settings), GMD and GMD_SGT (about 10 minutes)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from .conftest import copy_profile

pytestmark = [pytest.mark.cp2k, pytest.mark.integration]


def test_smoke_profile_end_to_end_and_cached_rerun(tmp_path: Path) -> None:
    if shutil.which("cp2k.psmp") is None:
        pytest.skip("cp2k.psmp not on PATH")
    from tools.experiments.alkane_repair.cli import main

    bench = str(copy_profile(tmp_path))
    art = tmp_path / "art"

    def run(*args: str) -> int:
        return main([*args[:1], "--benchmark", bench, *args[1:]])

    chain = [("generate",), ("freeze-protocol",), ("plan",), ("preflight",), ("label", "--stage", "initial_model", "--qualification-stage", "0"),
             ("qualify", "--stage", "0"), ("label", "--stage", "development"), ("fidelity-audit",), ("failures",), ("label", "--stage", "repair_val"),
             ("label", "--stage", "final_test"), ("calibrate-monitor", "--arm", "C_committee"), ("calibrate-monitor", "--arm", "D_phal"),
             ("repair-run",), ("evaluate", "--reason", "test"), ("report",)]
    for step in chain:
        assert run(*step) == 0, step
    labels = (art / "labels.jsonl").read_text()
    evaluation = json.loads((art / "evaluation" / "sealed_evaluation.json").read_text())
    assert not evaluation["partial"] and len(evaluation["runs"]) == 6
    assert all(json.loads((p / "repair_run.json").read_text())["final_stage"] == "done" for p in (art / "repair_runs").iterdir() if (p / "repair_run.json").is_file())
    report = json.loads((art / "report" / "report.json").read_text())
    assert report["assessment"]["implementation"]["repair_runs"] == 6 and report["assessment"]["scientific_status"].startswith("smoke profile")
    closed = [r["localisation"] for r in report["runs"] if r.get("localisation")]
    assert len(closed) == 4 and all(loc["all_label_weights_match_config"] and loc["end_to_end_wall_s"] > 0 for loc in closed)
    assert all(label["force_weights_by_role"]["core"] == [1.0] for loc in closed for label in loc["labels"])
    # cached re-execution: nothing is recomputed, the sealed evaluation is reproduced, and the access is logged again
    for step in chain:
        assert run(*step) == 0, step
    assert (art / "labels.jsonl").read_text() == labels
    again = json.loads((art / "evaluation" / "sealed_evaluation.json").read_text())
    assert [r["outcome"] for r in again["runs"]] == [r["outcome"] for r in evaluation["runs"]]
    assert len((art / "sealed_access.jsonl").read_text().splitlines()) == 2
