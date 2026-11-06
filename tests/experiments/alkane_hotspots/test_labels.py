"""Label runner with the real LabelingService + LocalScheduler and the scripted fake CP2K of
``tests/labeling/conftest.py``: record schema, idempotency, resume, budget refusal, settings hash."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ._helpers import ROOT, tiny_config
from tools.experiments.alkane_hotspots.families import generate, load_parents, load_positions, write_generation
from tools.experiments.alkane_hotspots.fidelity import label_requests, prepare
from tools.experiments.alkane_hotspots.labels import (
    LabelRequest,
    LabelRunner,
    LabelStore,
    build_service,
    cp2k_provenance,
    label_id_for,
    parent_region,
)

_spec = importlib.util.spec_from_file_location("labeling_fake_cp2k", ROOT / "tests" / "labeling" / "conftest.py")
assert _spec is not None and _spec.loader is not None
_fake_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fake_module)
FakeCP2K = _fake_module.FakeCP2K

RECORD_KEYS = {"label_id", "kind", "parent_id", "fragment", "region_path", "status", "forces_path", "energy_ev", "scf_steps", "cp2k", "cost",
               "task_id", "created_at"}


@pytest.fixture()
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeCP2K(tmp_path / "fake_cp2k")
    monkeypatch.setenv("FAKE_CP2K_DIR", str(fake.state))
    monkeypatch.delenv("CP2K_DATA_DIR", raising=False)

    def make(**overrides):
        overrides.setdefault("cp2k", {"executable": str(fake.executable)})
        loaded = tiny_config(tmp_path, **overrides)
        if not (loaded.artifacts_root / "parents.jsonl").exists():
            write_generation(generate(loaded.config), loaded.artifacts_root)
        return loaded

    return fake, make


def _runner(loaded, echo=None):
    service = build_service(loaded)
    meta = cp2k_provenance(service, smoke_only=True, probe=False)
    lines: list[str] = []
    return LabelRunner(service, LabelStore(loaded.artifacts_root), loaded, meta, echo=echo or lines.append), lines


def _full_requests(loaded, n: int | None = None) -> list[LabelRequest]:
    parents = [p for p in load_parents(loaded.artifacts_root) if p["split"] == "calibration"][:n]
    return [LabelRequest("full", p["parent_id"], p["chain_length"], parent_region(p, load_positions(loaded.artifacts_root, p))) for p in parents]


def test_records_follow_the_schema_with_measured_cost(setup) -> None:
    fake, make = setup
    loaded = make()
    runner, lines = _runner(loaded)
    requests = _full_requests(loaded, 2)
    summary = runner.run(requests)
    assert len(summary.completed) == 2 and not summary.failed
    store = LabelStore(loaded.artifacts_root)
    for request in requests:
        record = store.get(label_id_for(request.region, runner.service.settings_hash))
        assert RECORD_KEYS <= set(record) and record["status"] == "valid" and record["kind"] == "full" and record["fragment"] is None
        forces = np.load(loaded.artifacts_root / record["forces_path"])
        assert forces.shape == (request.region.n_atoms, 3)
        assert record["cp2k"]["settings_hash"] == runner.service.settings_hash and record["cp2k"]["settings"]["cutoff"] == 200.0
        assert record["cost"]["wall_s"] > 0 and record["cost"]["cpu_core_s"] == pytest.approx(record["cost"]["wall_s"] * 1)
        assert record["cost"]["timing_sources"] == ["local_job_record"] and record["cost"]["attempts"] == 1
        assert record["energy_ev"] is not None and record["cp2k"]["version"] == "2026.2"
    assert any(line.startswith("RUN") for line in lines) and any(line.startswith("DONE") for line in lines)


def test_rerun_reuses_finished_labels_without_new_jobs(setup) -> None:
    fake, make = setup
    loaded = make()
    runner, _ = _runner(loaded)
    runner.run(_full_requests(loaded, 2))
    calls = len(fake.calls())
    again, _ = _runner(loaded)
    summary = again.run(_full_requests(loaded, 2))
    assert len(summary.reused) == 2 and not summary.completed and len(fake.calls()) == calls


def test_resume_after_interruption_continues_the_existing_task(setup) -> None:
    fake, make = setup
    loaded = make()
    runner, _ = _runner(loaded)
    request = _full_requests(loaded, 1)[0]
    task = runner.service.submit(runner.service.prepare(request.region))  # an earlier invocation launched it, then died
    resumed, lines = _runner(loaded)
    summary = resumed.run([request])
    assert summary.completed == [label_id_for(request.region, resumed.service.settings_hash)]
    assert any(line.startswith("RESUME") for line in lines) and len(fake.calls()) == 1
    assert LabelStore(loaded.artifacts_root).get(summary.completed[0])["task_id"] == task.task_id


def test_budget_refusal_writes_budget_skipped_records(setup) -> None:
    fake, make = setup
    loaded = make(budgets={"max_full_labels": 1})
    runner, lines = _runner(loaded)
    requests = _full_requests(loaded, 2)
    summary = runner.run(requests)
    assert len(summary.completed) == 1 and len(summary.skipped) == 1
    skipped = LabelStore(loaded.artifacts_root).get(summary.skipped[0])
    assert skipped["status"] == "budget_skipped" and "max_full_labels=1" in skipped["reasons"][0]
    assert skipped["budget_estimate_core_s"] > 0 and skipped["task_id"] is None
    assert any(line.startswith("SKIP") and "core-h" in line for line in lines)
    assert len(fake.calls()) == 1


def test_core_hour_budget_uses_the_walltime_bound_before_any_measurement(setup) -> None:
    fake, make = setup
    loaded = make(budgets={"max_total_cpu_core_hours": 0.1, "per_job_walltime_s": 600})  # first-job bound 600 core-s > 0.1 h
    runner, _ = _runner(loaded)
    summary = runner.run(_full_requests(loaded, 1))
    assert len(summary.skipped) == 1 and not fake.calls()
    record = LabelStore(loaded.artifacts_root).get(summary.skipped[0])
    assert "upper bound" in record["budget_estimate_source"]


def test_parent_and_fragment_use_identical_settings(setup) -> None:
    fake, make = setup
    loaded = make()
    service = build_service(loaded)
    prepare(loaded, "calibration", service.settings_hash)
    requests = label_requests(loaded, "calibration")
    kinds = {r.kind for r in requests}
    assert kinds == {"full", "fragment"}
    hashes = set()
    for request in requests[:6]:
        task = service.prepare(request.region)
        hashes.add(task.settings_hash)
    assert hashes == {service.settings_hash}
    runner, _ = _runner(loaded)
    fragment = next(r for r in requests if r.kind == "fragment")
    full = next(r for r in requests if r.kind == "full")
    runner.run([full, fragment])
    store = LabelStore(loaded.artifacts_root)
    records = [store.get(label_id_for(r.region, service.settings_hash)) for r in (full, fragment)]
    assert records[0]["cp2k"]["settings_hash"] == records[1]["cp2k"]["settings_hash"] == service.settings_hash
    assert records[0]["cp2k"]["settings"] == records[1]["cp2k"]["settings"]
    frag = records[1]["fragment"]
    assert {"policy", "buffer", "core_parent_ids", "n_atoms", "n_core", "n_caps"} <= set(frag) and frag["n_caps"] >= 1
    region = json.loads((loaded.artifacts_root / records[1]["region_path"]).read_text())
    assert region["multiplicity"] == 1 and region["charge"] == 0


def test_mismatched_provenance_is_refused(setup) -> None:
    _, make = setup
    loaded = make()
    service = build_service(loaded)
    with pytest.raises(ValueError):
        LabelRunner(service, LabelStore(loaded.artifacts_root), loaded, {"settings_hash": "other", "cores": 1})


def test_launch_failure_is_recorded_not_dropped(setup, monkeypatch: pytest.MonkeyPatch) -> None:
    _, make = setup
    loaded = make()
    runner, _ = _runner(loaded)

    def broken(region):
        raise RuntimeError("input generation failed")

    monkeypatch.setattr(runner.service, "prepare", broken)
    summary = runner.run(_full_requests(loaded, 1))
    record = LabelStore(loaded.artifacts_root).get(summary.failed[0])
    assert record["status"] == "failed" and "input generation failed" in record["reasons"][0]
