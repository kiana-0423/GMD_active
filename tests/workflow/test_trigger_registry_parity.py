"""Retraining trigger vs PHAL (TRN-005) and model registry behaviour (TRN-006).

The PHAL trigger rule (retrain_trigger.py:_evaluate_trigger_condition) is re-implemented below as an
oracle. Every row of the table states PHAL's reason and GMD's reason; rows where they differ are
the deliberate deviations of finding F-13, each with its justification:

* initial_samples: PHAL fired on the very first labeled sample regardless of min_new_samples; GMD
  waits for min_new_samples unless ``train_on_first_samples`` is set (then it matches PHAL).
* manual with nothing new: PHAL retrained (re-merging old samples, then failed with no_samples when
  the directory was empty); GMD refuses with ``no_new_samples``.
* labels outstanding: PHAL had no notion of pending labels; GMD refuses to train on an incomplete
  selection (``labels_pending``) unless ``wait_for_pending_labels`` is false (then it matches PHAL).
* disabled: PHAL had no switch; GMD's ``enabled: false`` blocks automatic triggers, manual still works.
* state: PHAL advanced its state file in dry-run (swallowing the pending samples) and counted files;
  GMD counts ingested samples in the store and records ``last_trained_at`` only after a real training
  run (asserted in test_dry_run_free_state_advances_only_after_training).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gmd_active_learning.core.exceptions import WorkflowError
from gmd_active_learning.retraining.registry import GateReport, ModelRegistry
from gmd_active_learning.retraining.triggers import RetrainTriggerConfig, evaluate_trigger
from gmd_active_learning.workflow.store import WorkflowStore

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[2]


def phal_reason(*, force: bool, new_count: int, last_run_at: str | None, min_new_samples: int, interval_hours: float, now: datetime) -> str | None:
    """PHAL RetrainTrigger._evaluate_trigger_condition, verbatim logic."""

    if force:
        return "manual"
    if new_count >= min_new_samples:
        return "sample_count"
    if last_run_at is None and new_count > 0:
        return "initial_samples"
    if last_run_at is not None and new_count > 0:
        if now - datetime.fromisoformat(last_run_at) >= timedelta(hours=interval_hours):
            return "time_interval"
    return None


def _ago(hours: float) -> str:
    return (NOW - timedelta(hours=hours)).isoformat()


# (case, config overrides, manual, new samples, pending tasks, last trained, PHAL reason, GMD reason)
TABLE = [
    ("nothing new", {}, False, 0, 0, None, None, "no_new_samples"),
    ("nothing new, trained before", {}, False, 0, 0, _ago(48), None, "no_new_samples"),
    ("enough samples, never trained", {}, False, 10, 0, None, "sample_count", "sample_count"),
    ("enough samples, trained recently", {}, False, 12, 0, _ago(1), "sample_count", "sample_count"),
    ("few samples, interval elapsed", {}, False, 3, 0, _ago(24), "time_interval", "time_interval"),
    ("few samples, interval not elapsed", {}, False, 3, 0, _ago(23.9), None, "below_min_new_samples"),
    ("manual with new samples", {}, True, 2, 0, _ago(1), "manual", "manual"),
    ("manual beats sample_count", {}, True, 50, 0, None, "manual", "manual"),
    ("first sample (F-13)", {}, False, 1, 0, None, "initial_samples", "below_min_new_samples"),
    ("first sample, PHAL behaviour opted in", {"train_on_first_samples": True}, False, 1, 0, None, "initial_samples", "initial_samples"),
    ("manual with nothing new (F-13)", {}, True, 0, 0, _ago(1), "manual", "no_new_samples"),
    ("labels outstanding (F-13)", {}, False, 20, 3, None, "sample_count", "labels_pending"),
    ("labels outstanding, waiting disabled", {"wait_for_pending_labels": False}, False, 20, 3, None, "sample_count", "sample_count"),
    ("trigger disabled", {"enabled": False}, False, 20, 0, None, "sample_count", "disabled"),
    ("trigger disabled, manual", {"enabled": False}, True, 2, 0, None, "manual", "manual"),
    ("time trigger disabled", {"interval_hours": None}, False, 3, 0, _ago(1000), "time_interval", "below_min_new_samples"),
]
DEVIATIONS = {"first sample (F-13)", "manual with nothing new (F-13)", "labels outstanding (F-13)", "trigger disabled", "time trigger disabled"}


@pytest.mark.parametrize(("case", "overrides", "manual", "new", "pending", "last", "phal", "gmd"), TABLE, ids=[row[0] for row in TABLE])
def test_trigger_table_against_phal(case: str, overrides: dict, manual: bool, new: int, pending: int, last: str | None, phal: str | None, gmd: str) -> None:
    config = RetrainTriggerConfig(**{"min_new_samples": 10, "interval_hours": 24.0, **overrides})
    interval = 24.0 if config.interval_hours is None else config.interval_hours
    assert phal_reason(force=manual, new_count=new, last_run_at=last, min_new_samples=config.min_new_samples, interval_hours=interval, now=NOW) == phal
    decision = evaluate_trigger(config, new_samples=new, pending_tasks=pending, last_trained_at=last, manual=manual, now=NOW)
    assert decision.reason == gmd and decision.new_samples == new
    assert decision.due == (gmd in {"manual", "sample_count", "initial_samples", "time_interval"})
    if case in DEVIATIONS:
        assert (phal is not None) != decision.due or phal != gmd  # a documented deviation
    else:
        assert decision.due == (phal is not None)
        if decision.due:
            assert decision.reason == phal


def test_dry_run_free_state_advances_only_after_training(tmp_path: Path) -> None:
    """PHAL advanced last_run_at even in dry-run; GMD has no dry run and records it after training."""

    from tests_helpers_controller import run_one_cycle  # noqa: PLC0415 - local helper module

    store, trained = run_one_cycle(tmp_path, train=False)
    assert store.get_blob("trig:last_trained_at") is None and trained == []
    store, trained = run_one_cycle(tmp_path / "b", train=True)
    assert store.get_blob("trig:last_trained_at") is not None and len(trained) == 1


# ----------------------------------------------------------------------------- registry (TRN-006)
def _payload(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _register(registry: ModelRegistry, tmp_path: Path, tag: str, parent: str | None = None):  # noqa: ANN202
    return registry.register(checkpoint=_payload(tmp_path, f"{tag}.pt", tag.encode()), export=_payload(tmp_path, f"{tag}_x.pt", (tag + "x").encode()),
                             dataset_version=f"ds-{tag}", parent_model_id=parent, species=[1, 8], cutoff=4.0,
                             metrics={"force_rmse": 0.1}, train_mode="fresh", backend="gmd_sgt")


def test_register_promote_rollback_round_trip_publishes_the_export(tmp_path: Path) -> None:
    store = WorkflowStore(tmp_path / "s.sqlite")
    store.ensure_run("r", config_hash="c", initial_model_id=None, stage="sampling")
    registry = ModelRegistry(tmp_path / "models", store)
    v1 = _register(registry, tmp_path, "v1")
    registry.promote("r", v1.model_id, gate_report=None, reason="initial", force_initial=True)
    v2 = _register(registry, tmp_path, "v2", parent=v1.model_id)
    deployed = registry.promote("r", v2.model_id, gate_report=GateReport(True, {}), reason="gates passed")
    # The deployed GMD_SGT export is published through the ModelArtifact contract.
    assert deployed.export_path and Path(deployed.export_path).read_bytes() == b"v2x" and deployed.export_sha256
    assert registry.get(store.run("r").active_model_id).export_path == deployed.export_path
    assert registry.rollback("r", reason="unstable").model_id == v1.model_id
    assert [p["action"] for p in store.promotions("r")] == ["promote", "promote", "rollback"]


def test_reregistering_is_idempotent_and_never_overwrites(tmp_path: Path) -> None:
    store = WorkflowStore(tmp_path / "s.sqlite")
    registry = ModelRegistry(tmp_path / "models", store)
    first = _register(registry, tmp_path, "v1")
    again = _register(registry, tmp_path, "v1")
    assert again.model_id == first.model_id and again.created_at == first.created_at
    assert len(store.models()) == 1
    # Different bytes are a different model id: an existing id can never be overwritten.
    other = registry.register(checkpoint=_payload(tmp_path, "v1.pt", b"other weights"), export=None, dataset_version="ds-v1",
                              parent_model_id=None, species=[1, 8], cutoff=4.0, metrics={}, train_mode="fresh", backend="gmd_sgt")
    assert other.model_id != first.model_id and Path(first.checkpoint_path).read_bytes() == b"v1"


def test_sha_is_verified_at_promotion_and_rollback(tmp_path: Path) -> None:
    store = WorkflowStore(tmp_path / "s.sqlite")
    store.ensure_run("r", config_hash="c", initial_model_id=None, stage="sampling")
    registry = ModelRegistry(tmp_path / "models", store)
    v1 = _register(registry, tmp_path, "v1")
    registry.promote("r", v1.model_id, gate_report=None, reason="initial", force_initial=True)
    v2 = _register(registry, tmp_path, "v2", parent=v1.model_id)
    export = Path(v2.export_path)
    export.chmod(0o600)
    export.write_bytes(b"swapped behind the registry's back")
    with pytest.raises(WorkflowError, match="checksum"):
        registry.promote("r", v2.model_id, gate_report=GateReport(True, {}), reason="gates passed")
    assert store.run("r").active_model_id == v1.model_id


def test_concurrent_writers_in_threads_share_one_store(tmp_path: Path) -> None:
    WorkflowStore(tmp_path / "s.sqlite").close()
    errors: list[BaseException] = []
    ids: dict[str, str] = {}

    def writer(tag: str) -> None:
        try:
            registry = ModelRegistry(tmp_path / "models", WorkflowStore(tmp_path / "s.sqlite"))  # one connection per thread
            ids[tag] = _register(registry, tmp_path / tag, tag).model_id
            ids[f"common-{tag}"] = registry.register(checkpoint=_payload(tmp_path / tag, "common.pt", b"common"), export=None, dataset_version="ds",
                                                     parent_model_id=None, species=[1], cutoff=4.0, metrics={}, train_mode="fresh", backend="gmd_sgt").model_id
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    tags = [f"t{k}" for k in range(8)]
    for tag in tags:
        (tmp_path / tag).mkdir()
    threads = [threading.Thread(target=writer, args=(tag,)) for tag in tags]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    store = WorkflowStore(tmp_path / "s.sqlite")
    registry = ModelRegistry(tmp_path / "models", store)
    assert len({ids[f"common-{tag}"] for tag in tags}) == 1
    assert len(store.models()) == len(tags) + 1
    for model in store.models():
        registry.verify(model["model_id"])
    assert not list((tmp_path / "models").glob(".*.staging"))


_PROCESS_WRITER = """
import sys
from pathlib import Path
from gmd_active_learning.retraining.registry import ModelRegistry
from gmd_active_learning.workflow.store import WorkflowStore
root, tag = Path(sys.argv[1]), sys.argv[2]
registry = ModelRegistry(root / "models", WorkflowStore(root / "s.sqlite"))
(root / tag).mkdir()
for k in range(3):
    path = root / tag / f"m{k}.pt"
    path.write_bytes(f"{tag}-{k}".encode())
    registry.register(checkpoint=path, export=None, dataset_version="ds", parent_model_id=None, species=[1], cutoff=4.0, metrics={}, train_mode="fresh", backend="gmd_sgt")
shared = root / tag / "shared.pt"
shared.write_bytes(b"shared")
print(registry.register(checkpoint=shared, export=None, dataset_version="ds", parent_model_id=None, species=[1], cutoff=4.0, metrics={}, train_mode="fresh", backend="gmd_sgt").model_id)
"""


def test_concurrent_writer_processes_share_one_store(tmp_path: Path) -> None:
    WorkflowStore(tmp_path / "s.sqlite").close()
    env = {"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    processes = [subprocess.Popen([sys.executable, "-c", _PROCESS_WRITER, str(tmp_path), f"p{k}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                 for k in range(4)]
    outputs = [process.communicate(timeout=120) for process in processes]
    assert all(process.returncode == 0 for process in processes), [err for _, err in outputs]
    shared = {out.strip() for out, _ in outputs}
    assert len(shared) == 1
    store = WorkflowStore(tmp_path / "s.sqlite")
    registry = ModelRegistry(tmp_path / "models", store)
    assert len(store.models()) == 4 * 3 + 1
    for model in store.models():
        registry.verify(model["model_id"])
    assert json.loads((tmp_path / "models" / shared.pop() / "record.json").read_text())["dataset_version"] == "ds"
