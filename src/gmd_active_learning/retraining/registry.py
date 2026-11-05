"""Immutable model registry with gated promotion and rollback.

Replaces PHAL's filesystem registry (training/model_registry.py, ledger TRN-006), which stored no
checksums, overwrote a version registered twice and promoted without validation (F-14).

Layout: ``<root>/<model_id>/{checkpoint<ext>, export<ext>, record.json}``; files are copied in,
hashed, made read-only and re-verified before every promotion or rollback. States:
``registered`` → ``promoted`` (active) / ``rejected``; a superseded active model becomes ``retired``
and stays available as a rollback target.
"""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gmd_active_learning.contracts import UNITS, ModelArtifact, stable_id
from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.core.exceptions import WorkflowError
from gmd_active_learning.utils.atomic_io import write_json_atomic
from gmd_active_learning.utils.logging import get_logger
from gmd_active_learning.workflow.store import WorkflowStore, utc_now

LOGGER = get_logger(__name__)


class PromotionGateConfig(BaseModel):
    """``promotion`` config section: every gate is explicit and configurable."""

    model_config = ConfigDict(extra="forbid")

    max_force_rmse: float = Field(0.5, gt=0.0, description="eV/A on the validation split")
    max_energy_rmse_per_atom: float | None = Field(None, gt=0.0, description="eV/atom; None disables")
    max_relative_regression: float = Field(0.05, ge=0.0, description="allowed force-RMSE increase vs the active model")
    require_validation_samples: int = Field(1, ge=1)
    require_export_validation: bool = True
    export_max_force_deviation: float = Field(1.0e-4, gt=0.0, description="eV/A, export vs checkpoint")
    require_engine_smoke_test: bool = True


@dataclass(slots=True)
class GateReport:
    passed: bool
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks}


def evaluate_gates(
    config: PromotionGateConfig,
    *,
    metrics: dict[str, float],
    parent_metrics: dict[str, float] | None,
    export_check: dict[str, Any] | None,
    engine_smoke: dict[str, Any] | None,
) -> GateReport:
    """Decide whether a trained model may be promoted. Missing evidence fails the gate."""

    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, ok: bool, **detail: Any) -> None:
        checks[name] = {"passed": bool(ok), **detail}

    n_val = int(metrics.get("n_validation", 0))
    check("validation_samples", n_val >= config.require_validation_samples, value=n_val, required=config.require_validation_samples)
    force_rmse = metrics.get("force_rmse")
    check("force_rmse", force_rmse is not None and float(force_rmse) <= config.max_force_rmse, value=force_rmse, limit=config.max_force_rmse)
    if config.max_energy_rmse_per_atom is not None:
        value = metrics.get("energy_rmse_per_atom")
        check("energy_rmse_per_atom", value is not None and float(value) <= config.max_energy_rmse_per_atom, value=value, limit=config.max_energy_rmse_per_atom)
    if parent_metrics and parent_metrics.get("force_rmse") is not None and force_rmse is not None:
        limit = float(parent_metrics["force_rmse"]) * (1.0 + config.max_relative_regression)
        check("no_regression_vs_active", float(force_rmse) <= limit, value=force_rmse, limit=limit, parent=parent_metrics["force_rmse"])
    if config.require_export_validation:
        deviation = None if export_check is None else export_check.get("max_force_deviation")
        check("export_matches_checkpoint", deviation is not None and float(deviation) <= config.export_max_force_deviation, value=deviation, limit=config.export_max_force_deviation)
    if config.require_engine_smoke_test:
        check("engine_smoke_test", bool(engine_smoke and engine_smoke.get("passed")), detail=engine_smoke)
    return GateReport(passed=all(item["passed"] for item in checks.values()), checks=checks)


class ModelRegistry:
    def __init__(self, root: str | Path, store: WorkflowStore) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store

    # ---------------------------------------------------------------- registration
    def register(
        self,
        *,
        checkpoint: str | Path,
        export: str | Path | None,
        dataset_version: str | None,
        parent_model_id: str | None,
        species: list[int],
        cutoff: float | None,
        metrics: dict[str, float],
        train_mode: str,
        backend: str,
        metadata: dict[str, Any] | None = None,
        extra_files: dict[str, str | Path] | None = None,
    ) -> ModelArtifact:
        """Copy, hash and record a model. ``extra_files`` (e.g. committee members) are stored and
        verified exactly like the checkpoint and listed under ``metadata["extra_files"]``.

        The model id is content-derived (checkpoint/export/extra-file hashes, dataset version, parent),
        so registering the same artifacts again is idempotent: the existing record is re-verified and
        returned, never overwritten (PHAL overwrote an explicitly re-used version, F-14). Safe for
        concurrent writers (threads or processes) sharing one store."""

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise WorkflowError(f"cannot register missing or empty checkpoint {checkpoint}")
        checkpoint_sha = file_sha256(checkpoint)
        export_path = None if export is None else Path(export)
        if export_path is not None and (not export_path.is_file() or export_path.stat().st_size == 0):
            raise WorkflowError(f"cannot register missing or empty export {export_path}")
        export_sha = None if export_path is None else file_sha256(export_path)
        extras = {name: Path(path) for name, path in (extra_files or {}).items()}
        for name, path in extras.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise WorkflowError(f"cannot register missing or empty extra file {name}={path}")
        extra_shas = {name: file_sha256(path) for name, path in sorted(extras.items())}
        model_id = stable_id("model", checkpoint_sha, export_sha, dataset_version, parent_model_id, extra_shas)
        existing = self.store.model(model_id)
        if existing is not None:
            self.verify(model_id)
            return self._artifact(existing)
        target = self.root / model_id
        # A private staging directory per writer: concurrent registrations of the same model (two
        # controllers, or a CLI init racing a run) must not delete each other's half-written files.
        staging = self.root / f".{model_id}.{os.getpid()}.{uuid.uuid4().hex}.staging"
        staging.mkdir(parents=True)
        stored_checkpoint = staging / f"checkpoint{checkpoint.suffix or '.pt'}"
        shutil.copy2(checkpoint, stored_checkpoint)
        stored_export = None
        if export_path is not None:
            stored_export = staging / f"export{export_path.suffix or '.pt'}"
            shutil.copy2(export_path, stored_export)
        stored_extras: dict[str, dict[str, str]] = {}
        for name, path in sorted(extras.items()):
            shutil.copy2(path, staging / f"extra_{name}{path.suffix}")
            stored_extras[name] = {"path": str(target / f"extra_{name}{path.suffix}"), "sha256": extra_shas[name]}
        artifact = ModelArtifact(
            model_id=model_id,
            checkpoint_path=str(target / stored_checkpoint.name),
            checkpoint_sha256=checkpoint_sha,
            export_path=None if stored_export is None else str(target / stored_export.name),
            export_sha256=export_sha,
            dataset_version=dataset_version,
            parent_model_id=parent_model_id,
            species=tuple(sorted(int(z) for z in species)),
            cutoff=cutoff,
            units=dict(UNITS),
            metrics=dict(metrics),
            train_mode=train_mode,
            backend=backend,
            created_at=utc_now(),
            metadata={"source_checkpoint": str(checkpoint), "source_export": None if export_path is None else str(export_path),
                      "extra_files": stored_extras, **(metadata or {})},
        )
        write_json_atomic(staging / "record.json", artifact.to_dict())
        for path in staging.iterdir():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        if target.exists():  # another writer won the race, or a crash after rename but before the store insert
            self._discard(staging)
        else:
            try:
                os.replace(staging, target)
            except OSError:  # a concurrent writer renamed its identical copy first
                if not target.exists():
                    raise
                self._discard(staging)
        self.store.record_model(artifact.to_dict(), state="registered")
        self.verify(model_id)
        LOGGER.info("registered model %s (dataset %s, parent %s)", model_id, dataset_version, parent_model_id)
        return artifact

    @staticmethod
    def _discard(staging: Path) -> None:
        for path in staging.iterdir():
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        shutil.rmtree(staging)

    def register_external(self, *, checkpoint: str | Path, export: str | Path | None, species: list[int], cutoff: float | None, backend: str, metadata: dict[str, Any] | None = None) -> ModelArtifact:
        """Register an initial (externally trained) model; it still needs promotion."""

        return self.register(
            checkpoint=checkpoint, export=export, dataset_version=None, parent_model_id=None, species=species,
            cutoff=cutoff, metrics={}, train_mode="external", backend=backend, metadata=metadata,
        )

    # ---------------------------------------------------------------- integrity
    def verify(self, model_id: str) -> None:
        record = self.store.model(model_id)
        if record is None:
            raise WorkflowError(f"unknown model {model_id}")
        for key, digest in (("checkpoint_path", "checkpoint_sha256"), ("export_path", "export_sha256")):
            path = record.get(key)
            if path is None:
                continue
            if not Path(path).is_file():
                raise WorkflowError(f"model {model_id}: {key} {path} is missing")
            if file_sha256(path) != record[digest]:
                raise WorkflowError(f"model {model_id}: {key} {path} fails its checksum (artifact modified)")
        for name, extra in (record.get("metadata", {}).get("extra_files") or {}).items():
            if not Path(extra["path"]).is_file() or file_sha256(extra["path"]) != extra["sha256"]:
                raise WorkflowError(f"model {model_id}: extra file {name} is missing or fails its checksum")

    def get(self, model_id: str) -> ModelArtifact:
        record = self.store.model(model_id)
        if record is None:
            raise WorkflowError(f"unknown model {model_id}")
        return self._artifact(record)

    def state(self, model_id: str) -> str:
        record = self.store.model(model_id)
        if record is None:
            raise WorkflowError(f"unknown model {model_id}")
        return str(record["state"])

    @staticmethod
    def _artifact(record: dict[str, Any]) -> ModelArtifact:
        data = {key: value for key, value in record.items() if key != "state"}
        data["species"] = tuple(data["species"])
        return ModelArtifact(**data)

    # ---------------------------------------------------------------- promotion and rollback
    def promote(self, run_id: str, model_id: str, *, gate_report: GateReport | None, reason: str, force_initial: bool = False) -> ModelArtifact:
        """Make ``model_id`` the run's active model. Requires a passed gate report, except for the
        explicitly flagged initial model of a run (which has no parent to compare against)."""

        self.verify(model_id)
        run = self.store.run(run_id)
        if gate_report is None and not (force_initial and run.active_model_id is None):
            raise WorkflowError(f"model {model_id}: promotion requires a gate report")
        if gate_report is not None and not gate_report.passed:
            self.store.set_model_state(model_id, "rejected")
            self.store.record_promotion(run_id, model_id=model_id, previous_model_id=run.active_model_id, action="reject", reason=reason)
            raise WorkflowError(f"model {model_id} failed promotion gates: {gate_report.checks}")
        previous = run.active_model_id
        if previous == model_id:
            return self.get(model_id)
        if previous is not None and self.store.model(previous) is not None:
            self.store.set_model_state(previous, "retired")
        self.store.set_model_state(model_id, "promoted")
        self.store.record_promotion(run_id, model_id=model_id, previous_model_id=previous, action="promote", reason=reason)
        self.store.update_run(run_id, active_model_id=model_id)
        LOGGER.info("promoted %s (previous %s): %s", model_id, previous, reason)
        return self.get(model_id)

    def reject(self, run_id: str, model_id: str, *, reason: str) -> None:
        self.store.set_model_state(model_id, "rejected")
        self.store.record_promotion(run_id, model_id=model_id, previous_model_id=self.store.run(run_id).active_model_id, action="reject", reason=reason)

    def rollback(self, run_id: str, *, reason: str, to_model_id: str | None = None) -> ModelArtifact:
        """Re-activate the previously promoted model (or ``to_model_id``), after re-verifying it."""

        run = self.store.run(run_id)
        current = run.active_model_id
        if to_model_id is None:
            history = [item for item in self.store.promotions(run_id) if item["action"] in {"promote", "rollback"}]
            candidates = [item["previous_model_id"] for item in reversed(history) if item["model_id"] == current and item["previous_model_id"]]
            if not candidates:
                raise WorkflowError(f"run {run_id}: no rollback target for active model {current}")
            to_model_id = candidates[0]
        state = self.state(to_model_id)
        if state == "rejected":
            raise WorkflowError(f"cannot roll back to rejected model {to_model_id}")
        self.verify(to_model_id)
        if current is not None:
            self.store.set_model_state(current, "rejected")
        self.store.set_model_state(to_model_id, "promoted")
        self.store.record_promotion(run_id, model_id=to_model_id, previous_model_id=current, action="rollback", reason=reason)
        self.store.update_run(run_id, active_model_id=to_model_id)
        LOGGER.warning("rolled back %s -> %s: %s", current, to_model_id, reason)
        return self.get(to_model_id)
