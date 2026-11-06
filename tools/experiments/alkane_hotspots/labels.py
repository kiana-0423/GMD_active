"""CP2K labels for full parents and fragments through the GMD_active ``LabelingService``.

- One service (one CP2K configuration, one settings hash) labels parents and fragments, so both use
  identical electronic-structure settings; every record carries the hash and the runner refuses a
  task whose hash differs.
- ``label_id = stable_id("label", system content, settings hash)``: the same system with the same
  settings is one calculation. Finished labels are reused; tasks found on disk (a previous,
  interrupted invocation) are resumed through the service's own idempotency (``prepare`` returns the
  existing task, ``submit`` is a no-op once a job exists), so re-running never duplicates a job.
- Before a new job is launched the budgets are checked with the measured mean cost so far; a refused
  job gets a ``budget_skipped`` record with the reason and the estimate (also printed).
- Records hold the measured cost from the attempt directories (``cost.measured_cost``).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig
from gmd_active_learning.contracts import ExtractedRegion, RegionRole, content_hash, stable_id
from gmd_active_learning.contracts.labels import ExecutionStatus, ValidationStatus
from gmd_active_learning.labeling.service import LabelingService
from gmd_active_learning.utils.atomic_io import write_json_atomic
from tools.experiments.alkane_hotspots.config import LoadedConfig
from tools.experiments.alkane_hotspots.cost import BudgetLedger, measured_cost
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology
from tools.experiments.alkane_hotspots.manifest import utc_now

LOGGER = logging.getLogger(__name__)

LABELS_FILE = "labels.jsonl"
STATUSES = ("valid", "invalid", "failed", "budget_skipped")


# -- CP2K configuration ------------------------------------------------------------------------------
def cp2k_raw_config(loaded: LoadedConfig) -> dict[str, Any]:
    """The GMD_active configuration mapping (``backends``, ``scheduler``, ``cp2k``) for the service."""

    settings = loaded.config.cp2k
    budgets = loaded.config.budgets
    cp2k: dict[str, Any] = {
        "executable": settings.executable,
        "launcher": list(settings.launcher),
        "data_dir": settings.resolved_data_dir(),
        "environment": {"OMP_NUM_THREADS": str(settings.omp_threads)},
        "task_dir": str(loaded.artifacts_root / "cp2k_tasks"),
        "resources": {"ntasks": settings.mpi_ranks, "cpus_per_task": settings.omp_threads, "walltime": int(budgets.per_job_walltime_s)},
        "basis": settings.basis,
        "potential": settings.potential,
        "functional": settings.functional,
        "cutoff": settings.cutoff,
        "rel_cutoff": settings.rel_cutoff,
        "scf_eps": settings.eps_scf,
        "max_scf": settings.max_scf,
        "dispersion": settings.dispersion,
        "multiplicity": settings.multiplicity,
        "spin_policy": "strict",
        "poisson": {"non_periodic_solver": settings.poisson_solver},
        "cluster_padding": settings.box_padding,
        "uncapped_fragment_policy": "error",
        "h_only_opt": {"enabled": settings.h_only_opt.enabled, "max_iter": settings.h_only_opt.max_iter},
    }
    return {
        "backends": {"dft": {"engine": "cp2k"}, "scheduler": {"engine": "local"}, "discover_entry_points": False},
        "scheduler": {},
        "cp2k": cp2k,
    }


def build_service(loaded: LoadedConfig) -> LabelingService:
    return LabelingService.from_config(cp2k_raw_config(loaded), base_dir=loaded.artifacts_root)


def physics_settings(config: CP2KConfig) -> dict[str, Any]:
    dump = config.model_dump(mode="json")
    return {name: dump[name] for name in CP2KConfig.PHYSICS_FIELDS}


def cp2k_provenance(service: LabelingService, *, smoke_only: bool, probe: bool = True) -> dict[str, Any]:
    """Version (probed with ``--version``), settings hash and physics settings of the service."""

    version = None
    executable = None
    if probe:
        status = service.backend.check_runtime()
        version = (status.metadata or {}).get("version")
        executable = (status.metadata or {}).get("executable")
    return {
        "version": version,
        "executable": executable,
        "settings_hash": service.settings_hash,
        "settings": physics_settings(service.backend.config),
        "cores": int((service.resources.ntasks or 1) * (service.resources.cpus_per_task or 1)),
        "smoke_only": smoke_only,
    }


# -- systems and identities --------------------------------------------------------------------------
def system_hash(region: ExtractedRegion) -> str:
    return content_hash(
        {
            "numbers": region.numbers,
            "positions": np.round(region.positions, 10),
            "pbc": list(region.pbc),
            "cell": None if region.cell is None else np.round(region.cell, 10),
            "charge": int(region.charge),
            "multiplicity": region.multiplicity,
        },
        length=16,
    )


def label_id_for(region: ExtractedRegion, settings_hash: str) -> str:
    return stable_id("label", system_hash(region), settings_hash)


def parent_region(parent: dict[str, Any], positions: np.ndarray) -> ExtractedRegion:
    """The whole parent molecule as an isolated, uncapped region (every atom core)."""

    topology = AlkaneTopology(int(parent["chain_length"]))
    n = topology.n_atoms
    positions = np.asarray(positions, dtype=float)
    return ExtractedRegion(
        region_id=stable_id("alkane-full", parent["parent_id"], content_hash(np.round(positions, 10))),
        mode="full",
        run_id="alkane_hotspots",
        parent_step=0,
        event_id=None,
        numbers=topology.numbers,
        positions=positions,
        cell=None,
        pbc=(False, False, False),
        parent_atom_ids=np.arange(n),
        image_shifts=np.zeros((n, 3), dtype=np.int64),
        roles=[RegionRole.CORE] * n,
        force_weights=np.ones(n),
        frozen=np.zeros(n, dtype=bool),
        hotspot_atom_ids=list(parent["perturbation"]["atoms"]) or [0],
        charge=0,
        multiplicity=1,
        metadata={"parent_id": parent["parent_id"], "kind": "full", "family": parent["family"]},
    )


# -- record store ------------------------------------------------------------------------------------
class LabelStore:
    """``labels.jsonl``: one record per label_id (current state), rewritten atomically."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / LABELS_FILE
        self.records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["label_id"]] = record

    def get(self, label_id: str) -> dict[str, Any] | None:
        return self.records.get(label_id)

    def put(self, record: dict[str, Any]) -> None:
        if record["status"] not in STATUSES:
            raise ValueError(f"unknown label status {record['status']!r}")
        self.records[record["label_id"]] = record
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in self.records.values()), encoding="utf-8")
        temporary.replace(self.path)

    def all(self) -> list[dict[str, Any]]:
        return list(self.records.values())

    def label_dir(self, label_id: str) -> Path:
        return self.root / "labels" / label_id


@dataclass
class LabelRequest:
    kind: str  # full | fragment
    parent_id: str
    chain_length: int
    region: ExtractedRegion
    fragment: dict[str, Any] | None = None
    spec_ids: list[str] = field(default_factory=list)


@dataclass
class RunSummary:
    reused: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)  # still running when the invocation stopped

    def as_dict(self) -> dict[str, Any]:
        return {key: list(value) for key, value in self.__dict__.items()}


class LabelRunner:
    """Budgeted, resumable labeling of :class:`LabelRequest` objects."""

    def __init__(
        self,
        service: Any,
        store: LabelStore,
        loaded: LoadedConfig,
        cp2k_meta: dict[str, Any],
        *,
        sleep: Callable[[float], None] = time.sleep,
        echo: Callable[[str], None] = print,
    ) -> None:
        self.service = service
        self.store = store
        self.loaded = loaded
        self.cp2k_meta = cp2k_meta
        self.cores = int(cp2k_meta.get("cores") or loaded.config.cp2k.cores)
        self.sleep = sleep
        self.echo = echo
        if cp2k_meta["settings_hash"] != service.settings_hash:
            raise ValueError("cp2k provenance does not belong to this labeling service")

    # -- helpers -------------------------------------------------------------------------------------
    def _existing_task(self, region: ExtractedRegion) -> Any | None:
        task_id = self.service.task_id_for(region)
        if (Path(self.service.work_dir_for(task_id)) / "task.json").exists():
            return self.service.load(task_id)
        return None

    def _base_record(self, request: LabelRequest, label_id: str) -> dict[str, Any]:
        return {
            "label_id": label_id,
            "kind": request.kind,
            "parent_id": request.parent_id,
            "chain_length": request.chain_length,
            "fragment": request.fragment,
            "spec_ids": list(request.spec_ids),
            "n_atoms": request.region.n_atoms,
            "region_path": None,
            "status": None,
            "forces_path": None,
            "energy_ev": None,
            "scf_steps": None,
            "cp2k": {
                "version": self.cp2k_meta.get("version"),
                "settings_hash": self.service.settings_hash,
                "settings": self.cp2k_meta.get("settings"),
                "smoke_only": bool(self.cp2k_meta.get("smoke_only")),
            },
            "cost": None,
            "task_id": None,
            "reasons": [],
            "created_at": utc_now(),
        }

    def _write_region(self, label_id: str, region: ExtractedRegion) -> str:
        directory = self.store.label_dir(label_id)
        directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(directory / "region.json", region.to_dict())
        return str((directory / "region.json").relative_to(self.store.root))

    def _finish(self, request: LabelRequest, label_id: str, task: Any) -> dict[str, Any]:
        record = self._base_record(request, label_id)
        if task.settings_hash != self.service.settings_hash:
            raise RuntimeError(f"task {task.task_id} has settings hash {task.settings_hash}, service {self.service.settings_hash}")
        result = self.service.collect(task.task_id)
        task = self.service.load(task.task_id)
        record["task_id"] = task.task_id
        record["cost"] = measured_cost(task.work_dir, task.metadata.get("attempts", []), self.cores)
        labeled = request.region
        work = Path(task.work_dir)
        labeled_file = task.metadata.get("labeled_region_file")
        if labeled_file and (work / labeled_file).exists():
            labeled = ExtractedRegion.from_dict(json.loads((work / labeled_file).read_text(encoding="utf-8")))
        record["region_path"] = self._write_region(label_id, labeled)
        record["scf_steps"] = result.scf_steps
        record["energy_ev"] = result.energy
        if result.engine_version:
            record["cp2k"]["version"] = result.engine_version
        provenance = (result.metadata or {}).get("provenance") or {}
        record["cp2k"]["version_string"] = provenance.get("cp2k_version_string")
        record["cp2k"]["git_revision"] = provenance.get("cp2k_git_revision")
        record["caps_relaxed"] = bool((result.metadata or {}).get("caps_relaxed"))
        if result.validation is ValidationStatus.VALID:
            forces_path = self.store.label_dir(label_id) / "forces.npy"
            np.save(forces_path, np.asarray(result.forces, dtype=float))
            record["forces_path"] = str(forces_path.relative_to(self.store.root))
            record["status"] = "valid"
        else:
            record["status"] = "invalid" if task.execution is ExecutionStatus.COMPLETED else "failed"
            record["reasons"] = list(result.validation_reasons)
        self.store.put(record)
        return record

    # -- main loop -----------------------------------------------------------------------------------
    def run(self, requests: list[LabelRequest], *, max_wall_s: float | None = None) -> RunSummary:
        summary = RunSummary()
        config = self.loaded.config
        ledger = BudgetLedger(self.store.all(), config.budgets, self.cores)
        queue: list[tuple[LabelRequest, str]] = []
        seen: set[str] = set()
        for request in requests:
            label_id = label_id_for(request.region, self.service.settings_hash)
            if label_id in seen:
                continue
            seen.add(label_id)
            existing = self.store.get(label_id)
            if existing is not None and existing["status"] != "budget_skipped":
                merged = sorted(set(existing.get("spec_ids", [])) | set(request.spec_ids))
                if merged != existing.get("spec_ids", []):
                    existing["spec_ids"] = merged
                    self.store.put(existing)
                summary.reused.append(label_id)
                continue
            queue.append((request, label_id))

        active: dict[str, tuple[LabelRequest, str]] = {}
        started = time.monotonic()
        while queue or active:
            # Launch (or resume) while there is capacity.
            while queue and len(active) < config.labels.max_concurrent_jobs:
                request, label_id = queue.pop(0)
                task = self._existing_task(request.region)
                if task is None or not task.metadata.get("attempts"):
                    decision = ledger.decide(request.kind, request.region.n_atoms, request.chain_length)
                    message = (
                        f"{request.kind} {request.parent_id} ({request.region.n_atoms} atoms): estimate "
                        f"{decision.estimate_core_s / 3600.0:.3f} core-h [{decision.estimate_source}]"
                    )
                    if not decision.allowed:
                        record = self._base_record(request, label_id)
                        record.update({"status": "budget_skipped", "reasons": [decision.reason], "budget_estimate_core_s": decision.estimate_core_s,
                                       "budget_estimate_source": decision.estimate_source})
                        self.store.put(record)
                        ledger.records = self.store.all()
                        summary.skipped.append(label_id)
                        self.echo(f"SKIP {message}: {decision.reason}")
                        continue
                    self.echo(f"RUN  {message}")
                    try:
                        task = self.service.prepare(request.region)
                        task = self.service.submit(task)
                    except Exception as exc:  # noqa: BLE001 - recorded as a failed label, never silently dropped
                        record = self._base_record(request, label_id)
                        record.update({"status": "failed", "reasons": [f"{type(exc).__name__}: {exc}"]})
                        record["region_path"] = self._write_region(label_id, request.region)
                        self.store.put(record)
                        ledger.records = self.store.all()
                        summary.failed.append(label_id)
                        LOGGER.error("label %s could not be launched: %s", label_id, exc)
                        continue
                    ledger.inflight[label_id] = (request.kind, decision.estimate_core_s)
                else:
                    self.echo(f"RESUME {request.kind} {request.parent_id}: task {task.task_id} ({task.execution.value})")
                    ledger.inflight[label_id] = (request.kind, ledger.estimate(request.kind, request.chain_length)[0])
                active[label_id] = (request, task.task_id)
            if not active:
                break
            finished = []
            for label_id, (request, task_id) in active.items():
                task = self.service.poll(task_id)
                if task.metadata.get("stage") == "done":
                    record = self._finish(request, label_id, task)
                    finished.append(label_id)
                    (summary.completed if record["status"] == "valid" else summary.failed).append(label_id)
                    cost = record["cost"]
                    self.echo(
                        f"DONE {request.kind} {request.parent_id}: {record['status']} wall {cost['wall_s']:.1f} s, "
                        f"{cost['cpu_core_s'] / 3600.0:.3f} core-h, {cost['attempts']} attempt(s)"
                    )
            for label_id in finished:
                active.pop(label_id)
                ledger.inflight.pop(label_id, None)
            ledger.records = self.store.all()
            if max_wall_s is not None and time.monotonic() - started > max_wall_s and (active or queue):
                summary.pending.extend(active)
                summary.pending.extend(label_id for _, label_id in queue)
                self.echo(f"stopping after {max_wall_s:.0f} s; {len(active)} job(s) keep running detached, re-run to resume")
                break
            if active:
                self.sleep(config.labels.poll_interval_s)
        return summary


__all__ = [
    "LABELS_FILE",
    "LabelRequest",
    "LabelRunner",
    "LabelStore",
    "RunSummary",
    "build_service",
    "cp2k_provenance",
    "cp2k_raw_config",
    "label_id_for",
    "parent_region",
    "physics_settings",
    "system_hash",
]
