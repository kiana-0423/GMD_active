"""Compute allowances of an execution: durable ledger, pre-submission room, live enforcement.

Three layers bound the DFT expenditure of an execution:

1. **Hard per-attempt backstop.** Every CP2K attempt runs under ``/usr/bin/timeout -k <grace> <wall>``
   (the launcher in the alpha configuration), a supervisor inside the job's own session: SIGTERM at the
   wall limit, SIGKILL ``grace`` seconds later, even if every controlling process has died. An attempt
   therefore consumes at most ``(wall + grace) x cores`` core-seconds. (``ulimit -t`` was tried first and
   rejected: CP2K's Fortran runtime caught SIGXCPU, hung in its backtrace handler and kept spinning, and
   macOS did not escalate to SIGKILL at the hard limit.)
2. **Pre-submission room.** A new job is started only if every scope it charges (total, stage, the
   q0 sub-limit, the repair run) has room for a conservative estimate (``estimate_quantile`` of the
   measured costs of the same job kind) plus ``margin_core_s``.
3. **Live enforcement.** While a job runs, every poll recomputes consumption from the attempt
   directories on disk; when a scope's consumption reaches its allowance minus the margin, the job is
   cancelled (the scheduler terminates the process group) and the cancellation and its consumed
   core-seconds are journalled.

Consumption is always recomputed from disk (every attempt directory: completed, failed, cancelled,
retried and running; a running attempt counts its elapsed wall time x cores), so a restart cannot
reset it, and nothing is double counted. ``--max-wall-s`` is never used as a compute cap: it only
stops waiting and leaves jobs running.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from gmd_active_learning.core.exceptions import LabelingError

from tools.experiments.alkane_hotspots.manifest import utc_now

from .common import read_json, write_json
from .config import LoadedBenchmark

LOGGER = logging.getLogger(__name__)

STAGES = ("initial_model", "development", "fidelity_audit", "repair_val", "final_test", "repair_runs")
LOCAL_JOB_FILE = ".gmd_local_job.json"
EXIT_CODE_FILE = ".gmd_exit_code"


class BudgetRefused(LabelingError):
    """A job would not fit the remaining allowance (or was cancelled to stay inside it)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnforcementConfig(_Strict):
    margin_core_s: float = Field(..., ge=0.0, description="kept free in every scope; covers poll granularity and cancellation")
    estimate_quantile: float = Field(0.9, gt=0.0, le=1.0, description="of measured costs of the same job kind (pre-submission estimate)")
    attempt_wall_cap_s: float = Field(..., gt=0.0, description="wall-clock limit of one CP2K attempt (timeout in the launcher)")
    kill_grace_s: float = Field(..., gt=0.0, description="SIGKILL this many seconds after SIGTERM (timeout -k)")


class AuthorizationConfig(_Strict):
    granted: str
    statement: str = ""
    total_core_h: float = Field(..., gt=0.0)
    stages: dict[str, float]
    initial_model_q0_core_h: float = Field(..., gt=0.0)
    per_run_core_h: float = Field(..., gt=0.0)
    max_elapsed_h: float = Field(..., gt=0.0)
    final_reserve_h: float = Field(0.0, ge=0.0, description="hours of the elapsed window kept for the final test, evaluation and report: "
                                                             "no repair run starts if it would end inside the reserve (0 disables)")
    cp2k_max_concurrent_jobs: int = Field(1, ge=1, le=1)
    cores_per_job: int = Field(..., ge=1)
    enforcement: EnforcementConfig
    applied_amendments: list[dict[str, Any]] = Field(default_factory=list, description="set by load_authorization; never in the YAML")


#: allowance fields an amendment may change (``stages.<name>`` for a stage); enforcement settings, cores and
#: concurrency are not amendable.
AMENDABLE = ("total_core_h", "initial_model_q0_core_h", "per_run_core_h", "max_elapsed_h", "final_reserve_h", *(f"stages.{s}" for s in STAGES))


class AllowanceChange(_Strict):
    from_: float = Field(..., alias="from")
    to: float = Field(..., gt=0.0)


class Amendment(_Strict):
    id: str = Field(..., min_length=1)
    at: str = Field(..., min_length=1, description="UTC timestamp of the grant")
    granted_by: str = Field(..., min_length=1)
    statement: str = Field(..., min_length=1, description="the grant as given")
    rationale: str = Field(..., min_length=1)
    changes: dict[str, AllowanceChange]
    unchanged: dict[str, float] = Field(default_factory=dict, description="allowances the grant explicitly keeps, with their values (verified)")


def timeout_of_launcher(launcher: list[str]) -> tuple[float, float] | None:
    """(wall seconds, kill grace seconds) of a ``[/usr/bin/timeout, -k, GRACE, WALL]`` launcher (None if absent)."""

    if len(launcher) == 4 and Path(launcher[0]).name == "timeout" and launcher[1] == "-k":
        try:
            return float(launcher[3]), float(launcher[2])
        except ValueError:
            return None
    return None


def authorization_path(loaded: LoadedBenchmark) -> Path:
    return loaded.path.parent / f"{loaded.config.name}_authorization.yaml"


def amendments_path(loaded: LoadedBenchmark) -> Path:
    return loaded.path.parent / f"{loaded.config.name}_authorization_amendments.yaml"


def amendment_digest(amendment: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(amendment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _allowance(raw: dict[str, Any], key: str) -> float | None:
    if key.startswith("stages."):
        return (raw.get("stages") or {}).get(key.split(".", 1)[1])
    if key not in raw and key in AuthorizationConfig.model_fields and not AuthorizationConfig.model_fields[key].is_required():
        return AuthorizationConfig.model_fields[key].default  # an optional allowance absent from the original file has its default
    return raw.get(key)


def _apply_amendments(loaded: LoadedBenchmark, raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply the append-only amendment list in order; each change must chain from the current value."""

    path = amendments_path(loaded)
    if not path.is_file():
        return []
    entries = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("amendments") or []
    applied, seen = [], set()
    for entry in entries:
        amendment = Amendment.model_validate(entry)
        if amendment.id in seen:
            raise BudgetRefused(f"authorization amendment {amendment.id!r} appears twice")
        seen.add(amendment.id)
        for key, change in amendment.changes.items():
            if key not in AMENDABLE:
                raise BudgetRefused(f"amendment {amendment.id}: {key!r} is not an amendable allowance ({', '.join(AMENDABLE)})")
            current = _allowance(raw, key)
            if current is None or abs(float(current) - change.from_) > 1e-9:
                raise BudgetRefused(f"amendment {amendment.id}: {key} 'from' {change.from_} does not match the current allowance {current}")
            if key.startswith("stages."):
                raw["stages"][key.split(".", 1)[1]] = change.to
            else:
                raw[key] = change.to
        for key, value in amendment.unchanged.items():
            current = _allowance(raw, key)
            if key not in AMENDABLE or key in amendment.changes or current is None or abs(float(current) - value) > 1e-9:
                raise BudgetRefused(f"amendment {amendment.id}: {key} is declared unchanged at {value} but is {current}"
                                    + (" (and changed by the amendment)" if key in amendment.changes else ""))
        applied.append({**entry, "sha256": amendment_digest(entry)})
    return applied


def load_authorization(loaded: LoadedBenchmark) -> AuthorizationConfig:
    path = authorization_path(loaded)
    if not path.is_file():
        raise BudgetRefused(f"no execution authorization ({path}); expensive work needs an explicit compute allowance")
    raw = dict((yaml.safe_load(path.read_text(encoding="utf-8")) or {})["authorization"])
    if "applied_amendments" in raw:
        raise BudgetRefused("applied_amendments is set by the loader; amendments live in the amendments file")
    raw["stages"] = dict(raw.get("stages") or {})
    applied = _apply_amendments(loaded, raw)
    auth = AuthorizationConfig.model_validate({**raw, "applied_amendments": applied})
    unknown = set(auth.stages) - set(STAGES)
    if unknown:
        raise BudgetRefused(f"authorization: unknown stages {sorted(unknown)}")
    for stage in STAGES:
        protocol = float(getattr(loaded.config.budgets, stage))
        if auth.stages.get(stage, 0.0) > protocol + 1e-9:
            raise BudgetRefused(f"authorization for {stage} ({auth.stages[stage]}) exceeds the protocol budget ({protocol})")
    if auth.per_run_core_h > loaded.config.repair.budget_core_h_per_run + 1e-9:
        raise BudgetRefused("authorization per-run allowance exceeds the protocol's budget_core_h_per_run")
    if auth.initial_model_q0_core_h > auth.stages.get("initial_model", 0.0) + 1e-9:
        raise BudgetRefused(f"the q0 sub-allowance ({auth.initial_model_q0_core_h}) exceeds the initial-model allowance ({auth.stages.get('initial_model', 0.0)})")
    if auth.final_reserve_h >= auth.max_elapsed_h:
        raise BudgetRefused(f"the final reserve ({auth.final_reserve_h} h) must be shorter than the elapsed window ({auth.max_elapsed_h} h)")
    if sum(auth.stages.values()) > auth.total_core_h + 1e-9:
        raise BudgetRefused(f"stage allowances sum to {sum(auth.stages.values())} core-h, more than the total {auth.total_core_h}")
    limits = timeout_of_launcher(loaded.alpha.config.cp2k.launcher)
    expected = (auth.enforcement.attempt_wall_cap_s, auth.enforcement.kill_grace_s)
    if limits is None or abs(limits[0] - expected[0]) > 0.5 or abs(limits[1] - expected[1]) > 0.5:
        raise BudgetRefused(f"the CP2K launcher's timeout ({limits}) must equal enforcement (attempt_wall_cap_s, kill_grace_s) = {expected}; "
                            "every attempt needs the hard backstop")
    if auth.cores_per_job != loaded.alpha.config.cp2k.cores:
        raise BudgetRefused(f"authorization cores_per_job {auth.cores_per_job} != CP2K cores {loaded.alpha.config.cp2k.cores}")
    return auth


# ---------------------------------------------------------------------------------------------- consumption from disk


def attempt_cap_core_s(auth: AuthorizationConfig) -> float:
    """Most core-seconds one attempt can consume under the launcher's timeout."""

    return (auth.enforcement.attempt_wall_cap_s + auth.enforcement.kill_grace_s) * auth.cores_per_job


@dataclass
class Attempt:
    task_dir: Path
    attempt_dir: Path
    scope: dict[str, Any]
    started_at: float
    ended_at: float | None
    running: bool
    core_s: float


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def scan_attempt(attempt_dir: Path, cores: int, now: float) -> tuple[float, float | None, bool] | None:
    """(started_at, ended_at or None, running) of one attempt from the scheduler's records."""

    record_path = attempt_dir / LOCAL_JOB_FILE
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        started = float(record["started_at"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None
    exit_path = attempt_dir / EXIT_CODE_FILE
    if exit_path.is_file():
        return started, exit_path.stat().st_mtime, False
    if _pid_alive(int(record.get("pid", -1))):
        return started, None, True
    mtimes = [p.stat().st_mtime for p in attempt_dir.iterdir() if p.is_file()]  # ended without an exit code (killed)
    return started, max([started, *mtimes]), False


class ComputeLedger:
    """Consumption of every CP2K attempt of a benchmark, classified into allowance scopes."""

    def __init__(self, loaded: LoadedBenchmark, auth: AuthorizationConfig) -> None:
        self.loaded = loaded
        self.auth = auth
        self.cores = auth.cores_per_job
        self._parents: dict[str, dict[str, Any]] | None = None

    # -- classification ------------------------------------------------------------------------------
    def parents(self) -> dict[str, dict[str, Any]]:
        if self._parents is None:
            from .generate import load_records

            records = {}
            for root in (self.loaded.root, self.loaded.sealed_root):
                if (root / "parents.jsonl").is_file():
                    records.update({r["parent_id"]: r for r in load_records(root)})
            self._parents = records
        return self._parents

    def q0_bases(self) -> set[str]:
        n = self.loaded.config.initial_model.stages[0].id_train_bases
        return set(sorted({r["base_id"] for r in self.parents().values() if r["partition"] == "id_train"})[:n])

    def classify(self, task_dir: Path) -> dict[str, Any]:
        """Scope of a CP2K task directory: stage, q-stage for initial-model labels, repair run."""

        root = self.loaded.root
        relative = task_dir.resolve().relative_to(root.resolve())
        parts = relative.parts
        if parts[0] == "repair_runs":
            return {"stage": "repair_runs", "run": parts[1]}
        if parts[0] == self.loaded.config.sealing.sealed_subdir:
            return {"stage": "final_test"}
        region = read_json(task_dir / "region.json") if (task_dir / "region.json").is_file() else {}
        parent_id = (region.get("metadata") or {}).get("parent_id") or region.get("run_id")
        parent = self.parents().get(str(parent_id))
        if parent is None:
            return {"stage": "unclassified", "parent_id": parent_id}
        full = region.get("mode") in ("full", "full_system")
        partition = parent["partition"]
        if partition in ("id_train", "id_val") and full:
            q = 0 if partition == "id_val" or parent["base_id"] in self.q0_bases() else 1
            return {"stage": "initial_model", "q": q, "parent_id": parent_id}
        if partition == "development":
            return {"stage": "development" if full else "fidelity_audit", "parent_id": parent_id}
        if partition == "repair_val" and full:
            return {"stage": "repair_val", "parent_id": parent_id}
        return {"stage": "unclassified", "parent_id": parent_id}

    # -- scanning ------------------------------------------------------------------------------------
    def task_roots(self) -> list[Path]:
        root = self.loaded.root
        roots = [root / "cp2k_tasks", self.loaded.sealed_root / "cp2k_tasks"]
        roots += sorted((root / "repair_runs").glob("*/run/cp2k_tasks"))
        return [r for r in roots if r.is_dir()]

    def attempts(self, now: float | None = None) -> list[Attempt]:
        now = time.time() if now is None else now
        out = []
        for task_root in self.task_roots():
            for task_dir in sorted(p for p in task_root.iterdir() if p.is_dir()):
                scope = None
                for attempt_dir in sorted((task_dir / "attempts").glob("*")) if (task_dir / "attempts").is_dir() else []:
                    scanned = scan_attempt(attempt_dir, self.cores, now)
                    if scanned is None:
                        continue
                    started, ended, running = scanned
                    scope = scope or self.classify(task_dir)
                    seconds = (now if running else float(ended)) - started
                    out.append(Attempt(task_dir, attempt_dir, scope, started, ended, running, max(0.0, seconds) * self.cores))
        return out

    def consumption(self, now: float | None = None) -> dict[str, Any]:
        attempts = self.attempts(now)
        stages: dict[str, float] = {}
        runs: dict[str, float] = {}
        q0 = 0.0
        for a in attempts:
            stages[a.scope["stage"]] = stages.get(a.scope["stage"], 0.0) + a.core_s
            if a.scope.get("run"):
                runs[a.scope["run"]] = runs.get(a.scope["run"], 0.0) + a.core_s
            if a.scope["stage"] == "initial_model" and a.scope.get("q") == 0:
                q0 += a.core_s
        return {"total_core_s": sum(a.core_s for a in attempts), "stages_core_s": stages, "runs_core_s": runs, "q0_core_s": q0,
                "n_attempts": len(attempts), "running": [{"task": a.task_dir.name, "attempt": a.attempt_dir.name, "scope": a.scope,
                                                          "elapsed_core_s": a.core_s} for a in attempts if a.running]}

    # -- allowances ----------------------------------------------------------------------------------
    def limits(self, scope: Mapping[str, Any]) -> list[tuple[str, float]]:
        """(name, allowance core-s) of every scope a job of ``scope`` charges."""

        auth = self.auth
        out = [("total", auth.total_core_h * 3600.0)]
        stage = scope.get("stage")
        if stage in auth.stages:
            out.append((f"stage:{stage}", auth.stages[stage] * 3600.0))
        if stage == "initial_model" and scope.get("q") == 0:
            out.append(("initial_model_q0", auth.initial_model_q0_core_h * 3600.0))
        if scope.get("run"):
            out.append((f"run:{scope['run']}", auth.per_run_core_h * 3600.0))
        if stage not in auth.stages:
            out.append((f"stage:{stage}", 0.0))  # an unclassified job has no allowance
        return out

    def used(self, name: str, consumption: Mapping[str, Any]) -> float:
        if name == "total":
            return consumption["total_core_s"]
        if name == "initial_model_q0":
            return consumption["q0_core_s"]
        kind, _, key = name.partition(":")
        if kind == "stage":
            return consumption["stages_core_s"].get(key, 0.0)
        if kind == "run":
            return consumption["runs_core_s"].get(key, 0.0)
        raise KeyError(name)

    def remaining(self, scope: Mapping[str, Any], consumption: Mapping[str, Any] | None = None) -> dict[str, float]:
        consumption = consumption or self.consumption()
        return {name: limit - self.used(name, consumption) for name, limit in self.limits(scope)}

    def snapshot(self) -> dict[str, Any]:
        consumption = self.consumption()
        auth = self.auth
        remaining = {"total_core_h": (auth.total_core_h * 3600.0 - consumption["total_core_s"]) / 3600.0,
                     "stages_core_h": {s: (auth.stages.get(s, 0.0) * 3600.0 - consumption["stages_core_s"].get(s, 0.0)) / 3600.0 for s in STAGES},
                     "initial_model_q0_core_h": (auth.initial_model_q0_core_h * 3600.0 - consumption["q0_core_s"]) / 3600.0}
        return {"at": utc_now(), "consumed_core_h": {"total": consumption["total_core_s"] / 3600.0,
                                                      "stages": {k: v / 3600.0 for k, v in consumption["stages_core_s"].items()},
                                                      "initial_model_q0": consumption["q0_core_s"] / 3600.0,
                                                      "runs": {k: v / 3600.0 for k, v in consumption["runs_core_s"].items()}},
                "remaining": remaining, "running": consumption["running"], "n_attempts": consumption["n_attempts"],
                "allowances": {"total_core_h": auth.total_core_h, "stages_core_h": dict(auth.stages), "initial_model_q0_core_h": auth.initial_model_q0_core_h,
                               "amendments": [a["id"] for a in auth.applied_amendments]},
                "note": "measured from every attempt directory (completed, failed, cancelled, retried, running); all of it is newly consumed by this "
                        "benchmark (no cached label of another root is charged here)"}


# ---------------------------------------------------------------------------------------------- guard


@dataclass
class BudgetGuard:
    """Pre-submission room and live enforcement against the ledger; every decision is journalled."""

    ledger: ComputeLedger
    estimate: Callable[[str], float]
    journal_path: Path
    decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def margin(self) -> float:
        return self.ledger.auth.enforcement.margin_core_s

    def _journal(self, event: dict[str, Any]) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": utc_now(), **event}, sort_keys=True, default=str) + "\n")
        write_json(self.journal_path.parent / "ledger_snapshot.json", self.ledger.snapshot())

    def room(self, scope: Mapping[str, Any], kind: str) -> dict[str, Any]:
        """Whether a new ``kind`` job charging ``scope`` fits every remaining allowance."""

        estimate = float(self.estimate(kind))
        remaining = self.ledger.remaining(scope)
        need = estimate + self.margin
        short = {name: value for name, value in remaining.items() if value < need}
        return {"allowed": not short, "kind": kind, "estimate_core_s": estimate, "margin_core_s": self.margin, "remaining_core_s": remaining,
                "insufficient": short, "scope": dict(scope)}

    def require_room(self, scope: Mapping[str, Any], kind: str, *, label: str) -> dict[str, Any]:
        decision = self.room(scope, kind)
        self._journal({"event": "submission_check", "label": label, **decision})
        if not decision["allowed"]:
            raise BudgetRefused(f"{label}: no room for a {kind} job ({decision['estimate_core_s']:.0f} + margin {self.margin:.0f} core-s) in "
                                f"{sorted(decision['insufficient'])}")
        return decision

    def violation(self, task_dir: Path) -> dict[str, Any] | None:
        """For a running task: the scopes whose consumption has reached allowance - margin (None if fine)."""

        consumption = self.ledger.consumption()
        running = [r for r in consumption["running"] if r["task"] == task_dir.name]
        if not running:
            return None
        scope = self.ledger.classify(task_dir)
        remaining = self.ledger.remaining(scope, consumption)
        exceeded = {name: value for name, value in remaining.items() if value <= self.margin}
        if not exceeded:
            return None
        return {"task": task_dir.name, "scope": scope, "remaining_core_s": remaining, "exceeded": exceeded}

    def record_cancellation(self, task_dir: Path, violation: Mapping[str, Any]) -> None:
        consumed = sum(a.core_s for a in self.ledger.attempts() if a.task_dir == task_dir)
        self._journal({"event": "cancelled_for_budget", "task": task_dir.name, "violation": dict(violation), "task_consumed_core_s": consumed})


class GuardedService:
    """``LabelingService`` proxy for reference labels: room check on submit, enforcement on poll."""

    def __init__(self, inner: Any, guard: BudgetGuard, *, kind_of: Callable[[Path], str]) -> None:
        self._inner = inner
        self._guard = guard
        self._kind_of = kind_of

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def submit(self, task: Any) -> Any:
        task_id = task if isinstance(task, str) else task.task_id
        work = Path(self._inner.work_dir_for(task_id))
        if would_launch(work):  # an existing job (running or finished) is reconciled, never charged twice
            self._guard.require_room(self._guard.ledger.classify(work), self._kind_of(work), label=task_id)
        return self._inner.submit(task)

    def poll(self, task: Any) -> Any:
        result = self._inner.poll(task)
        work = Path(result.work_dir)
        violation = self._guard.violation(work)
        if violation is not None:
            LOGGER.warning("cancelling %s: allowance reached %s", result.task_id, sorted(violation["exceeded"]))
            result = self._inner.cancel(result.task_id)
            self._guard.record_cancellation(work, violation)
        return result


class GuardedLabeler:
    """Closed-loop ``Labeler`` proxy (around ``BudgetedLabeler``): room check on submit, enforcement on poll."""

    def __init__(self, inner: Any, guard: BudgetGuard, run: str) -> None:
        self.inner = inner
        self.service = inner.service
        self.guard = guard
        self.run = run

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def submit(self, task: Any) -> Any:
        work = Path(task.work_dir)
        if would_launch(work):
            self.guard.require_room(self.guard.ledger.classify(work), "fragment" if _is_fragment(work) else "full", label=task.task_id)
        return self.inner.submit(task)

    def poll(self, task: Any) -> Any:
        result = self.inner.poll(task)
        work = Path(result.work_dir)
        violation = self.guard.violation(work)
        if violation is not None:
            LOGGER.warning("cancelling %s: allowance reached %s", result.task_id, sorted(violation["exceeded"]))
            result = self.inner.cancel(result)
            self.guard.record_cancellation(work, violation)
        return result


def would_launch(task_dir: Path) -> bool:
    """Whether a labeling submit for this task directory would start a new job (no task or a never-submitted one).

    The labeling service's ``submit`` is a no-op for a task that already has (or had) a job; such a task is
    reconciled by polling, its attempts are already in the ledger, and it must not be refused or charged again.
    """

    record = read_json(task_dir / "task.json") if (task_dir / "task.json").is_file() else None
    if record is None:
        return True
    return record.get("execution") == "prepared" and not (record.get("metadata") or {}).get("attempts")


def _is_fragment(task_dir: Path) -> bool:
    region = read_json(task_dir / "region.json") if (task_dir / "region.json").is_file() else {}
    return region.get("mode") not in ("full", "full_system")


def estimator(loaded: LoadedBenchmark, auth: AuthorizationConfig) -> Callable[[str], float]:
    """Conservative pre-submission estimate: a quantile of measured costs of the kind (this root + timing sources)."""

    import numpy as np

    from .plan import measured_timings

    timings = measured_timings(loaded)
    q = auth.enforcement.estimate_quantile
    pools: dict[str, list[float]] = {"full": [], "fragment": []}
    for root in (loaded.root, loaded.sealed_root):
        path = root / "labels.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                record = json.loads(line) if line.strip() else None
                cost = ((record or {}).get("cost") or {}).get("cpu_core_s")
                if record and cost and record.get("kind") in pools and (record["kind"] == "fragment" or int(record.get("chain_length") or 0) == loaded.config.chain_length):
                    pools[record["kind"]].append(float(cost))
    fallback = {"full": timings["full_label_core_s"], "fragment": timings["fragment_label_core_s"]}

    def estimate(kind: str) -> float:
        values = pools.get(kind) or []
        if len(values) >= 3:
            return float(np.quantile(values, q))
        stats = fallback[kind]
        if not stats.get("n"):
            raise BudgetRefused(f"no measured cost of {kind} labels to estimate from")
        return float(stats.get("p90") if q >= 0.9 else stats["mean"])

    return estimate


def build_guard(loaded: LoadedBenchmark) -> BudgetGuard:
    auth = load_authorization(loaded)
    return BudgetGuard(ComputeLedger(loaded, auth), estimator(loaded, auth), loaded.root / "execution" / "budget_journal.jsonl")


__all__ = ["AMENDABLE", "Amendment", "AuthorizationConfig", "BudgetGuard", "BudgetRefused", "ComputeLedger", "GuardedLabeler", "GuardedService", "STAGES", "build_guard",
           "amendment_digest", "amendments_path", "estimator", "load_authorization", "scan_attempt", "would_launch"]
