"""Resumable execution of a benchmark under its authorization, with a durable journal.

``execute`` walks the protocol's stages in order. Each stage is idempotent (labels, models, runs and
reports are reused when they exist) and records a terminal status in ``execution/state.json``;
``execution/journal.jsonl`` is the append-only history. On every start the existing work is reconciled
first: label runners and controllers adopt running CP2K jobs before submitting anything new.

Stopping conditions (protocol section 12 and the authorization): elapsed time beyond
``max_elapsed_h`` from the recorded execution start; a compute allowance that cannot cover the next
job; both qualification stages failing; no eligible challenge family; a dependency failing
deterministically (recorded as blocked). A stage that could not finish for budget or infrastructure
reasons is recorded as incomplete, never as a scientific failure.

The repair-run order is fixed once, before any repair outcome exists: seed-major (a budget interruption
leaves complete seeds), cases interleaved across families, and the arm order rotated per (seed, case)
so that no arm is systematically last. Before the sealed final-test labels are produced, the list of
runs and their final model versions is frozen (``execution/frozen_outcomes.json``).
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gmd_active_learning.contracts.ids import file_sha256

from tools.experiments.alkane_hotspots.manifest import utc_now

from .common import read_json, write_json
from .config import LoadedBenchmark

LOGGER = logging.getLogger(__name__)

EXEC_DIR = "execution"


class ExecutionStop(RuntimeError):
    """A stopping condition was reached; ``status`` names it."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------------------------- state and journal


class Journal:
    def __init__(self, loaded: LoadedBenchmark) -> None:
        self.loaded = loaded
        self.dir = loaded.root / EXEC_DIR
        self.state_path = self.dir / "state.json"
        self.journal_path = self.dir / "journal.jsonl"
        self.state: dict[str, Any] = read_json(self.state_path) if self.state_path.is_file() else {"stages": {}}

    def save(self) -> None:
        self.state["updated_at"] = utc_now()
        write_json(self.state_path, self.state)

    def log(self, event: str, **payload: Any) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": utc_now(), "event": event, **payload}, sort_keys=True, default=str) + "\n")
        LOGGER.info("journal: %s %s", event, {k: v for k, v in payload.items() if k not in ("traceback",)})

    def stage(self, name: str) -> dict[str, Any]:
        return self.state["stages"].get(name, {})

    def done(self, name: str) -> bool:
        return self.stage(name).get("status") in ("done", "unmet", "skipped")

    def set_stage(self, name: str, status: str, **detail: Any) -> None:
        entry = self.state["stages"].setdefault(name, {})
        if status == "running" and "started_at" not in entry:
            entry["started_at"] = utc_now()
        if status != "running":
            entry["finished_at"] = utc_now()
        entry.update(status=status, **detail)
        self.save()
        self.log("stage", stage=name, status=status, **{k: v for k, v in detail.items() if k != "traceback"})

    def next_action(self, text: str) -> None:
        self.state["next_action"] = text
        self.save()


def _git(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=30).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    return {"path": str(path), "commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def start_record(loaded: LoadedBenchmark) -> dict[str, Any]:
    from tools.experiments.alkane_hotspots.manifest import environment_info

    from .budget import authorization_path

    workspace = loaded.repo_path(".").parent
    gmd = loaded.repo_path(loaded.config.repair.executable)
    cp2k = shutil.which(loaded.alpha.config.cp2k.executable)
    version = None
    if cp2k:
        try:
            version = subprocess.run([cp2k, "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[:2]
        except (OSError, subprocess.SubprocessError):
            version = None
    return {"execution_start": utc_now(), "execution_start_epoch": time.time(), "host": platform.node(), "platform": platform.platform(),
            "repositories": {name: _git(workspace / name) for name in ("GMD_active", "GMD", "GMD_SGT", "cp2k-ml-scf") if (workspace / name).is_dir()},
            "environment": environment_info(loaded.repo_path(".")), "cp2k": {"executable": cp2k, "version": version},
            "gmd": {"executable": str(gmd), "sha256": file_sha256(gmd) if gmd.is_file() else None},
            "artifacts_root": str(loaded.root), "sealed_root": str(loaded.sealed_root),
            "authorization": {"path": str(authorization_path(loaded)), "sha256": file_sha256(authorization_path(loaded))},
            "benchmark_config": {"path": str(loaded.path), "sha256": file_sha256(loaded.path)}}


# ---------------------------------------------------------------------------------------------- run order


def fixed_run_order(loaded: LoadedBenchmark, families: list[str], arms: list[str]) -> list[dict[str, Any]]:
    """Seed-major; cases interleaved across families; arm order rotated per (seed, case). Fixed before outcomes."""

    from .repair import repair_cases, run_matrix

    by_family: dict[str, list[dict[str, Any]]] = {}
    for case in repair_cases(loaded):
        if case["family"] in families:
            by_family.setdefault(case["family"], []).append(case)
    interleaved = []
    for index in range(max((len(v) for v in by_family.values()), default=0)):
        for family in families:
            if index < len(by_family.get(family, [])):
                interleaved.append(by_family[family][index])
    specs = {(s.arm, s.case, s.seed): s for s in run_matrix(loaded, arms=arms)}
    order = []
    for s_index, seed in enumerate(loaded.config.repair.seeds):
        for c_index, case in enumerate(interleaved):
            shift = (s_index + c_index) % len(arms)
            for arm in arms[shift:] + arms[:shift]:
                spec = specs[(arm, case["parent_id"], int(seed))]
                order.append({"run": spec.name, "arm": arm, "case": spec.case, "family": spec.family, "seed": spec.seed, "kind": spec.kind})
    return order


# ---------------------------------------------------------------------------------------------- execution


def _elapsed_h(journal: Journal) -> float:
    return (time.time() - float(journal.state["start"]["execution_start_epoch"])) / 3600.0


def verify_authorization_record(loaded: LoadedBenchmark, journal: Journal, auth: Any) -> None:
    """The original authorization is unchanged; amendments are append-only and journalled once each.

    The authorization file recorded at the execution start must keep its hash (an amendment never edits it).
    Every amendment in the amendments file is journalled (with the code commit that applies it) the first
    time it is seen; afterwards its content may not change and it may not be removed.
    """

    from .budget import authorization_path

    original = journal.state["start"]["authorization"]["sha256"]
    current = file_sha256(authorization_path(loaded))
    if current != original:
        raise ExecutionStop("blocked", f"the original authorization file changed since the execution start ({original[:12]} -> {current[:12]}); "
                                       "allowances change only through the amendments file")
    recorded = journal.state.setdefault("amendments", {})
    applied = {a["id"]: a for a in auth.applied_amendments}
    for amendment_id, entry in recorded.items():
        if amendment_id not in applied:
            raise ExecutionStop("blocked", f"authorization amendment {amendment_id} was recorded but is missing from the amendments file")
        if applied[amendment_id]["sha256"] != entry["sha256"]:
            raise ExecutionStop("blocked", f"authorization amendment {amendment_id} changed after it was recorded")
    order = list(recorded)
    if list(applied)[: len(order)] != order:
        raise ExecutionStop("blocked", f"authorization amendments reordered: recorded {order}, file {list(applied)}")
    for amendment_id, amendment in applied.items():
        if amendment_id in recorded:
            continue
        recorded[amendment_id] = {"sha256": amendment["sha256"], "recorded_at": utc_now()}
        journal.save()
        journal.log("authorization_amendment", amendment={k: v for k, v in amendment.items() if k != "sha256"}, sha256=amendment["sha256"],
                    code=_git(loaded.repo_path(".")),
                    effective={"total_core_h": auth.total_core_h, "stages": dict(auth.stages), "initial_model_q0_core_h": auth.initial_model_q0_core_h,
                               "per_run_core_h": auth.per_run_core_h, "max_elapsed_h": auth.max_elapsed_h})


# ---------------------------------------------------------------------------------------------- projections before stages


_FINISHED_BY = {"initial_model": "qualified", "development": "development", "fidelity_audit": "development", "repair_val": "repair_val",
                "repair_runs": "repair_runs", "final_test": "final_test"}


def _closed_loop_runs(loaded: LoadedBenchmark, journal: Journal) -> int:
    if "run_order" in journal.state:
        runs = journal.state.get("runs", {})
        return sum(1 for i in journal.state["run_order"] if i["kind"] == "closed_loop" and runs.get(i["run"], {}).get("status") not in ("done", "failed_terminal"))
    from .plan import build_plan

    return int(sum(v["n_runs"] for v in build_plan(loaded, write=False)["stages"]["repair_runs"]["by_arm"].values() if v["kind"] == "closed_loop"))


def _finished(journal: Journal, stage: str) -> bool:
    key = _FINISHED_BY[stage]
    if key == "qualified":
        return any(journal.stage(f"qualification_q{q}").get("accepted") for q in range(4))
    return journal.done(key)


def gate(loaded: LoadedBenchmark, journal: Journal, label: str, stage: str, *, qualification_stage: int | None = None) -> dict[str, Any]:
    """Re-estimate a stage before it starts; stop (unaffordable, with a redistribution proposal) if it does not fit."""

    from .projection import project, redistribution_proposal, write_projection

    runs = _closed_loop_runs(loaded, journal) if stage == "repair_runs" else None
    projection = project(loaded, stage, qualification_stage=qualification_stage, closed_loop_runs=runs)
    summary = [{k: s[k] for k in ("scope", "expected_core_h", "upper90_core_h", "remaining_core_h", "fits")} for s in projection["scopes"]]
    costs = {k: {m: v.get(m) for m in ("n", "mean", "sd", "source")} for k, v in projection["costs_core_s"].items()}
    if projection["fits"]:
        write_projection(loaded, label, projection)
        journal.log("projection", stage=label, fits=True, scopes=summary, costs_core_s=costs)
        return projection
    future: dict[str, float] = {}
    current = {s["scope"] for s in projection["scopes"]}
    for other in ("development", "repair_val", "repair_runs", "final_test"):
        if other in current or _finished(journal, other):
            continue
        item = project(loaded, other, closed_loop_runs=_closed_loop_runs(loaded, journal) if other == "repair_runs" else None)
        for s in item["scopes"]:
            # a donor keeps its hard bound where one exists (repair runs: runs x per-run cap), else its 90% upper projection
            bound = (s["by_kind"].get("fragment") or {}).get("per_run_cap_bound_core_h")
            future[s["scope"]] = max(s["upper90_core_h"], bound) if bound is not None else s["upper90_core_h"]
    finished = {s for s in _FINISHED_BY if s not in current and s not in future and _finished(journal, s)}
    proposal = redistribution_proposal(loaded, projection, future=future, finished=finished)
    projection["redistribution_proposal"] = proposal
    path = write_projection(loaded, label, projection)
    journal.log("projection", stage=label, fits=False, scopes=summary, costs_core_s=costs, proposal=proposal, path=str(path))
    journal.set_stage(label, "unaffordable", scopes=summary, proposal=proposal)
    short = "; ".join(f"{s['scope']}: upper {s['upper90_core_h']:.1f} vs remaining {s['remaining_core_h']:.1f} core-h" for s in summary if not s["fits"])
    raise ExecutionStop("unaffordable", f"{label} is projected not to fit under the current authorization ({short}); "
                                        f"no allowance was moved; proposal in {path}")


def _check_time(journal: Journal, auth: Any) -> None:
    elapsed = _elapsed_h(journal)
    if elapsed >= auth.max_elapsed_h:
        raise ExecutionStop("elapsed_allowance_exhausted", f"{elapsed:.1f} h elapsed >= {auth.max_elapsed_h} h")


def eligibility(loaded: LoadedBenchmark) -> dict[str, Any]:
    """A family is eligible for repair when a development structure shows a reference-confirmed failure and the
    frozen fragment setting passed the audit for it."""

    failures = read_json(loaded.root / "development" / "failures.json")
    audit = read_json(loaded.root / "fidelity" / "audit.json")
    out = {}
    for family in loaded.config.challenges.enabled():
        rows = [r for r in failures["rows"] if r.get("family") == family and "region" in r]
        failed = sum(1 for r in rows if r["failure"])
        audit_ok = bool((audit["families"].get(family) or {}).get("passes")) or not loaded.config.fidelity.require_audit
        reasons = []
        if not failed:
            reasons.append("no reference-confirmed failure on the development structures (uninformative for repair)")
        if not audit_ok:
            reasons.append("the frozen fragment setting failed the fidelity audit (no validated local labels)")
        out[family] = {"development_failures": f"{failed}/{len(rows)}", "audit_passes": audit_ok, "eligible": not reasons, "reasons": reasons}
    return out


def execute(loaded: LoadedBenchmark, *, echo: Callable[[str], None] = print, max_runs: int | None = None) -> dict[str, Any]:
    from .budget import BudgetRefused, build_guard, load_authorization

    journal = Journal(loaded)
    auth = load_authorization(loaded)
    if "start" not in journal.state:
        journal.state["start"] = start_record(loaded)
        journal.save()
        journal.log("execution_start", record=journal.state["start"])
    try:
        verify_authorization_record(loaded, journal, auth)
    except ExecutionStop as stop:
        journal.log("stop", status=stop.status, message=str(stop))
        echo(f"STOP {stop.status}: {stop}")
        return journal.state
    journal.log("resume", elapsed_h=_elapsed_h(journal), ledger=build_guard(loaded).ledger.snapshot())
    try:
        _run_stages(loaded, journal, auth, echo=echo, max_runs=max_runs)
    except ExecutionStop as stop:
        journal.state["terminal"] = {"status": stop.status, "message": str(stop), "at": utc_now()}
        journal.save()
        journal.log("stop", status=stop.status, message=str(stop))
        echo(f"STOP {stop.status}: {stop}")
    except BudgetRefused as exc:
        journal.state["terminal"] = {"status": "allowance_exhausted", "message": str(exc), "at": utc_now()}
        journal.save()
        journal.log("stop", status="allowance_exhausted", message=str(exc))
        echo(f"STOP allowance: {exc}")
    journal.state["ledger"] = build_guard(loaded).ledger.snapshot()
    journal.save()
    return journal.state


def _run_stages(loaded: LoadedBenchmark, journal: Journal, auth: Any, *, echo: Callable[[str], None], max_runs: int | None) -> None:
    from .failure import analyse_development
    from .plan import build_plan
    from .preflight import preflight
    from .qualify import qualify
    from .references import fidelity_audit, freeze_protocol, label_stage
    from .report import write_report

    # 0. consistency checks and protocol freeze
    if not journal.done("freeze"):
        journal.set_stage("freeze", "running")
        checks = preflight(loaded)
        plan = build_plan(loaded)
        problems = [] if checks["passed"] else ["geometry preflight failed"]
        problems += [f"plan exceeds budget {name}" for name, item in plan["budgets"].items() if not item["within_budget"]]
        if problems:
            journal.set_stage("freeze", "blocked", problems=problems)
            raise ExecutionStop("blocked", f"consistency checks failed: {problems}")
        frozen = freeze_protocol(loaded, note="execution under repair_c16_authorization.yaml")
        journal.state["protocol_sha256"] = frozen["protocol"]["sha256"]
        journal.set_stage("freeze", "done", protocol_sha256=frozen["protocol"]["sha256"], structures=frozen["structures"])

    # 1. initial-model qualification (q0, then q1 only if q0 is scientifically not accepted)
    qualified = None
    for q in range(len(loaded.config.initial_model.stages)):
        _check_time(journal, auth)
        name = f"qualification_q{q}"
        if journal.done(name):
            if journal.stage(name).get("accepted"):
                qualified = q
                break
            if journal.stage(name).get("status") == "done":
                continue
        if q >= 1 and journal.stage(name).get("status") != "running":  # an expansion starts only if it is projected to fit whole
            gate(loaded, journal, name, "initial_model", qualification_stage=q)
        journal.set_stage(name, "running")
        journal.next_action(f"label --stage initial_model --qualification-stage {q}; qualify --stage {q}")
        labels = label_stage(loaded, "initial_model", qualification_stage=q, echo=echo)
        if labels["missing"]:
            journal.set_stage(name, "incomplete", missing=labels["missing"], not_started=labels["not_started"], failed=labels["failed"])
            raise ExecutionStop("qualification_incomplete", f"stage {q}: {len(labels['missing'])} reference labels missing (budget or infrastructure)")
        report = qualify(loaded, q)
        journal.set_stage(name, "done", accepted=report["accepted"], criteria=report["criteria"], id_val=report["id_val_accuracy"])
        if report["accepted"]:
            qualified = q
            break
    if qualified is None:
        journal.set_stage("prerequisite", "unmet", reason="no qualification stage met the acceptance criteria")
        write_report(loaded, publish=True)
        raise ExecutionStop("prerequisite_unmet", "neither prescribed qualification attempt met the frozen acceptance criteria")

    # 2. development references, failures, fidelity audit, eligibility
    _check_time(journal, auth)
    if not journal.done("development"):
        gate(loaded, journal, "development", "development")
        journal.set_stage("development", "running")
        journal.next_action("label --stage development; failures; fidelity-audit")
        labels = label_stage(loaded, "development", echo=echo)
        if labels["missing"]:
            journal.set_stage("development", "incomplete", missing=labels["missing"], not_started=labels["not_started"])
            raise ExecutionStop("development_incomplete", "development reference labels missing")
        analyse_development(loaded)
        audit = fidelity_audit(loaded, echo=echo)
        incomplete = [f for f, v in audit["families"].items() if not v["complete"]]
        if incomplete and any(r.get("status") == "not_started" for r in audit["fragments"]):
            journal.set_stage("development", "incomplete", audit_incomplete=incomplete)
            raise ExecutionStop("audit_incomplete", f"fidelity audit incomplete for {incomplete}")
        journal.set_stage("development", "done", eligibility=eligibility(loaded))
    families = [f for f, v in journal.stage("development")["eligibility"].items() if v["eligible"]]
    if not families:
        write_report(loaded, publish=True)
        raise ExecutionStop("no_eligible_family", "no challenge family is eligible for repair (see development eligibility)")

    # 3. repair-validation references (promotion evidence)
    _check_time(journal, auth)
    if not journal.done("repair_val"):
        gate(loaded, journal, "repair_val", "repair_val")
        journal.set_stage("repair_val", "running")
        labels = label_stage(loaded, "repair_val", echo=echo)
        if labels["missing"]:
            journal.set_stage("repair_val", "incomplete", missing=labels["missing"], not_started=labels["not_started"])
            raise ExecutionStop("repair_val_incomplete", "repair-validation labels missing")
        journal.set_stage("repair_val", "done")

    # 4. monitor calibration per closed-loop arm
    if not journal.done("calibration"):
        journal.set_stage("calibration", "running")
        from .repair import calibrate_arm

        arms = {}
        for arm in loaded.config.arms:
            if not arm.enabled or arm.kind != "closed_loop":
                continue
            try:
                record = calibrate_arm(loaded, arm.name)
                arms[arm.name] = {"available": record["ok"], "alarm_fraction": record["alarm_fraction"], "cold": record["cold_required_metrics"]}
            except Exception as exc:  # noqa: BLE001 - an arm that cannot pass preflight is recorded unavailable, never retried blindly
                arms[arm.name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        journal.set_stage("calibration", "done", arms=arms)
    arm_status = journal.stage("calibration")["arms"]
    arms = [a.name for a in loaded.config.arms if a.enabled and a.kind == "continued_training"]
    arms += [name for name, v in arm_status.items() if v["available"]]

    # 5. repair runs in the fixed order
    if "run_order" not in journal.state:
        journal.state["run_order"] = fixed_run_order(loaded, families, arms)
        journal.state["run_order_fixed_at"] = utc_now()
        journal.save()
        journal.log("run_order_fixed", n_runs=len(journal.state["run_order"]), families=families, arms=arms)
    if not journal.done("repair_runs"):
        gate(loaded, journal, "repair_runs_projection", "repair_runs")
        _repair_runs(loaded, journal, auth, echo=echo, max_runs=max_runs)

    # 6. freeze outcomes, then sealed final-test labels and evaluation
    if not journal.done("freeze_outcomes"):
        freeze_outcomes(loaded, journal)
    _check_time(journal, auth)
    if not journal.done("final_test"):
        gate(loaded, journal, "final_test_projection", "final_test")
        journal.set_stage("final_test", "running")
        labels = label_stage(loaded, "final_test", echo=echo)
        if labels["missing"]:
            journal.set_stage("final_test", "incomplete", missing=labels["missing"], not_started=labels["not_started"])
            raise ExecutionStop("final_test_incomplete", "sealed final-test labels missing")
        journal.set_stage("final_test", "done")
    if not journal.done("evaluation"):
        from .evaluate import evaluate

        journal.set_stage("evaluation", "running")
        result = evaluate(loaded, allow_incomplete=False, reason="final evaluation of the frozen outcomes")
        journal.set_stage("evaluation", "done", n_runs=len(result["runs"]))
    write_report(loaded, publish=True)
    journal.set_stage("report", "done")
    journal.next_action("none: the prescribed experiment and reporting are complete")


def _repair_runs(loaded: LoadedBenchmark, journal: Journal, auth: Any, *, echo: Callable[[str], None], max_runs: int | None) -> None:
    from .budget import BudgetRefused, build_guard
    from .repair import run_matrix, run_repair

    journal.set_stage("repair_runs", "running")
    specs = {s.name: s for s in run_matrix(loaded)}
    runs = journal.state.setdefault("runs", {})
    consecutive_errors: dict[str, int] = {}
    started = 0
    order = journal.state["run_order"]
    for item in order + [dict(i, retry_pass=True) for i in order]:  # second pass: one retry of runs that failed once
        name = item["run"]
        if runs.get(name, {}).get("status") in ("done", "failed_terminal"):
            continue
        if item.get("retry_pass") and runs.get(name, {}).get("status") != "failed":
            continue
        _check_time(journal, auth)
        if max_runs is not None and started >= max_runs:
            journal.next_action(f"repair-run {name} (resume `execute`)")
            raise ExecutionStop("paused", f"paused after {started} runs (--max-runs)")
        if item["kind"] == "closed_loop":
            guard = build_guard(loaded)
            room = guard.room({"stage": "repair_runs", "run": name}, "fragment")
            if not room["allowed"]:
                runs[name] = {"status": "not_started", "reason": f"allowance: {sorted(room['insufficient'])}"}
                journal.save()
                journal.set_stage("repair_runs", "incomplete", reason="repair-run allowance exhausted")
                raise ExecutionStop("allowance_exhausted", f"no room for repair run {name}: {sorted(room['insufficient'])}")
        journal.next_action(f"repair-run {name}")
        journal.log("run_start", run=name)
        started += 1
        try:
            report = run_repair(loaded, specs[name])
            status = "done" if report.get("final_stage") == "done" else "incomplete"
            runs[name] = {"status": status, "final_stage": report.get("final_stage"), "dft_core_h": report.get("measured_dft_cpu_core_hours"),
                          "finished_at": utc_now()}
        except BudgetRefused as exc:
            runs[name] = {"status": "budget_refused", "reason": str(exc)}
            journal.save()
            raise
        except Exception as exc:  # noqa: BLE001 - recorded; a deterministic error is not retried endlessly
            signature = f"{type(exc).__name__}: {str(exc)[:200]}"
            consecutive_errors[signature] = consecutive_errors.get(signature, 0) + 1
            attempts = runs.get(name, {}).get("attempts", 0) + 1
            runs[name] = {"status": "failed_terminal" if attempts >= 2 else "failed", "error": signature, "attempts": attempts,
                          "traceback": traceback.format_exc()[-4000:]}
            journal.log("run_error", run=name, error=signature)
            if consecutive_errors[signature] >= 3:
                journal.save()
                journal.set_stage("repair_runs", "blocked", error=signature)
                raise ExecutionStop("blocked", f"the same error ended 3 runs: {signature}") from exc
        journal.save()
        journal.log("run_end", run=name, **{k: v for k, v in runs[name].items() if k != "traceback"})
    pending = [i["run"] for i in journal.state["run_order"] if runs.get(i["run"], {}).get("status") not in ("done", "failed_terminal")]
    if pending:
        journal.set_stage("repair_runs", "incomplete", pending=pending)
        raise ExecutionStop("repair_runs_incomplete", f"{len(pending)} runs not finished")
    journal.set_stage("repair_runs", "done", n_done=sum(1 for v in runs.values() if v["status"] == "done"),
                      n_failed=sum(1 for v in runs.values() if v["status"] == "failed_terminal"))


def freeze_outcomes(loaded: LoadedBenchmark, journal: Journal) -> dict[str, Any]:
    """The run list and final model versions, fixed before any final-test label exists."""

    from .repair import RUNS_SUBDIR

    sealed_labels = loaded.sealed_root / "labels.jsonl"
    if sealed_labels.is_file() and sealed_labels.read_text().strip():
        raise ExecutionStop("blocked", "final-test labels exist before the outcomes were frozen")
    runs = []
    for item in journal.state["run_order"]:
        record = journal.state.get("runs", {}).get(item["run"], {})
        path = loaded.root / RUNS_SUBDIR / item["run"] / "repair_run.json"
        final = read_json(path).get("final_model") if path.is_file() else None
        export = (final or {}).get("export")
        runs.append({**item, "status": record.get("status"), "final_model": final,
                     "export_sha256": file_sha256(export) if export and Path(export).is_file() else None})
    payload = {"frozen_at": utc_now(), "runs": [r for r in runs if r["status"] == "done"], "not_evaluated": [r for r in runs if r["status"] != "done"],
               "statement": "fixed before any sealed final-test label existed; no run is continued or replaced after this point"}
    write_json(loaded.root / EXEC_DIR / "frozen_outcomes.json", payload)
    journal.set_stage("freeze_outcomes", "done", n_runs=len(payload["runs"]), n_not_evaluated=len(payload["not_evaluated"]))
    return payload


__all__ = ["ExecutionStop", "Journal", "eligibility", "execute", "fixed_run_order", "freeze_outcomes", "gate", "start_record", "verify_authorization_record"]
