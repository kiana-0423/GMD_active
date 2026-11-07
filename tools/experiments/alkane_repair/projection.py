"""Re-estimation of a stage's DFT cost before it starts, from this execution's own measurements.

Full-system and fragment labels are costed separately. The cost of a task is the sum of all its attempts
(failed attempts, retries and cancellations included) as recorded in the attempt directories, i.e. the
same numbers the budget ledger charges. A kind with fewer than three measured tasks in this execution
falls back to the plan's timing sources (earlier runs on the same machine).

The projection of a stage that needs ``n`` more labels of a kind with measured mean ``m`` and standard
deviation ``s`` is ``n m`` (expected) and ``n m + z s sqrt(n)`` (one-sided upper bound of the sum, normal
approximation), with ``z`` the normal quantile of the authorization's ``enforcement.estimate_quantile`` (0.9 for C16,
the same level as the pre-submission estimate). A stage is started only if its upper bound plus the enforcement margin fits the
stage's remaining allowance and the remaining total; otherwise it is reported as unaffordable under the
current authorization, with a concrete redistribution proposal that keeps the total unchanged. No allowance
is moved automatically.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.experiments.alkane_hotspots.manifest import utc_now

from .budget import STAGES, ComputeLedger, _is_fragment, load_authorization
from .common import write_json
from .config import LoadedBenchmark



def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    return {"n": int(array.size), "mean": float(array.mean()), "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "p90": float(np.percentile(array, 90)), "max": float(array.max())}


def measured_costs(loaded: LoadedBenchmark) -> dict[str, Any]:
    """Core-seconds per finished task of each kind in this execution (every attempt), with the fallback used."""

    from .plan import measured_timings

    auth = load_authorization(loaded)
    per_task: dict[Path, float] = defaultdict(float)
    running: set[Path] = set()
    for attempt in ComputeLedger(loaded, auth).attempts():
        per_task[attempt.task_dir] += attempt.core_s
        if attempt.running:
            running.add(attempt.task_dir)
    pools: dict[str, list[float]] = {"full": [], "fragment": []}
    for task_dir, core_s in per_task.items():
        if task_dir not in running:
            pools["fragment" if _is_fragment(task_dir) else "full"].append(core_s)
    out: dict[str, Any] = {}
    fallback = measured_timings(loaded)
    for kind, key in (("full", "full_label_core_s"), ("fragment", "fragment_label_core_s")):
        own = _stats(pools[kind])
        if own["n"] >= 3:
            out[kind] = {**own, "source": "this execution (all attempts per task)"}
        else:
            fb = fallback[key]
            sd = (fb["p90"] - fb["mean"]) / NormalDist().inv_cdf(0.9) if fb.get("n") else 0.0  # the timing sources report a p90
            out[kind] = {"n": fb.get("n", 0), "mean": fb.get("mean"), "sd": max(sd, 0.0), "p90": fb.get("p90"), "max": fb.get("max"),
                         "source": f"plan timing sources (only {own['n']} measured in this execution)", "own": own}
    return out


def _sum_projection(n: float, stats: dict[str, Any], z: float) -> dict[str, float]:
    if n <= 0:
        return {"n": 0, "expected_core_h": 0.0, "upper90_core_h": 0.0}
    if stats.get("mean") is None:
        raise ValueError("no measured cost to project from")
    expected = n * stats["mean"]
    upper = expected + z * stats["sd"] * math.sqrt(n)
    return {"n": n, "expected_core_h": expected / 3600.0, "upper90_core_h": upper / 3600.0}


def stage_need(loaded: LoadedBenchmark, stage: str, *, qualification_stage: int | None = None, closed_loop_runs: int | None = None) -> dict[str, float]:
    """Labels a stage still needs, by kind (exact for public full-label stages; protocol counts otherwise)."""

    from tools.experiments.alkane_hotspots.labels import LabelStore

    from .plan import build_plan
    from .references import _full_records, stage_structures

    if stage in ("initial_model", "development", "repair_val"):
        _, structures = stage_structures(loaded, stage, qualification_stage=qualification_stage)
        ids = {r["parent_id"] for r in structures}
        valid = {r["parent_id"] for r in _full_records(LabelStore(loaded.root), ids) if r["status"] == "valid"}
        need = {"full": float(len(ids - valid)), "fragment": 0.0}
        if stage == "development":  # audit fragments of the protocol, less the valid ones already measured (resume)
            planned = build_plan(loaded, write=False)["stages"]["fidelity_audit"]["dft"]["n_fragment"]
            done = sum(1 for r in LabelStore(loaded.root).all() if r.get("kind") == "fragment" and r.get("parent_id") in ids and r.get("status") == "valid")
            need["fragment_audit"] = float(max(planned - done, 0))
        return need
    plan = build_plan(loaded, write=False)
    if stage == "final_test":  # counted from the protocol (the sealed structures are not opened), less valid sealed labels (resume)
        sealed = loaded.sealed_root / "labels.jsonl"
        done = sum(1 for r in LabelStore(loaded.sealed_root).all() if r.get("kind") == "full" and r.get("status") == "valid") if sealed.is_file() else 0
        return {"full": float(max(plan["stages"]["final_test"]["dft"]["n_full"] - done, 0)), "fragment": 0.0}
    if stage == "repair_runs":
        r = loaded.config.repair
        per_run = r.cycles * r.labels_per_cycle * loaded.config.plan.expected_labels_per_run_fraction
        return {"full": 0.0, "fragment": float(closed_loop_runs or 0) * per_run, "runs": float(closed_loop_runs or 0)}
    raise ValueError(f"no projection for stage {stage!r}")


def project(loaded: LoadedBenchmark, stage: str, *, qualification_stage: int | None = None, closed_loop_runs: int | None = None) -> dict[str, Any]:
    """Projection of one stage against its remaining allowance (and the remaining total)."""

    auth = load_authorization(loaded)
    remaining = ComputeLedger(loaded, auth).snapshot()["remaining"]
    costs = measured_costs(loaded)
    need = stage_need(loaded, stage, qualification_stage=qualification_stage, closed_loop_runs=closed_loop_runs)
    margin_h = auth.enforcement.margin_core_s / 3600.0
    z = NormalDist().inv_cdf(auth.enforcement.estimate_quantile)
    parts = {}
    if stage == "development":
        parts["development"] = {"full": _sum_projection(need["full"], costs["full"], z)}
        parts["fidelity_audit"] = {"fragment": _sum_projection(need["fragment_audit"], costs["fragment"], z)}
    elif stage == "repair_runs":
        frag = _sum_projection(need["fragment"], costs["fragment"], z)
        cap = need["runs"] * auth.per_run_core_h
        frag = {**frag, "expected_core_h": min(frag["expected_core_h"], cap), "upper90_core_h": min(frag["upper90_core_h"], cap),
                "per_run_cap_bound_core_h": cap}
        parts["repair_runs"] = {"fragment": frag}
    else:
        parts[stage] = {"full": _sum_projection(need["full"], costs["full"], z)}
    scopes = []
    for scope, kinds in parts.items():
        expected = sum(k["expected_core_h"] for k in kinds.values())
        upper = sum(k["upper90_core_h"] for k in kinds.values())
        room = remaining["stages_core_h"].get(scope, 0.0)
        scopes.append({"scope": scope, "by_kind": kinds, "expected_core_h": expected, "upper90_core_h": upper, "remaining_core_h": room,
                       "fits": upper + margin_h <= room + 1e-9, "fits_expected": expected + margin_h <= room + 1e-9})
    total_upper = sum(s["upper90_core_h"] for s in scopes)
    total_ok = total_upper + margin_h <= remaining["total_core_h"] + 1e-9
    result = {"at": utc_now(), "stage": stage, "qualification_stage": qualification_stage, "need": need, "costs_core_s": costs,
              "scopes": scopes, "remaining_total_core_h": remaining["total_core_h"], "fits_total": total_ok,
              "fits": total_ok and all(s["fits"] for s in scopes), "margin_core_h": margin_h,
              "upper_quantile": auth.enforcement.estimate_quantile,
              "rule": "start only if the one-sided upper bound (enforcement.estimate_quantile) of the stage's DFT plus the enforcement margin fits every remaining allowance"}
    return result


def redistribution_proposal(loaded: LoadedBenchmark, projection: dict[str, Any], *, future: dict[str, float], finished: set[str]) -> dict[str, Any]:
    """A concrete transfer that covers the projection's deficits from slack elsewhere, keeping the total unchanged.

    ``future`` maps a not-yet-finished stage to its own projected upper bound (core-h); a finished stage's whole
    remaining allowance is slack. Slack is never taken below a stage's projected need. Nothing is applied.
    """

    auth = load_authorization(loaded)
    remaining = ComputeLedger(loaded, auth).snapshot()["remaining"]["stages_core_h"]
    margin_h = auth.enforcement.margin_core_s / 3600.0
    deficits = {s["scope"]: s["upper90_core_h"] + margin_h - s["remaining_core_h"] for s in projection["scopes"] if not s["fits"]}
    deficits = {k: math.ceil(v * 10.0) / 10.0 for k, v in deficits.items() if v > 0}  # round up to 0.1 core-h
    slack = {}
    for stage in STAGES:
        if stage in deficits:
            continue
        if stage in finished:
            slack[stage] = remaining.get(stage, 0.0)
        elif stage in future:
            slack[stage] = remaining.get(stage, 0.0) - future[stage] - margin_h
    slack = {k: math.floor(v * 10.0) / 10.0 for k, v in slack.items() if v >= 0.1}
    transfers, new = [], dict(auth.stages)
    for target, amount in deficits.items():
        left = amount
        for donor in sorted(slack, key=lambda k: -slack[k]):
            if left <= 1e-9:
                break
            take = min(slack[donor], left)
            if take <= 0:
                continue
            transfers.append({"from": donor, "to": target, "core_h": round(take, 1)})
            slack[donor] -= take
            new[donor] = round(new[donor] - take, 1)
            new[target] = round(new[target] + take, 1)
            left -= take
        if left > 1e-9:
            return {"feasible": False, "deficits": deficits, "uncovered_core_h": round(left, 1), "transfers": transfers,
                    "statement": f"the deficit of {target} cannot be covered from projected slack without exceeding the total of "
                                 f"{auth.total_core_h} core-h; the stage is unaffordable under the current authorization"}
    return {"feasible": True, "deficits": deficits, "transfers": transfers, "proposed_stages": new, "total_core_h": auth.total_core_h,
            "sum_after": round(sum(new.values()), 6), "future_upper90_core_h": future,
            "statement": "proposal only; it changes nothing until granted as an authorization amendment"}


def write_projection(loaded: LoadedBenchmark, name: str, payload: dict[str, Any]) -> Path:
    path = loaded.root / "execution" / "projections" / f"{name}.json"
    write_json(path, payload)
    return path


__all__ = ["measured_costs", "project", "redistribution_proposal", "stage_need", "write_projection"]
