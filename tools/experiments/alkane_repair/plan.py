"""Bounded execution manifest of the repair benchmark, costed from measured timings only.

Measured inputs (``plan.timing_sources``, existing artifacts roots of the alkane demonstration):

* DFT: CPU core-seconds of valid full labels (by chain length) and fragment labels, including failed
  attempts and retries (the label records' measured cost);
* training: seconds per (structure x epoch) from GMD_SGT ``training_log.csv`` files and their datasets;
* MD: GMD steps per second from completed segments (``status.json`` vs the segment's ``run.in``).

The manifest counts every calculation the frozen protocol implies, per stage, and compares the
projected DFT with the stage budgets in the YAML. Nothing is launched. Repair-run DFT is an upper
bound (every cycle uses its full label allowance) unless ``plan.expected_labels_per_run_fraction``
says otherwise; actual consumption is measured and reported separately per run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from tools.experiments.alkane_hotspots.manifest import utc_now

from .common import read_json, read_jsonl, write_json
from .config import LoadedBenchmark

LOGGER = logging.getLogger(__name__)


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    return {"n": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
            "max": float(array.max())}


def measured_timings(loaded: LoadedBenchmark) -> dict[str, Any]:
    full: list[float] = []
    fragment: list[float] = []
    attempts_full: list[int] = []
    per_structure_epoch: list[float] = []
    md_rates: list[float] = []
    sources = []
    n = loaded.config.chain_length
    for relative in loaded.config.plan.timing_sources:
        root = loaded.repo_path(relative)
        if not root.is_dir():
            sources.append({"root": relative, "status": "missing"})
            continue
        sources.append({"root": relative, "status": "read"})
        for record in read_jsonl(root / "labels.jsonl"):
            cost = (record.get("cost") or {}).get("cpu_core_s")
            if record.get("status") != "valid" or cost is None:
                continue
            if record.get("kind") == "full" and int(record.get("chain_length") or 0) == n:
                full.append(float(cost))
                attempts_full.append(int((record.get("cost") or {}).get("attempts") or 1))
            elif record.get("kind") == "fragment":
                fragment.append(float(cost))
        for log in root.glob("model_study*/models/*/train/member_0/training_log.csv"):
            manifest = log.parents[2] / "dataset" / "manifest.json"
            if not manifest.is_file():
                continue
            n_train = len(read_json(manifest).get("train", []))
            rows = [line.split(",") for line in log.read_text().splitlines()[1:] if line.strip()]
            times = [float(r[-1]) for r in rows if r and r[-1] not in ("", "nan")]
            if n_train and times:
                per_structure_epoch.append(float(np.median(times)) / n_train)
        for status in root.glob("**/seed_*/status.json"):
            run_in = status.parent / "run.in"
            data = read_json(status)
            if not run_in.is_file() or data.get("state") != "completed":
                continue
            seconds = status.stat().st_mtime - run_in.stat().st_mtime
            steps = int(data.get("step", 0)) - int(data.get("start_step", 0))
            if seconds > 0 and steps > 0:
                md_rates.append(steps / seconds)
    return {"sources": sources, "full_label_core_s": _stats(full), "full_label_attempts": _stats([float(a) for a in attempts_full]),
            "fragment_label_core_s": _stats(fragment), "train_s_per_structure_epoch": _stats(per_structure_epoch),
            "md_steps_per_s": _stats(md_rates)}


def build_plan(loaded: LoadedBenchmark) -> dict[str, Any]:
    from .repair import run_matrix

    config = loaded.config
    timings = measured_timings(loaded)
    missing = [k for k in ("full_label_core_s", "fragment_label_core_s", "train_s_per_structure_epoch", "md_steps_per_s") if not timings[k].get("n")]
    if missing:
        raise ValueError(f"no measured timings for {missing} in plan.timing_sources {config.plan.timing_sources}; the plan uses measurements only")
    full_cs, frag_cs = timings["full_label_core_s"]["mean"], timings["fragment_label_core_s"]["mean"]
    full_p90 = timings["full_label_core_s"]["p90"]
    epoch_s = timings["train_s_per_structure_epoch"]["median"]
    md_rate = timings["md_steps_per_s"]["median"]
    cores = loaded.alpha.config.cp2k.cores
    part = config.partitions
    families = len(config.challenges.enabled())
    training = loaded.model.config.training
    members = 1 + training.sgt.committee_size
    epochs = int({**training.train, **config.initial_model.train_overrides}.get("n_epochs", 1))
    ft_epochs = int({**training.train, **config.repair.finetune_train}.get("n_epochs", epochs))

    def dft(n_full: int, n_frag: int = 0) -> dict[str, Any]:
        core_s = n_full * full_cs + n_frag * frag_cs
        return {"n_full": n_full, "n_fragment": n_frag, "core_h": core_s / 3600.0, "core_h_if_full_at_p90": (n_full * full_p90 + n_frag * frag_cs) / 3600.0,
                "wall_h_serial": core_s / cores / 3600.0}

    stages: dict[str, Any] = {}
    for index, stage in enumerate(config.initial_model.stages):
        n_train = stage.id_train_bases * part.id_train.ordinary_per_base
        n_val = part.id_val.bases * part.id_val.ordinary_per_base
        previous = 0 if index == 0 else config.initial_model.stages[index - 1].id_train_bases * part.id_train.ordinary_per_base
        n_new = n_train - previous + (n_val if index == 0 else 0)
        md_steps = (part.calibration.bases * part.calibration.ordinary_per_base + n_val) * len(config.initial_model.acceptance.md.seeds) * config.initial_model.acceptance.md.steps
        stages[f"initial_model_q{index}"] = {
            "condition": "always" if index == 0 else f"only if qualification stage {index - 1} fails",
            "dft": dft(n_new), "train_wall_h": n_train * epochs * epoch_s * members / 3600.0, "md_wall_h": md_steps / md_rate / 3600.0}
    dev_challenges = part.development.bases * part.development.challenges_per_family * families
    stages["development"] = {"dft": dft(part.development.bases * part.development.ordinary_per_base + dev_challenges)}
    stages["fidelity_audit"] = {"dft": dft(0, dev_challenges * config.plan.fidelity_audit_fragments_per_challenge)}
    stages["repair_val"] = {"dft": dft(part.repair_val.bases * (part.repair_val.ordinary_per_base + part.repair_val.challenges_per_family * families))}
    stages["final_test"] = {"dft": dft(part.final_test.bases * (part.final_test.ordinary_per_base + part.final_test.challenges_per_family * families)),
                            "note": "labels are produced in the sealed root and read only by the sealed evaluation"}
    cal_steps = part.calibration.bases * part.calibration.ordinary_per_base * len(config.monitor_calibration.seeds) * config.monitor_calibration.steps
    stages["monitor_calibration"] = {"dft": dft(0), "md_wall_h": cal_steps / md_rate / 3600.0}

    n_cases = part.repair.bases * part.repair.challenges_per_family * families
    r = config.repair
    final_train = config.initial_model.stages[-1].id_train_bases * part.id_train.ordinary_per_base
    runs_by_arm: dict[str, Any] = {}
    total_runs_core_h = 0.0
    for arm in config.arms:
        if not arm.enabled or arm.kind == "frozen":
            continue
        n_runs = n_cases * len(r.seeds)
        labels = r.cycles * r.labels_per_cycle * config.plan.expected_labels_per_run_fraction if arm.kind == "closed_loop" else 0.0
        core_h = min(labels * frag_cs / 3600.0, r.budget_core_h_per_run) * n_runs
        train_h = n_runs * r.cycles * ft_epochs * (final_train + labels) * epoch_s * members / 3600.0
        md_h = n_runs * r.cycles * r.segment_steps / md_rate / 3600.0 if arm.kind == "closed_loop" else 0.0
        runs_by_arm[arm.name] = {"kind": arm.kind, "n_runs": n_runs, "labels_per_run_upper": labels, "dft_core_h": core_h,
                                 "dft_wall_h_serial": core_h * 3600.0 / cores / 3600.0, "train_wall_h": train_h, "md_wall_h": md_h}
        total_runs_core_h += core_h
    stages["repair_runs"] = {"cases": n_cases, "seeds": list(r.seeds), "by_arm": runs_by_arm, "dft": {"core_h": total_runs_core_h,
                             "wall_h_serial": total_runs_core_h / cores}, "per_run_cap_core_h": r.budget_core_h_per_run}
    n_models = 1 + sum(v["n_runs"] for v in runs_by_arm.values())
    starts_per_family = part.final_test.bases * part.final_test.challenges_per_family
    st = config.evaluation.stability
    stages["sealed_evaluation"] = {"dft": dft(0), "n_models_upper": n_models,
                                   "stability_md_wall_h_upper": n_models * starts_per_family * len(st.seeds) * st.steps / md_rate / 3600.0,
                                   "note": "models identical to the initial model (not promoted) share its stability runs"}

    budget_map = {"initial_model": [k for k in stages if k.startswith("initial_model")], "development": ["development"], "fidelity_audit": ["fidelity_audit"],
                  "repair_val": ["repair_val"], "final_test": ["final_test"], "repair_runs": ["repair_runs"]}
    budgets = {}
    for name, keys in budget_map.items():
        projected = sum(stages[k]["dft"]["core_h"] for k in keys)
        limit = getattr(config.budgets, name)
        budgets[name] = {"projected_core_h": projected, "budget_core_h": limit, "within_budget": projected <= limit + 1e-9}
    first_q = stages["initial_model_q0"]["dft"]["core_h"]
    all_q = budgets["initial_model"]["projected_core_h"]
    total_upper = sum(v["projected_core_h"] for v in budgets.values())
    total_if_first_qualifies = total_upper - (all_q - first_q)
    plan = {"created_at": utc_now(), "benchmark": config.name, "protocol_version": config.protocol_version, "measured_timings": timings,
            "cpu_cores_per_cp2k_job": cores, "stages": stages, "budgets": budgets,
            "totals": {"dft_core_h_upper": total_upper, "dft_core_h_if_first_qualification_stage_passes": total_if_first_qualifies,
                       "dft_wall_h_serial_upper": total_upper / cores, "budget_total_core_h": config.budgets.total(),
                       "n_planned_runs": len(run_matrix(loaded)) if (loaded.root / "parents.jsonl").is_file() else None},
            "stop_conditions": ["qualification stage fails and no further stage -> prerequisite unmet, stop",
                                "development failures absent for a family -> report; the family's repair claim is untestable",
                                "fidelity audit fails for a family -> no local labels for that family",
                                "a stage budget would be exceeded -> the job is not started",
                                "budgets.repair_runs reached -> no further repair run"],
            "statement": "costs from measured timings of earlier runs on this machine; repair-run DFT is an upper bound"}
    write_json(loaded.root / "plan" / "execution_plan.json", plan)
    return plan


def render_plan(plan: dict[str, Any]) -> str:
    lines = [f"# Execution plan: {plan['benchmark']} (protocol {plan['protocol_version']})", "",
             "Costed from measured timings only (DFT label records, GMD_SGT training logs, GMD segments). Nothing is launched by this plan.", "",
             "| Stage | Condition | Full labels | Fragments | DFT core-h | DFT wall h (serial) | Training h | MD h |", "|---|---|---|---|---|---|---|---|"]
    for name, stage in plan["stages"].items():
        if name == "repair_runs":
            continue
        d = stage["dft"]
        lines.append(f"| {name} | {stage.get('condition', '')} | {d.get('n_full', 0)} | {d.get('n_fragment', 0)} | {d['core_h']:.1f} | "
                     f"{d.get('wall_h_serial', 0.0):.1f} | {stage.get('train_wall_h', 0.0):.1f} | "
                     f"{stage.get('md_wall_h', stage.get('stability_md_wall_h_upper', 0.0)):.1f} |")
    lines += ["", "Repair runs (upper bounds):", "", "| Arm | Kind | Runs | Labels/run | DFT core-h | Training h | MD h |", "|---|---|---|---|---|---|---|"]
    for arm, item in plan["stages"]["repair_runs"]["by_arm"].items():
        lines.append(f"| {arm} | {item['kind']} | {item['n_runs']} | {item['labels_per_run_upper']:.0f} | {item['dft_core_h']:.1f} | "
                     f"{item['train_wall_h']:.1f} | {item['md_wall_h']:.1f} |")
    lines += ["", "| Budget | Projected core-h | Budget core-h | Within |", "|---|---|---|---|"]
    for name, item in plan["budgets"].items():
        lines.append(f"| {name} | {item['projected_core_h']:.1f} | {item['budget_core_h']:.1f} | {'yes' if item['within_budget'] else 'NO'} |")
    t = plan["totals"]
    lines += ["", f"Total DFT upper bound: {t['dft_core_h_upper']:.1f} core-h ({t['dft_wall_h_serial_upper']:.1f} h serial wall); "
              f"if the first qualification stage passes: {t['dft_core_h_if_first_qualification_stage_passes']:.1f} core-h.", ""]
    timing = plan["measured_timings"]
    lines += ["Measured inputs: full label "
              f"{timing['full_label_core_s']['mean']:.0f} core-s mean (n={timing['full_label_core_s']['n']}, p90 {timing['full_label_core_s']['p90']:.0f}), "
              f"fragment {timing['fragment_label_core_s']['mean']:.0f} core-s mean (n={timing['fragment_label_core_s']['n']}), "
              f"training {timing['train_s_per_structure_epoch']['median']:.3f} s per structure-epoch (n={timing['train_s_per_structure_epoch']['n']}), "
              f"GMD {timing['md_steps_per_s']['median']:.0f} steps/s (n={timing['md_steps_per_s']['n']}).", ""]
    return "\n".join(lines)


__all__ = ["build_plan", "measured_timings", "render_plan"]
