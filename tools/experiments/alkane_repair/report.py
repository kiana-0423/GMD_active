"""Benchmark status and report: what was executed, what was observed, and what remains untested.

Implementation completion and scientific outcome are reported separately. Each of the seven
assessment questions is answered only from recorded evidence; a question whose evidence does not
exist yet is answered "not yet tested" with the missing stage named.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from tools.experiments.alkane_hotspots.geometry import AlkaneTopology
from tools.experiments.alkane_hotspots.manifest import utc_now

from .common import challenged_region, read_json, write_json
from .config import LoadedBenchmark
from .generate import load_records

LOGGER = logging.getLogger(__name__)

RESULTS_DIR = "docs/demos/alkane_repair/results"


def _maybe(path: Path) -> Any:
    return read_json(path) if path.is_file() else None


def localisation(loaded: LoadedBenchmark, name: str, case: dict[str, Any]) -> dict[str, Any]:
    """Did the run's labels cover the challenged region of its MD start? (atom identities are preserved)."""

    from gmd_active_learning.contracts import ExtractedRegion, RegionRole

    from .repair import RUNS_SUBDIR

    report = _maybe(loaded.root / RUNS_SUBDIR / name / "closed_loop_report.json")
    if report is None:
        return {"status": "no_closed_loop_report"}
    region = set(challenged_region(case, AlkaneTopology(int(case["chain_length"])), loaded.config.failure.region_bond_hops))
    expected = loaded.model.config.training.force_weights.as_dict()
    labels = []
    for task in report.get("label_tasks", []):
        path = loaded.root / RUNS_SUBDIR / name / "run" / "cp2k_tasks" / task["task_id"] / "region.json"
        if not path.is_file():
            continue
        extracted = ExtractedRegion.from_dict(read_json(path))
        core = {int(p) for p, r in zip(extracted.parent_atom_ids, extracted.roles) if r is RegionRole.CORE and int(p) >= 0}
        weights: dict[str, list[float]] = {}
        for role, weight in zip(extracted.roles, extracted.force_weights):
            weights.setdefault(role.value, []).append(float(weight))
        by_role = {role: sorted(set(values)) for role, values in weights.items()}
        mismatched = sorted(role for role, values in by_role.items() if expected.get(role) is not None and values != [float(expected[role])])
        labels.append({"task_id": task["task_id"], "validation": task.get("validation"), "n_atoms": extracted.n_atoms, "n_core": len(core),
                       "covers_challenged_region": bool(core & region), "cycle": task.get("cycle"), "attempts": task.get("attempts"),
                       "force_weights_by_role": by_role, "weights_match_config": not mismatched, "mismatched_roles": mismatched})
    events = report.get("events", [])
    by_cycle: dict[int, dict[str, int]] = {}
    for event in events:
        by_cycle.setdefault(int(event["cycle"]), {"events": 0, "labels": 0})["events"] += 1
    for label in labels:
        by_cycle.setdefault(int(label["cycle"]), {"events": 0, "labels": 0})["labels"] += 1
    transitions = report.get("transitions", [])
    from datetime import datetime

    end_to_end = None
    if len(transitions) >= 2:
        end_to_end = (datetime.fromisoformat(transitions[-1]["at"]) - datetime.fromisoformat(transitions[0]["at"])).total_seconds()
    return {"status": "evaluated", "n_events": len(events), "first_event_step": min((e["trigger_step"] for e in events), default=None),
            "per_cycle": {str(k): v for k, v in sorted(by_cycle.items())},
            "train_wall_s": sum(float(t.get("wall_s") or 0.0) for t in report.get("trainings", [])), "end_to_end_wall_s": end_to_end,
            "all_label_weights_match_config": all(x["weights_match_config"] for x in labels),
            "n_labels": len(labels), "n_valid_labels": sum(1 for x in labels if x["validation"] == "valid"),
            "n_labels_covering_challenged_region": sum(1 for x in labels if x["covers_challenged_region"]), "labels": labels,
            "trigger_reasons": sorted({r for e in events for r in (e.get("trigger_reasons") or [])}),
            "events_rejected_before_labeling": len(report.get("events_rejected_before_labeling", [])),
            "promotions": [p for p in report["summary"].get("promotions", []) if p.get("previous_model_id")],
            "model_changes": [{k: j.get(k) for k in ("boundary_step", "jump_ev", "jump_ev_per_atom", "same_coordinates")} for j in report.get("model_changes", [])],
            "dft_core_h": report.get("measured_dft_cpu_core_hours")}


def collect(loaded: LoadedBenchmark) -> dict[str, Any]:
    from .qualify import FROZEN_FILE, STATUS_FILE
    from .references import FREEZE_FILE
    from .repair import RUNS_SUBDIR, run_matrix

    root = loaded.root
    out: dict[str, Any] = {"benchmark": loaded.config.name, "protocol_version": loaded.config.protocol_version, "smoke_only": loaded.config.smoke_only,
                           "generated_at": utc_now()}
    out["protocol_freeze"] = _maybe(root / FREEZE_FILE)
    out["structures"] = _maybe(root / "structures_summary.json")
    out["plan"] = _maybe(root / "plan" / "execution_plan.json")
    preflight = _maybe(root / "preflight.json")
    out["preflight"] = None if preflight is None else {"passed": preflight["passed"], "coverage": preflight["coverage"],
                                                       "boxes": {k: v for k, v in preflight["boxes"].items() if k != "rows"},
                                                       "fragments": {k: v for k, v in preflight["fragments"].items() if k != "rows"}}
    out["reference_labels"] = {p.stem: read_json(p) for p in sorted((root / "reference_labels").glob("*.json"))}
    out["qualification"] = {"status": _maybe(root / STATUS_FILE), "frozen": _maybe(root / FROZEN_FILE),
                            "stages": [read_json(p) for p in sorted((root / "initial_model").glob("stage*/qualification.json"))]}
    out["fidelity_audit"] = _maybe(root / "fidelity" / "audit.json")
    out["development_failures"] = _maybe(root / "development" / "failures.json")
    out["monitor_calibration"] = {p.stem: {k: v for k, v in read_json(p).items() if k != "state"} for p in sorted((root / "monitor_calibration").glob("*.json"))}
    runs = []
    if (root / "parents.jsonl").is_file():
        cases = {r["parent_id"]: r for r in load_records(root)}
        for spec in run_matrix(loaded):
            record = _maybe(root / RUNS_SUBDIR / spec.name / "repair_run.json")
            row = {"run": spec.name, **spec.__dict__, "status": "planned" if record is None else record.get("final_stage")}
            if record is not None:
                row["dft_core_h"] = record.get("measured_dft_cpu_core_hours")
                row["final_model_promoted"] = not (record.get("final_model") or {}).get("initial", True)
                if spec.kind == "closed_loop":
                    row["localisation"] = localisation(loaded, spec.name, cases[spec.case])
                else:
                    row["rounds"] = [{k: r[k] for k in ("round", "promoted", "train_wall_s")} | {"metrics": r["metrics"]} for r in record.get("rounds", [])]
            runs.append(row)
    out["runs"] = runs
    out["sealed_evaluation"] = _maybe(root / "evaluation" / "sealed_evaluation.json")
    out["sealed_access_log"] = [json.loads(line) for line in (root / "sealed_access.jsonl").read_text().splitlines() if line.strip()] \
        if (root / "sealed_access.jsonl").is_file() else []
    out["compute"] = compute_accounting(out)
    out["assessment"] = assess(loaded, out)
    return out


def compute_accounting(data: dict[str, Any]) -> dict[str, Any]:
    stages = {name: item.get("measured_stage_core_h") for name, item in data["reference_labels"].items()}
    audit = data.get("fidelity_audit") or {}
    audit_core_h = sum((row.get("cpu_core_s") or 0.0) for row in audit.get("fragments", []) if row.get("status") == "evaluated") / 3600.0
    runs = sum(r.get("dft_core_h") or 0.0 for r in data["runs"])
    return {"reference_label_stages_core_h": stages, "fidelity_audit_core_h": audit_core_h, "repair_runs_core_h": runs,
            "note": "measured CP2K CPU core-hours (wall x cores), failed attempts and retries included; repair-run labels are always newly computed "
                    "(MD frames), so attributed and newly consumed compute coincide for them"}


def assess(loaded: LoadedBenchmark, data: dict[str, Any]) -> dict[str, Any]:
    """The seven assessment questions, answered only from recorded evidence."""

    q = data["qualification"]
    ev = data.get("sealed_evaluation")
    status = (q.get("status") or {}).get("prerequisite", "pending")
    answers: dict[str, Any] = {}
    if q.get("frozen"):
        f = q["frozen"]
        answers["1_initial_model_reliable_in_domain"] = (f"qualified at stage {f['qualification_stage']}: id_val force RMSE {f['id_val_force_rmse']:.3f} eV/A "
                                                         f"(validation set; sealed in-domain test: {'see evaluation' if ev else 'not yet evaluated'})")
    else:
        answers["1_initial_model_reliable_in_domain"] = f"not established ({status})"
    if ev:
        fails = [r for r in ev["initial_model"]["failures"] if "region" in r]
        answers["2_reference_confirmed_failure"] = {fam: f"{sum(1 for r in fails if r['family'] == fam and r['failure'])}/{sum(1 for r in fails if r['family'] == fam)} "
                                                         "sealed challenge structures" for fam in loaded.config.challenges.enabled()}
    elif data.get("development_failures"):
        answers["2_reference_confirmed_failure"] = {"development_only (not confirmatory)": {f: f"{v['n_failures']}/{v['n_structures']}"
                                                                                           for f, v in data["development_failures"]["families"].items()}}
    else:
        answers["2_reference_confirmed_failure"] = "not yet tested (no reference labels of challenge structures)"
    closed = [r for r in data["runs"] if r.get("localisation", {}).get("status") == "evaluated"]
    if closed:
        by_arm: dict[str, Any] = {}
        for r in closed:
            item = by_arm.setdefault(r["arm"], {"runs": 0, "runs_with_events": 0, "runs_with_label_in_region": 0, "labels": 0, "labels_in_region": 0})
            loc = r["localisation"]
            item["runs"] += 1
            item["runs_with_events"] += int(loc["n_events"] > 0)
            item["runs_with_label_in_region"] += int(loc["n_labels_covering_challenged_region"] > 0)
            item["labels"] += loc["n_labels"]
            item["labels_in_region"] += loc["n_labels_covering_challenged_region"]
        answers["3_detected_and_labeled_relevant_region"] = by_arm
    else:
        answers["3_detected_and_labeled_relevant_region"] = "not yet tested (no completed closed-loop repair run)"
    if ev and not ev.get("partial"):
        answers["4_local_labels_improved_independent_predictions"] = {arm: {fam: {"success": f"{v['n_success']}/{v['n_runs']}",
                                                                                 "mean_relative_improvement_over_bases": v["relative_improvement_by_base"]["mean"],
                                                                                 "n_bases": v["n_bases"]} for fam, v in fams.items()}
                                                                       for arm, fams in ev["summary"].items()}
        answers["5_retention"] = {r["run"]: r["outcome"]["relative_retention_change"] for r in ev["runs"]}
        answers["6_physical_signals"] = [{k: d[k] for k in ("a", "b", "n_pairs")} for d in ev["paired_differences"]]
    else:
        for key in ("4_local_labels_improved_independent_predictions", "5_retention", "6_physical_signals"):
            answers[key] = "not yet tested (sealed evaluation not executed" + (", partial evaluation only)" if ev else ")")
    answers["7_unsupported"] = [
        "equilibrium conformational statistics (no reference ensemble)",
        "chain lengths other than the benchmark's (C64/C128 not attempted)",
        "event-buffering benefit, unless the enabled no-buffering ablation was executed and evaluated",
        "efficiency (labels or core-hours to repair) beyond the measured, small-sample values",
        "any equivalence or 'no difference' between arms from this sample size",
    ]
    implementation = {"structures": data["structures"] is not None, "protocol_frozen": data["protocol_freeze"] is not None,
                      "plan": data["plan"] is not None, "qualification_run": bool(q["stages"]), "repair_runs": sum(1 for r in data["runs"] if r["status"] == "done"),
                      "planned_runs": len(data["runs"]), "sealed_evaluation": ev is not None}
    if loaded.config.smoke_only:
        scientific = "smoke profile: mechanics only, no scientific result"
    else:
        scientific = "no confirmatory result yet" if not ev or ev.get("partial") else "sealed evaluation complete"
    return {"implementation": implementation, "answers": answers, "scientific_status": scientific}


def render_markdown(data: dict[str, Any]) -> str:
    lines = [f"# Alkane repair benchmark: {data['benchmark']} (protocol {data['protocol_version']})", ""]
    if data["smoke_only"]:
        lines += ["**Smoke profile: smoke-only CP2K settings and toy budgets. Mechanics only; no scientific claim.**", ""]
    a = data["assessment"]
    lines += ["## Implementation status", ""] + [f"- {k}: {v}" for k, v in a["implementation"].items()] + ["", f"Scientific status: **{a['scientific_status']}**", ""]
    freeze = data.get("protocol_freeze")
    lines += ["## Protocol", "", f"- frozen: {freeze['frozen_at'] if freeze else 'NO'}; protocol sha256 `{freeze['protocol']['sha256'][:16] if freeze else '-'}`", ""]
    s = data.get("structures")
    if s:
        lines += ["## Structures", "", "| Partition | Counts |", "|---|---|"] + [f"| {p} | {json.dumps(c)} |" for p, c in sorted(s["counts"].items())] + [""]
    pf = data.get("preflight")
    if pf:
        lines += ["## Geometry preflight (no DFT)", "", f"- passed: {pf['passed']}; MD boxes: {pf['boxes']['n']} (max {pf['boxes']['max_length_A']:.1f} A); "
                  f"fragments: {pf['fragments']['n']} ({pf['fragments']['n_failed']} failed, {pf['fragments']['n_over_atom_limit']} over the atom limit, "
                  f"atoms {pf['fragments']['atoms_range']}, caps {pf['fragments']['caps_range']})", ""]
    q = data["qualification"]
    lines += ["## Initial model", ""]
    for stage in q["stages"]:
        acc = stage["id_val_accuracy"]
        lines.append(f"- stage {stage['qualification_stage']}: {stage['n_train']} train / {stage['n_val']} val structures; id_val force RMSE "
                     f"{acc['force_rmse']:.4f} eV/A, atom-error p95 {acc['atom_error_p95']:.4f}; MD trajectories intact "
                     f"{sum(1 for t in stage['md_trajectories'] if t['intact_fraction'] >= 1.0)}/{len(stage['md_trajectories'])}; "
                     f"criteria {stage['criteria']}; **{'accepted' if stage['accepted'] else 'not accepted'}**")
    lines += [f"- prerequisite: {(q.get('status') or {}).get('prerequisite', 'pending')}", ""]
    if data.get("fidelity_audit"):
        lines += ["## Fragment-fidelity audit (development challenges, DFT)", "", "| Family | Fragments | RMSE | Max | Passes |", "|---|---|---|---|---|"]
        for fam, v in data["fidelity_audit"]["families"].items():
            rmse = "-" if v["pooled_core_force_rmse"] is None else f"{v['pooled_core_force_rmse']:.4f}"
            worst = "-" if v["max_core_force_error"] is None else f"{v['max_core_force_error']:.4f}"
            lines.append(f"| {fam} | {v['n_evaluated']}/{v['n_fragments']} | {rmse} | {worst} | {v['passes']} |")
        lines.append("")
    if data["monitor_calibration"]:
        lines += ["## Monitor calibration (in-domain trajectories of the frozen model)", "", "| Arm | Frames | Alarm fraction | Cold required metrics | OK |", "|---|---|---|---|---|"]
        for arm, v in data["monitor_calibration"].items():
            lines.append(f"| {arm} | {v['n_frames']} | {v['alarm_fraction']:.4f} | {v['cold_required_metrics']} | {v['ok']} |")
        lines.append("")
    if data["runs"]:
        lines += ["## Repair runs", "", "| Run | Status | Events | Labels (in challenged region) | Promotions | DFT core-h |", "|---|---|---|---|---|---|"]
        for r in data["runs"]:
            loc = r.get("localisation") or {}
            events = loc.get("n_events", "-")
            labels = f"{loc.get('n_labels', '-')} ({loc.get('n_labels_covering_challenged_region', '-')})" if loc else "-"
            promotions = len(loc.get("promotions", [])) if loc else sum(1 for x in r.get("rounds", []) if x["promoted"]) if r.get("rounds") else "-"
            dft = r.get("dft_core_h")
            lines.append(f"| {r['run']} | {r['status']} | {events} | {labels} | {promotions} | {'-' if dft is None else f'{dft:.3f}'} |")
        lines.append("")
    ev = data.get("sealed_evaluation")
    lines += ["## Sealed evaluation", ""]
    if ev is None:
        lines += ["Not executed.", ""]
    else:
        lines += [f"- partial: {ev['partial']} (missing runs: {len(ev['missing_runs'])})", "", "| Arm | Family | Runs | Bases | Success | Promoted | Mean rel. improvement (bases) |",
                  "|---|---|---|---|---|---|---|"]
        for arm, fams in ev["summary"].items():
            for fam, v in fams.items():
                mean = v["relative_improvement_by_base"]["mean"]
                lines.append(f"| {arm} | {fam} | {v['n_runs']} | {v['n_bases']} | {v['n_success']} | {v['n_promoted']} | {'-' if mean is None else f'{mean:.3f}'} |")
        lines.append("")
    lines += ["## Assessment (from recorded evidence only)", ""] + [f"- **{k}**: {json.dumps(v, default=str)}" for k, v in a["answers"].items()] + [""]
    c = data["compute"]
    lines += ["## Measured compute", "", f"- reference labels by stage (core-h): {json.dumps(c['reference_label_stages_core_h'])}",
              f"- fidelity audit: {c['fidelity_audit_core_h']:.3f} core-h; repair runs: {c['repair_runs_core_h']:.3f} core-h", f"- {c['note']}", ""]
    return "\n".join(lines) + "\n"


def write_figures(loaded: LoadedBenchmark, data: dict[str, Any]) -> dict[str, list[str]]:
    from . import figures as fg

    out = loaded.root / "report" / "figures"
    arms = [a.name for a in loaded.config.arms if a.enabled and a.kind != "frozen"]
    families = loaded.config.challenges.enabled()
    ev = data.get("sealed_evaluation")
    made = {}
    made["qualification"] = fg.save(fg.fig_qualification(data["qualification"]["stages"], target=loaded.config.initial_model.acceptance.id_val_force_rmse_max),
                                    out, "qualification")
    failure_rows = (ev or {}).get("initial_model", {}).get("failures") or (data.get("development_failures") or {}).get("rows")
    made["failures"] = fg.save(fg.fig_failures(failure_rows, families=families), out, "failures")
    made["region_errors"] = fg.save(fg.fig_region_errors(ev, arms=arms, families=families, threshold=loaded.config.evaluation.success.challenge_region_rmse_max),
                                    out, "region_errors")
    made["retention"] = fg.save(fg.fig_retention(ev, arms=arms, tolerance=loaded.config.evaluation.success.id_retention_max_relative_regression), out, "retention")
    made["stability"] = fg.save(fg.fig_stability(ev, arms=arms, families=families), out, "stability")
    return made


def _portable(text: str, loaded: LoadedBenchmark) -> str:
    return text.replace(str(loaded.root), "<artifacts_root>").replace(str(loaded.repo_path(".")), "<GMD_active>")


def write_report(loaded: LoadedBenchmark, *, publish: bool = False) -> dict[str, Any]:
    data = collect(loaded)
    figures = write_figures(loaded, data)
    data["figures"] = {k: [str(Path(p).name) for p in v] for k, v in figures.items()}
    write_json(loaded.root / "report" / "report.json", data)
    markdown = render_markdown(data)
    (loaded.root / "report" / "report.md").write_text(markdown, encoding="utf-8")
    published = None
    if publish:
        target = loaded.repo_path(RESULTS_DIR) / loaded.config.name
        (target / "figures").mkdir(parents=True, exist_ok=True)
        (target / "report.md").write_text(_portable(markdown, loaded), encoding="utf-8")
        (target / "report.json").write_text(_portable(json.dumps(data, indent=1, sort_keys=True, default=str), loaded) + "\n", encoding="utf-8")
        for paths in figures.values():
            for path in paths:
                shutil.copyfile(path, target / "figures" / Path(path).name)
        if data.get("plan"):
            from .plan import render_plan

            (target / "execution_plan.md").write_text(_portable(render_plan(data["plan"]), loaded), encoding="utf-8")
        published = str(target)
    return {"report": str(loaded.root / "report" / "report.md"), "published": published, "assessment": data["assessment"]}


def status(loaded: LoadedBenchmark) -> dict[str, Any]:
    data = collect(loaded)
    return {"implementation": data["assessment"]["implementation"], "qualification": (data["qualification"].get("status") or {}),
            "compute": data["compute"], "scientific_status": data["assessment"]["scientific_status"]}


__all__ = ["assess", "collect", "localisation", "render_markdown", "status", "write_report"]
