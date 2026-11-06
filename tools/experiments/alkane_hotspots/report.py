"""Machine-readable report (report.json) and markdown summary, plus the CLI of the model study.

The report separates what was **executed** (artifact path, sha256, config hash, label counts, measured
costs, the numbers read from that artifact) from what is **planned / not executed** (configured but no
artifact). Every number in the summary is copied from a named artifact; nothing is computed from
assumptions. Negative and null results are reported as they are (e.g. a bootstrap interval that
contains 0 is stated as "no detectable difference").

CLI subcommands, registered on alpha's CLI (``python -m tools.experiments.alkane_hotspots <cmd> --config
configs/demos/alkane_hotspots/<profile>.yaml --study configs/demos/alkane_hotspots/arms_<profile>.yaml``):
``receptive-field``, ``ground-truth``, ``arms-run`` (resumable), ``evaluate``, ``figures``, ``report``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .arms import (
    REPO_ROOT,
    ArmsRunner,
    LoadedConfig,
    collect_progress,
    default_context,
    learning_curves,
    load_beta_config,
    write_json_atomic,
)
from .evaluation import curve_analysis
from .ground_truth import read_jsonl

LOGGER = logging.getLogger(__name__)

FILES = {
    "receptive_field": "receptive_field.json",
    "ground_truth": "ground_truth.json",
    "regions": "ground_truth_regions.json",
    "per_atom": "ground_truth_per_atom.json",
    "evaluation": "evaluation.json",
    "figures": "figures/index.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)} if path.is_file() else {"path": str(path), "exists": False}


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


# ---------------------------------------------------------------------------------------------- label accounting


def label_accounting(labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by = Counter((str(r.get("kind")), str(r.get("status"))) for r in labels)
    cost = {}
    for kind in ("full", "fragment"):
        values = [float((r.get("cost") or {}).get("cpu_core_s")) for r in labels
                  if r.get("kind") == kind and (r.get("cost") or {}).get("cpu_core_s") is not None]
        cost[kind] = {"n_with_cost": len(values), "total_cpu_core_s": float(sum(values)),
                      "median_cpu_core_s": float(np.median(values)) if values else None}
    smoke = sum(1 for r in labels if (r.get("cp2k") or {}).get("smoke_only"))
    return {"n_records": len(labels), "by_kind_status": {f"{k}/{s}": n for (k, s), n in sorted(by.items())}, "measured_cost": cost,
            "n_smoke_only": smoke,
            "smoke_only_note": "smoke-only labels (reduced CP2K settings) are pipeline checks; their numbers are not results" if smoke else None}


# ---------------------------------------------------------------------------------------------- report


def _gt_findings(gt: Mapping[str, Any]) -> list[str]:
    findings = []
    for tau, block in (gt.get("per_threshold") or {}).items():
        for comp in block.get("comparisons", []):
            if comp.get("delta") is None:
                findings.append(f"tau={tau}: AP({comp['a']}) - AP({comp['b']}) undefined ({comp.get('reason', 'no positives')})")
                continue
            ci = comp.get("delta_ci") or {}
            low, high = ci.get("low"), ci.get("high")
            verdict = ("no detectable difference (interval contains 0)" if low is not None and high is not None and low <= 0 <= high
                       else "interval excludes 0" if low is not None else "no bootstrap interval")
            findings.append(f"tau={tau}: AP({comp['a']}) - AP({comp['b']}) = {comp['delta']:.3f} "
                            f"[{'n/a' if low is None else f'{low:.3f}'}, {'n/a' if high is None else f'{high:.3f}'}]: {verdict}")
        lj = (block.get("signals") or {}).get("lj_residual") or {}
        normal = lj.get("normal") or {}
        if normal.get("fpr_normal") is not None:
            findings.append(f"tau={tau}: LJ residual on normal configurations: FPR {normal['fpr_normal']:.3f}, "
                            f"alarm rate {normal['alarm_rate_normal']:.3f}")
    return findings


def build_report(loaded: LoadedConfig) -> dict[str, Any]:
    out = loaded.output_dir
    beta = loaded.config
    executed: dict[str, Any] = {}
    planned: dict[str, Any] = {}

    rf = _load(out / FILES["receptive_field"])
    if rf is not None:
        executed["receptive_field"] = {"artifact": _artifact(out / FILES["receptive_field"]), "nominal": rf.get("nominal"),
                                       "measured": rf.get("measured"), "model_source": (rf.get("model") or {}).get("source"),
                                       "interpretation": rf.get("interpretation")}
    elif beta.receptive_field.enabled:
        planned["receptive_field"] = {"reason": "no receptive_field.json"}

    gt = _load(out / FILES["ground_truth"])
    if gt is not None and gt.get("status") == "executed":
        summary = {}
        for tau, block in gt["per_threshold"].items():
            summary[tau] = {"n_positive": block["n_positive"], "average_precision": {
                name: s.get("average_precision") for name, s in block["signals"].items()}}
        executed["ground_truth"] = {"artifact": _artifact(out / FILES["ground_truth"]), "n_configurations": gt["n_configurations"],
                                    "n_regions": gt["n_regions"], "model": gt.get("model"), "per_threshold": summary,
                                    "findings": _gt_findings(gt), "localisation": gt.get("localisation")}
    elif beta.ground_truth.enabled:
        planned["ground_truth"] = {"reason": "no ground_truth.json" if gt is None else gt.get("reason", gt.get("status"))}

    progress = collect_progress(out)
    if beta.arms is not None:
        variants = {}
        for variant in beta.arms.variants():
            reps = progress.get(variant.name, [])
            if not reps:
                planned.setdefault("arms", {})[variant.name] = {"reason": "not run", "repetitions_planned": beta.arms.repetitions,
                                                                 "rounds_planned": beta.arms.n_rounds}
                continue
            labels_n = [rep["rounds"][-1]["n_labels"] for rep in reps if rep["rounds"]]
            cost = [rep["rounds"][-1]["cumulative_cpu_core_s"] for rep in reps if rep["rounds"]]
            variants[variant.name] = {
                "kind": variant.kind,
                "repetitions_complete": sum(1 for rep in reps if rep.get("status") == "complete"),
                "repetitions_found": len(reps), "repetitions_planned": beta.arms.repetitions,
                "rounds_completed": [len(rep["rounds"]) - 1 for rep in reps],
                "labels_acquired": labels_n, "cumulative_cpu_core_s": cost,
                "final_force_rmse": [rep["rounds"][-1].get("force_rmse") for rep in reps if rep["rounds"]],
                "progress_files": [_artifact(Path(rep["path"])) for rep in reps],
            }
        if variants:
            executed["arms"] = variants
    evaluation = _load(out / FILES["evaluation"])
    if evaluation is not None:
        executed["evaluation"] = {"artifact": _artifact(out / FILES["evaluation"]), "analysis": evaluation.get("analysis")}
    elif beta.arms is not None:
        planned["evaluation"] = {"reason": "no evaluation.json (run `evaluate` after `arms-run`)"}
    figures = _load(out / FILES["figures"])
    if figures is not None:
        with_data = {name: item for name, item in figures.items() if item.get("has_data")}
        without = {name: item.get("missing") for name, item in figures.items() if not item.get("has_data")}
        if with_data:
            executed["figures"] = {"index": _artifact(out / FILES["figures"]), "figures": {n: i["paths"] for n, i in with_data.items()},
                                   "partial": {n: i["missing"] for n, i in with_data.items() if i.get("missing")}}
        if without:
            planned["figures"] = {"rendered_without_data (panels say NOT EXECUTED)": without}
    else:
        planned["figures"] = {"reason": "figures not rendered"}

    labels_path = loaded.artifacts_root / beta.inputs.labels
    labels = read_jsonl(labels_path) if labels_path.is_file() else []
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"name": beta.name, "path": str(loaded.path), "config_hash": loaded.config_hash, "artifacts_root": str(loaded.artifacts_root)},
        "labels": {"artifact": _artifact(labels_path), **label_accounting(labels)},
        "executed": executed,
        "planned_not_executed": planned,
        "rules": ["costs are measured wall time x allocated cores from label records",
                  "final_eval is used for evaluation only", "fragment energies never supervise parents (energy_weight 0 for capped fragments)"],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [f"# Alkane hotspot model study: {report['config']['name']}", "",
             f"Config `{report['config']['path']}` (hash `{report['config']['config_hash'][:16]}`), generated {report['generated_at']}.", ""]
    labels = report["labels"]
    lines += ["## Label records", "", f"- file: `{labels['artifact']['path']}` ({labels['n_records']} records)"]
    for key, n in labels["by_kind_status"].items():
        lines.append(f"  - {key}: {n}")
    for kind, cost in labels["measured_cost"].items():
        lines.append(f"  - measured {kind} cost: {cost['total_cpu_core_s']:.0f} CPU core-s over {cost['n_with_cost']} calculations")
    if labels.get("n_smoke_only"):
        lines.append(f"  - **{labels['n_smoke_only']} smoke-only records**: {labels['smoke_only_note']}")
    lines += ["", "## Executed", ""]
    executed = report["executed"]
    if not executed:
        lines.append("Nothing executed yet.")
    if "receptive_field" in executed:
        rf = executed["receptive_field"]
        lines += ["### Receptive field", "", f"- artifact: `{rf['artifact']['path']}`", f"- model source: {rf['model_source']}",
                  f"- nominal: {json.dumps(rf['nominal'])}", f"- measured: {json.dumps(rf['measured'])}", f"- {rf['interpretation']}", ""]
    if "ground_truth" in executed:
        gt = executed["ground_truth"]
        lines += ["### Hotspot ground truth", "", f"- artifact: `{gt['artifact']['path']}`",
                  f"- {gt['n_configurations']} configurations, {gt['n_regions']} regions", ""]
        for tau, block in gt["per_threshold"].items():
            lines.append(f"Reference threshold {tau} eV/A ({block['n_positive']} positive regions), average precision:")
            lines.append("")
            lines.append("| signal | AP |")
            lines.append("|---|---|")
            for name, ap in block["average_precision"].items():
                lines.append(f"| {name} | {'undefined' if ap is None else f'{ap:.3f}'} |")
            lines.append("")
        lines += [f"- {item}" for item in gt["findings"]] + [""]
    if "arms" in executed:
        lines += ["### Acquisition arms", "", "| variant | reps complete/planned | labels | cumulative CPU core-s | final force RMSE (eV/A) |",
                  "|---|---|---|---|---|"]
        for name, item in executed["arms"].items():
            rmse = ", ".join("n/a" if v is None else f"{v:.4f}" for v in item["final_force_rmse"])
            cost = ", ".join(f"{v:.0f}" for v in item["cumulative_cpu_core_s"])
            lines.append(f"| {name} | {item['repetitions_complete']}/{item['repetitions_planned']} | {item['labels_acquired']} | {cost} | {rmse} |")
        lines.append("")
    if "evaluation" in executed:
        lines += ["### Budget analysis", "", f"- artifact: `{executed['evaluation']['artifact']['path']}`", ""]
    if "figures" in executed:
        lines += ["### Figures", ""] + [f"- {name}: {', '.join(Path(p).name for p in paths)}" for name, paths in
                                        (executed["figures"]["figures"] or {}).items()] + [""]
    lines += ["## Planned / not executed", ""]
    planned = report["planned_not_executed"]
    if not planned:
        lines.append("Nothing pending.")
    for name, item in planned.items():
        lines.append(f"- {name}: {json.dumps(item)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------- commands


def _parents_and_labels(loaded: LoadedConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    beta = loaded.config
    parents = read_jsonl(loaded.input_path(beta.inputs.parents))
    labels_path = loaded.artifacts_root / beta.inputs.labels
    labels = read_jsonl(labels_path) if labels_path.is_file() else []
    return parents, labels


def cmd_receptive_field(loaded: LoadedConfig, args: argparse.Namespace) -> Path:
    from .ground_truth import parent_frame
    from .receptive_field import build_model, run_receptive_field
    from .training import make_backend

    beta = loaded.config
    cfg = beta.receptive_field
    make_backend(beta.training, root=REPO_ROOT)  # imports GMD_SGT from the configured checkout
    parents, _ = _parents_and_labels(loaded)
    selector = cfg.structure
    matches = [p for p in parents if (selector.parent_id is None and p.get("split") == selector.split and p.get("family") == selector.family)
               or p.get("parent_id") == selector.parent_id]
    if len(matches) <= selector.index:
        raise SystemExit(f"receptive-field: no parent matches {selector.model_dump()}")
    record = matches[selector.index]
    frame = parent_frame(record, *loaded.bases())
    checkpoint = None if cfg.checkpoint is None else loaded.artifacts_root / cfg.checkpoint
    model, model_config, provenance = build_model(beta.training.model, cfg, checkpoint=checkpoint)
    result = run_receptive_field(model, model_config, frame.numbers, frame.positions, cfg, provenance=provenance,
                                 structure={"parent_id": record["parent_id"], "family": record.get("family"), "chain_length": record.get("chain_length")})
    result["config_hash"] = loaded.config_hash
    return write_json_atomic(loaded.output_dir / FILES["receptive_field"], result)


def initial_model(loaded: LoadedConfig, context: Any) -> Any:
    """The round-0 model of repetition 0 (initial data only) -- identical to the arms' round-0 model."""

    runner = ArmsRunner(loaded.config, context)
    training = loaded.config.training
    seed = loaded.config.arms.seed_for(0) if loaded.config.arms is not None else 0
    model, key = runner._train(runner._split_samples(training.initial_splits), runner._split_samples(training.validation_splits), seed)
    return model, key


def cmd_ground_truth(loaded: LoadedConfig, args: argparse.Namespace) -> Path:
    from .arms import _predict_frames
    from .ground_truth import full_labels_by_parent, per_atom_dump, region_table, run_ground_truth

    beta = loaded.config
    cfg = beta.ground_truth
    context = default_context(loaded)
    if args.checkpoints:
        from .arms import _ModelRef

        model, key = _ModelRef(list(args.checkpoints)), "cli"
    else:
        model, key = initial_model(loaded, context)
    predictor = context.predictor_factory(model)
    full = full_labels_by_parent(context.labels)
    frames = _predict_frames(context.parents, cfg.splits, predictor, context.bases, full)
    calibration = _predict_frames(context.parents, cfg.calibration.splits, predictor, context.bases, None, families=cfg.calibration.families)
    references = None
    if cfg.temporal_reference == "group_normal":
        normals = {f.split_group: f for f in _predict_frames(context.parents, cfg.splits, predictor, context.bases, None, families=["normal"])}
        references = {f.parent_id: normals.get(f.split_group) for f in frames if f.family != "normal"}
    report, tables = run_ground_truth(frames, calibration, beta.monitor, cfg, references=references)
    report.update(config_hash=loaded.config_hash, model={"checkpoints": list(model.checkpoints), "key": key})
    table = region_table(frames, tables, cfg, cfg.signal_names()) if frames else None
    if table is not None:
        write_json_atomic(loaded.output_dir / FILES["regions"], {"error": table.error, "family": table.family, "parent_id": table.parent_id,
                                                                  "scores": dict(table.scores), "contains_site": table.contains_site})
        write_json_atomic(loaded.output_dir / FILES["per_atom"], per_atom_dump(frames, tables))
    return write_json_atomic(loaded.output_dir / FILES["ground_truth"], report)


def cmd_arms_run(loaded: LoadedConfig, args: argparse.Namespace) -> Path:
    runner = ArmsRunner(loaded.config, default_context(loaded))
    results = runner.run(only=args.only or None, repetitions=args.repetitions or None)
    return write_json_atomic(loaded.output_dir / "arms_status.json", {name: [{"status": r["status"], "rounds": len(r["rounds"]) - 1}
                                                                             for r in reps] for name, reps in results.items()})


def cmd_evaluate(loaded: LoadedConfig, args: argparse.Namespace) -> Path:
    progress = collect_progress(loaded.output_dir)
    curves = learning_curves(progress)
    analysis = curve_analysis(curves, loaded.config.evaluation) if curves else {}
    return write_json_atomic(loaded.output_dir / FILES["evaluation"], {"config_hash": loaded.config_hash, "curves": curves,
                                                                       "analysis": analysis, "status": "executed" if curves else "no_data"})


def render_figures(loaded: LoadedConfig) -> dict[str, dict[str, Any]]:
    """Render every figure; the index records, per figure, whether any panel lacks executed data."""

    from . import figures as fg

    beta = loaded.config
    out = loaded.output_dir
    fig_dir = out / "figures"
    fmt, dpi = beta.figures.formats, beta.figures.dpi
    index: dict[str, dict[str, Any]] = {}

    labels_path = loaded.artifacts_root / beta.inputs.labels
    smoke = labels_path.is_file() and any((r.get("cp2k") or {}).get("smoke_only") for r in read_jsonl(labels_path))

    def emit(name: str, fig: Any, sources: list[Path]) -> None:
        missing = fg.missing_reasons(fig)
        if smoke and name not in ("receptive_field",):
            fg.stamp(fig, "SMOKE-ONLY CP2K settings present: pipeline check, not a result")
        index[name] = {"paths": fg.save(fig, fig_dir, name, fmt, dpi), "missing": missing, "has_data": not missing,
                       "sources": [_artifact(p) for p in sources if p.is_file()]}

    emit("receptive_field", fg.fig_receptive_field(_load(out / FILES["receptive_field"])), [out / FILES["receptive_field"]])
    per_atom = _load(out / FILES["per_atom"])
    signal = next(iter(beta.ground_truth.monitor_variants), "committee")
    emit("hotspots", fg.fig_hotspots(per_atom, signal=signal), [out / FILES["per_atom"]])
    regions = _load(out / FILES["regions"])
    signals = [s for s in ("committee", "lj_residual", signal) if s in beta.ground_truth.signal_names()]
    emit("risk_vs_error", fg.fig_risk_vs_error(regions, signals=signals, thresholds=beta.ground_truth.reference_thresholds), [out / FILES["regions"]])
    fidelity_path = loaded.artifacts_root / beta.figures.fidelity_results
    fidelity = _load(fidelity_path)
    rows = [dict(r, policy=str(r.get("setting_id", "")).split("-")[0]) for r in (fidelity or {}).get("settings", [])]
    targets = (fidelity or {}).get("targets") or {}
    emit("fidelity", fg.fig_fidelity(rows, x_key="mean_n_atoms", y_key="pooled_core_force_rmse", target=targets.get("core_force_rmse"),
                                     x_label="fragment size (atoms, mean over specs)"), [fidelity_path])
    emit("fidelity_max", fg.fig_fidelity(rows, x_key="mean_n_atoms", y_key="max_core_force_error", target=targets.get("max_core_force_error"),
                                         x_label="fragment size (atoms, mean over specs)"), [fidelity_path])
    evaluation = _load(out / FILES["evaluation"]) or {}
    curves = evaluation.get("curves")
    arms = [v.name for v in beta.arms.variants() if v.name in ("random_local", "committee_local", "phal_local", "phal_full_system")] if beta.arms else []
    emit("learning_curves", fg.fig_learning_curves(curves, variants=arms), [out / FILES["evaluation"]])
    emit("learning_curves_labels", fg.fig_learning_curves(curves, variants=arms, x_key="n_labels"), [out / FILES["evaluation"]])
    finals = {name: [rep[-1].get("force_rmse") if rep else None for rep in reps] for name, reps in (curves or {}).items()}
    emit("ablations", fg.fig_ablations(finals), [out / FILES["evaluation"]])
    timeline_path = None if beta.figures.timeline_report is None else loaded.artifacts_root / beta.figures.timeline_report
    emit("timeline", fg.fig_timeline(None if timeline_path is None else _load(timeline_path)), [] if timeline_path is None else [timeline_path])
    write_json_atomic(out / FILES["figures"], index)
    return index


def cmd_figures(loaded: LoadedConfig, args: argparse.Namespace) -> Path:
    render_figures(loaded)
    return loaded.output_dir / FILES["figures"]


def portable(report: Any, loaded: LoadedConfig) -> Any:
    """Absolute paths -> ``<artifacts_root>/...`` and ``<GMD_active>/...`` so committed summaries carry no machine paths."""

    text = json.dumps(report)
    for path, token in ((str(loaded.artifacts_root), "<artifacts_root>"), (str(REPO_ROOT.resolve()), "<GMD_active>")):
        text = text.replace(path, token)
    data = json.loads(text)
    data["config"]["artifacts_root_setting"] = str(Path("..") / loaded.artifacts_root.relative_to(REPO_ROOT.resolve().parent)) \
        if loaded.artifacts_root.is_relative_to(REPO_ROOT.resolve().parent) else "<outside GMD_system>"
    return data


def cmd_report(loaded: LoadedConfig, args: argparse.Namespace) -> Path:
    report = portable(build_report(loaded), loaded)
    path = write_json_atomic(loaded.output_dir / "report.json", report)
    (loaded.output_dir / "report.md").write_text(render_markdown(report))
    if args.publish:
        target = REPO_ROOT / loaded.config.report.results_dir / f"model_study_{loaded.config.name}"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / "report.json")
        shutil.copy2(loaded.output_dir / "report.md", target / "report.md")
    return path


COMMANDS = {
    "receptive-field": (cmd_receptive_field, "measure the model's receptive field on a parent structure"),
    "ground-truth": (cmd_ground_truth, "hotspot ground truth vs monitor signals (needs full-DFT labels)"),
    "arms-run": (cmd_arms_run, "run / resume the acquisition arms and ablations"),
    "evaluate": (cmd_evaluate, "learning curves, accuracy at budget, cost to target, seed statistics"),
    "figures": (cmd_figures, "render every figure from the results that exist"),
    "report": (cmd_report, "report.json + report.md (executed vs planned)"),
}


def _add_arguments(name: str) -> Any:
    def add(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--study", default=None,
                            help="model-study YAML (configs/demos/alkane_hotspots/arms_<profile>.yaml); default: the model_side section of --config")
        if name == "ground-truth":
            parser.add_argument("--checkpoints", nargs="*", default=None, help="primary + committee checkpoints (default: round-0 model)")
        if name == "arms-run":
            parser.add_argument("--only", nargs="*", default=None, help="variant names")
            parser.add_argument("--repetitions", nargs="*", type=int, default=None)
        if name == "report":
            parser.add_argument("--publish", action="store_true", help="copy report.json/md to the committed results directory")

    return add


def study_config(args: argparse.Namespace, alpha: Any) -> LoadedConfig:
    """The model-study configuration for alpha's loaded demo config (``--study`` or its ``model_side``)."""

    from .arms import beta_from_mapping

    if getattr(args, "study", None):
        return load_beta_config(args.study, alpha=alpha)
    section = dict(getattr(alpha.config, "model_side", {}) or {})
    if not section:
        raise SystemExit("no model-study configuration: pass --study arms_<profile>.yaml or fill model_side in the demo config")
    section.setdefault("name", alpha.config.name)
    return beta_from_mapping(section, path=Path(alpha.path).resolve(), alpha=alpha)


def register_cli(register: Any) -> None:
    """alpha's CLI hook: ``python -m tools.experiments.alkane_hotspots <command> --config <demo.yaml> [--study <arms.yaml>]``."""

    for name, (handler, help_text) in COMMANDS.items():
        def run(args: argparse.Namespace, alpha: Any, handler: Any = handler) -> int:
            path = handler(study_config(args, alpha), args)
            print(path)
            return 0

        register(name, help_text, _add_arguments(name), run)


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry (the study YAML names alpha's config in ``base_config``)."""

    parser = argparse.ArgumentParser(prog="alkane_hotspots model study")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (handler, help_text) in COMMANDS.items():
        child = sub.add_parser(name, help=help_text)
        _add_arguments(name)(child)
        child.set_defaults(handler=handler)
    args = parser.parse_args(argv)
    if not args.study:
        parser.error("--study is required for the standalone entry")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(args.handler(load_beta_config(args.study), args))
    return 0


__all__ = ["COMMANDS", "build_report", "label_accounting", "main", "register_cli", "render_markdown", "study_config"]
