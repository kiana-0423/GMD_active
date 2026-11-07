"""Sealed final evaluation: the only code path that reads final-test labels.

Every model is evaluated on the same sealed structures (common references, not each model's own
trajectories):

* arm A is the frozen initial model; every other run's final (promoted) model is compared with it on
  the same structures, so every comparison is paired;
* challenged-region force errors on the sealed challenge structures of the run's family, whole-structure
  errors, and in-domain retention on the sealed ordinary structures;
* reference-confirmed failure of the initial model on each sealed challenge (``failure`` rule);
* stability: fresh GMD runs of each model from every sealed challenge structure of the family with the
  configured paired seeds; every trajectory counts in the denominator;
* repair success per run (``evaluation.success``), success counts per arm x family, per-base summaries
  (independent base conformations are the unit), raw paired differences between arms;
* trained-model consistency for local labels: the final model's core forces on each acquired capped
  fragment vs on the full parent frame it was cut from (independent of the DFT fidelity audit).

Each invocation appends a record to ``sealed_access.jsonl`` (who/what/when, protocol hash, the runs
evaluated, whether every planned run was complete).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.core.exceptions import WorkflowError

from tools.experiments.alkane_hotspots.families import load_positions
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology
from tools.experiments.alkane_hotspots.manifest import utc_now

from .common import challenged_region, force_errors, model_artifact, pooled_errors, read_json, read_jsonl, run_md, write_json
from .config import LoadedBenchmark
from .failure import region_rows
from .generate import load_records
from .qualify import require_qualified, training_config
from .references import full_label_for, label_forces, require_frozen
from .repair import RUNS_SUBDIR, RunSpec, run_dir, run_matrix

LOGGER = logging.getLogger(__name__)

EVAL_SUBDIR = "evaluation"


# ---------------------------------------------------------------------------------------------- inputs


def collect_runs(loaded: LoadedBenchmark) -> tuple[list[dict[str, Any]], list[str]]:
    """Finished runs (repair_run.json) and the planned runs that are missing or unfinished."""

    finished, missing = [], []
    for spec in run_matrix(loaded):
        path = run_dir(loaded, spec) / "repair_run.json"
        if not path.is_file():
            missing.append(spec.name)
            continue
        record = read_json(path)
        if record.get("final_stage") != "done":
            missing.append(spec.name)
            continue
        finished.append({**record, "name": spec.name})
    return finished, missing


def log_access(loaded: LoadedBenchmark, payload: dict[str, Any]) -> None:
    path = loaded.root / "sealed_access.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": utc_now(), **payload}, sort_keys=True, default=str) + "\n")


# ---------------------------------------------------------------------------------------------- predictions


class ModelBank:
    """Per-model predictions on the sealed structures, cached by export/checkpoint hash."""

    def __init__(self, loaded: LoadedBenchmark) -> None:
        from tools.experiments.alkane_hotspots.training import make_backend

        self.loaded = loaded
        self.backend = make_backend(training_config(loaded), root=loaded.repo_path("."))
        self.cache: dict[tuple[str, str], np.ndarray] = {}

    def forces(self, checkpoint: str, record: dict[str, Any], root: Path) -> np.ndarray:
        key = (file_sha256(checkpoint), record["parent_id"])
        if key not in self.cache:
            topology = AlkaneTopology(int(record["chain_length"]))
            self.cache[key] = np.asarray(self.backend.predict(checkpoint, load_positions(root, record), topology.numbers).forces, dtype=float)
        return self.cache[key]

    def predictor(self, checkpoint: str, root: Path, records: dict[str, dict[str, Any]]) -> Any:
        by_positions = {}

        def predict(numbers: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, None]:
            for record in records.values():
                key = record["parent_id"]
                if key not in by_positions:
                    by_positions[key] = load_positions(root, record)
                if by_positions[key].shape == positions.shape and np.allclose(by_positions[key], positions):
                    return self.forces(checkpoint, record, root), None
            raise WorkflowError("prediction requested for a structure that is not a sealed structure")

        return predict


def structure_errors(bank: ModelBank, checkpoint: str, records: Sequence[dict[str, Any]], root: Path, store: Any, bond_hops: int) -> dict[str, Any]:
    rows = {}
    for record in records:
        label = full_label_for(store, record)
        if label is None:
            rows[record["parent_id"]] = {"status": "no_reference_label"}
            continue
        reference = label_forces(root, label)
        forces = bank.forces(checkpoint, record, root)
        row = {"status": "evaluated", "whole": force_errors(forces, reference)}
        if record["kind"] == "challenge":
            topology = AlkaneTopology(int(record["chain_length"]))
            row["region"] = force_errors(forces, reference, challenged_region(record, topology, bond_hops))
        rows[record["parent_id"]] = row
    return rows


def pooled(bank: ModelBank, checkpoint: str, records: Sequence[dict[str, Any]], root: Path, store: Any, bond_hops: int, *, region: bool) -> dict[str, Any]:
    pairs = []
    for record in records:
        label = full_label_for(store, record)
        if label is None:
            continue
        reference = label_forces(root, label)
        forces = bank.forces(checkpoint, record, root)
        if region:
            atoms = challenged_region(record, AlkaneTopology(int(record["chain_length"])), bond_hops)
            pairs.append((forces[atoms], reference[atoms]))
        else:
            pairs.append((forces, reference))
    return pooled_errors(pairs)


# ---------------------------------------------------------------------------------------------- stability


def stability(loaded: LoadedBenchmark, model: dict[str, Any], starts: Sequence[dict[str, Any]], frozen: dict[str, Any]) -> dict[str, Any]:
    """Fresh GMD runs of a model from every start with every stability seed (cached by export hash)."""

    config = loaded.config
    st = config.evaluation.stability
    export = Path(model["export"])
    digest = file_sha256(export)
    trajectories = []
    for start in starts:
        topology = AlkaneTopology(int(start["chain_length"]))
        for seed in st.seeds:
            name = f"{start['parent_id']}-s{seed}"
            result = run_md(artifact=model_artifact(export, Path(model["checkpoint"]), frozen["species"], f"eval-{digest[:12]}"),
                            positions=load_positions(loaded.sealed_root, start), topology=topology, directives=config.repair.md_directives, seed=seed,
                            steps=st.steps, frame_interval=st.frame_interval, out_dir=loaded.root / EVAL_SUBDIR / "stability" / digest[:16] / name,
                            run_id=f"stability-{digest[:8]}-{name}", executable=config.repair.executable, env=config.repair.env,
                            md_box=config.repair.md_box, model_section=frozen["model_section"], covalent_scale=st.covalent_scale)
            trajectories.append({"start": start["parent_id"], "seed": seed, **{k: result[k] for k in ("status", "n_frames", "n_broken", "first_broken_step",
                                                                                                         "intact_fraction")},
                                 "min_image_distance": result["image_check"].get("min_image_distance"),
                                 "below_margin_frames": len(result["image_check"].get("below_margin") or [])})
    stable = [t["status"] == "completed" and t["intact_fraction"] >= st.min_intact_fraction for t in trajectories]
    return {"export_sha256": digest, "n_trajectories": len(trajectories), "n_stable": int(sum(stable)),
            "stable_fraction": float(np.mean(stable)) if stable else None, "all_stable": bool(stable) and all(stable),
            "denominator": "every trajectory (completed or not) counts", "trajectories": trajectories}


# ---------------------------------------------------------------------------------------------- consistency


def consistency(loaded: LoadedBenchmark, bank: ModelBank, run: dict[str, Any]) -> dict[str, Any]:
    """Final model: core forces on each acquired capped fragment vs on the full parent frame (same atoms)."""

    from gmd_active_learning.adapters.gmd.engine import parse_frames
    from gmd_active_learning.contracts import ExtractedRegion, RegionRole

    from tools.experiments.alkane_hotspots.closed_loop import isolated_frame

    report_path = loaded.root / RUNS_SUBDIR / run["name"] / "closed_loop_report.json"
    if not report_path.is_file():
        return {"status": "not_applicable", "reason": "no closed-loop report (arm without acquisition)"}
    report = read_json(report_path)
    checkpoint = (run.get("final_model") or {}).get("checkpoint")
    rows = []
    topology = AlkaneTopology(loaded.config.chain_length)
    for task in report.get("label_tasks", []):
        region_file = loaded.root / RUNS_SUBDIR / run["name"] / "run" / "cp2k_tasks" / task["task_id"] / "region.json"
        if not region_file.is_file():
            rows.append({"task_id": task["task_id"], "status": "unavailable", "reason": "region.json not found"})
            continue
        region = ExtractedRegion.from_dict(read_json(region_file))
        if region.mode == "full_system":
            rows.append({"task_id": task["task_id"], "status": "not_applicable", "reason": "full-system label"})
            continue
        event_dir = loaded.root / RUNS_SUBDIR / run["name"] / "run" / "events" / str(region.event_id)
        frames = parse_frames((event_dir / "frames.extxyz").read_text(), run_id=run["name"]) if (event_dir / "frames.extxyz").is_file() else []
        trigger = [f for f in frames if f.step == region.parent_step]
        if not trigger:
            rows.append({"task_id": task["task_id"], "status": "unavailable", "reason": "event trigger frame not found"})
            continue
        parent = isolated_frame(trigger[0], topology)
        full = np.asarray(bank.backend.predict(checkpoint, parent.positions, parent.numbers).forces)
        fragment = np.asarray(bank.backend.predict(checkpoint, region.positions, region.numbers).forces)
        core = [(row, int(pid)) for row, (pid, role) in enumerate(zip(region.parent_atom_ids, region.roles)) if role is RegionRole.CORE]
        diff = np.asarray([fragment[row] - full[pid] for row, pid in core])
        norms = np.linalg.norm(diff, axis=1)
        rows.append({"task_id": task["task_id"], "status": "evaluated", "n_core": len(core), "core_force_rmse": float(np.sqrt(np.mean(diff * diff))),
                     "max_core_force_difference": float(norms.max())})
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    return {"status": "evaluated" if evaluated else "no_fragments", "rows": rows,
            "max_core_force_rmse": max((r["core_force_rmse"] for r in evaluated), default=None),
            "statement": "model-side consistency of fragment vs parent inputs for the final model; distinct from the DFT fidelity audit"}


# ---------------------------------------------------------------------------------------------- outcomes


def run_outcome(config: Any, before: dict[str, Any], after: dict[str, Any], retention_before: dict[str, Any], retention_after: dict[str, Any],
                stable: dict[str, Any] | None) -> dict[str, Any]:
    success = config.evaluation.success
    rmse_before, rmse_after = before.get("force_rmse"), after.get("force_rmse")
    improvement = None if not rmse_before else (rmse_before - rmse_after) / rmse_before
    retention = None if not retention_before.get("force_rmse") else (retention_after["force_rmse"] - retention_before["force_rmse"]) / retention_before["force_rmse"]
    checks = {"challenge_region_rmse": rmse_after is not None and rmse_after <= success.challenge_region_rmse_max,
              "improvement": improvement is not None and improvement >= success.min_relative_improvement,
              "retention": retention is not None and retention <= success.id_retention_max_relative_regression}
    if success.require_stability:
        checks["stability"] = bool(stable and stable["all_stable"])
    return {"region_rmse_before": rmse_before, "region_rmse_after": rmse_after, "relative_improvement": improvement,
            "id_test_rmse_before": retention_before.get("force_rmse"), "id_test_rmse_after": retention_after.get("force_rmse"),
            "relative_retention_change": retention, "checks": checks, "success": all(checks.values())}


def bootstrap_mean(values_by_base: dict[str, list[float]], config: dict[str, Any]) -> dict[str, Any]:
    """Mean over bases of each base's mean over seeds; percentile CI resampling bases (needs >= 2 bases)."""

    bases = sorted(b for b, v in values_by_base.items() if v)
    means = np.asarray([np.mean(values_by_base[b]) for b in bases])
    out: dict[str, Any] = {"n_bases": len(bases), "per_base_mean": {b: float(m) for b, m in zip(bases, means)},
                           "mean": float(means.mean()) if len(means) else None, "ci": None}
    if len(bases) >= 2:
        rng = np.random.default_rng(int(config.get("seed", 0)))
        draws = rng.choice(means, size=(int(config.get("n", 2000)), len(means)), replace=True).mean(axis=1)
        alpha = (1.0 - float(config.get("confidence", 0.95))) / 2.0
        out["ci"] = [float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))]
        out["ci_note"] = f"percentile bootstrap over {len(bases)} bases; with this few bases the interval is crude"
    else:
        out["ci_note"] = "fewer than 2 bases: no interval"
    return out


def evaluate(loaded: LoadedBenchmark, *, allow_incomplete: bool = False, reason: str = "final evaluation") -> dict[str, Any]:
    from tools.experiments.alkane_hotspots.labels import LabelStore

    frozen_protocol = require_frozen(loaded)
    frozen = require_qualified(loaded)
    config = loaded.config
    runs, missing = collect_runs(loaded)
    if missing and not allow_incomplete:
        raise WorkflowError(f"{len(missing)} planned runs are missing or unfinished ({missing[:3]}...); the sealed evaluation runs once all are done "
                            "(--allow-incomplete records an explicitly partial evaluation)")
    log_access(loaded, {"reason": reason, "protocol_sha256": frozen_protocol["protocol"]["sha256"], "n_runs": len(runs), "missing_runs": missing,
                        "partial": bool(missing)})
    sealed = {r["parent_id"]: r for r in load_records(loaded.sealed_root)}
    store = LabelStore(loaded.sealed_root)
    bank = ModelBank(loaded)
    hops = config.failure.region_bond_hops
    ordinary = [r for r in sealed.values() if r["kind"] == "ordinary"]
    challenges = {family: sorted((r for r in sealed.values() if r["kind"] == "challenge" and r["family"] == family), key=lambda r: r["parent_id"])
                  for family in config.challenges.enabled()}
    initial = {"checkpoint": frozen["checkpoints"][0]["path"], "export": frozen["export"]["path"], "checkpoints": [c["path"] for c in frozen["checkpoints"]]}

    # Arm A: the frozen initial model, and the reference-confirmed failures it shows on the sealed challenges.
    predict_initial = bank.predictor(initial["checkpoint"], loaded.sealed_root, sealed)
    failures = region_rows(config, loaded.sealed_root, list(sealed.values()), store, predict_initial, baseline_rmse=float(frozen["id_val_force_rmse"]))
    retention_initial = pooled(bank, initial["checkpoint"], ordinary, loaded.sealed_root, store, hops, region=False)
    initial_family = {f: pooled(bank, initial["checkpoint"], recs, loaded.sealed_root, store, hops, region=True) for f, recs in challenges.items()}
    stability_initial = {f: stability(loaded, initial, recs, frozen) for f, recs in challenges.items()}

    outcomes = []
    for run in runs:
        spec = run["spec"]
        family = spec["family"]
        model = run["final_model"] or initial
        after = pooled(bank, model["checkpoint"], challenges[family], loaded.sealed_root, store, hops, region=True)
        retention_after = pooled(bank, model["checkpoint"], ordinary, loaded.sealed_root, store, hops, region=False)
        stable = stability(loaded, model, challenges[family], frozen) if config.evaluation.success.require_stability else None
        outcome = run_outcome(config, initial_family[family], after, retention_initial, retention_after, stable)
        row = {"run": run["name"], **spec, "promoted": not model.get("initial", False), "outcome": outcome,
               "per_structure": structure_errors(bank, model["checkpoint"], challenges[family], loaded.sealed_root, store, hops),
               "stability": stable, "dft_core_h": run.get("measured_dft_cpu_core_hours")}
        if config.evaluation.consistency_check and spec["kind"] == "closed_loop":
            row["consistency"] = consistency(loaded, bank, run)
        if spec["kind"] == "continued_training" and run.get("ungated_last"):
            ungated = run["ungated_last"]["checkpoints"][0]
            row["ungated_last"] = {"region": pooled(bank, ungated, challenges[family], loaded.sealed_root, store, hops, region=True),
                                   "id_test": pooled(bank, ungated, ordinary, loaded.sealed_root, store, hops, region=False)}
        outcomes.append(row)

    summary: dict[str, Any] = {}
    by_arm_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        by_arm_family[(row["arm"], row["family"])].append(row)
    for (arm, family), rows in sorted(by_arm_family.items()):
        improvements = defaultdict(list)
        for row in rows:
            if row["outcome"]["relative_improvement"] is not None:
                improvements[row["base_id"]].append(row["outcome"]["relative_improvement"])
        summary.setdefault(arm, {})[family] = {
            "n_runs": len(rows), "n_cases": len({r["case"] for r in rows}), "n_bases": len({r["base_id"] for r in rows}),
            "n_success": sum(1 for r in rows if r["outcome"]["success"]), "n_promoted": sum(1 for r in rows if r["promoted"]),
            "relative_improvement_by_base": bootstrap_mean(improvements, config.evaluation.bootstrap),
            "success_by_case": {case: [r["outcome"]["success"] for r in rows if r["case"] == case] for case in sorted({r["case"] for r in rows})}}
    paired = paired_differences(outcomes)
    result = {"created_at": utc_now(), "protocol_sha256": frozen_protocol["protocol"]["sha256"], "partial": bool(missing), "missing_runs": missing,
              "initial_model": {"id_val_force_rmse": frozen["id_val_force_rmse"], "id_test": retention_initial, "challenge_regions": initial_family,
                                "stability": stability_initial, "failures": failures},
              "runs": outcomes, "summary": summary, "paired_differences": paired,
              "units": "eV/A; region RMSE per Cartesian component over the challenged region; independent unit = base conformation"}
    write_json(loaded.root / EVAL_SUBDIR / "sealed_evaluation.json", result)
    return result


def paired_differences(outcomes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Raw paired differences of the after-repair region RMSE between arms on the same (case, seed)."""

    by_key: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in outcomes:
        value = row["outcome"]["region_rmse_after"]
        if value is not None:
            by_key[(row["case"], int(row["seed"]))][row["arm"]] = value
    arms = sorted({row["arm"] for row in outcomes})
    out = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            pairs = [(case, seed, v[a], v[b]) for (case, seed), v in sorted(by_key.items()) if a in v and b in v]
            out.append({"a": a, "b": b, "n_pairs": len(pairs), "pairs": [{"case": c, "seed": s, "a": x, "b": y, "a_minus_b": x - y} for c, s, x, y in pairs],
                        "note": "raw paired values; no equivalence or 'no difference' is inferred from a small sample"})
    return out


__all__ = ["EVAL_SUBDIR", "bootstrap_mean", "collect_runs", "evaluate", "paired_differences", "run_outcome"]
