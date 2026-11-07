"""Freeze completed pilot evidence for the PHAL manuscript; never run simulations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any
import yaml

LOG = logging.getLogger(__name__)
ARMS = ("random_local", "committee_local", "phal_local", "phal_full_system", "no_lj")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(repo: Path, out: Path) -> dict[str, Any]:
    """Copy only an explicit allowlist of completed pilot results."""
    evidence = out / "data" / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    published = repo / "docs/demos/alkane_hotspots/results"
    artifacts = repo.parent / "alkane_artifacts"
    records: list[dict[str, Any]] = []

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, str):
            return value.replace(str(artifacts), "<alkane_artifacts>").replace(str(repo), "<GMD_active>")
        return value

    def capture(path: Path, name: str) -> Any:
        data = sanitize(read_json(path))
        target = evidence / name
        write_json(target, data)
        source = str(path.relative_to(repo)) if path.is_relative_to(repo) else "../" + str(path.relative_to(repo.parent))
        records.append({"source": source, "source_sha256": digest(path),
                        "snapshot": str(target.relative_to(out)), "snapshot_sha256": digest(target),
                        "transformation": "JSON formatting and replacement of workstation paths"})
        return data

    calibration = capture(published / "fidelity_pilot/report_calibration.json", "fidelity_calibration.json")
    c16 = capture(published / "fidelity_pilot/report_final_eval.json", "fidelity_c16.json")
    c32 = capture(published / "fidelity_pilot_c32/report_final_eval.json", "fidelity_c32.json")
    costs = capture(published / "fidelity_pilot_c32/label_costs.json", "c32_costs.json")
    closed = capture(published / "closed_loop_pilot_v2/closed_loop_report.json", "closed_loop.json")
    frozen = capture(published / "closed_loop_pilot_v2/frozen_eval.json", "frozen_eval.json")
    recovery = capture(published / "closed_loop_pilot_v2/recovery_report.json", "recovery.json")
    capture(published / "model_study_pilot_v2/report.json", "model_study_v2.json")
    capture(published / "model_study_pilot_v2c/report.json", "model_study_v2c.json")
    capture(published / "timing_probe.json", "timing_probe.json")
    for name in ("pilot.yaml", "pilot_c32.yaml", "arms_pilot_v2.yaml", "arms_pilot_v2c.yaml", "closed_loop_pilot_v2.yaml"):
        path = repo / "configs/demos/alkane_hotspots" / name
        target = out / "data" / "configs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        records.append({"source": str(path.relative_to(repo)), "source_sha256": digest(path),
                        "snapshot": str(target.relative_to(out)), "snapshot_sha256": digest(target),
                        "transformation": "byte-identical configuration snapshot; provenance, not a relocated runnable config"})

    arms: dict[str, Any] = {}
    for arm in ARMS:
        arms[arm] = []
        for rep in range(2):
            path = artifacts / "pilot/model_study_v2c/arms" / arm / f"rep{rep}/progress.json"
            data = capture(path, f"arms/{arm}_rep{rep}.json")
            if data["status"] != "complete":
                raise ValueError(f"Refusing incomplete evidence: {path}")
            arms[arm].append({"seed": data["seed"], "rounds": [
                {key: row[key] for key in ("round", "n_labels", "force_rmse", "cumulative_cpu_core_s", "acquisitions")}
                for row in data["rounds"]]})

    boundary: list[dict[str, Any]] = []
    for system, source in (("C16 calibration", artifacts / "pilot/fidelity/per_atom_calibration.csv"),
                           ("C16 held-out", artifacts / "pilot/fidelity/per_atom_final_eval.csv"),
                           ("C32 transfer", artifacts / "pilot_c32/fidelity/per_atom_final_eval.csv")):
        target = evidence / (system.lower().replace(" ", "_") + ".csv")
        target.write_bytes(source.read_bytes())
        records.append({"source": "../" + str(source.relative_to(repo.parent)), "source_sha256": digest(source),
                        "snapshot": str(target.relative_to(out)), "snapshot_sha256": digest(target),
                        "transformation": "byte-identical per-atom metrics"})
        for row in csv.DictReader(source.read_text().splitlines()):
            if row["setting_id"] != "cluster-c1.5-b3":
                continue
            boundary.append({"system": system, "spec_id": row["spec_id"], "parent_id": row["parent_id"],
                             "atom_id": int(row["parent_atom_id"]), "element": "C" if int(row["parent_atom_id"]) < (32 if system.startswith("C32") else 16) else "H",
                             "distance": float(row["graph_distance_to_cut"]), "error": float(row["force_error"])})

    parents_path = artifacts / "pilot/parents.jsonl"
    parents = [json.loads(line) for line in parents_path.read_text().splitlines()]
    geometries = []
    for parent in parents:
        if parent["split"] != "final_eval":
            continue
        source = artifacts / "pilot" / parent["structure_path"]
        lines = source.read_text().splitlines()
        atoms = [line.split() for line in lines[2:2 + int(lines[0])]]
        bonds = [[i, i + 1] for i in range(15)]
        index = 16
        for carbon in range(16):
            count = 3 if carbon in (0, 15) else 2
            bonds.extend([[carbon, h] for h in range(index, index + count)])
            index += count
        geometries.append({"parent_id": parent["parent_id"], "family": parent["family"],
                           "symbols": [a[0] for a in atoms],
                           "positions": [[float(x) for x in a[1:4]] for a in atoms],
                           "bonds": bonds, "perturbation": parent["perturbation"]})
        records.append({"source": "../" + str(source.relative_to(repo.parent)), "source_sha256": digest(source),
                        "snapshot": "data/plot_data.json#geometries", "transformation": "coordinates and symbols parsed from extxyz; fixed alkane connectivity"})
    records.append({"source": "../" + str(parents_path.relative_to(repo.parent)), "source_sha256": digest(parents_path),
                    "snapshot": "data/plot_data.json#geometries", "transformation": "held-out parent metadata selection"})

    metrics = [closed["initial_model"]["metrics_validation"]["force_rmse"]]
    metrics.extend(g["metrics"]["force_rmse"] for g in closed["gate_reports"])
    arm_config = yaml.safe_load((repo / "configs/demos/alkane_hotspots/arms_pilot_v2c.yaml").read_text())
    result = {"schema": 1, "scope": "completed alkane_hotspots pilot only; no alkane_repair inputs",
              "accuracy_targets": arm_config["evaluation"]["accuracy_targets"],
              "calibration": calibration["settings"], "c16": c16["settings"][0], "c32": c32["settings"][0],
              "targets": calibration["targets"], "c32_costs": costs["labels"], "boundary": boundary,
              "arms": arms, "geometries": geometries,
              "closed_loop": {"event_steps": closed["event_steps"], "events": closed["events"],
                              "validation_force_rmse": metrics, "cumulative_new_labels": [0, 2, 3],
                              "labels": closed["label_tasks"], "dft_core_h": closed["measured_dft_cpu_core_hours"],
                              "energy_jump_ev": closed["model_changes"][0]["jump_ev"],
                              "frozen_evaluation": {name: {
                                  "per_seed": [{"seed": int(seed), "n_frames": row["n_frames"],
                                                "n_intact": row["topology"]["n_intact"]}
                                               for seed, row in block["per_seed"].items()],
                                  "n_frames": block["pooled"]["n_frames"],
                                  "n_intact": block["pooled"]["topology"]["n_intact"]}
                                  for name, block in frozen["per_model"].items()},
                              "recovery_adopted": recovery["resume"]["verification"]["passed"],
                              "cancellation_passed": recovery["cancellation"]["passed"]}}
    write_json(out / "data/plot_data.json", result)
    manifest = {"repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "source_experiments": ["fidelity_pilot", "fidelity_pilot_c32", "model_study_pilot_v2c", "closed_loop_pilot_v2"],
                "excluded": ["ongoing alkane_repair", "campaign", "C64/C128", "unexecuted panels"],
                "records": records, "plot_data_sha256": digest(out / "data/plot_data.json")}
    write_json(out / "data/source_manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        data = snapshot(args.repo.resolve(), args.out.resolve())
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Snapshot failed: {exc}") from exc
    LOG.info("Captured %d acquisition arms and %d molecular structures", len(data["arms"]), len(data["geometries"]))


if __name__ == "__main__":
    main()
