"""Initial-model qualification: is the model reliable inside its intended (in-domain) coverage?

For qualification stage ``q`` (``initial_model.stages[q]``), a GMD_SGT model (primary + committee)
is trained on the full-system labels of the first N id_train bases, with id_val as the validation
(early-stopping) set, and accepted only if every pre-registered criterion holds:

* id_val force RMSE (per Cartesian component, all atoms) <= ``acceptance.id_val_force_rmse_max``;
* id_val 95th-percentile per-atom error <= ``acceptance.id_val_atom_error_p95_max``;
* short GMD runs from every calibration/id_val start, every seed: the bond topology survives in at
  least ``min_intact_fraction`` of the frames of *every* trajectory;
* the GMD setup uses the ML force field only (no fallback potential) and no directive that clips
  forces or constrains the dynamics.

An accepted model is frozen (``initial_model/frozen.json``): checkpoint, committee and export hashes,
dataset manifest, species, model and training sections, atomic energies. Repair runs use exactly
these files. If every stage fails, ``initial_model/status.json`` records the unmet prerequisite and
no repair command runs. final_test is never read here.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.contracts.ids import content_hash, file_sha256
from gmd_active_learning.core.exceptions import WorkflowError

from tools.experiments.alkane_hotspots.families import load_positions
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology
from tools.experiments.alkane_hotspots.manifest import utc_now

from .common import force_errors, model_artifact, pooled_errors, read_json, run_md, scan_directives, write_json
from .config import LoadedBenchmark
from .generate import load_records
from .references import full_label_for, label_forces, require_frozen, stage_structures

LOGGER = logging.getLogger(__name__)

STATUS_FILE = "initial_model/status.json"
FROZEN_FILE = "initial_model/frozen.json"


class QualificationError(WorkflowError):
    """The initial model is not qualified (or inputs are missing)."""


def training_config(loaded: LoadedBenchmark) -> Any:
    training = loaded.model.config.training
    overrides = loaded.config.initial_model.train_overrides
    if overrides:
        training = training.model_copy(update={"train": {**training.train, **overrides}})
    return training


def _samples(loaded: LoadedBenchmark, structures: list[dict[str, Any]], training: Any) -> tuple[list[Any], list[Any], dict[str, Any]]:
    from tools.experiments.alkane_hotspots.labels import LabelStore
    from tools.experiments.alkane_hotspots.training import full_sample

    store = LabelStore(loaded.root)
    train, val, record, missing = [], [], {"train": [], "validation": []}, []
    for parent in sorted(structures, key=lambda r: r["parent_id"]):
        label = full_label_for(store, parent)
        if label is None:
            missing.append(parent["parent_id"])
            continue
        sample = full_sample(label, parent, training, loaded.root)
        target, key = (train, "train") if parent["partition"] == "id_train" else (val, "validation")
        if parent["partition"] not in ("id_train", "id_val"):
            raise QualificationError(f"{parent['parent_id']} ({parent['partition']}) may not enter initial-model training or selection")
        target.append(sample)
        record[key].append({"parent_id": parent["parent_id"], "base_id": parent["base_id"], "label_id": label["label_id"], "sample_id": sample.sample_id})
    if missing:
        raise QualificationError(f"missing valid full labels for {len(missing)} structures ({missing[:4]}...): run `label --stage initial_model`")
    return train, val, record


def qualify(loaded: LoadedBenchmark, qualification_stage: int) -> dict[str, Any]:
    from tools.experiments.alkane_hotspots.training import make_backend, predictor_for, train_model, write_dataset

    require_frozen(loaded)
    config = loaded.config
    acceptance = config.initial_model.acceptance
    _, structures = stage_structures(loaded, "initial_model", qualification_stage=qualification_stage)
    training = training_config(loaded)
    train, val, record = _samples(loaded, structures, training)
    out = loaded.root / "initial_model" / f"stage{qualification_stage}"
    training_hash = content_hash({"training": training.model_dump(mode="json"), "seed": config.initial_model.seed}, length=16)
    finished = out / "qualification.json"
    if finished.is_file():  # idempotent: a finished stage is never retrained, re-exported or re-judged
        report = read_json(finished)
        if (report["training_hash"], report["dataset"]) != (training_hash, record):
            raise QualificationError(f"{finished} was produced with other data or settings; a qualification stage is never redone in place")
        write_status(loaded)
        return report
    if (loaded.root / FROZEN_FILE).is_file() and read_json(loaded.root / FROZEN_FILE)["qualification_stage"] == qualification_stage:
        raise QualificationError(f"stage {qualification_stage} is frozen but its qualification record is missing; refusing to retrain it")
    dataset = write_dataset(train, val, out / "dataset", version=f"q{qualification_stage}-{training_hash}")
    backend = make_backend(training, root=loaded.repo_path("."))
    started = time.monotonic()
    model = train_model(backend, dataset, out / "train", seed=config.initial_model.seed)
    train_wall_s = time.monotonic() - started

    # Accuracy on id_val (validation; never final_test).
    predict = predictor_for(backend, model)
    by_id = {p["parent_id"]: p for p in structures}
    pairs, per_structure = [], []
    from tools.experiments.alkane_hotspots.labels import LabelStore

    store = LabelStore(loaded.root)
    for item in record["validation"]:
        parent = by_id[item["parent_id"]]
        topology = AlkaneTopology(int(parent["chain_length"]))
        forces, _ = predict(topology.numbers, load_positions(loaded.root, parent))
        reference = label_forces(loaded.root, full_label_for(store, parent))
        pairs.append((forces, reference))
        per_structure.append({"parent_id": parent["parent_id"], "base_id": parent["base_id"], **force_errors(forces, reference)})
    accuracy = pooled_errors(pairs)
    accuracy_ok = {"id_val_force_rmse": accuracy["force_rmse"] <= acceptance.id_val_force_rmse_max,
                   "id_val_atom_error_p95": accuracy["atom_error_p95"] <= acceptance.id_val_atom_error_p95_max}

    # Stability: short GMD runs with the exported primary model from calibration/id_val starts.
    export = backend.export(Path(model.primary), out / "export")
    artifact = model_artifact(export, Path(model.primary), model.species, f"qualification-q{qualification_stage}")
    md_cfg = acceptance.md
    starts = [r for r in load_records(loaded.root) if r["partition"] in md_cfg.start_partitions and r["kind"] == "ordinary"]
    trajectories = []
    for start in sorted(starts, key=lambda r: r["parent_id"]):
        topology = AlkaneTopology(int(start["chain_length"]))
        for seed in md_cfg.seeds:
            result = run_md(artifact=artifact, positions=load_positions(loaded.root, start), topology=topology, directives=config.repair.md_directives,
                            seed=seed, steps=md_cfg.steps, frame_interval=md_cfg.frame_interval, out_dir=out / "md" / f"{start['parent_id']}-s{seed}",
                            run_id=f"q{qualification_stage}-{start['parent_id']}-s{seed}", executable=config.repair.executable, env=config.repair.env,
                            md_box=config.repair.md_box, model_section=training.model, covalent_scale=md_cfg.covalent_scale)
            trajectories.append({"start": start["parent_id"], "partition": start["partition"], **{k: result[k] for k in
                                 ("seed", "status", "n_frames", "n_broken", "first_broken_step", "intact_fraction")},
                                 "min_image_distance": result["image_check"].get("min_image_distance")})
    stable = bool(trajectories) and all(t["status"] == "completed" and t["intact_fraction"] >= md_cfg.min_intact_fraction for t in trajectories)

    # No hidden stabilisation: ML force field only, no clipping/constraint directives.
    flagged = scan_directives(config.repair.md_directives, acceptance.forbidden_directive_keys)
    honest = {"forbidden_directives_found": flagged, "force_field": "ml (GMD force_field ml, the exported model only)",
              "monitor_stop_actions_note": "monitor stop actions end a trajectory; they never modify forces",
              "passed": not flagged}

    criteria = {**accuracy_ok, "md_topology": stable, "no_hidden_stabilisation": honest["passed"]}
    accepted = all(criteria.values())
    report = {"created_at": utc_now(), "qualification_stage": qualification_stage, "accepted": accepted, "criteria": criteria,
              "acceptance": acceptance.model_dump(mode="json"), "n_train": len(train), "n_val": len(val),
              "train_bases": sorted({i["base_id"] for i in record["train"]}), "val_bases": sorted({i["base_id"] for i in record["validation"]}),
              "id_val_accuracy": accuracy, "id_val_per_structure": per_structure, "md_trajectories": trajectories,
              "md_denominator": "every trajectory counts; intact_fraction is over all frames of a trajectory",
              "hidden_stabilisation_check": honest, "train_wall_s": train_wall_s, "training_hash": training_hash, "dataset": record,
              "checkpoints": model.checkpoints, "export": str(export), "best_val": model.summary.get("best_val")}
    write_json(out / "qualification.json", report)
    if accepted:
        freeze_initial_model(loaded, qualification_stage, model, export, dataset, training, record, report)
    write_status(loaded)
    return report


def freeze_initial_model(loaded: LoadedBenchmark, stage: int, model: Any, export: Path, dataset: Any, training: Any, record: dict[str, Any],
                         report: dict[str, Any]) -> dict[str, Any]:
    path = loaded.root / FROZEN_FILE
    summary_atomic = (model.summary or {}).get("atomic_energies")
    frozen = {"frozen_at": utc_now(), "qualification_stage": stage, "checkpoints": [{"path": p, "sha256": file_sha256(p)} for p in model.checkpoints],
              "export": {"path": str(export), "sha256": file_sha256(export)}, "species": list(model.species), "cutoff": model.cutoff,
              "dataset_manifest": {"path": dataset.metadata["manifest"], "sha256": file_sha256(dataset.metadata["manifest"])},
              "dataset_version": dataset.version, "samples": record, "model_section": training.model, "train_section": training.train,
              "training_hash": report["training_hash"], "seed": loaded.config.initial_model.seed, "atomic_energies": summary_atomic,
              "force_weights": training.force_weights.as_dict(), "id_val_force_rmse": report["id_val_accuracy"]["force_rmse"],
              "id_val_atom_error_p95": report["id_val_accuracy"]["atom_error_p95"]}
    if path.is_file():
        existing = read_json(path)
        if existing["checkpoints"] != frozen["checkpoints"]:
            raise QualificationError(f"{path} already freezes another initial model; the frozen initial model is never replaced")
        return existing
    write_json(path, frozen)
    return frozen


def write_status(loaded: LoadedBenchmark) -> dict[str, Any]:
    stages = loaded.config.initial_model.stages
    reports = {}
    for index in range(len(stages)):
        path = loaded.root / "initial_model" / f"stage{index}" / "qualification.json"
        if path.is_file():
            data = read_json(path)
            reports[index] = {"accepted": data["accepted"], "criteria": data["criteria"], "id_val_force_rmse": data["id_val_accuracy"]["force_rmse"],
                              "id_val_atom_error_p95": data["id_val_accuracy"]["atom_error_p95"], "n_train": data["n_train"]}
    frozen = (loaded.root / FROZEN_FILE).is_file()
    last_failed = len(reports) == len(stages) and not any(r["accepted"] for r in reports.values())
    status = {"updated_at": utc_now(), "stages": reports, "qualified": frozen,
              "prerequisite": "met" if frozen else ("unmet: no qualification stage met the acceptance criteria within budget" if last_failed else "pending"),
              "next": None if frozen or last_failed else f"label and qualify stage {len(reports)}"}
    write_json(loaded.root / STATUS_FILE, status)
    return status


def require_qualified(loaded: LoadedBenchmark) -> dict[str, Any]:
    path = loaded.root / FROZEN_FILE
    if not path.is_file():
        status = read_json(loaded.root / STATUS_FILE) if (loaded.root / STATUS_FILE).is_file() else {"prerequisite": "pending"}
        raise QualificationError(f"no qualified initial model ({status['prerequisite']}); repair experiments are not run on an unqualified baseline")
    frozen = read_json(path)
    for item in [*frozen["checkpoints"], frozen["export"]]:
        if file_sha256(item["path"]) != item["sha256"]:
            raise QualificationError(f"frozen initial-model file {item['path']} changed since it was frozen")
    return frozen


def id_val_baseline(loaded: LoadedBenchmark) -> float:
    return float(require_qualified(loaded)["id_val_force_rmse"])


__all__ = ["FROZEN_FILE", "QualificationError", "STATUS_FILE", "id_val_baseline", "qualify", "require_qualified", "training_config", "write_status"]
