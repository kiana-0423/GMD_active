"""Initial-model diagnostic: is a small-data GMD_SGT model limited by optimiser steps or by data?

Trains single models (no committee) on growing sets of EXISTING labels and reports per-atom force
errors on the validation split's full-system labels. final_eval is never read. No DFT is run.

Usage (from the GMD_active root, after the pilot labels and fidelity gate exist):

    python examples/alkane_hotspot_demo/initial_model_diagnostic.py \
        --study configs/demos/alkane_hotspots/arms_pilot.yaml [--threads 2] [--seed 11]

Variants (fixed; the training section of ``--study`` is the baseline):
  1. initial_train full labels, study training settings;
  2. same data, more optimiser steps (n_epochs 1500, batch_size 1, patience 300);
  3. + calibration full labels;
  4. + calibration fragment labels of the frozen fidelity setting (fidelity/selection.json).

Writes ``<artifacts_root>/diagnostics/initial_model/summary.json``. Validation labels are also the
early-stopping set, so the errors are mildly optimistic; this is a sizing diagnostic, not a result.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

LOGGER = logging.getLogger("alkane.initial_model_diagnostic")
MORE_STEPS = {"n_epochs": 1500, "batch_size": 1, "patience": 300, "warmup_steps": 50}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", required=True, help="model-study YAML (arms_<profile>.yaml)")
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads (keep low next to a running CP2K job)")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch

    from tools.experiments.alkane_hotspots.arms import load_beta_config
    from tools.experiments.alkane_hotspots.training import (TrainingConfig, make_backend, predictor_for, sample_for_label,
                                                            train_model, write_dataset)

    torch.set_num_threads(args.threads)
    loaded = load_beta_config(Path(args.study))
    art = loaded.artifacts_root
    parents = {p["parent_id"]: p for p in _read_jsonl(art / "parents.jsonl")}
    labels = _read_jsonl(art / "labels.jsonl")
    selection = json.loads((art / "fidelity" / "selection.json").read_text())
    if selection.get("status") != "selected":
        raise SystemExit("no frozen fidelity selection: run the fidelity gate first")
    setting_id = selection["setting"]["setting_id"]
    approved = {s.get("label_id") for s in _read_jsonl(art / "fidelity" / "fragments.jsonl") if s.get("setting_id") == setting_id}

    def split(label: dict) -> str:
        return parents[label["parent_id"]]["split"]

    full = [l for l in labels if l["kind"] == "full" and l["status"] == "valid"]
    fragments = [l for l in labels if l["kind"] == "fragment" and l["status"] == "valid" and split(l) == "calibration"
                 and l["label_id"] in approved]
    validation = [l for l in full if split(l) == "validation"]
    if not validation:
        raise SystemExit("no validation full labels")
    sets = {"initial_train": [l for l in full if split(l) == "initial_train"]}
    sets["initial_train+calibration_full"] = sets["initial_train"] + [l for l in full if split(l) == "calibration"]
    sets["initial_train+calibration_full+fragments"] = sets["initial_train+calibration_full"] + fragments
    base = loaded.config.training.model_dump()
    runs = [("initial_train", "study"), ("initial_train", "more_steps"), ("initial_train+calibration_full", "more_steps"),
            ("initial_train+calibration_full+fragments", "more_steps")]
    out_dir = art / "diagnostics" / "initial_model"
    summary: dict[str, dict] = {}
    for set_name, variant in runs:
        raw = dict(base)
        raw["train"] = dict(base["train"]) if variant == "study" else {**base["train"], **MORE_STEPS}
        raw["sgt"] = {**base["sgt"], "committee_size": 0}
        config = TrainingConfig.model_validate(raw)
        backend = make_backend(config, root=ROOT)
        train = [sample_for_label(l, parents, config, art, ROOT) for l in sets[set_name]]
        val = [sample_for_label(l, parents, config, art, ROOT) for l in validation]
        key = f"{set_name}__{variant}"
        dataset = write_dataset(train, val, out_dir / key / "dataset", version=key)
        model = train_model(backend, dataset, out_dir / key / "train", seed=args.seed, config=config)
        predict = predictor_for(backend, model)
        errors, elements = [], []
        for sample in val:
            forces, _ = predict(np.asarray(sample.region.numbers), np.asarray(sample.region.positions))
            errors.append(np.linalg.norm(np.asarray(forces) - np.asarray(sample.forces), axis=1))
            elements.append(np.asarray(sample.region.numbers))
        err, z = np.concatenate(errors), np.concatenate(elements)
        summary[key] = {"train_set": set_name, "train_variant": variant, "train_settings": raw["train"], "n_train_samples": len(train),
                        "n_train_atoms": int(sum(s.region.n_atoms for s in train)), "n_validation_atoms": int(len(err)),
                        "median_atom_error": float(np.median(err)), "median_atom_error_C": float(np.median(err[z == 6])),
                        "median_atom_error_H": float(np.median(err[z == 1])),
                        "fraction_atoms_below": {f"{t:g}": float(np.mean(err < t)) for t in (0.1, 0.2, 0.4)}}
        LOGGER.info("%s: %s", key, json.dumps(summary[key]))
    result = {"description": __doc__.splitlines()[0], "seed": args.seed, "fidelity_setting": setting_id,
              "evaluated_on": "validation full labels (also the early-stopping set)", "units": "eV/A, |F_model - F_DFT| per atom",
              "runs": summary}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
