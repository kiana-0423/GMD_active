#!/usr/bin/env python3
# Ported from PHAL (scripts.train_density_guess) at 96263b6; MIT License, see NOTICE.
"""Train and deploy the ML initial-density model from a collected dataset.

    python -m tools.scf.scripts.train_density_guess.py \\
        --dataset runtime/cp2k/pulay_dataset \\
        --checkpoint runtime/cp2k_ml/models/density.pt \\
        --deploy runtime/cp2k_ml/models/active/guess.ts.pt

Reads converged densities the mixing dataset already recorded, trains the
density GNN on them, and scripts the result for CP2K's ``SCF_GUESS ML``.

The split is by system, and the report prints both losses: with a dataset this
small, the gap between them is the most important number on the page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.scf.training.dataset_adapter import (
    chemistry_key,
    samples_from_dataset,
    split_by_chemistry,
)
from tools.scf.training.dataset_loader import build_examples
from tools.scf.training.density_gnn import (
    DensityTrainingConfig,
    train_density_matrix_model,
)


def _cost_summary(samples) -> dict:
    """What the SCF still had to do from each target density.

    Recorded rather than optimised: no gradient reaches CP2K, so these labels
    describe the training set and are the quantity the benchmark then measures
    for real. A target frame whose median remaining cost is high is one the
    model would have to be very good at to be worth anything.
    """

    import numpy as np

    remaining = [
        item.metadata["scf_cost"]["scf_steps_remaining"]
        for item in samples
        if item.metadata.get("scf_cost", {}).get("scf_steps_remaining") is not None
    ]
    residual_rates = [
        item.metadata["scf_cost"]["residual_reduction_rate"]
        for item in samples
        if item.metadata.get("scf_cost", {}).get("residual_reduction_rate")
    ]
    energy_rates = [
        item.metadata["scf_cost"]["energy_reduction_rate"]
        for item in samples
        if item.metadata.get("scf_cost", {}).get("energy_reduction_rate")
    ]
    if not remaining:
        return {}
    return {
        "n": len(remaining),
        "scf_steps_remaining": {
            "min": int(np.min(remaining)),
            "median": float(np.median(remaining)),
            "max": int(np.max(remaining)),
            "mean": float(np.mean(remaining)),
        },
        "residual_reduction_rate_median": (
            float(np.median(residual_rates)) if residual_rates else None
        ),
        "energy_reduction_rate_median": float(np.median(energy_rates)) if energy_rates else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=Path("runtime/cp2k/pulay_dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runtime/cp2k_ml/models/density.pt"))
    parser.add_argument("--deploy", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.3)
    parser.add_argument(
        "--target-frame",
        default="final",
        help="'final' for the converged density, or k for the density after k SCF iterations",
    )
    parser.add_argument(
        "--train-on",
        choices=("split", "all"),
        default="split",
        help=(
            "'split' holds out whole chemistries and reports honest generalisation; "
            "'all' fits every family, for a model that has to cover them at run time"
        ),
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    target_frame = args.target_frame if args.target_frame == "final" else int(args.target_frame)
    samples, tally = samples_from_dataset(args.dataset, target_frame=target_frame)
    print(f"dataset: {tally['n_samples']} usable, {tally['n_skipped']} skipped "
          f"(target frame: {target_frame})")
    if not samples:
        print("nothing to train on")
        return 1

    # Whole chemistries are held out, never individual geometries. A random
    # split over samples scores the model on other geometries of molecules it
    # was fitted on, which is what made the first model's validation loss look
    # reasonable while it failed on anything new.
    train_samples, validation_samples = split_by_chemistry(
        samples, holdout_fraction=args.validation_fraction
    )
    if args.train_on == "all":
        # A deployable model has to cover every element it will meet; the split
        # above still runs, so the reported generalisation number comes from a
        # genuine holdout even though this model is not the one it measured.
        print(
            f"fitting all {len(samples)} samples for deployment; the held-out families "
            f"({sorted({chemistry_key(item) for item in validation_samples})}) are reported "
            "from the split model, not this one"
        )
        train_samples, validation_samples = samples, []
    train_set = build_examples(train_samples, target="density_matrix")
    validation_set = build_examples(validation_samples, target="density_matrix")
    print(f"train {len(train_set)}  validation {len(validation_set)}")
    for rejected in list(train_set.rejected) + list(validation_set.rejected):
        print(f"  rejected {rejected}")
    if not len(train_set):
        print("no usable training examples")
        return 1

    config = DensityTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
    )
    result = train_density_matrix_model(
        train_set, validation_set if len(validation_set) else None, args.checkpoint, config=config
    )

    payload = result.to_dict()
    train_history = [item for item in result.history if item.split == "train"]
    validation_history = [item for item in result.history if item.split == "validation"]
    summary = {
        "target_frame": target_frame,
        "train_on": args.train_on,
        "train_families": sorted({chemistry_key(item) for item in train_samples}),
        "holdout_families": sorted({chemistry_key(item) for item in validation_samples}),
        "n_train": len(train_set),
        "n_validation": len(validation_set),
        "train_systems": [example.sample_id for example in train_set],
        "validation_systems": [example.sample_id for example in validation_set],
        "best_epoch": result.best_epoch,
        "best_density_mae": result.best_density_mae,
        "final_train_mae": train_history[-1].density_mae if train_history else None,
        "final_validation_mae": validation_history[-1].density_mae if validation_history else None,
        "wall_time_seconds": result.wall_time_seconds,
        "scf_cost_labels": _cost_summary(train_samples),
    }
    print(json.dumps(summary, indent=1))

    if args.deploy:
        from gmd_active_learning.adapters.dft.scf.deploy import export_density_guess

        deployed = export_density_guess(args.checkpoint, args.deploy)
        summary["deployed"] = {"path": str(deployed.path), "digest": deployed.digest}
        print(f"deployed {deployed.path} ({deployed.digest})")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"summary": summary, "training": payload}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
