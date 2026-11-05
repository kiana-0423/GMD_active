#!/usr/bin/env python3
# Ported from PHAL (scripts.benchmark_guess_variants) at 96263b6; MIT License, see NOTICE.
"""Compare initial-guess strategies by what they cost the SCF.

    python -m tools.scf.scripts.benchmark_guess_variants.py \\
        --structures data/structures/benchmark/molecules.extxyz \\
        --template configs/cp2k/benchmark_guess.inp.template \\
        --cp2k-data ../cp2k-ml-scf/cp2k/data \\
        --binary build/cp2k-ml/bin/cp2k.psmp \\
        --model final=runtime/cp2k_ml/models/active/guess.ts.pt \\
        --model early=runtime/cp2k_ml/models/active/guess_iter1.ts.pt \\
        --alpha 0.25 0.5 0.75 1.0

Arms: the atomic guess, then every (model, alpha) pair, where

    P_initial = (1 - alpha) * P_atomic + alpha * P_ML

Reported per arm: SCF steps, wall time, convergence rate, and the energy and
residual trajectories. Density error is deliberately *not* the headline: it was
measured to correlate with SCF cost at Spearman +0.29
(docs/ml_density_failure_analysis.md), which is why this script exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gmd_active_learning.adapters.dft.scf.execution import LocalSchedulerBackend
from tools.scf.benchmark.guess import (
    GUESS_ATOMIC,
    GUESS_ML,
    GuessArm,
    run_arms,
)
from tools.scf.structures import read_collection, render_input, resolve_kinds

REPORT_NAME = "guess_variant_benchmark.json"


def arm_summary(label: str, runs, baseline=None) -> dict:
    converged = [item for item in runs if item.converged]
    steps = [float(item.scf_steps) for item in converged if item.scf_steps]
    runtime = [float(item.runtime_s) for item in converged]
    fell_back = [item for item in runs if item.initial_guess == GUESS_ML and not item.ml_guess_accepted]

    summary = {
        "arm": label,
        "n_runs": len(runs),
        "n_converged": len(converged),
        "convergence_rate": len(converged) / len(runs) if runs else 0.0,
        "n_fallback": len(fell_back),
        "total_scf_steps": int(sum(steps)) if steps else 0,
        "mean_scf_steps": float(np.mean(steps)) if steps else None,
        "mean_runtime_s": float(np.mean(runtime)) if runtime else None,
        "per_system": {
            item.system: {
                "scf_steps": item.scf_steps,
                "runtime_s": item.runtime_s,
                "converged": item.converged,
                "guess_source": item.guess_source,
            }
            for item in runs
        },
    }

    if baseline is not None:
        paired = [
            (item.scf_steps, other.scf_steps)
            for item, other in zip(
                sorted(runs, key=lambda run: run.system),
                sorted(baseline, key=lambda run: run.system),
            )
            if item.converged and other.converged and item.scf_steps and other.scf_steps
        ]
        if paired:
            summary["mean_speedup"] = float(np.mean([base / arm for arm, base in paired]))
            summary["total_steps_vs_baseline"] = int(
                sum(arm for arm, _ in paired) - sum(base for _, base in paired)
            )
            summary["n_systems_improved"] = int(sum(1 for arm, base in paired if arm < base))
            summary["n_systems_worsened"] = int(sum(1 for arm, base in paired if arm > base))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--structures", required=True, type=Path, nargs="+")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--cp2k-data", required=True, type=Path)
    parser.add_argument("--binary", required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a deployed model to benchmark; repeatable",
    )
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--work-dir", type=Path, default=Path("runtime/cp2k_ml/variants"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()

    models: dict[str, Path] = {}
    for entry in args.model:
        name, _, path = entry.partition("=")
        if not path:
            parser.error(f"--model expects NAME=PATH, got {entry!r}")
        models[name] = Path(path).resolve()

    structures = read_collection(args.structures)
    if args.only:
        wanted = set(args.only)
        structures = [item for item in structures if item.label in wanted]

    template = args.template.read_text(encoding="utf-8")
    elements = sorted({symbol for item in structures for symbol in item.symbols})
    kinds = resolve_kinds(elements, data_dir=args.cp2k_data)

    input_dir = args.work_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, dict[str, Path]] = {}
    for structure in structures:
        for guess in (GUESS_ATOMIC, GUESS_ML):
            path = input_dir / f"{structure.label}-{guess.lower()}.inp"
            path.write_text(render_input(template, structure, kinds, guess=guess), encoding="utf-8")
            inputs.setdefault(structure.label, {})[guess] = path

    arms = [GuessArm(label="atomic", guess=GUESS_ATOMIC)]
    for name, path in models.items():
        for alpha in args.alpha:
            arms.append(
                GuessArm(label=f"{name}-a{alpha:g}", guess=GUESS_ML, model=path, alpha=alpha)
            )

    print(f"{len(arms)} arms x {len(structures)} systems = {len(arms) * len(structures)} runs")
    results = run_arms(
        inputs,
        arms,
        binary=args.binary,
        scheduler=LocalSchedulerBackend(),
        work_dir=args.work_dir,
        timeout=args.timeout,
        poll_interval=2.0,
        max_parallel=args.max_parallel,
    )

    baseline = results["atomic"]
    summaries = [arm_summary(arm.label, results[arm.label], baseline=baseline) for arm in arms]

    trajectories = {
        arm.label: {
            item.system: {
                "residuals": list(item.residuals),
                "energies": list(item.energies),
            }
            for item in results[arm.label]
        }
        for arm in arms
    }

    report = {
        "benchmark_format": "phal-guess-variant-benchmark-v1",
        "systems": [item.label for item in structures],
        "alphas": list(args.alpha),
        "models": {name: str(path) for name, path in models.items()},
        "arms": summaries,
        "trajectories": trajectories,
        "note": (
            "P_initial = (1 - alpha) P_atomic + alpha P_ML. Speedup is against the atomic arm "
            "on the same systems. Density error is not reported here on purpose: it predicts "
            "SCF cost at Spearman +0.29, so it is not the quantity to optimise."
        ),
    }

    path = args.report or (args.work_dir / REPORT_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\n  {'arm':16s} {'steps':>6s} {'conv':>6s} {'speedup':>8s} {'better':>7s} {'worse':>6s}")
    for summary in summaries:
        speedup = summary.get("mean_speedup")
        print(
            f"  {summary['arm']:16s} {summary['total_scf_steps']:6d} "
            f"{summary['n_converged']:3d}/{summary['n_runs']:<2d} "
            f"{(f'{speedup:.3f}' if speedup else '-'):>8s} "
            f"{summary.get('n_systems_improved', '-')!s:>7s} "
            f"{summary.get('n_systems_worsened', '-')!s:>6s}"
            + (f"  ({summary['n_fallback']} fallback)" if summary["n_fallback"] else "")
        )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
