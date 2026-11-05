#!/usr/bin/env python3
# Ported from PHAL (scripts.benchmark_controller) at 96263b6; MIT License, see NOTICE.
"""Native CP2K against the adaptive SCF controller, on difficult systems.

    python -m tools.scf.scripts.benchmark_controller.py \\
        --structures data/structures/pulay_sampling/transition_metal_oxides.extxyz \\
        --template configs/cp2k/pulay_dataset_smeared.inp.template \\
        --cp2k-data ../cp2k-ml-scf/cp2k/data \\
        --binary build/cp2k-ml/bin/cp2k.psmp

Two arms, identical inputs, one environment variable apart. Reported per arm:
convergence success rate, failed runs, total SCF iterations, wall time, and
what the controller actually did.

The interesting systems are the ones the native mixer struggles with or fails
on -- an easy SCF has nothing to control. `--label-mode` instead sweeps forced
actions at chosen iterations to produce counterfactual training labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gmd_active_learning.adapters.dft.scf.execution import LocalSchedulerBackend
from tools.scf.benchmark.guess import GUESS_ATOMIC, GuessArm, run_arms
from tools.scf.structures import read_collection, render_input, resolve_kinds

ACTIONS = ("keep", "reset_diis", "switch_mixing", "increase_stability")


def summarise(label: str, runs, baseline=None) -> dict:
    converged = [item for item in runs if item.converged]
    steps = [float(item.scf_steps) for item in converged if item.scf_steps]
    runtime = [float(item.runtime_s) for item in converged]

    actions = sum((item.controller or {}).get("n_actions", 0) for item in runs)
    rejected = sum((item.controller or {}).get("n_rejected", 0) for item in runs)
    fell_back = sum(1 for item in runs if (item.controller or {}).get("fell_back"))

    summary = {
        "arm": label,
        "n_runs": len(runs),
        "n_converged": len(converged),
        "n_failed": len(runs) - len(converged),
        "convergence_rate": len(converged) / len(runs) if runs else 0.0,
        "total_scf_steps": int(sum(steps)) if steps else 0,
        "mean_scf_steps": float(np.mean(steps)) if steps else None,
        "mean_runtime_s": float(np.mean(runtime)) if runtime else None,
        "controller": {"n_actions": actions, "n_rejected": rejected, "n_fell_back": fell_back},
        "per_system": {
            item.system: {
                "scf_steps": item.scf_steps,
                "converged": item.converged,
                "runtime_s": item.runtime_s,
                "actions": (item.controller or {}).get("actions", []),
                "failure": item.failure,
            }
            for item in runs
        },
    }

    if baseline is not None:
        pairs = [
            (item, other)
            for item, other in zip(
                sorted(runs, key=lambda run: run.system),
                sorted(baseline, key=lambda run: run.system),
            )
        ]
        # Rescues and regressions are the metrics that matter on hard systems:
        # a run that converges where the baseline aborted is worth more than
        # any number of iterations saved elsewhere.
        summary["n_rescued"] = sum(1 for a, b in pairs if a.converged and not b.converged)
        summary["n_lost"] = sum(1 for a, b in pairs if b.converged and not a.converged)
        both = [(a, b) for a, b in pairs if a.converged and b.converged and a.scf_steps and b.scf_steps]
        if both:
            summary["total_steps_vs_native"] = int(
                sum(a.scf_steps for a, _ in both) - sum(b.scf_steps for _, b in both)
            )
            summary["n_faster"] = sum(1 for a, b in both if a.scf_steps < b.scf_steps)
            summary["n_slower"] = sum(1 for a, b in both if a.scf_steps > b.scf_steps)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--structures", required=True, type=Path, nargs="+")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--cp2k-data", required=True, type=Path)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("runtime/cp2k_ml/controller"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--only", nargs="+")
    parser.add_argument(
        "--label-mode",
        nargs="+",
        type=int,
        metavar="ITER",
        help="force each action at these iterations and record the cost, for training labels",
    )
    args = parser.parse_args()

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
        path = input_dir / f"{structure.label}.inp"
        path.write_text(render_input(template, structure, kinds), encoding="utf-8")
        inputs[structure.label] = {GUESS_ATOMIC: path}

    if args.label_mode:
        # Counterfactual labels: the same SCF replayed with each action forced
        # at the same iteration. The label is whichever action left the fewest
        # iterations to do, measured rather than assumed -- a recording alone
        # cannot say what an action it never took would have done.
        arms = [GuessArm(label="native", guess=GUESS_ATOMIC)]
        for iteration in args.label_mode:
            for action in ACTIONS:
                arms.append(
                    GuessArm(
                        label=f"i{iteration}-{action}",
                        guess=GUESS_ATOMIC,
                        controller=True,
                        force=f"{iteration}:{action}",
                    )
                )
    else:
        arms = [
            GuessArm(label="native", guess=GUESS_ATOMIC),
            GuessArm(label="controller", guess=GUESS_ATOMIC, controller=True),
        ]

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

    baseline = results["native"]
    summaries = [summarise(arm.label, results[arm.label], baseline=baseline) for arm in arms]

    report: dict = {
        "benchmark_format": "phal-controller-benchmark-v1",
        "systems": [item.label for item in structures],
        "arms": summaries,
        "note": (
            "One environment variable apart. The controller changes strategy only: it never "
            "touches the density, the KS solve, or the convergence test."
        ),
    }

    if args.label_mode:
        report["labels"] = counterfactual_labels(results, args.label_mode)

    path = args.report or (args.work_dir / "controller_benchmark.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\n  {'arm':22s} {'steps':>6s} {'conv':>7s} {'rescued':>8s} {'lost':>5s} {'actions':>8s}")
    for summary in summaries:
        print(
            f"  {summary['arm']:22s} {summary['total_scf_steps']:6d} "
            f"{summary['n_converged']:3d}/{summary['n_runs']:<3d} "
            f"{summary.get('n_rescued', '-')!s:>8s} {summary.get('n_lost', '-')!s:>5s} "
            f"{summary['controller']['n_actions']:8d}"
        )
    print(f"wrote {path}")
    return 0


def counterfactual_labels(results: dict, iterations: list[int]) -> list[dict]:
    """Which action left the fewest iterations to do, per (system, iteration)."""

    native = {item.system: item for item in results["native"]}
    labels: list[dict] = []
    for iteration in iterations:
        systems = {item.system for item in results[f"i{iteration}-keep"]}
        for system in sorted(systems):
            baseline = native.get(system)
            if baseline is None or not baseline.converged or baseline.scf_steps is None:
                continue
            if baseline.scf_steps <= iteration:
                # The run had already finished; there was no decision to make.
                continue

            options: dict[str, int | None] = {}
            for action in ACTIONS:
                run = next(
                    (item for item in results[f"i{iteration}-{action}"] if item.system == system),
                    None,
                )
                # A run that failed gets no cost, not an optimistic one.
                options[action] = (
                    run.scf_steps - iteration
                    if run is not None and run.converged and run.scf_steps is not None
                    else None
                )

            usable = {key: value for key, value in options.items() if value is not None}
            if not usable:
                continue
            best = min(usable, key=lambda key: usable[key])
            labels.append(
                {
                    "system": system,
                    "iteration": iteration,
                    "native_steps_remaining": baseline.scf_steps - iteration,
                    "remaining_by_action": options,
                    "label": best,
                    "saving_vs_keep": (
                        usable["keep"] - usable[best] if "keep" in usable else None
                    ),
                }
            )
    return labels


if __name__ == "__main__":
    raise SystemExit(main())
