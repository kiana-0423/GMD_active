#!/usr/bin/env python3
# Ported from PHAL (scripts.benchmark_scf_strategies) at 96263b6; MIT License, see NOTICE.
"""Run every SCF configuration on every system, and record what happened.

    python -m tools.scf.scripts.benchmark_scf_strategies.py \\
        --structures data/structures/pulay_sampling/*.extxyz \\
        --template configs/cp2k/strategy.inp.template \\
        --cp2k-data ../cp2k-ml-scf/cp2k/data \\
        --binary build/cp2k-ml/bin/cp2k.psmp

This is the dataset the strategy predictor is fitted on: structure features in,
and for every configuration the convergence, SCF steps, wall time, final energy
and failure reason. A configuration that fails is recorded as a failure, not
dropped -- the systems nothing converges on are the ones worth predicting for.

With --evaluate it also runs leave-one-out: each system predicted by a model
that never saw it, scored on measured SCF steps rather than on label accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gmd_active_learning.adapters.dft.scf.execution import LocalSchedulerBackend
from tools.scf.benchmark.guess import GUESS_ATOMIC, GuessArm, run_arms
from tools.scf.strategy_predictor import (
    StrategyDataset,
    StrategyOutcome,
    apply_strategy,
    leave_one_out,
    strategy_grid,
    structure_features,
)
from tools.scf.structures import (
    descriptors,
    read_collection,
    render_input,
    resolve_kinds,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--structures", required=True, type=Path, nargs="+")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--cp2k-data", required=True, type=Path)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("runtime/cp2k_ml/strategy"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--only", nargs="+")
    parser.add_argument(
        "--smeared",
        nargs="*",
        default=None,
        help="labels needing smearing and empty states; default: anything with a d-block metal",
    )
    parser.add_argument("--evaluate", action="store_true", help="also run leave-one-out")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    structures = read_collection(args.structures)
    if args.only:
        wanted = set(args.only)
        structures = [item for item in structures if item.label in wanted]

    from ase.data import atomic_numbers as ase_numbers

    template = args.template.read_text(encoding="utf-8")
    elements = sorted({symbol for item in structures for symbol in item.symbols})
    kinds = resolve_kinds(elements, data_dir=args.cp2k_data)
    valence = {
        symbol: int(spec.potential.rsplit("-q", 1)[1]) for symbol, spec in kinds.items()
    }

    input_dir = args.work_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    dataset = StrategyDataset()
    inputs: dict[str, dict[str, Path]] = {}
    strategies: dict[str, list] = {}

    for structure in structures:
        numbers = [int(ase_numbers[symbol]) for symbol in structure.symbols]
        detail = descriptors(structure)
        # A d-block metal means partial occupations at the Fermi level; without
        # smearing those systems do not converge at all, so the grid for them
        # is a different grid rather than a worse one.
        smearing = (
            structure.label in set(args.smeared)
            if args.smeared is not None
            else bool(set(numbers) & set(range(21, 31)) | set(numbers) & set(range(39, 49)))
        )
        dataset.features[structure.label] = structure_features(
            numbers,
            cell=structure.cell,
            periodic=structure.periodic,
            valence_electrons=sum(valence[symbol] for symbol in structure.symbols),
            min_distance=detail.get("min_distance_A"),
        )

        grid = strategy_grid(smearing=smearing)
        strategies[structure.label] = grid
        base = render_input(template, structure, kinds)
        for strategy in grid:
            path = input_dir / f"{structure.label}-{strategy.name}.inp"
            path.write_text(apply_strategy(base, strategy), encoding="utf-8")
            inputs.setdefault(structure.label, {})[strategy.name] = path

    # One arm per configuration. Systems that do not offer a configuration
    # simply have no run for it, which run_arms already handles.
    names = sorted({name for per_system in inputs.values() for name in per_system})
    arms = [GuessArm(label=name, guess=name) for name in names]

    print(f"{len(names)} configurations x {len(structures)} systems")
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

    for name, runs in results.items():
        for run in runs:
            dataset.outcomes.append(
                StrategyOutcome(
                    system=run.system,
                    strategy=name,
                    converged=run.converged,
                    scf_steps=run.scf_steps,
                    runtime_s=run.runtime_s,
                    energy=run.energy,
                    failure=run.failure,
                )
            )

    path = args.report or (args.work_dir / "scf_strategy_dataset.json")
    payload = dataset.to_dict()
    if args.evaluate:
        payload["leave_one_out"] = leave_one_out(dataset, k=args.k)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\n  {'system':22s}" + "".join(f"{name:>18s}" for name in names) + "   best")
    for system in dataset.systems():
        outcomes = dataset.by_system(system)
        row = f"  {system:22s}"
        for name in names:
            item = outcomes.get(name)
            if item is None:
                row += f"{'-':>18s}"
            else:
                row += f"{(str(item.scf_steps) if item.converged else 'FAIL'):>18s}"
        print(row + f"   {dataset.best_strategy(system)}")

    if args.evaluate:
        summary = payload["leave_one_out"]["summary"]
        print(f"\nleave-one-out over {summary['n_systems']} systems")
        print(f"  label accuracy        {summary['accuracy']:.0%}")
        print(f"  default total steps   {summary['default_total_steps']}")
        print(f"  predicted total steps {summary['recommended_total_steps']}")
        print(f"  oracle total steps    {summary['oracle_total_steps']}")
        print(f"  improved / degraded   {summary['n_improved']} / {summary['n_degraded']}")
        if "step_saving" in summary:
            print(f"  saving vs default     {summary['step_saving']:+.1%}")

    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
