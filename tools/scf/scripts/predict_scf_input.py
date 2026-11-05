#!/usr/bin/env python3
# Ported from PHAL (scripts.predict_scf_input) at 96263b6; MIT License, see NOTICE.
"""Write a CP2K input with SCF settings chosen for the structure.

    python -m tools.scf.scripts.predict_scf_input.py \\
        --structure system.xyz \\
        --template configs/cp2k/strategy.inp.template \\
        --cp2k-data ../cp2k-ml-scf/cp2k/data \\
        --dataset runtime/cp2k_ml/validation/scf_strategy_dataset.json \\
        --output system.inp

Predicts, renders, and stops. CP2K runs the input and remains the authority on
the result; nothing here inspects or overrides one.

Safety, in order:

* a structure with no similar system in the dataset gets CP2K's defaults;
* a split vote gets CP2K's defaults;
* a recommendation whose nearest neighbours *failed* under it gets CP2K's
  defaults.

The default is never a regression: it is the configuration everything else is
measured against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.scf.strategy_predictor import (
    DEFAULT_SMEARED,
    DEFAULT_STRATEGY,
    StrategyDataset,
    StrategyPredictor,
    apply_strategy,
    strategy_grid,
    structure_features,
)
from tools.scf.structures import (
    descriptors,
    read_structures,
    render_input,
    resolve_kinds,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--cp2k-data", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--smeared", action="store_true", help="force smearing and empty states")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    structures = read_structures(args.structure)
    structure = structures[args.index]

    from ase.data import atomic_numbers as ase_numbers

    numbers = [int(ase_numbers[symbol]) for symbol in structure.symbols]
    kinds = resolve_kinds(sorted(set(structure.symbols)), data_dir=args.cp2k_data)
    valence = sum(int(kinds[symbol].potential.rsplit("-q", 1)[1]) for symbol in structure.symbols)
    detail = descriptors(structure)

    smearing = args.smeared or bool(
        set(numbers) & (set(range(21, 31)) | set(range(39, 49)))
    )
    features = structure_features(
        numbers,
        cell=structure.cell,
        periodic=structure.periodic,
        valence_electrons=valence,
        min_distance=detail.get("min_distance_A"),
    )

    dataset = StrategyDataset.from_dict(json.loads(args.dataset.read_text(encoding="utf-8")))
    predictor = StrategyPredictor(k=args.k, min_confidence=args.min_confidence).fit(dataset)
    recommendation = predictor.recommend(features)

    grid = {item.name: item for item in strategy_grid(smearing=smearing)}
    fallback = DEFAULT_SMEARED if smearing else DEFAULT_STRATEGY
    strategy = grid.get(recommendation.strategy)

    note = recommendation.reason
    if strategy is None:
        # The recommendation names a configuration this system cannot run --
        # a non-smeared setting for a metal, say. Defaults, and say why.
        strategy = fallback
        note = (
            f"{recommendation.strategy!r} is not available for this system "
            f"({'smeared' if smearing else 'unsmeared'} grid); using CP2K's defaults"
        )
    elif not _neighbours_succeeded(dataset, recommendation, strategy.name):
        strategy = fallback
        note = (
            f"{recommendation.strategy!r} failed on the systems it was recommended from; "
            "using CP2K's defaults"
        )

    base = render_input(args.template.read_text(encoding="utf-8"), structure, kinds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(apply_strategy(base, strategy), encoding="utf-8")

    sidecar = args.output.with_suffix(args.output.suffix + ".strategy.json")
    sidecar.write_text(
        json.dumps(
            {
                "system": structure.label,
                "features": features,
                "recommendation": recommendation.to_dict(),
                "applied": strategy.to_dict(),
                "note": note,
                "smearing": smearing,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"{structure.label}: {strategy.name} (alpha {strategy.alpha}, {strategy.mixing_method})")
    print(f"  {note}")
    print(f"wrote {args.output}")
    return 0


def _neighbours_succeeded(dataset: StrategyDataset, recommendation, strategy: str) -> bool:
    """Did this configuration actually converge on the systems it came from?

    A vote can name a configuration that was the *best available* on a
    neighbour where everything struggled. Recommending it onward would spread
    one system's bad luck.
    """

    if not recommendation.neighbours:
        return False
    for neighbour in recommendation.neighbours:
        outcome = dataset.by_system(neighbour).get(strategy)
        if outcome is not None and outcome.converged:
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
