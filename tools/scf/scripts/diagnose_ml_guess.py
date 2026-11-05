# Ported from PHAL (scripts.diagnose_ml_guess) at 96263b6; MIT License, see NOTICE.
#!/usr/bin/env python3
"""Measure why the ML initial guess is slower than the atomic one.

    scripts/diagnose_ml_guess.py \\
        --structures data/structures/benchmark/molecules.extxyz \\
        --template configs/cp2k/diagnostic_guess.inp.template \\
        --cp2k-data ../cp2k-ml-scf/cp2k/data \\
        --binary build/cp2k-ml/bin/cp2k.psmp \\
        --guess-model runtime/cp2k_ml/models/active/guess.ts.pt \\
        --training-systems runtime/cp2k_ml/validation/density_guess_training.json

Runs each system under both guesses with the density printed every iteration,
then reports for each: ``||P_guess - P_final||`` for both guesses, the residual
and energy path, and which of the four candidate causes the numbers support.

Writes ml_density_failure_report.json. Trains nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gmd_active_learning.adapters.dft.scf.execution import LocalSchedulerBackend
from gmd_active_learning.adapters.dft.scf.ao_matrices import parse_density_trajectory, parse_overlap_matrix
from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history
from tools.scf.difficulty import assess_sample, parse_abort
from tools.scf.benchmark.guess import (
    GUESS_ATOMIC,
    GUESS_ML,
    run_benchmark,
)
from tools.scf.diagnostics.guess import (
    REPORT_NAME,
    SystemDiagnosis,
    classify,
    density_distance,
    diagnose_set,
    trajectory_metrics,
    write_report,
)
from tools.scf.structures import read_collection, render_input, resolve_kinds


def _formula(symbols) -> str:
    """The same formula string the dataset records, for membership matching."""

    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return "".join(f"{element}{counts[element]}" for element in sorted(counts))


def read_run(path: Path) -> dict:
    """Everything the diagnosis needs from one CP2K output."""

    import re

    text = path.read_text(encoding="utf-8", errors="replace")
    frames = parse_density_trajectory(text)
    overlap = parse_overlap_matrix(text)
    history = parse_scf_history(text)
    steps = None
    match = re.search(r"SCF run converged in\s+(\d+)\s+steps", text)
    if match:
        steps = int(match.group(1))

    source = re.search(r"ML-SCF\|\s*ml_guess_source:\s*(\S+)", text)
    accepted = re.search(r"ML-SCF\|\s*prediction_success:\s*([TF])", text)
    reason = re.search(r"ML-SCF\|\s*fallback used:\s*(.+)", text)

    return {
        "frames": frames,
        "overlap": overlap,
        "guess_source": source.group(1) if source else None,
        "guess_accepted": bool(accepted and accepted.group(1) == "T"),
        "fallback_reason": reason.group(1).strip() if reason else None,
        "residuals": [item.residual for item in history.iterations],
        "energies": [item.energy for item in history.iterations],
        "scf_steps": steps,
        "converged": steps is not None,
        "failure": None if steps is not None else parse_abort(text),
        "text": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--structures", required=True, type=Path, nargs="+")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--cp2k-data", required=True, type=Path)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--guess-model", required=True, type=Path)
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=Path("runtime/cp2k/pulay_dataset"),
        help="dataset the model was fitted on; membership is matched by chemistry",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("runtime/cp2k_ml/diagnosis"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="systems in flight at once; each is two CP2K processes",
    )
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()

    structures = read_collection(args.structures)
    if args.only:
        wanted = set(args.only)
        structures = [item for item in structures if item.label in wanted]

    # Membership is matched by chemical formula, not by name: the benchmark
    # calls it "h2o" and the dataset calls it "mol-h2o", and what the
    # generalisation question actually asks is whether this *chemistry* was
    # fitted on, not whether this string was.
    fitted_formulas: set[str] = set()
    if args.training_dataset and args.training_dataset.is_dir():
        for sample_file in sorted(args.training_dataset.glob("*/sample.json")):
            try:
                payload = json.loads(sample_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            formula = (payload.get("structure") or {}).get("formula")
            if formula:
                fitted_formulas.add(str(formula))

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

    # In chunks. Every system here prints its density each iteration, so the
    # whole set at once is 2N processes each holding several nao^2 buffers; the
    # first attempt lost three runs to the machine rather than to the physics,
    # and a truncated output looks exactly like a system that did not converge.
    labels = list(inputs)
    for start in range(0, len(labels), max(1, args.max_parallel)):
        chunk = labels[start : start + max(1, args.max_parallel)]
        print(f"running {', '.join(chunk)}")
        run_benchmark(
            {label: inputs[label] for label in chunk},
            binary=args.binary,
            scheduler=LocalSchedulerBackend(),
            work_dir=args.work_dir,
            guess_model=args.guess_model,
            timeout=args.timeout,
            poll_interval=2.0,
        )

    diagnoses: list[SystemDiagnosis] = []
    for structure in structures:
        system = structure.label
        runs = {}
        for guess in (GUESS_ATOMIC, GUESS_ML):
            output = args.work_dir / f"{system}-{guess.lower()}" / "cp2k.out"
            runs[guess] = read_run(output) if output.is_file() else None

        atomic, ml = runs[GUESS_ATOMIC], runs[GUESS_ML]
        notes: dict = {}
        distances: dict = {GUESS_ATOMIC: None, GUESS_ML: None}
        metrics: dict = {GUESS_ATOMIC: None, GUESS_ML: None}

        # The converged density of the reference run is the target for both:
        # comparing each guess against its own run's endpoint would measure two
        # different things and could not be compared.
        reference_final = None
        if atomic and atomic["frames"] and atomic["overlap"] is not None:
            reference_final = atomic["frames"][-1]
            n_electrons = float((reference_final * atomic["overlap"]).sum())
            notes["n_electrons"] = n_electrons

            for guess, run in ((GUESS_ATOMIC, atomic), (GUESS_ML, ml)):
                if not run or not run["frames"]:
                    continue
                # Frame 0 is the density CP2K started from.
                distances[guess] = density_distance(
                    run["frames"][0], reference_final, atomic["overlap"], n_electrons=n_electrons
                )
                metrics[guess] = trajectory_metrics(run["residuals"], run["energies"])

            if ml and ml["frames"]:
                # Did the two runs converge to the same place? If not, the
                # distances are being measured against different answers.
                drift = float(np.abs(ml["frames"][-1] - reference_final).max())
                notes["converged_density_drift"] = drift

        difficulty = None
        sample_dir = args.work_dir / f"{system}-atomic"
        if sample_dir.is_dir():
            difficulty = assess_sample(sample_dir, sample_id=system).difficulty

        # A run whose ML arm fell back is the atomic run twice over; saying
        # anything about the model from it would be inventing evidence.
        ml_used = bool(ml and ml["guess_source"] == "torchscript" and ml["guess_accepted"])
        if ml and not ml_used:
            notes["ml_guess_source"] = ml["guess_source"]
            notes["ml_fallback_reason"] = ml["fallback_reason"]
        both_converged = bool(atomic and atomic["converged"] and ml and ml["converged"])
        if not both_converged:
            notes["convergence"] = {
                "atomic": bool(atomic and atomic["converged"]),
                "ml": bool(ml and ml["converged"]),
            }

        classification, evidence = classify(
            atomic_distance=distances[GUESS_ATOMIC],
            ml_distance=distances[GUESS_ML],
            atomic_trajectory=metrics[GUESS_ATOMIC],
            ml_trajectory=metrics[GUESS_ML],
            scf_steps_atomic=atomic["scf_steps"] if atomic else None,
            scf_steps_ml=ml["scf_steps"] if ml else None,
            ml_guess_used=ml_used,
            both_converged=both_converged,
        )

        diagnoses.append(
            SystemDiagnosis(
                system=system,
                difficulty=difficulty,
                in_training_set=(
                    _formula(structure.symbols) in fitted_formulas if fitted_formulas else None
                ),
                atomic_distance=distances[GUESS_ATOMIC],
                ml_distance=distances[GUESS_ML],
                scf_steps_atomic=atomic["scf_steps"] if atomic else None,
                scf_steps_ml=ml["scf_steps"] if ml else None,
                atomic_trajectory=metrics[GUESS_ATOMIC],
                ml_trajectory=metrics[GUESS_ML],
                classification=classification,
                evidence=tuple(evidence),
                notes=notes,
            )
        )

    path = write_report(diagnoses, args.report or (args.work_dir / REPORT_NAME))

    header = f"  {'system':14s} {'diff':7s} {'fit':4s} {'d_atomic':>9s} {'d_ml':>9s} {'steps':>9s}"
    print(header)
    for item in diagnoses:
        atomic_d = f"{item.atomic_distance.metric:9.4f}" if item.atomic_distance else "        -"
        ml_d = f"{item.ml_distance.metric:9.4f}" if item.ml_distance else "        -"
        steps = f"{item.scf_steps_atomic}->{item.scf_steps_ml}"
        fitted = "-" if item.in_training_set is None else ("yes" if item.in_training_set else "no")
        print(
            f"  {item.system:14s} {str(item.difficulty):7s} {fitted:4s} {atomic_d} {ml_d} "
            f"{steps:>9s}  {item.classification}"
        )

    summary = diagnose_set(diagnoses)
    print(f"\nverdict: {summary['verdict']}")
    for reason in summary["reasons"]:
        print(f"  {reason}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
