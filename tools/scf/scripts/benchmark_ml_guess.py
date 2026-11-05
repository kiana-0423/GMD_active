# Ported from PHAL (scripts.benchmark_ml_guess) at 96263b6; MIT License, see NOTICE.
#!/usr/bin/env python3
"""Benchmark the ML initial density guess against CP2K's atomic guess.

    scripts/benchmark_ml_guess.py \\
        --structures data/structures/benchmark/molecules.extxyz \\
        --template configs/cp2k/benchmark_guess.inp.template \\
        --cp2k-data ../cp2k-ml-scf/cp2k/data \\
        --binary build/cp2k-ml/bin/cp2k.psmp \\
        --guess-model runtime/cp2k_ml/models/active/guess.ts.pt

One system, two runs, one difference: ``SCF_GUESS ATOMIC`` vs ``SCF_GUESS ML``.
Without ``--guess-model`` only the atomic arm runs, which is a baseline, not a
comparison -- and the report says so rather than implying a result.

Writes benchmark_ml_guess.json: per run the system, guess, SCF steps, runtime,
energy error, force error and convergence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gmd_active_learning.adapters.dft.scf.execution import resolve_scheduler
from tools.scf.benchmark.guess import (
    GUESS_ATOMIC,
    GUESS_ML,
    REPORT_NAME,
    compare,
    run_benchmark,
    write_report,
)
from tools.scf.structures import (
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
    parser.add_argument("--guess-model", type=Path, help="deployed TorchScript density model")
    parser.add_argument("--work-dir", type=Path, default=Path("runtime/cp2k_ml/benchmark"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--scheduler",
        default="local",
        help="'local' or a 'module:Class' path to a GMD_active scheduler adapter (Slurm/PBS)",
    )
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--only", nargs="+", help="restrict to these system labels")
    args = parser.parse_args()

    structures = read_collection(args.structures)
    if args.only:
        wanted = set(args.only)
        structures = [item for item in structures if item.label in wanted]
        if not structures:
            parser.error(f"no structure matched {sorted(wanted)}")

    template = args.template.read_text(encoding="utf-8")
    elements = sorted({symbol for item in structures for symbol in item.symbols})
    kinds = resolve_kinds(elements, data_dir=args.cp2k_data)

    guesses = [GUESS_ATOMIC] + ([GUESS_ML] if args.guess_model else [])
    input_dir = args.work_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    inputs: dict[str, dict[str, Path]] = {}
    for structure in structures:
        for guess in guesses:
            path = input_dir / f"{structure.label}-{guess.lower()}.inp"
            path.write_text(render_input(template, structure, kinds, guess=guess), encoding="utf-8")
            inputs.setdefault(structure.label, {})[guess] = path

    runs = run_benchmark(
        inputs,
        binary=args.binary,
        scheduler=resolve_scheduler(args.scheduler),
        work_dir=args.work_dir,
        guess_model=args.guess_model,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
    )

    report = compare(runs)
    if not args.guess_model:
        report["note"] = (
            "Baseline only: no --guess-model was given, so the ML arm did not run. "
            "These numbers are the reference to beat, not a comparison. "
        ) + str(report["note"])

    path = write_report(report, args.report or (args.work_dir / REPORT_NAME))

    for entry in report["runs"]:
        mark = "ok " if entry["converged"] else "FAIL"
        steps = entry["scf_steps"] if entry["scf_steps"] is not None else "-"
        source = entry.get("ml_guess_source") or "-"
        print(
            f"  {mark} {entry['system']:12s} {entry['initial_guess']:6s} "
            f"steps={str(steps):>4s} {entry['runtime']:8.1f}s "
            f"src={source:11s} dE={_fmt(entry['energy_error'])} dF={_fmt(entry['force_error'])}"
            + ("" if entry["converged"] else f"  {entry['failure']}")
        )
    print(f"\n{json.dumps(report['summary'], indent=1)}")
    print(f"wrote {path}")

    # A fallback run is not a slow result, it is no result: the ML arm measured
    # the baseline. Failing here is what stops that becoming a reported speedup.
    if args.guess_model and not report["valid"]:
        print("\nBENCHMARK INVALID: ML was enabled but the guess fell back")
        for warning in report["warnings"]:
            print(f"  {warning}")
        return 2

    return 0 if any(entry["converged"] for entry in report["runs"]) else 1


def _fmt(value: float | None) -> str:
    return "      -" if value is None else f"{value:.1e}"


if __name__ == "__main__":
    raise SystemExit(main())
