# Ported from PHAL (hotspot_al.cp2k_ml_scf.cli) at 96263b6; MIT License, see NOTICE.
"""``scf`` command group (PHAL: ``phal cp2k-scf``).

Read-mostly: inspect, validate and export the SCF dataset, probe what a CP2K
binary can do, and compare recorded benchmark runs. The only command that
launches CP2K is ``capabilities --validate-native``, which runs one tiny H2
calculation to *validate* (not merely read from the banner) native SCF and
AO-matrix printing.

Standalone: ``python -m gmd_active_learning.adapters.dft.scf.cli --help``.
:func:`add_subparser` registers the group on the main ``gmd-al`` CLI (shared
file, owned elsewhere). ``benchmark`` needs the source-checkout tooling under
``tools/scf``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from gmd_active_learning.adapters.dft.scf.dataset import (
    DEFAULT_DATASET_ROOT,
    SPLITS,
    DatasetError,
    SCFDataset,
    ingest_scf_root,
)

DEFAULT_SCF_ROOT = Path("runtime/cp2k/scf")


def build_parser() -> argparse.ArgumentParser:
    """A standalone parser whose only command group is ``scf``."""

    parser = argparse.ArgumentParser(prog="gmd-scf")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


def add_subparser(subparsers: Any, *, name: str = "scf") -> argparse.ArgumentParser:
    """Register the SCF command group on an existing CLI."""

    parser = subparsers.add_parser(
        name,
        help="Inspect the CP2K ML-SCF dataset, capabilities, and benchmarks.",
        description=(
            "Tools for the ML-accelerated CP2K SCF workflow. CP2K remains the "
            "electronic structure engine; these commands only manage the data "
            "and provenance around it."
        ),
    )
    groups = parser.add_subparsers(dest="cp2k_scf_command", required=True)

    dataset_parser = groups.add_parser("dataset", help="Manage the SCF training dataset.")
    dataset_actions = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    init_parser = dataset_actions.add_parser("init", help="Create the dataset directory layout.")
    _add_root_argument(init_parser)

    ingest_parser = dataset_actions.add_parser(
        "ingest", help="Import recorded scf_history.json runs into the dataset."
    )
    _add_root_argument(ingest_parser)
    ingest_parser.add_argument(
        "--scf-root",
        default=None,
        help=f"Directory holding <task-id>/scf_history.json "
        f"(default: scf.recording.output_dir from --config, else {DEFAULT_SCF_ROOT}).",
    )
    ingest_parser.add_argument(
        "--task-root",
        default=None,
        help="Directory holding CP2K task dirs with region.extxyz (default: --scf-root).",
    )
    ingest_parser.add_argument("--overwrite", action="store_true", help="Replace samples that already exist.")
    _add_json_argument(ingest_parser)

    validate_parser = dataset_actions.add_parser("validate", help="Validate every sample in the dataset.")
    _add_root_argument(validate_parser)
    _add_split_argument(validate_parser)
    _add_json_argument(validate_parser)
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail on warnings (non-converged trajectories, growing residuals).",
    )

    inspect_parser = dataset_actions.add_parser("inspect", help="Summarise the dataset's contents.")
    _add_root_argument(inspect_parser)
    _add_split_argument(inspect_parser)
    _add_json_argument(inspect_parser)

    export_parser = dataset_actions.add_parser("export", help="Export training-ready artifacts.")
    _add_root_argument(export_parser)
    export_parser.add_argument("--output", required=True, help="Directory to write the export into.")
    export_parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Splits to export (default: all).",
    )
    export_parser.add_argument("--no-features", action="store_true", help="Skip the .npz feature archives.")
    export_parser.add_argument(
        "--require-converged",
        action="store_true",
        help="Drop trajectories CP2K did not converge.",
    )
    export_parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Abort instead of skipping records that fail validation.",
    )
    _add_json_argument(export_parser)

    capabilities_parser = groups.add_parser(
        "capabilities", help="Detect the installed CP2K version and ML-SCF support."
    )
    capabilities_parser.add_argument(
        "--executable", default=None, help="CP2K executable to probe (default: search PATH)."
    )
    capabilities_parser.add_argument("--config", default=None, help="Config file supplying the executable.")
    capabilities_parser.add_argument(
        "--require-ml-scf",
        action="store_true",
        help="Return a non-zero status unless ML-SCF support is detected.",
    )
    capabilities_parser.add_argument(
        "--validate-native",
        metavar="WORK_DIR",
        default=None,
        help="Run one tiny H2 calculation in WORK_DIR to validate native SCF and AO-matrix printing.",
    )
    _add_json_argument(capabilities_parser)

    benchmark_parser = groups.add_parser(
        "benchmark", help="Compare native CP2K against the ML-accelerated modes."
    )
    benchmark_parser.add_argument(
        "--runs",
        required=True,
        help="JSON file describing the recorded runs to compare (see docs/cp2k_ml_architecture.md).",
    )
    benchmark_parser.add_argument("--output", default=None, help="Write the benchmark report JSON here.")
    benchmark_parser.add_argument(
        "--results-dir",
        default=None,
        help="Write benchmark_results.json into this directory.",
    )
    benchmark_parser.add_argument(
        "--energy-tolerance", type=float, default=None, help="Acceptance tolerance in Hartree."
    )
    benchmark_parser.add_argument(
        "--force-tolerance", type=float, default=None, help="Acceptance tolerance in Hartree/Bohr."
    )
    benchmark_parser.add_argument(
        "--require-accepted",
        action="store_true",
        help="Return a non-zero status unless every accelerated mode is accepted.",
    )
    _add_json_argument(benchmark_parser)

    return parser


def run(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``cp2k-scf`` invocation."""

    command = getattr(args, "cp2k_scf_command", None)
    if command == "dataset":
        return _run_dataset(args)
    if command == "capabilities":
        return _run_capabilities(args)
    if command == "benchmark":
        return _run_benchmark(args)
    print(f"unknown cp2k-scf command: {command}")
    return 2


# -- dataset commands ---------------------------------------------------


def _run_dataset(args: argparse.Namespace) -> int:
    action = args.dataset_command
    if getattr(args, "root", None) is None:
        # scf.dataset.root (PHAL's cp2k.ml_scf.dataset_dir was declared but never read).
        from gmd_active_learning.adapters.dft.scf.config import scf_config

        args.root = scf_config(_load_config(getattr(args, "config", None)) or None).dataset.root
    if action == "init":
        dataset = SCFDataset.create(args.root, split_ratios=_split_ratios(args))
        print(f"initialised SCF dataset at {dataset.root}")
        for split in SPLITS:
            print(f"  {split}/")
        return 0

    if action == "ingest":
        return _run_ingest(args)

    try:
        dataset = SCFDataset.open(args.root)
    except DatasetError as exc:
        print(f"error: {exc}")
        return 2

    if action == "validate":
        return _run_validate(dataset, args)
    if action == "inspect":
        return _run_inspect(dataset, args)
    if action == "export":
        return _run_export(dataset, args)
    print(f"unknown dataset command: {action}")
    return 2


def _run_ingest(args: argparse.Namespace) -> int:
    from gmd_active_learning.adapters.dft.scf.config import scf_config

    config = _load_config(args.config)
    settings = scf_config(config or None)
    dataset = SCFDataset.create(args.root, split_ratios=settings.dataset.split_ratios)
    summary = ingest_scf_root(
        args.scf_root or settings.recording.output_dir,
        dataset,
        task_root=args.task_root,
        config=config,
        overwrite=args.overwrite,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"ingested {summary['n_ingested']} run(s) into {dataset.root}")
        for skipped in summary["skipped"]:
            print(f"  skipped {skipped['task_id']}: {skipped['reason']}")
    return 0


def _run_validate(dataset: SCFDataset, args: argparse.Namespace) -> int:
    reports = dataset.validate(args.split)
    failed = [report for report in reports if not report.ok]
    warned = [report for report in reports if report.ok and report.warnings]

    if args.json:
        print(
            json.dumps(
                {
                    "root": str(dataset.root),
                    "split": args.split,
                    "strict": bool(args.strict),
                    **_summarize(reports),
                },
                indent=2,
            )
        )
    else:
        print(f"validated {len(reports)} sample(s) in {dataset.root}")
        for report in failed:
            print(f"FAIL {report.subject}")
            for issue in report.errors:
                print(f"  {issue}")
        for report in warned:
            print(f"WARN {report.subject}")
            for issue in report.warnings:
                print(f"  {issue}")
        print(f"{len(reports) - len(failed)} ok, {len(failed)} failed, {len(warned)} with warnings")

    if failed:
        return 1
    if args.strict and warned:
        return 1
    return 0


def _run_inspect(dataset: SCFDataset, args: argparse.Namespace) -> int:
    summary = dataset.inspect(args.split)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"dataset: {summary['root']}")
    print(f"  dataset_version: {summary['dataset_version']}")
    print(f"  schema_version:  {summary['schema_version']}")
    counts = summary["counts"]
    print(f"  samples: {summary['n_total']} total " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if summary["n_undecodable"]:
        print(f"  undecodable: {summary['n_undecodable']}")
        for detail in summary["undecodable"]:
            print(f"    {detail}")
    atoms = summary["atoms"]
    iterations = summary["scf_iterations"]
    print(f"  atoms/sample: min={atoms['min']} max={atoms['max']} mean={atoms['mean']}")
    print(f"  scf iterations: min={iterations['min']} max={iterations['max']} mean={iterations['mean']}")
    print(f"  converged fraction: {summary['converged_fraction']:.3f}")
    print(f"  initial guesses: {summary['initial_guess_counts']}")
    print(f"  samples with density artifacts: {summary['n_with_density_artifacts']}")
    return 0


def _run_export(dataset: SCFDataset, args: argparse.Namespace) -> int:
    from gmd_active_learning.adapters.dft.scf.export import ExportError, export_dataset

    try:
        result = export_dataset(
            dataset,
            args.output,
            splits=tuple(args.splits),
            include_features=not args.no_features,
            skip_invalid=not args.fail_on_invalid,
            require_converged=args.require_converged,
        )
    except ExportError as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"exported to {result.output_dir}")
    for split, count in result.counts.items():
        print(f"  {split}: {count} sample(s)")
    if result.n_rejected:
        print(f"  rejected {result.n_rejected} sample(s):")
        for detail in result.rejected:
            print(f"    {detail}")
    print(f"  manifest: {result.manifest_path}")
    return 0


# -- capabilities and benchmark ----------------------------------------


def _run_capabilities(args: argparse.Namespace) -> int:
    from gmd_active_learning.adapters.dft.scf.capabilities import detect_cp2k_capabilities

    config = _load_config(args.config)
    capabilities = detect_cp2k_capabilities(executable=args.executable, config=config or None)
    if args.validate_native:
        from gmd_active_learning.adapters.dft.scf.capabilities import run_native_validation

        capabilities = run_native_validation(capabilities, args.validate_native, config=config or None)

    if args.json:
        print(json.dumps(capabilities.to_dict(), indent=2))
    else:
        print(f"executable:   {capabilities.executable or 'not found'}")
        print(f"resolved:     {capabilities.resolved_path or '-'}")
        print(f"version:      {capabilities.version or 'unknown'}")
        print(f"git commit:   {capabilities.git_commit or 'unknown'}")
        print(f"ml-scf:       {'yes' if capabilities.ml_scf_supported else 'no'}")
        print(f"ml mixing:    {'yes' if capabilities.ml_mixing_supported else 'no'}")
        print(f"torch (cp2k): {'yes' if capabilities.torch_supported else 'no'}")
        print(f"torch (py):   {'yes' if capabilities.python_torch_available else 'no'}")
        print(f"contract:     {capabilities.contract_version}")
        for name, status in sorted(capabilities.features.items()):
            state = "validated" if status.validated else ("advertised" if status.advertised else "unsupported")
            print(f"  {name:20s} {state:11s} {status.evidence}")
        for detail in capabilities.details:
            print(f"  {detail}")

    if args.require_ml_scf and not capabilities.ml_scf_supported:
        return 1
    return 0 if capabilities.available else 1


def _run_benchmark(args: argparse.Namespace) -> int:
    try:
        from tools.scf.benchmark.modes import (
            MODE_NATIVE,
            AcceptanceCriteria,
            BenchmarkError,
            load_runs,
            run_benchmark,
            write_benchmark_results,
        )
    except ImportError as exc:
        print(f"error: the benchmark needs the tools/scf tooling of a source checkout on sys.path ({exc})")
        return 2

    overrides: dict[str, float] = {}
    if args.energy_tolerance is not None:
        overrides["energy_tolerance"] = args.energy_tolerance
    if args.force_tolerance is not None:
        overrides["force_tolerance"] = args.force_tolerance
    criteria = AcceptanceCriteria(**overrides)

    try:
        runs = load_runs(args.runs)
        report = run_benchmark(runs, criteria=criteria)
    except BenchmarkError as exc:
        print(f"error: {exc}")
        return 1

    payload = report.to_dict()
    written: list[str] = []
    if args.output:
        from gmd_active_learning.utils.atomic_io import write_json_atomic

        write_json_atomic(args.output, payload)
        written.append(str(args.output))
    if args.results_dir:
        written.append(str(write_benchmark_results(report, args.results_dir)))

    if args.json or not written:
        print(json.dumps(payload, indent=2))
    else:
        for path in written:
            print(f"wrote {path}")
        for result in report.results:
            status = "ACCEPT" if result.accepted else "REJECT"
            print(f"  [{status}] {result.mode}: iterations={result.iterations} speedup={result.speedup}")
            for reason in result.rejection_reasons:
                print(f"      {reason}")

    if args.require_accepted:
        # The baseline is the reference, not a candidate, so it is exempt.
        rejected = [
            result.mode
            for result in report.results
            if result.mode != MODE_NATIVE and not result.accepted
        ]
        if rejected:
            print(f"rejected mode(s): {', '.join(rejected)}")
            return 1
    return 0


# -- helpers ------------------------------------------------------------


def _add_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        default=None,
        help=f"Dataset root directory (default: scf.dataset.root from --config, else {DEFAULT_DATASET_ROOT}).",
    )
    if not any(action.dest == "config" for action in parser._actions):
        parser.add_argument("--config", default=None, help="YAML config with an scf section.")


def _add_split_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        default=None,
        choices=list(SPLITS),
        help="Restrict to one split (default: all).",
    )


def _add_json_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def _split_ratios(args: argparse.Namespace) -> dict[str, float]:
    """``scf.dataset.split_ratios`` from ``--config`` (defaults otherwise)."""

    from gmd_active_learning.adapters.dft.scf.config import scf_config

    return dict(scf_config(_load_config(getattr(args, "config", None)) or None).dataset.split_ratios)


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    from gmd_active_learning.adapters.dft.scf.config import load_config

    return load_config(path)


def _summarize(reports: Sequence[Any]) -> dict[str, Any]:
    from gmd_active_learning.adapters.dft.scf.validation import summarize

    summary = summarize(reports)
    summary["warned"] = [report.to_dict() for report in reports if report.ok and report.warnings]
    return summary


__all__ = ["DEFAULT_SCF_ROOT", "add_subparser", "build_parser", "main", "run"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
