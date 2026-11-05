# Ported from PHAL (scripts.generate_pulay_dataset) at 96263b6; MIT License, see NOTICE.
#!/usr/bin/env python3
"""Generate a CP2K Pulay-matrix dataset.

Three ways to say what to run, in increasing order of convenience:

    # an explicit list of CP2K inputs
    scripts/generate_pulay_dataset.py --tasks tasks.json --binary .../cp2k.psmp

    # a structure collection (xyz / extxyz / cif) plus one input template
    scripts/generate_pulay_dataset.py \\
        --structures data/structures/pulay_sampling/small_molecules.extxyz \\
        --template configs/cp2k/pulay_dataset.inp.template \\
        --cp2k-data ../cp2k-ml-scf/cp2k/data --binary .../cp2k.psmp

    # a sampling plan: several families, each with the settings that suit it
    scripts/generate_pulay_dataset.py \\
        --plan configs/cp2k/pulay_sampling_plan.json \\
        --cp2k-data ../cp2k-ml-scf/cp2k/data --binary .../cp2k.psmp

All tasks are submitted before any is polled, so ``--scheduler slurm`` or
``pbs`` runs them concurrently on a cluster; ``local`` runs them as concurrent
subprocesses. A task that fails becomes a rejected sample with a reason, not an
exception -- one bad geometry must not discard the rest.

Generates and validates only. Training is a separate step, deliberately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gmd_active_learning.adapters.dft.scf.execution import resolve_scheduler
from tools.scf.pulay_dataset import (
    generate_dataset,
    load_task_list,
    write_dataset_report,
)



def build_tasks(args: argparse.Namespace) -> list:
    if args.tasks:
        return load_task_list(args.tasks)

    # Imported lazily: this is the only path that needs ASE, and --tasks should
    # keep working in an environment without it.
    from tools.scf.structures import tasks_from_collection, tasks_from_plan

    # Beside the dataset, not inside it: every directory under the root is a
    # sample, and a directory of rendered inputs would be validated as one.
    input_dir = args.input_dir or (args.root.parent / f"{args.root.name}_inputs")
    if args.plan:
        return tasks_from_plan(
            args.plan,
            input_dir=input_dir,
            data_dir=args.cp2k_data,
            groups=args.group or None,
            vacuum=args.vacuum,
        )
    return tasks_from_collection(
        args.structures,
        template_path=args.template,
        input_dir=input_dir,
        data_dir=args.cp2k_data,
        vacuum=args.vacuum,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasks", type=Path, help="explicit task list (JSON)")
    source.add_argument("--plan", type=Path, help="sampling plan (JSON)")
    source.add_argument(
        "--structures", type=Path, nargs="+", help="xyz / extxyz / cif collection files"
    )
    parser.add_argument("--template", type=Path, help="CP2K input template, required with --structures")
    parser.add_argument("--cp2k-data", type=Path, help="CP2K data directory (BASIS_MOLOPT, POTENTIAL)")
    parser.add_argument("--group", action="append", help="restrict a plan to these groups")
    parser.add_argument("--input-dir", type=Path, help="where rendered inputs go (default <root>/inputs)")
    parser.add_argument("--vacuum", type=float, default=6.0, help="padding for non-periodic systems (A)")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--root", type=Path, default=Path("runtime/cp2k/pulay_dataset"))
    parser.add_argument(
        "--scheduler",
        default="local",
        help="'local' or a 'module:Class' path to a GMD_active scheduler adapter (Slurm/PBS)",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    args = parser.parse_args()

    if args.structures and not args.template:
        parser.error("--structures requires --template")
    if (args.structures or args.plan) and not args.cp2k_data:
        parser.error("--structures/--plan require --cp2k-data")

    tasks = build_tasks(args)
    config = {
        "cp2k": {
            "scf_recording": {"enabled": True, "record_density": True, "record_trajectory": True},
        }
    }

    results = generate_dataset(
        tasks,
        root=args.root,
        binary=args.binary,
        scheduler=resolve_scheduler(args.scheduler),
        config=config,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )

    accepted = [result for result in results if result.accepted]
    for result in results:
        mark = "ok " if result.accepted else "REJ"
        detail = "" if result.accepted else f"  {result.reasons[0]}"
        print(f"  {mark} {result.sample_id:28s} frames={result.n_frames:3d} "
              f"pulay={result.n_pulay:3d} nao={result.n_ao:3d}{detail}")

    report_path = write_dataset_report(args.root, results)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    chemistry = report["chemistry"]
    print(f"\n{len(accepted)}/{len(results)} accepted; "
          f"{report['iterations']['n_pulay_matrices']} Pulay matrices; "
          f"{chemistry['n_distinct_elements']} elements, "
          f"{chemistry['n_distinct_formulas']} formulas, "
          f"{chemistry['n_periodic']} periodic")
    print(f"wrote {report_path}")

    # A dataset with nothing in it is a failure, however cleanly it ran.
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
