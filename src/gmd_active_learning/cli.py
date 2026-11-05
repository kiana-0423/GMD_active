"""``gmd-al`` command line interface.

    gmd-al validate-config --config run.yaml
    gmd-al doctor          --config run.yaml
    gmd-al init            --config run.yaml --work-dir runs/a --checkpoint ckpt_best.pt --export model.pt [--committee m1.pt ...]
    gmd-al run             --config run.yaml --work-dir runs/a
    gmd-al status          --work-dir runs/a [--run-id ID]
    gmd-al stop | retrain  --config run.yaml --work-dir runs/a
    gmd-al rollback        --config run.yaml --work-dir runs/a --reason "..." [--to MODEL_ID]
    gmd-al resolve-unknown --config run.yaml --work-dir runs/a --task TASK_ID --reason "..."
    gmd-al train-reliability-model --dataset features.npz --output-dir artifacts/
    gmd-al scf ...         (experimental CP2K ML-SCF tooling)

Replaces PHAL's ``hotspot-al``/``phal`` CLI (ledger CLI-001/002) and GMD_active's former CLI, whose
``run --config`` argument was ignored and whose ``import-labels`` was a placeholder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from gmd_active_learning.core.exceptions import GMDActiveLearningError
from gmd_active_learning.utils.logging import configure_logging


def _add_config(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument("--config", type=Path, required=required, help="YAML configuration (see configs/example_run.yaml)")


def _add_work_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, required=True, help="persistent run directory (state, models, datasets)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gmd-al", description="GMD active-learning controller")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_config(sub.add_parser("validate-config", help="validate a configuration file"))
    _add_config(sub.add_parser("doctor", help="check that every configured backend is usable"))

    init = sub.add_parser("init", help="register and promote the initial model of a run")
    _add_config(init)
    _add_work_dir(init)
    init.add_argument("--checkpoint", type=Path, required=True)
    init.add_argument("--export", type=Path, required=True, help="TorchScript export GMD will run")
    init.add_argument("--committee", type=Path, nargs="*", default=[], help="extra committee member checkpoints")
    init.add_argument("--dataset-version", default=None, help="published dataset the model was trained on, if any")

    run = sub.add_parser("run", help="advance the persistent workflow (resumes after interruption)")
    _add_config(run)
    _add_work_dir(run)

    status = sub.add_parser("status", help="print the run report as JSON")
    _add_work_dir(status)
    status.add_argument("--run-id", default=None)

    for name, help_text in (("stop", "request a cooperative stop"), ("retrain", "request retraining at the next decision point")):
        command = sub.add_parser(name, help=help_text)
        _add_config(command)
        _add_work_dir(command)

    rollback = sub.add_parser("rollback", help="re-activate the previous (or a given) promoted model")
    _add_config(rollback)
    _add_work_dir(rollback)
    rollback.add_argument("--reason", required=True)
    rollback.add_argument("--to", default=None, help="model id to restore")

    unknown = sub.add_parser("resolve-unknown", help="declare a lost job failed after manual reconciliation")
    _add_config(unknown)
    _add_work_dir(unknown)
    unknown.add_argument("--task", required=True)
    unknown.add_argument("--reason", required=True)

    reliability = sub.add_parser("train-reliability-model", help="train the optional reliability classifier")
    reliability.add_argument("--dataset", type=Path, required=True, help="NPZ with features/labels (no synthetic fallback)")
    reliability.add_argument("--output-dir", type=Path, required=True)

    try:
        from gmd_active_learning.adapters.dft.scf.cli import add_subparser as add_scf
    except ImportError:  # pragma: no cover - optional tooling
        pass
    else:
        add_scf(sub, name="scf")
    return parser


def _controller(args: argparse.Namespace) -> Any:
    from gmd_active_learning.config import load_config
    from gmd_active_learning.workflow.assembly import build_controller

    return build_controller(load_config(args.config), work_dir=args.work_dir)


def _store(work_dir: Path) -> Any:
    from gmd_active_learning.workflow.store import WorkflowStore

    path = work_dir / "workflow.sqlite"
    if not path.is_file():
        raise GMDActiveLearningError(f"no workflow state in {work_dir}")
    return WorkflowStore(path)


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=1, default=str))


def _doctor(args: argparse.Namespace) -> int:
    from gmd_active_learning.adapters.gmd.engine import GMDEngine
    from gmd_active_learning.adapters.registry import runtime_report
    from gmd_active_learning.adapters.sgt.backend import import_gmd_sgt
    from gmd_active_learning.config import load_config

    config = load_config(args.config)
    rows: list[dict[str, Any]] = []
    for name, check in (("md:gmd", lambda: GMDEngine(config["md"])), ("mlip:gmd_sgt", lambda: import_gmd_sgt(config["mlip"]))):  # type: ignore[arg-type]
        try:
            check()
            rows.append({"backend": name, "available": True})
        except Exception as exc:  # noqa: BLE001 - doctor lists every failure
            rows.append({"backend": name, "available": False, "reason": f"{type(exc).__name__}: {exc}"})
    for status in runtime_report(config.raw):
        rows.append({"backend": f"{status.role.value}:{status.backend}", "available": status.available, "reason": status.detail})
    _print(rows)
    return 0 if all(row["available"] for row in rows) else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import logging

    configure_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))
    try:
        if args.command == "validate-config":
            from gmd_active_learning.config import load_config

            loaded = load_config(args.config)
            _print({"valid": True, "sections": sorted(loaded.sections), "config_hash": loaded.config_hash})
            return 0
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "init":
            controller = _controller(args)
            from gmd_active_learning.adapters.sgt.backend import GMDSGTBackend

            trainer = controller.trainer
            assert isinstance(trainer, GMDSGTBackend)
            metadata = trainer.api.load_export_metadata(args.export) or {}
            species = [int(z) for z in metadata.get("species") or []]
            if not species:
                raise GMDActiveLearningError(f"{args.export} carries no species metadata (export it with GMD_SGT >= 0.4)")
            cutoff = metadata.get("local_cutoff", (metadata.get("cutoffs") or {}).get("local_cutoff"))
            artifact = controller.registry.register(
                checkpoint=args.checkpoint, export=args.export, dataset_version=args.dataset_version, parent_model_id=None,
                species=species, cutoff=None if cutoff is None else float(cutoff), metrics={}, train_mode="external", backend="gmd_sgt",
                extra_files={f"committee_{k + 1}": path for k, path in enumerate(args.committee)},
            )
            controller.initialise(artifact)
            if args.dataset_version:
                controller.store.update_run(controller.run_id, active_dataset_version=args.dataset_version)
            _print({"run_id": controller.run_id, "initial_model": artifact.model_id})
            return 0
        if args.command == "run":
            controller = _controller(args)
            stage = controller.run()
            from gmd_active_learning.workflow.controller import summarize

            _print(summarize(controller.store, controller.run_id))
            return 0 if stage in {"done", "waiting", "stopped"} else 1
        if args.command == "status":
            from gmd_active_learning.workflow.controller import summarize

            store = _store(args.work_dir)
            run_ids = [args.run_id] if args.run_id else [row[0] for row in store._connection.execute("SELECT run_id FROM runs")]  # noqa: SLF001
            _print([summarize(store, run_id) for run_id in run_ids])
            return 0
        if args.command in {"stop", "retrain"}:
            controller = _controller(args)
            (controller.request_stop if args.command == "stop" else controller.request_retraining)()
            _print({"run_id": controller.run_id, "requested": args.command})
            return 0
        if args.command == "rollback":
            controller = _controller(args)
            artifact = controller.registry.rollback(controller.run_id, reason=args.reason, to_model_id=args.to)
            _print({"run_id": controller.run_id, "active_model": artifact.model_id})
            return 0
        if args.command == "resolve-unknown":
            controller = _controller(args)
            task = controller.labeler.service.resolve_unknown(args.task, reason=args.reason)
            _print({"task_id": task.task_id, "execution": task.execution.value})
            return 0
        if args.command == "train-reliability-model":
            from gmd_active_learning.reliability_model.dataset import ReliabilityDataset
            from gmd_active_learning.reliability_model.train import train_reliability_model

            dataset = ReliabilityDataset.from_file(args.dataset)
            _print({"checkpoint": str(train_reliability_model(dataset, input_dim=dataset.features.shape[1], output_dir=args.output_dir))})
            return 0
        if args.command == "scf":
            from gmd_active_learning.adapters.dft.scf.cli import run as run_scf

            return int(run_scf(args) or 0)
    except GMDActiveLearningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
