"""``gmd-al`` command line interface.

    gmd-al validate-config --config run.yaml
    gmd-al convert-phal-config --phal phal.yaml --output run.yaml [--gmd-executable ...]
    gmd-al doctor          [--config run.yaml] [--strict] [--write-config [PATH]] [--json]
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

Startup is light: ``gmd-al --help`` and the configuration/doctor commands import neither torch,
GMD_SGT nor the SCF tooling; ``gmd-al scf ...`` loads the SCF CLI only when it is invoked.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from gmd_active_learning.core.exceptions import GMDActiveLearningError
from gmd_active_learning.utils.logging import configure_logging


DEFAULT_RUNTIME_CONFIG = Path("configs") / "runtime.local.yaml"
GMD_CANDIDATES = ("gmd",)
CP2K_CANDIDATES = ("cp2k.psmp", "cp2k.popt", "cp2k")


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One doctor row. ``required=False`` rows are reported but never fail ``--strict``."""

    name: str
    ok: bool
    detail: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.name, "available": self.ok, "reason": self.detail, "required": self.required}


def _format_result(result: CheckResult) -> str:
    status = "OK" if result.ok else "MISS"
    suffix = "" if result.required or result.ok else " (optional)"
    return f"[{status}] {result.name}: {result.detail}{suffix}"


def _executable(value: str | None) -> str | None:
    """Absolute path of an executable given as a path or a bare command name, or None."""

    if not value:
        return None
    if "/" in value or os.sep in value:
        path = Path(value).expanduser()
        return str(path.resolve()) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(value)


def _find(candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _check_executable(name: str, configured: str | None, env_var: str, candidates: Sequence[str]) -> tuple[CheckResult, str | None]:
    if configured:
        found, origin = _executable(configured), f"configured {configured}"
    elif os.environ.get(env_var):
        found, origin = _executable(os.environ[env_var]), f"${env_var}={os.environ[env_var]}"
    else:
        found, origin = _find(candidates), f"PATH ({', '.join(candidates)})"
    if found:
        return CheckResult(name, True, found), found
    hint = f"set ${env_var}, put {candidates[0]} on PATH or set it in the config"
    return CheckResult(name, False, f"not found via {origin}; {hint}"), None


def _check_gmd_sgt(source_path: str | None) -> tuple[CheckResult, str | None]:
    source = source_path or os.environ.get("GMD_SGT_PATH")
    try:
        from gmd_active_learning.adapters.sgt.backend import SGTConfig, import_gmd_sgt

        module = import_gmd_sgt(SGTConfig(source_path=source))
    except Exception as exc:  # noqa: BLE001 - a doctor lists every failure
        return CheckResult("mlip:gmd_sgt", False, f"{type(exc).__name__}: {exc}"), None
    location = str(Path(module.__file__).resolve().parents[1]) if getattr(module, "__file__", None) else source
    return CheckResult("mlip:gmd_sgt", True, f"gmd_sgt {module.__version__} ({location})"), source


def _check_torch(require_cuda: bool) -> list[CheckResult]:
    if importlib.util.find_spec("torch") is None:
        return [CheckResult("torch", False, "not importable"), CheckResult("cuda", False, "torch is not importable", require_cuda)]
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001 - binary import failures are reported
        return [CheckResult("torch", False, f"import failed: {exc}"), CheckResult("cuda", False, "torch import failed", require_cuda)]
    version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    rows = [CheckResult("torch", True, f"torch {version}")]
    if torch.cuda.is_available():
        rows.append(CheckResult("cuda", True, f"cuda {cuda_version}, device {torch.cuda.get_device_name(0)}", require_cuda))
    else:
        rows.append(CheckResult("cuda", False, f"torch {version}, cuda {cuda_version}, cuda unavailable", require_cuda))
    return rows


def collect_runtime_checks(config: Any | None = None) -> tuple[list[CheckResult], dict[str, str]]:
    """Doctor checks (PHAL ``collect_runtime_checks``): GMD binary, CP2K, GMD_SGT, torch and CUDA.

    Discovery order: the configured value (``--config``), then ``$GMD_EXECUTABLE`` /
    ``$CP2K_EXECUTABLE`` / ``$GMD_SGT_PATH``, then ``PATH``. CUDA is required only when
    ``mlip.device`` asks for it (GMD runs TorchScript models on the CPU). With a configuration the
    configured DFT and scheduler backends also report their own ``check_runtime()``.
    """

    md = config.get("md") if config is not None else None
    mlip = config.get("mlip") if config is not None else None
    cp2k = config.get("cp2k") if config is not None else None
    results: list[CheckResult] = []
    paths: dict[str, str] = {}
    gmd_row, gmd_path = _check_executable("md:gmd", getattr(md, "executable", None), "GMD_EXECUTABLE", GMD_CANDIDATES)
    results.append(gmd_row)
    if gmd_path:
        paths["gmd"] = gmd_path
    cp2k_row, cp2k_path = _check_executable("dft:cp2k", getattr(cp2k, "executable", None), "CP2K_EXECUTABLE", CP2K_CANDIDATES)
    if cp2k_path:
        paths["cp2k"] = cp2k_path
    sgt_row, sgt_source = _check_gmd_sgt(getattr(mlip, "source_path", None))
    if sgt_row.ok and sgt_source:
        paths["gmd_sgt"] = str(Path(sgt_source).expanduser().resolve())
    device = str(getattr(mlip, "device", "cpu"))
    torch_rows = _check_torch(require_cuda=device.startswith("cuda"))
    if config is not None and cp2k_row.ok:
        from gmd_active_learning.adapters.registry import BackendRole, runtime_report

        for status in runtime_report(config.raw, (BackendRole.DFT, BackendRole.SCHEDULER)):
            name = f"{status.role.value}:{status.backend}"
            row = CheckResult(name, bool(status.available), str(status.detail))
            if name == "dft:cp2k":
                cp2k_row = row if not row.ok else CheckResult(name, True, f"{cp2k_path} ({status.detail})")
            else:
                results.append(row)
    results.insert(1, cp2k_row)
    results.extend([sgt_row, *torch_rows])
    return results, paths


def write_runtime_config(path: str | Path, paths: dict[str, str], *, base: dict[str, Any] | None = None) -> Path:
    """Write a complete, validated configuration (``base`` or the packaged defaults) with the
    discovered executables filled in. Refuses to write a file that would not load."""

    import yaml

    from gmd_active_learning.config import build_config, default_config_dict

    target = Path(path)
    raw = json.loads(json.dumps(base if base is not None else default_config_dict(), default=str))
    if "gmd" in paths:
        raw.setdefault("md", {})["executable"] = paths["gmd"]
    if "cp2k" in paths:
        raw.setdefault("cp2k", {})["executable"] = paths["cp2k"]
    if "gmd_sgt" in paths:
        raw.setdefault("mlip", {})["source_path"] = paths["gmd_sgt"]
    build_config(raw, base_dir=target.parent.resolve())  # raises ConfigurationError if it would not validate
    target.parent.mkdir(parents=True, exist_ok=True)
    header = "# Written by `gmd-al doctor --write-config`: configuration with the discovered runtime paths.\n"
    target.write_text(header + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return target


def _add_config(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument("--config", type=Path, required=required, help="YAML configuration (see configs/example_run.yaml)")


def _add_work_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, required=True, help="persistent run directory (state, models, datasets)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gmd-al", description="GMD active-learning controller")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_config(sub.add_parser("validate-config", help="validate a configuration file"))
    doctor = sub.add_parser("doctor", help="check GMD, CP2K, GMD_SGT, torch and CUDA ([OK]/[MISS] per check)")
    _add_config(doctor, required=False)
    doctor.add_argument("--strict", action="store_true", help="exit 1 if any required check is missing")
    doctor.add_argument("--write-config", nargs="?", const=str(DEFAULT_RUNTIME_CONFIG), default=None, metavar="PATH",
                        help=f"write a validated configuration with the discovered paths (default {DEFAULT_RUNTIME_CONFIG})")
    doctor.add_argument("--json", action="store_true", help="print the checks as JSON instead of [OK]/[MISS] lines")

    convert = sub.add_parser("convert-phal-config", help="convert a PHAL (hotspot_al) YAML configuration")
    convert.add_argument("--phal", type=Path, required=True, help="PHAL configuration (complete, e.g. its default.yaml)")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--gmd-executable", default="gmd")
    convert.add_argument("--initial-structure", default="initial_structure.xyz")
    convert.add_argument("--run-id", default=None)

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

    # Placeholder: the SCF tooling (and its parser) is imported only when `gmd-al scf` runs.
    scf = sub.add_parser("scf", add_help=False, help="experimental CP2K ML-SCF tooling (see `gmd-al scf --help`)")
    scf.add_argument("scf_args", nargs=argparse.REMAINDER)
    return parser


def _run_scf(argv: list[str]) -> int:
    from gmd_active_learning.adapters.dft.scf.cli import add_subparser, run

    parser = argparse.ArgumentParser(prog="gmd-al")
    add_subparser(parser.add_subparsers(dest="command", required=True), name="scf")
    return int(run(parser.parse_args(["scf", *argv])) or 0)


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


def run_doctor(args: argparse.Namespace) -> int:
    """PHAL ``run_doctor`` semantics: report every check; exit 1 only with ``--strict`` when a
    required check is missing; ``--write-config`` writes a configuration that loads and validates."""

    from gmd_active_learning.config import load_config

    config = load_config(args.config) if args.config is not None else None
    results, paths = collect_runtime_checks(config)
    if args.json:
        _print([result.to_dict() for result in results])
    else:
        for result in results:
            print(_format_result(result))
    if args.write_config is not None:
        written = write_runtime_config(args.write_config, paths, base=None if config is None else config.raw)
        print(f"wrote runtime config: {written}", file=sys.stderr if args.json else sys.stdout)
    missing = [result for result in results if result.required and not result.ok]
    return 1 if args.strict and missing else 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, extra = build_parser().parse_known_args(argv)
    if args.command == "scf":  # delegate everything after "scf" (options included) to the SCF CLI
        return _run_scf(argv[argv.index("scf") + 1 :])
    if extra:
        build_parser().error(f"unrecognized arguments: {' '.join(extra)}")
    import logging

    configure_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))
    try:
        if args.command == "validate-config":
            from gmd_active_learning.config import load_config

            loaded = load_config(args.config)
            _print({"valid": True, "sections": sorted(loaded.sections), "config_hash": loaded.config_hash})
            return 0
        if args.command == "doctor":
            return run_doctor(args)
        if args.command == "convert-phal-config":
            import yaml

            from gmd_active_learning.config import build_config
            from gmd_active_learning.config_phal import convert_phal_config

            try:
                phal = yaml.safe_load(args.phal.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as exc:
                raise GMDActiveLearningError(f"cannot read PHAL config {args.phal}: {exc}") from exc
            raw, report = convert_phal_config(phal, gmd_executable=args.gmd_executable,
                                              initial_structure=args.initial_structure, run_id=args.run_id)
            build_config(raw, base_dir=args.output.parent.resolve())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            _print({"output": str(args.output), "mapped": len(report.mapped), "dropped": len(report.dropped),
                    "dropped_with_content": report.dropped_with_content, "warnings": report.warnings})
            return 0
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
    except GMDActiveLearningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
