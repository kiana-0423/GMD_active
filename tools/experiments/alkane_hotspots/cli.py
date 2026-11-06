"""Command line: ``python -m tools.experiments.alkane_hotspots <command> --config <yaml>``.

Part alpha registers ``generate``, ``label-full``, ``fidelity-prepare``, ``fidelity-run``,
``fidelity-report`` and ``status``. Other modules add commands through the registry:

    from tools.experiments.alkane_hotspots.cli import register_command

    def register_cli(register):            # called by the CLI for every module in EXTENSION_MODULES
        register("train-arms", "Train the acquisition arms", add_arguments, handler)

``handler(args, loaded) -> int`` receives the parsed arguments and the :class:`LoadedConfig`.
Modules listed in ``EXTENSION_MODULES`` are imported only if they exist; each may define
``register_cli(register)``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from tools.experiments.alkane_hotspots.config import SPLITS, LoadedConfig, load_config

LOGGER = logging.getLogger(__name__)
PACKAGE = "tools.experiments.alkane_hotspots"
EXTENSION_MODULES: tuple[str, ...] = ("receptive_field", "ground_truth", "arms", "training", "evaluation", "figures", "report",
                                         "closed_loop", "recovery")

Handler = Callable[[argparse.Namespace, LoadedConfig], int]


@dataclass
class Command:
    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None] | None
    handler: Handler


COMMANDS: dict[str, Command] = {}


def register_command(
    name: str,
    help: str,  # noqa: A002 - argparse vocabulary
    add_arguments: Callable[[argparse.ArgumentParser], None] | None,
    handler: Handler,
    *,
    replace: bool = False,
) -> None:
    """Register a subcommand (every command gets ``--config`` and ``--artifacts-root``)."""

    if name in COMMANDS and not replace:
        raise ValueError(f"command {name!r} is already registered")
    COMMANDS[name] = Command(name, help, add_arguments, handler)


_LOADED_EXTENSIONS: set[str] = set()


def load_extensions(modules: Sequence[str] = EXTENSION_MODULES) -> list[str]:
    """Import the extension modules that exist and call their ``register_cli`` once per process."""

    loaded = []
    for name in modules:
        full = f"{PACKAGE}.{name}"
        if full in _LOADED_EXTENSIONS:
            loaded.append(name)
            continue
        if importlib.util.find_spec(full) is None:
            continue
        module = importlib.import_module(full)
        hook = getattr(module, "register_cli", None)
        if callable(hook):
            hook(register_command)
            _LOADED_EXTENSIONS.add(full)
            loaded.append(name)
    return loaded


# -- part alpha commands -------------------------------------------------------------------------------
def _echo_json(payload: Any) -> None:
    print(json.dumps(payload, indent=1, sort_keys=True, default=str))


def cmd_generate(args: argparse.Namespace, loaded: LoadedConfig) -> int:
    from tools.experiments.alkane_hotspots.families import generate, write_generation
    from tools.experiments.alkane_hotspots.manifest import write_manifest

    result = generate(loaded.config)
    paths = write_generation(result, loaded.artifacts_root)
    counts: dict[str, dict[str, int]] = {}
    for parent in result.parents:
        counts.setdefault(parent["split"], {}).setdefault(parent["family"], 0)
        counts[parent["split"]][parent["family"]] += 1
    summary = {
        "n_parents": len(result.parents),
        "n_rejected": len(result.rejected),
        "by_split_family": counts,
        "parents_file": str(paths["parents"]),
    }
    write_manifest(loaded, "generate", {"outputs": summary})
    _echo_json(summary)
    return 0


def _runner(loaded: LoadedConfig, *, probe: bool = True) -> Any:
    from tools.experiments.alkane_hotspots.labels import LabelRunner, LabelStore, build_service, cp2k_provenance

    service = build_service(loaded)
    meta = cp2k_provenance(service, smoke_only=loaded.config.cp2k.smoke_only, probe=probe)
    return LabelRunner(service, LabelStore(loaded.artifacts_root), loaded, meta), meta


def cmd_label_full(args: argparse.Namespace, loaded: LoadedConfig) -> int:
    from tools.experiments.alkane_hotspots.families import load_parents, load_positions
    from tools.experiments.alkane_hotspots.labels import LabelRequest, parent_region
    from tools.experiments.alkane_hotspots.manifest import write_manifest

    splits = args.splits or loaded.config.labels.full_splits
    parents = [p for p in load_parents(loaded.artifacts_root) if p["split"] in splits]
    if args.parents:
        parents = [p for p in parents if p["parent_id"] in set(args.parents)]
    runner, meta = _runner(loaded)
    requests = [LabelRequest("full", p["parent_id"], int(p["chain_length"]), parent_region(p, load_positions(loaded.artifacts_root, p))) for p in parents]
    write_manifest(loaded, "label-full", {"cp2k": meta, "splits": splits, "n_requests": len(requests)})
    summary = runner.run(requests, max_wall_s=args.max_wall_s)
    _echo_json(summary.as_dict())
    return 0


def cmd_fidelity_prepare(args: argparse.Namespace, loaded: LoadedConfig) -> int:
    from tools.experiments.alkane_hotspots.fidelity import prepare
    from tools.experiments.alkane_hotspots.labels import build_service
    from tools.experiments.alkane_hotspots.manifest import write_manifest

    service = build_service(loaded)
    specs = prepare(loaded, args.split, service.settings_hash)
    by_status: dict[str, int] = {}
    for spec in specs:
        by_status[spec["status"]] = by_status.get(spec["status"], 0) + 1
    unique = {s["label_id"] for s in specs if s["status"] == "ready" and not s["covers_parent"]}
    summary = {"split": args.split, "n_specs": len(specs), "by_status": by_status, "n_fragment_calculations": len(unique),
               "max_atoms": max((s.get("n_atoms") or 0 for s in specs), default=0)}
    write_manifest(loaded, f"fidelity-prepare-{args.split}", {"outputs": summary, "settings_hash": service.settings_hash})
    _echo_json(summary)
    return 0


def cmd_fidelity_run(args: argparse.Namespace, loaded: LoadedConfig) -> int:
    from tools.experiments.alkane_hotspots.fidelity import label_requests
    from tools.experiments.alkane_hotspots.manifest import write_manifest

    runner, meta = _runner(loaded)
    requests = label_requests(loaded, args.split)
    if args.only:
        requests = [r for r in requests if r.kind == args.only]
    write_manifest(loaded, f"fidelity-run-{args.split}", {"cp2k": meta, "n_requests": len(requests)})
    summary = runner.run(requests, max_wall_s=args.max_wall_s)
    _echo_json(summary.as_dict())
    return 0


def cmd_fidelity_report(args: argparse.Namespace, loaded: LoadedConfig) -> int:
    from tools.experiments.alkane_hotspots.fidelity import report
    from tools.experiments.alkane_hotspots.manifest import write_manifest

    result = report(loaded, args.split)
    write_manifest(loaded, f"fidelity-report-{args.split}", {"decision": result.get("decision", {}).get("status")})
    keys = ("setting_id", "complete", "n_evaluated", "n_failed", "n_pending", "pooled_core_force_rmse", "max_core_force_error",
            "mean_n_atoms", "mean_fragment_cost_core_s", "passes")
    _echo_json({"split": args.split, "settings": [{k: s[k] for k in keys} for s in result["settings"]],
                "decision": (result.get("decision") or {}).get("status")})
    return 0


def status_payload(loaded: LoadedConfig) -> dict[str, Any]:
    from tools.experiments.alkane_hotspots.cost import BudgetLedger
    from tools.experiments.alkane_hotspots.fidelity import load_selection, load_specs
    from tools.experiments.alkane_hotspots.families import load_parents
    from tools.experiments.alkane_hotspots.labels import LabelStore

    config = loaded.config
    root = loaded.artifacts_root
    store = LabelStore(root)
    records = store.all()
    ledger = BudgetLedger(records, config.budgets, config.cp2k.cores)
    parents = load_parents(root) if (root / "parents.jsonl").exists() else []
    by_status: dict[str, dict[str, int]] = {}
    for record in records:
        by_status.setdefault(record["kind"], {}).setdefault(record["status"], 0)
        by_status[record["kind"]][record["status"]] += 1
    timing: dict[str, Any] = {}
    for kind in ("full", "fragment"):
        lengths = sorted({p["chain_length"] for p in parents}) if kind == "full" else [None]
        for n in lengths:
            values = ledger.measured(kind, n)
            walls = [r["cost"]["wall_s"] for r in records if r["kind"] == kind and r.get("cost") and r["status"] in {"valid", "invalid"}
                     and (n is None or r.get("chain_length") == n)]
            key = f"{kind}" + (f"_C{n}" if n is not None else "")
            timing[key] = {"n": len(values), "mean_cpu_core_s": (sum(values) / len(values)) if values else None,
                           "mean_wall_s": (sum(walls) / len(walls)) if walls else None}
    # Remaining work: full labels of configured splits + fragment calculations of prepared specs not yet done.
    done = {r["label_id"] for r in records if r["status"] in {"valid", "invalid", "failed"}}
    remaining = {"full": 0, "fragment": 0}
    remaining_cost = 0.0
    from tools.experiments.alkane_hotspots.families import load_positions
    from tools.experiments.alkane_hotspots.labels import build_service, label_id_for, parent_region

    settings_hash = build_service(loaded).settings_hash
    for parent in parents:
        if parent["split"] not in config.labels.full_splits:
            continue
        lid = label_id_for(parent_region(parent, load_positions(root, parent)), settings_hash)
        if lid not in done:
            remaining["full"] += 1
            remaining_cost += ledger.estimate("full", parent["chain_length"])[0]
    fragment_ids = {s["label_id"] for s in load_specs(loaded).values() if s["status"] == "ready" and not s["covers_parent"]}
    pending_fragments = fragment_ids - done
    remaining["fragment"] = len(pending_fragments)
    remaining_cost += len(pending_fragments) * ledger.estimate("fragment")[0]
    selection = load_selection(loaded)
    return {
        "config": config.name,
        "artifacts_root": str(root),
        "smoke_only": config.cp2k.smoke_only,
        "n_parents": len(parents),
        "labels": by_status,
        "budgets": {
            "spent_cpu_core_hours": ledger.spent_core_s() / 3600.0,
            "max_total_cpu_core_hours": config.budgets.max_total_cpu_core_hours,
            "full_labels_launched": ledger.launched("full"),
            "max_full_labels": config.budgets.max_full_labels,
            "fragment_labels_launched": ledger.launched("fragment"),
            "max_fragment_labels": config.budgets.max_fragment_labels,
            "per_job_walltime_s": config.budgets.per_job_walltime_s,
            "cores_per_job": config.cp2k.cores,
        },
        "measured_timing": timing,
        "remaining": {**remaining, "estimated_cpu_core_hours": remaining_cost / 3600.0,
                      "estimated_wall_hours_serial": remaining_cost / 3600.0 / max(config.cp2k.cores, 1)},
        "fidelity_selection": None if selection is None else selection["setting"]["setting_id"],
    }


def cmd_status(args: argparse.Namespace, loaded: LoadedConfig) -> int:
    _echo_json(status_payload(loaded))
    return 0


def _split_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=["calibration", "final_eval"], default="calibration")


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-wall-s", type=float, default=None, help="stop waiting after this many seconds (jobs keep running; re-run resumes)")


def _label_full_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=None, help="default: labels.full_splits")
    parser.add_argument("--parents", nargs="+", default=None, help="restrict to these parent IDs")
    _run_arguments(parser)


def _fidelity_run_arguments(parser: argparse.ArgumentParser) -> None:
    _split_argument(parser)
    parser.add_argument("--only", choices=["full", "fragment"], default=None)
    _run_arguments(parser)


register_command("generate", "Generate parents, checks and splits (deterministic)", None, cmd_generate)
register_command("label-full", "CP2K labels of full parents (budgeted, resumable)", _label_full_arguments, cmd_label_full)
register_command("fidelity-prepare", "Build fragment specs for the fidelity gate", _split_argument, cmd_fidelity_prepare)
register_command("fidelity-run", "CP2K labels for the fidelity gate (parents + fragments)", _fidelity_run_arguments, cmd_fidelity_run)
register_command("fidelity-report", "Fidelity metrics; calibration selects and freezes, final_eval evaluates", _split_argument, cmd_fidelity_report)
register_command("status", "Budgets, measured timing and remaining-cost estimate", None, cmd_status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.experiments.alkane_hotspots", description="Alkane hotspot demonstration (PHAL in GMD_active)")
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, command in COMMANDS.items():
        child = sub.add_parser(name, help=command.help)
        child.add_argument("--config", required=True, help="YAML under configs/demos/alkane_hotspots/")
        child.add_argument("--artifacts-root", default=None, help="override artifacts_root")
        if command.add_arguments is not None:
            command.add_arguments(child)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_extensions()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.WARNING), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    loaded = load_config(args.config, artifacts_root=args.artifacts_root)
    return int(COMMANDS[args.command].handler(args, loaded))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
