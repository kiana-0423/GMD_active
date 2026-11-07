"""``python -m tools.experiments.alkane_repair <command> --benchmark <yaml>`` (run from the GMD_active root).

Order of a benchmark (each command is resumable; every command after ``freeze-protocol`` refuses to run
if a protocol file changed):

    generate -> freeze-protocol -> plan
    label --stage initial_model --qualification-stage 0 -> qualify --stage 0 [-> stage 1 ...]
    label --stage development -> fidelity-audit -> failures
    label --stage repair_val -> label --stage final_test (sealed)
    calibrate-monitor --arm <each closed-loop arm>
    repair-run [--arm/--case/--seed]
    evaluate -> report [--publish]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

LOGGER = logging.getLogger("alkane_repair")


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=1, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.experiments.alkane_repair", description=__doc__.splitlines()[0])
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--benchmark", required=True, help="configs/demos/alkane_repair/<name>.yaml")
        return p

    command("generate", "structures and partitions (no DFT)")
    p = command("freeze-protocol", "hash the protocol, configuration and structure sets")
    p.add_argument("--note", default="")
    command("plan", "costed execution manifest from measured timings (nothing is launched)")
    p = command("label", "full-system CP2K reference labels of one stage (within its budget)")
    p.add_argument("--stage", required=True, choices=["initial_model", "development", "repair_val", "final_test"])
    p.add_argument("--qualification-stage", type=int, default=None)
    p.add_argument("--max-wall-s", type=float, default=None)
    p = command("qualify", "train and qualify the initial model of one qualification stage")
    p.add_argument("--stage", type=int, required=True)
    p = command("fidelity-audit", "frozen fragment setting on development challenges (CP2K fragments)")
    p.add_argument("--no-labels", action="store_true", help="evaluate existing labels only")
    command("failures", "reference-confirmed failures of the frozen initial model on development challenges")
    p = command("calibrate-monitor", "calibrate and check one closed-loop arm's monitor on in-domain trajectories")
    p.add_argument("--arm", required=True)
    p = command("repair-run", "repair runs (all planned, or a subset)")
    p.add_argument("--arm", action="append")
    p.add_argument("--case", action="append")
    p.add_argument("--seed", type=int, action="append")
    p.add_argument("--max-wall-s", type=float, default=None)
    p.add_argument("--dry-run", action="store_true", help="list the runs only")
    p = command("evaluate", "sealed final evaluation (reads final-test labels; logged)")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--reason", default="final evaluation")
    p = command("report", "report.json/report.md and figures")
    p.add_argument("--publish", action="store_true", help="copy the report and figures to docs/demos/alkane_repair/results/<name>")
    command("status", "what exists and what is next")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .config import load_benchmark

    loaded = load_benchmark(args.benchmark)
    if args.command == "generate":
        from .generate import generate, write_structures

        _print(write_structures(loaded, generate(loaded))["counts"])
    elif args.command == "freeze-protocol":
        from .references import freeze_protocol

        _print(freeze_protocol(loaded, note=args.note))
    elif args.command == "plan":
        from .plan import build_plan, render_plan

        plan = build_plan(loaded)
        text = render_plan(plan)
        (loaded.root / "plan" / "execution_plan.md").write_text(text, encoding="utf-8")
        print(text)
    elif args.command == "label":
        from .references import label_stage

        _print(label_stage(loaded, args.stage, qualification_stage=args.qualification_stage, max_wall_s=args.max_wall_s))
    elif args.command == "qualify":
        from .qualify import qualify

        report = qualify(loaded, args.stage)
        _print({k: report[k] for k in ("qualification_stage", "accepted", "criteria", "n_train", "n_val")} | {"id_val": report["id_val_accuracy"]})
    elif args.command == "fidelity-audit":
        from .references import fidelity_audit

        _print(fidelity_audit(loaded, run_labels=not args.no_labels)["families"])
    elif args.command == "failures":
        from .failure import analyse_development

        _print(analyse_development(loaded)["families"])
    elif args.command == "calibrate-monitor":
        from .repair import calibrate_arm

        record = calibrate_arm(loaded, args.arm)
        _print({k: v for k, v in record.items() if k != "state"})
        return 0 if record["ok"] else 1
    elif args.command == "repair-run":
        from .repair import run_all, run_matrix

        specs = run_matrix(loaded, arms=args.arm, cases=args.case, seeds=args.seed)
        if args.dry_run:
            _print([s.name for s in specs])
        else:
            _print(run_all(loaded, specs, max_wall_s=args.max_wall_s))
    elif args.command == "evaluate":
        from .evaluate import evaluate

        result = evaluate(loaded, allow_incomplete=args.allow_incomplete, reason=args.reason)
        _print(result["summary"])
    elif args.command == "report":
        from .report import write_report

        _print(write_report(loaded, publish=args.publish))
    elif args.command == "status":
        from .report import status

        _print(status(loaded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
