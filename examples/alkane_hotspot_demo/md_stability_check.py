"""MD stability check: does a model keep the alkane's bond topology in short GMD runs? (no CP2K)

Runs the closed loop's MD setup (box, directives, parent structure from ``--closed-loop``) with each
given model for ``--steps`` steps and several velocity seeds, and records for every frame whether the
parent bond topology is intact (``label_geometry.covalent_scale``). A model that dissociates the chain
cannot drive the closed loop: its events are refused before labeling.

Usage (from the GMD_active root):

    python examples/alkane_hotspot_demo/md_stability_check.py \
        --closed-loop configs/demos/alkane_hotspots/closed_loop_pilot.yaml \
        --model v1_initial=closed_loop/pilot/initial_model/export/model.pt \
        --model more_steps=diagnostics/initial_model/initial_train__more_steps/train/member_0/ckpt_best.pt

``--model name=path``: a TorchScript export (``.pt`` under ``export/``) or a GMD_SGT checkpoint (exported
first); paths are relative to the artifacts root. Writes ``<artifacts_root>/diagnostics/md_stability/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

LOGGER = logging.getLogger("alkane.md_stability_check")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--closed-loop", required=True, help="closed-loop YAML (MD box, directives, parent, label geometry)")
    parser.add_argument("--model", action="append", required=True, help="name=path (relative to the artifacts root)")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8])
    parser.add_argument("--frame-interval", type=int, default=10)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from gmd_active_learning.adapters.gmd.engine import GMDConfig, GMDEngine
    from gmd_active_learning.contracts import FrameAction, ModelArtifact
    from gmd_active_learning.contracts.ids import file_sha256

    from tools.experiments.alkane_hotspots.closed_loop import assemble, chain_observables, load_closed_loop
    from tools.experiments.alkane_hotspots.training import make_backend
    from gmd_active_learning.workflow.protocols import SegmentRequest

    cl = load_closed_loop(args.closed_loop)
    art = cl.alpha.artifacts_root
    asm = assemble(cl, "main")  # MD box and GMD settings only; nothing is trained or labeled here
    md = asm.gmd_config.raw["md"]
    scale = cl.config.label_geometry.covalent_scale
    out = art / "diagnostics" / "md_stability"
    backend = None
    results: dict[str, dict] = {}
    for spec in args.model:
        name, _, rel = spec.partition("=")
        path = art / rel
        if not path.is_file():
            raise SystemExit(f"model {name}: {path} does not exist")
        export = path
        if "export" not in path.parts:
            if backend is None:
                backend = make_backend(cl.beta.config.training, root=ROOT)
            export = backend.export(path, out / "exports" / name)
        for seed in args.seeds:
            artifact = ModelArtifact(model_id=f"stability-{name}", checkpoint_path=str(path), checkpoint_sha256="", export_path=str(export),
                                     export_sha256=file_sha256(export), dataset_version=None, parent_model_id=None,
                                     species=tuple(sorted(set(asm.topology.numbers.tolist()))), cutoff=None, units={}, metrics={},
                                     train_mode="fresh", backend="gmd_sgt", created_at="")
            directives = {**md["directives"], "velocity_seed": int(seed), "velocity_init": "random"}
            engine = GMDEngine(GMDConfig.model_validate({**md, "directives": directives, "frame_interval": args.frame_interval}))
            rows: list[dict] = []

            def on_frame(frame, rows=rows):  # noqa: ANN001, ANN202
                rows.append(chain_observables(frame, asm.topology, covalent_scale=scale))
                return FrameAction.CONTINUE

            request = SegmentRequest(f"stability-{name}-s{seed}", 0, out / name / f"seed_{seed}", artifact, args.steps, None, None)
            result = engine.run_segment(request, on_frame)
            broken = [r["step"] for r in rows if r["topology_intact"] is False]
            results[f"{name}|seed{seed}"] = {"model": rel, "seed": seed, "status": result.status, "n_frames": len(rows),
                                             "first_broken_step": min(broken) if broken else None,
                                             "intact_fraction": 1.0 - len(broken) / max(1, len(rows))}
            LOGGER.info("%s seed %d: %s", name, seed, json.dumps(results[f"{name}|seed{seed}"]))
    summary = {"description": __doc__.splitlines()[0], "closed_loop_config": str(args.closed_loop), "steps": args.steps,
               "time_step_fs": md["directives"].get("time_step"), "temperature_K": md["directives"].get("velocity"),
               "covalent_scale": scale, "parent_id": asm.parent["parent_id"], "runs": results}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
