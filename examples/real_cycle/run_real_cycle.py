"""Real closed active-learning cycle: GMD -> monitoring -> extraction -> CP2K -> dataset ->
GMD_SGT training -> export -> gated promotion -> resumed GMD with the new model.

Usage (from the GMD_active root, in the gmd-active environment):

    python examples/real_cycle/run_real_cycle.py --work-dir /tmp/gmd_al_cycle \
        --gmd ../GMD/build-torch/gmd --gmd-sgt ../GMD_SGT --cp2k-data /opt/homebrew/share/cp2k/data

Everything is real: CP2K 2026.2 labels (PBE/DZVP), GMD_SGT training/export, GMD MD (TorchScript).
The tiny model and short runs keep the run to a few minutes; the science is a smoke-scale demo.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase.build import molecule

from gmd_active_learning.config import build_config
from gmd_active_learning.contracts import ExtractedRegion, RegionRole, ValidationStatus, stable_id
from gmd_active_learning.workflow.assembly import build_controller
from gmd_active_learning.workflow.controller import summarize

BOX = 7.5
SITES = [(1.9, 1.9, 1.9), (5.6, 5.6, 1.9), (1.9, 5.6, 5.6), (5.6, 1.9, 5.6)]
TINY_MODEL = {
    "model": {"n_species": 10, "n_blocks": 1, "scalar_dim": 16, "irreps": "16x0e", "n_basis": 6, "l_max": 0,
              "local_cutoff": 4.0, "lr_cutoff": 4.0, "long_range_type": "none", "n_heads": 1, "avg_neighbors": 6.0},
    "train": {"n_epochs": 30, "batch_size": 4, "warmup_steps": 1, "lr": 5.0e-3, "patience": 0, "w_energy": 0.0, "w_force": 1.0},
}


def config_dict(args: argparse.Namespace, work: Path) -> dict[str, Any]:
    return {
        "workflow": {
            "run_id": "water-cycle", "max_cycles": 2, "segment_steps": 40, "poll_interval_s": 1.0, "max_poll_wait_s": 1800.0,
            "max_pending_tasks": 8,
            "candidates": {"max_per_cycle": 3, "max_per_event": 1, "diversity_threshold": 0.0},
            "retraining": {"min_new_samples": 2},
            "promotion": {"max_force_rmse": 10.0, "max_relative_regression": 10.0, "require_export_validation": True,
                          "require_engine_smoke_test": True},
        },
        "md": {"executable": str(Path(args.gmd).resolve()), "initial_structure": str(work / "water_box.xyz"), "frame_interval": 2,
               "directives": {"time_step": 0.25, "velocity": 300.0, "velocity_init": "random", "velocity_seed": 7}},
        "mlip": {"source_path": str(Path(args.gmd_sgt).resolve()), "train": TINY_MODEL, "committee_size": 1, "seed": 11},
        "monitor": {
            "metrics": {name: {"enabled": False} for name in ("force", "delta_force", "displacement", "delta_q", "lj_residual", "mlip_force_deviation")}
            | {"rmin": {"enabled": True, "threshold": 0.6}, "committee": {"enabled": True, "weight": 1.0, "min_members": 2}},
            "running_stats": {"enabled": False},
            "stages": {"light_interval": 1, "physics_interval": "always", "full_interval": "always", "screen_threshold": 0.02,
                       "physics_threshold": 0.02, "label_threshold": 0.02, "min_trigger_atoms": 1},
        },
        "hotspots": {"merge_radius": 1.6},  # one hotspot cluster per water molecule
        "events": {"pre_trigger_frames": 1, "post_trigger_frames": 1, "cooldown_steps": 8},
        # Molecular system: the bonded subgraph around the hotspot is a whole, neutral molecule, so no
        # covalent bond is cut (a radial cluster in this small box cut O-H bonds and produced radicals
        # whose SCF did not converge; see docs/real_cycle.md).
        "extraction": {"mode": "graph", "graph": {"hops": 2, "radial_shell": False}, "vacuum_padding": 4.0, "mlip_cutoff": 3.5},
        "datasets": {"split": {"val_fraction": 0.5, "group_by": "event"}},
        "backends": {"dft": {"engine": "cp2k"}, "scheduler": {"engine": "local"}},
        "cp2k": {"executable": args.cp2k, "data_dir": args.cp2k_data, "cutoff": 280.0, "rel_cutoff": 40.0,
                 "environment": {"OMP_NUM_THREADS": "1"}, "task_dir": str(work / "cp2k_tasks")},
    }


def write_water_box(path: Path) -> None:
    lines = [str(3 * len(SITES)), f"{BOX} {BOX} {BOX}"]
    rng = np.random.default_rng(0)
    for site in SITES:
        mol = molecule("H2O")
        mol.rotate(float(rng.uniform(0, 360)), tuple(rng.normal(size=3)), center="COM")
        mol.translate(np.asarray(site) - mol.get_center_of_mass())
        lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(mol.get_chemical_symbols(), mol.positions)]
    path.write_text("\n".join(lines) + "\n")


def seed_regions(n: int) -> list[ExtractedRegion]:
    """Perturbed water dimers as initial (non-periodic) training regions, all atoms trusted."""

    rng = np.random.default_rng(1)
    regions = []
    for k in range(n):
        a, b = molecule("H2O"), molecule("H2O")
        b.rotate(float(rng.uniform(0, 360)), tuple(rng.normal(size=3)), center="COM")
        b.translate([2.8 + rng.uniform(-0.2, 0.4), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)])
        a += b
        a.positions += rng.normal(0, 0.04, a.positions.shape)
        n_atoms = len(a)
        regions.append(ExtractedRegion(stable_id("seed", k), "seed", "seed", 0, f"seed-{k}", a.numbers, a.positions, None, (False,) * 3,
                                       np.arange(n_atoms), np.zeros((n_atoms, 3), int), [RegionRole.CORE] * n_atoms, np.ones(n_atoms),
                                       np.zeros(n_atoms, bool), [0], metadata={"seed": True}))
    return regions


def label_seed(controller: Any, regions: list[ExtractedRegion]) -> list[Path]:
    service = controller.labeler.service
    tasks = [service.submit(service.prepare(region).task_id) for region in regions]
    paths = []
    import time

    pending = {task.task_id for task in tasks}
    while pending:
        for task_id in sorted(pending):
            task = service.poll(task_id)
            if task.metadata.get("stage") == "done":
                pending.discard(task_id)
        time.sleep(1.0)
    for region, task in zip(regions, tasks):
        result = service.collect(task.task_id)
        if result.validation is not ValidationStatus.VALID:
            raise SystemExit(f"seed label {task.task_id} invalid: {result.validation_reasons}")
        _sample_id, path = controller.datasets.add_sample(region, result, provenance={"task_id": task.task_id, "seed": True})
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gmd", required=True)
    parser.add_argument("--gmd-sgt", required=True)
    parser.add_argument("--cp2k", default="cp2k.psmp")
    parser.add_argument("--cp2k-data", default=None)
    parser.add_argument("--n-seed", type=int, default=8)
    parser.add_argument("--fresh", action="store_true", help="delete the work dir first")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if args.fresh and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    write_water_box(work / "water_box.xyz")
    raw = config_dict(args, work)
    (work / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    config = build_config(raw, base_dir=work, path=work / "config.yaml")
    controller = build_controller(config, work_dir=work / "run")

    run = controller.store.ensure_run("water-cycle", config_hash=controller.config_hash, initial_model_id=None, stage="sampling")
    if run.active_model_id is None:
        seed_paths = label_seed(controller, seed_regions(args.n_seed))
        groups = {Path(p).stem: f"seed-{i % 4}" for i, p in enumerate(seed_paths)}
        seed_dataset = controller.datasets.publish(seed_paths, parent=None, split_groups=groups)
        output = controller.trainer.train(seed_dataset, parent=None, work_dir=work / "seed_training")
        export = controller.trainer.export(output.checkpoint, work / "seed_training" / "export")
        initial = controller.registry.register(
            checkpoint=output.checkpoint, export=export, dataset_version=seed_dataset.version, parent_model_id=None,
            species=output.species, cutoff=output.cutoff, metrics=controller.trainer.evaluate(output.checkpoint, seed_dataset),
            train_mode=output.mode, backend="gmd_sgt",
            extra_files={f"committee_{k + 1}": path for k, path in enumerate(output.summary["committee_checkpoints"])},
        )
        controller.initialise(initial)
        controller.store.update_run("water-cycle", active_dataset_version=seed_dataset.version)
    final = controller.run()
    report = summarize(controller.store, "water-cycle")
    (work / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"final_stage": final, **{k: report[k] for k in ("cycle", "events", "samples", "candidates", "tasks", "promotions", "segments")}}, indent=1, default=str))
    print(f"report written to {work / 'report.json'}")
    return 0 if final == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
