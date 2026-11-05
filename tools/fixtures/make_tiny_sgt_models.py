"""Train and export tiny seeded GMD_SGT model fixtures (ledger FIX-002).

Synthetic: the models are trained for a few epochs on ``tests/fixtures/datasets/tiny_h2o.extxyz``
(a toy harmonic H2O set, not DFT). They exist to exercise loading, species/cutoff reporting,
inference, committee spread and TorchScript export through the GMD_SGT adapter. No physical
accuracy is claimed. Committee members differ only by seed.

    PYTHONPATH=src python tools/fixtures/make_tiny_sgt_models.py [--out tests/fixtures/sgt] \
        [--gmd-sgt PATH] [--seeds 11 12]

Writes ``member_<seed>/checkpoint.pt``, ``member_<seed>/model.pt`` (TorchScript) and
``manifest.json`` (seeds, sha256, species, cutoff, GMD_SGT version, command). Training scratch
files go to a temporary directory and are discarded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "tests" / "fixtures" / "datasets" / "tiny_h2o.extxyz"
DEFAULT_OUT = REPO / "tests" / "fixtures" / "sgt"
DEFAULT_SEEDS = (11, 12)
VALIDATION_GROUPS = ("event:evt-3",)
TINY_TRAIN_CONFIG: dict[str, Any] = {
    "model": {"n_species": 10, "n_blocks": 1, "scalar_dim": 8, "irreps": "8x0e", "n_basis": 4, "l_max": 0,
              "local_cutoff": 4.0, "lr_cutoff": 4.0, "long_range_type": "none", "n_heads": 1, "avg_neighbors": 2.0},
    "train": {"n_epochs": 3, "batch_size": 2, "warmup_steps": 1, "lr": 3.0e-3, "patience": 0},
}


def _default_gmd_sgt() -> str:
    return os.environ.get("GMD_SGT_PATH", str(REPO.parent / "GMD_SGT"))


def _split(work: Path) -> tuple[Path, Path]:
    from ase.io import read, write

    frames = read(DATASET, index=":")
    train = [atoms for atoms in frames if atoms.info.get("split_group") not in VALIDATION_GROUPS]
    val = [atoms for atoms in frames if atoms.info.get("split_group") in VALIDATION_GROUPS]
    write(work / "train.extxyz", train, format="extxyz")
    write(work / "val.extxyz", val, format="extxyz")
    return work / "train.extxyz", work / "val.extxyz"


def make_models(out: Path, *, gmd_sgt: str, seeds: Sequence[int] = DEFAULT_SEEDS, n_epochs: int | None = None) -> dict[str, Any]:
    """Train one model per seed through the adapter, export it, and write the manifest."""

    sys.path.insert(0, str(REPO / "src"))
    from gmd_active_learning.adapters.sgt.backend import GMDSGTBackend, SGTConfig
    from gmd_active_learning.contracts import DatasetVersion
    from gmd_active_learning.contracts.ids import file_sha256

    train_config = json.loads(json.dumps(TINY_TRAIN_CONFIG))
    if n_epochs is not None:
        train_config["train"]["n_epochs"] = int(n_epochs)
    backend = GMDSGTBackend(SGTConfig(source_path=gmd_sgt, train=train_config, device="cpu"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    members = []
    with tempfile.TemporaryDirectory(prefix="tiny-sgt-") as scratch:
        work = Path(scratch)
        train_file, val_file = _split(work)
        dataset = DatasetVersion("tiny-h2o", str(work), file_sha256(DATASET), 8, 6, 2, None, (), "fixture",
                                 {"train_file": str(train_file), "val_file": str(val_file)})
        for seed in seeds:
            output = backend.train(dataset, parent=None, work_dir=work / f"train_{seed}", seed=int(seed))
            target = out / f"member_{seed}"
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True)
            checkpoint = target / "checkpoint.pt"
            shutil.copy2(output.checkpoint, checkpoint)
            export = Path(backend.export(checkpoint, target))
            info = backend.model_info(export)
            members.append({
                "seed": int(seed),
                "checkpoint": checkpoint.relative_to(out).as_posix(),
                "checkpoint_sha256": file_sha256(checkpoint),
                "export": export.relative_to(out).as_posix(),
                "export_sha256": file_sha256(export),
                "species": list(info.species),
                "cutoff": info.cutoff,
                "best_epoch": output.summary.get("best_epoch"),
            })
    manifest = {
        "description": "Tiny seeded GMD_SGT models trained on the synthetic tiny_h2o fixture (not DFT; no accuracy claims).",
        "dataset": DATASET.relative_to(REPO).as_posix(),
        "dataset_sha256": file_sha256(DATASET),
        "validation_groups": list(VALIDATION_GROUPS),
        "gmd_sgt_version": str(backend.sgt.__version__),
        "train_config": train_config,
        "command": "PYTHONPATH=src python tools/fixtures/make_tiny_sgt_models.py --seeds " + " ".join(str(s) for s in seeds),
        "members": members,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gmd-sgt", default=_default_gmd_sgt(), help="GMD_SGT checkout (default: $GMD_SGT_PATH or ../GMD_SGT)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=None, help="override the number of epochs (tests use fewer)")
    args = parser.parse_args(argv)
    manifest = make_models(args.out, gmd_sgt=args.gmd_sgt, seeds=args.seeds, n_epochs=args.epochs)
    print(json.dumps({"out": str(args.out), "members": [m["seed"] for m in manifest["members"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
