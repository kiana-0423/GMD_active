#!/usr/bin/env python
# Ported from PHAL (scripts/make_tiny_h2o_dataset.py, train_tiny_allegro_h2o.py,
# smoke_allegro_h2o_inference.py) at 96263b6; MIT License, see NOTICE.
"""Tiny GMD_SGT H2O smoke chain (ledger OPS-003), CPU, well under 90 s.

1. dataset: the synthetic harmonic H2O set (``tests/fixtures/datasets/make_tiny_h2o.py``; toy
   model, NOT DFT) written in GMD_active's published layout and split by event group;
2. train: a few epochs of a tiny GMD_SGT model through the GMD_SGT adapter (same settings as the
   model fixtures, ``tools/fixtures/make_tiny_sgt_models.py``);
3. export: TorchScript export (what GMD runs), with metadata checked by the adapter;
4. infer: one energy/force evaluation of an H2O molecule with the export *and* the checkpoint;
   both must be finite and agree (float32 export vs eager checkpoint).

Exit codes keep PHAL's inference-smoke contract: 0 ok, 1 runtime error, 2 import error (GMD_SGT
unavailable), 4 model path error, 5 model load failure, 6 inference failure, 7 element mismatch.
PHAL's chain needed a hard-coded conda env (allegro-mac) and trained a NequIP model under an
"Allegro" name; this one uses only the GMD_SGT adapter. No accuracy is claimed.

    PYTHONPATH=src python tools/smoke/sgt_tiny_h2o/run_chain.py --work-dir /tmp/sgt_chain [--gmd-sgt ../GMD_SGT]
    PYTHONPATH=src python tools/smoke/sgt_tiny_h2o/run_chain.py --infer-only --model model.pt [--symbols OHH]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATASET_MAKER = REPO / "tests" / "fixtures" / "datasets" / "make_tiny_h2o.py"
MODEL_MAKER = REPO / "tools" / "fixtures" / "make_tiny_sgt_models.py"
EXIT_OK, EXIT_RUNTIME_ERROR, EXIT_IMPORT_ERROR = 0, 1, 2
EXIT_MODEL_PATH_ERROR, EXIT_MODEL_LOAD_FAILED, EXIT_INFERENCE_FAILED, EXIT_ELEMENT_MISMATCH = 4, 5, 6, 7
H2O_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2400, 0.9266, 0.0]])


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _backend(gmd_sgt: str | None, epochs: int) -> Any:
    from gmd_active_learning.adapters.sgt.backend import GMDSGTBackend, SGTConfig

    train = json.loads(json.dumps(_load(MODEL_MAKER, "_tiny_models").TINY_TRAIN_CONFIG))
    train["train"]["n_epochs"] = int(epochs)
    return GMDSGTBackend(SGTConfig(source_path=gmd_sgt, train=train, device="cpu"))


def make_dataset(work: Path) -> Any:
    """Write train/val files from the synthetic set and return a DatasetVersion pointing at them."""

    from ase.io import read, write

    from gmd_active_learning.contracts import DatasetVersion
    from gmd_active_learning.contracts.ids import file_sha256

    maker = _load(DATASET_MAKER, "_tiny_h2o")
    work.mkdir(parents=True, exist_ok=True)
    full = work / "tiny_h2o.extxyz"
    maker.write_samples_extxyz(maker.build(), full)
    frames = read(full, index=":")
    groups = sorted({atoms.info["split_group"] for atoms in frames})
    val_groups = set(groups[-1:])
    train = [atoms for atoms in frames if atoms.info["split_group"] not in val_groups]
    val = [atoms for atoms in frames if atoms.info["split_group"] in val_groups]
    write(work / "train.extxyz", train, format="extxyz")
    write(work / "val.extxyz", val, format="extxyz")
    return DatasetVersion("tiny-h2o-smoke", str(work), file_sha256(full), len(frames), len(train), len(val), None, (), "smoke",
                          {"train_file": str(work / "train.extxyz"), "val_file": str(work / "val.extxyz")})


def infer(backend: Any, model: Path, symbols: str) -> tuple[int, dict[str, Any]]:
    from ase.data import atomic_numbers

    from gmd_active_learning.core.exceptions import AdapterError, ConfigurationError

    if not model.is_file():
        return EXIT_MODEL_PATH_ERROR, {"error": f"model file not found: {model}"}
    try:
        info = backend.model_info(model)
    except Exception as exc:  # noqa: BLE001
        return EXIT_MODEL_LOAD_FAILED, {"error": f"cannot load {model.name}: {type(exc).__name__}: {exc}"}
    numbers = np.array([atomic_numbers[symbol] for symbol in _split_symbols(symbols)], dtype=np.int64)
    unknown = sorted(set(numbers.tolist()) - set(info.species))
    if unknown:
        return EXIT_ELEMENT_MISMATCH, {"error": f"species {unknown} not in model species {list(info.species)}"}
    positions = H2O_POSITIONS[: len(numbers)] if len(numbers) <= 3 else np.arange(len(numbers) * 3, dtype=float).reshape(-1, 3)
    try:
        prediction = backend.predict(model, positions, numbers)
    except (AdapterError, ConfigurationError) as exc:
        return EXIT_INFERENCE_FAILED, {"error": str(exc)}
    finite = bool(np.isfinite(prediction.energy) and np.all(np.isfinite(prediction.forces)))
    return (EXIT_OK if finite else EXIT_INFERENCE_FAILED), {
        "model": str(model), "kind": info.kind, "species": list(info.species), "cutoff": info.cutoff,
        "energy_ev": prediction.energy, "forces_ev_per_a": prediction.forces.tolist(), "finite": finite,
    }


def _split_symbols(text: str) -> list[str]:
    symbols, current = [], ""
    for char in text:
        if char.isupper() and current:
            symbols.append(current)
            current = char
        else:
            current += char
    return [*symbols, current] if current else symbols


def run(work: Path, *, gmd_sgt: str | None, epochs: int, symbols: str) -> tuple[int, dict[str, Any]]:
    from gmd_active_learning.core.exceptions import CapabilityUnavailableError

    report: dict[str, Any] = {"work_dir": str(work)}
    started = time.monotonic()
    try:
        backend = _backend(gmd_sgt, epochs)
    except CapabilityUnavailableError as exc:
        return EXIT_IMPORT_ERROR, {**report, "error": str(exc)}
    try:
        dataset = make_dataset(work / "dataset")
        report["dataset"] = {"n_train": dataset.n_train, "n_val": dataset.n_val}
        output = backend.train(dataset, parent=None, work_dir=work / "train", seed=11)
        report["checkpoint"] = str(output.checkpoint)
        export = Path(backend.export(Path(output.checkpoint), work / "export"))
        report["export"] = str(export)
    except Exception as exc:  # noqa: BLE001 - every failure is reported with its stage
        return EXIT_RUNTIME_ERROR, {**report, "error": f"{type(exc).__name__}: {exc}"}
    code, exported = infer(backend, export, symbols)
    report["export_inference"] = exported
    if code != EXIT_OK:
        return code, report
    code, eager = infer(backend, Path(output.checkpoint), symbols)
    report["checkpoint_inference"] = eager
    if code != EXIT_OK:
        return code, report
    deviation = float(np.max(np.abs(np.asarray(exported["forces_ev_per_a"]) - np.asarray(eager["forces_ev_per_a"]))))
    report["export_vs_checkpoint_max_force_deviation"] = deviation
    report["seconds"] = round(time.monotonic() - started, 2)
    if deviation > 1.0e-3:
        report["error"] = f"export and checkpoint disagree by {deviation:.2e} eV/A"
        return EXIT_INFERENCE_FAILED, report
    return EXIT_OK, report


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if not args.infer_only:
        return run(args.work_dir, gmd_sgt=args.gmd_sgt, epochs=args.epochs, symbols=args.symbols)
    if args.model is None:
        parser.error("--infer-only needs --model")
    from gmd_active_learning.core.exceptions import CapabilityUnavailableError

    try:
        backend = _backend(args.gmd_sgt, args.epochs)
    except CapabilityUnavailableError as exc:
        return EXIT_IMPORT_ERROR, {"error": str(exc)}
    return infer(backend, args.model, args.symbols)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", type=Path, default=Path("sgt_tiny_h2o_smoke"))
    parser.add_argument("--gmd-sgt", default=os.environ.get("GMD_SGT_PATH"), help="GMD_SGT checkout (default $GMD_SGT_PATH or installed)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--symbols", default="OHH", help="species of the inference structure (e.g. OHH; C triggers exit 7)")
    parser.add_argument("--infer-only", action="store_true", help="only run step 4 on --model")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="also write the JSON report to this file")
    args = parser.parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):  # GMD_SGT training/export logs; stdout carries only the report
        code, report = _dispatch(parser, args)
    report["exit_code"] = code
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
