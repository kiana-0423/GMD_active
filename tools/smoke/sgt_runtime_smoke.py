#!/usr/bin/env python
# Ported from PHAL (scripts/smoke_allegro.py) at 96263b6; MIT License, see NOTICE.
"""GMD_SGT runtime smoke check (ledger OPS-002).

Reports torch / device / GMD_SGT import status and runs a tiny tensor and a GMD_SGT import probe.
Keeps PHAL's exit-code contract:

    0 ok   1 runtime error   2 import error   3 requested device unavailable   4 wrong environment

Environment: ``GMD_SGT_SMOKE_DEVICE`` = auto | cpu | cuda | mps (default auto),
``GMD_SGT_REQUIRED_CONDA_ENV`` (optional guard), ``GMD_SGT_PATH`` (optional GMD_SGT checkout).
``--json`` prints one JSON document instead of text.

    python tools/smoke/sgt_runtime_smoke.py [--json] [--gmd-sgt PATH]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_IMPORT_ERROR = 2
EXIT_DEVICE_UNAVAILABLE = 3
EXIT_ENV_ERROR = 4
SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "mps")
MIN_GMD_SGT = (0, 4, 0)


def _version(distribution: str, module: Any | None = None) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown")) if module is not None else "unknown"


def select_device(torch: Any, requested: str) -> tuple[str | None, str | None]:
    """(device, error). ``auto`` prefers CUDA, then MPS, then CPU."""

    if requested not in SUPPORTED_DEVICES:
        return None, f"unsupported GMD_SGT_SMOKE_DEVICE={requested!r}; expected one of {SUPPORTED_DEVICES}"
    cuda = bool(torch.cuda.is_available())
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "cuda" and not cuda:
        return None, "GMD_SGT_SMOKE_DEVICE=cuda was requested, but CUDA is unavailable"
    if requested == "mps" and not mps:
        return None, "GMD_SGT_SMOKE_DEVICE=mps was requested, but MPS is unavailable"
    if requested == "auto":
        return ("cuda" if cuda else "mps" if mps else "cpu"), None
    return requested, None


def run(gmd_sgt_path: str | None = None) -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {"python": sys.version.split()[0], "conda_env": os.environ.get("CONDA_DEFAULT_ENV")}
    required = os.environ.get("GMD_SGT_REQUIRED_CONDA_ENV")
    if required and report["conda_env"] != required:
        report["error"] = f"this smoke check must run inside conda environment {required}"
        return EXIT_ENV_ERROR, report
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001 - report any import failure
        report["error"] = f"torch import failed: {type(exc).__name__}: {exc}"
        return EXIT_IMPORT_ERROR, report
    report["torch"] = str(torch.__version__)
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_version"] = getattr(torch.version, "cuda", None)
    report["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    device, error = select_device(torch, os.environ.get("GMD_SGT_SMOKE_DEVICE", "auto").lower())
    if error:
        report["error"] = error
        return EXIT_DEVICE_UNAVAILABLE, report
    report["device"] = device
    try:
        x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3).requires_grad_(True)
        (x**2).sum().backward()
        report["tensor_check"] = bool(torch.allclose(x.grad, 2 * x.detach()))
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"tensor check failed on {device}: {type(exc).__name__}: {exc}"
        return EXIT_RUNTIME_ERROR, report
    source = gmd_sgt_path or os.environ.get("GMD_SGT_PATH")
    if source:
        resolved = str(Path(source).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        report["gmd_sgt_path"] = resolved
    try:
        sgt = importlib.import_module("gmd_sgt")
        for module in ("gmd_sgt.api", "gmd_sgt.inference.calculator"):
            importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        report["gmd_sgt_import"] = False
        report["error"] = f"gmd_sgt import failed: {type(exc).__name__}: {exc}"
        return EXIT_IMPORT_ERROR, report
    version = _version("gmd-sgt", sgt)
    report.update({"gmd_sgt_import": True, "gmd_sgt": version, "gmd_sgt_file": str(getattr(sgt, "__file__", ""))})
    for dependency in ("e3nn", "ase", "numpy"):
        try:
            report[dependency] = _version(dependency, importlib.import_module(dependency))
        except Exception as exc:  # noqa: BLE001
            report[dependency] = f"unavailable ({type(exc).__name__})"
    try:
        parts = tuple(int(part) for part in version.split(".")[:3])
    except ValueError:
        parts = (0, 0, 0)
    if parts < MIN_GMD_SGT:
        report["error"] = f"GMD_SGT {version} < {'.'.join(map(str, MIN_GMD_SGT))} (needs per-atom force weights)"
        return EXIT_IMPORT_ERROR, report
    return EXIT_OK, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--gmd-sgt", default=None, help="GMD_SGT checkout (default $GMD_SGT_PATH or installed package)")
    args = parser.parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):  # GMD_SGT prints notices on import; keep stdout for the report
        code, report = run(args.gmd_sgt)
    report["exit_code"] = code
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        for key, value in sorted(report.items()):
            print(f"{key}: {value}")
        print("GMD_SGT smoke: OK" if code == EXIT_OK else f"GMD_SGT smoke: FAILED ({code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
