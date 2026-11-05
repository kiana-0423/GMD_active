#!/usr/bin/env python3
# Ported from PHAL (containers/scripts/validate_md_stack.py and validate_md_binary_static.py) at 96263b6; MIT License, see NOTICE.
"""Validate the GMD MD stack: a static binary check or a real 0-/10-step run.

Modes
-----
``--mode static``  (image build time; PHAL's validate_md_binary_static.py)
    Never executes ``gmd``. Build hosts usually lack ``libcuda.so.1``, which the
    NGC LibTorch links, so the binary is checked for existence, executability,
    architecture, LibTorch linkage and unresolved libraries (``libcuda.so.1``
    is the one expected gap, reported as ``not_available_on_local_builder``).

``--mode runtime``  (PBS job under ``apptainer exec --nv``; PHAL's validate_md_stack.py)
    Copies the probe inputs and the random GMD_SGT probe model into a writable
    work directory and runs ``gmd h2.xyz run0.run`` and ``gmd h2.xyz run10.run``.
    GMD refuses ``run 0``, so the single-point stage runs one step and gates on
    the step-0 row of ``output.log``. Gates: exit status 0, ``status.json``
    state ``completed`` with the probe model's sha256, rows for every expected
    step, and finite PE / KE / E_total / T on each row (P is ``nan`` by design
    when ``P_valid`` is 0 and is only checked when valid).

Output: one JSON document on stdout (and --output PATH); exit 0 on PASS.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROBE_DIR = Path(os.environ.get("GMD_AL_PROBE_DIR", "/opt/gmd_al/probes/gmd"))
MODEL_NAME = "sgt-probe.pt"
PROBE_INPUTS = ("h2.xyz", "run0.run", "run10.run")
EXPECTED_DRIVER_GAPS = ("libcuda.so.1", "libnvidia-ml.so.1")
FINITE_COLUMNS = ("PE[eV]", "KE[eV]", "E_total[eV]", "T[K]")
LOG_FAILURE = re.compile(r"segmentation fault|illegal memory access|cuda error|terminate called", re.I)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None, timeout: float | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                                   check=False, timeout=timeout)
        return {"command": command, "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}"}


def resolve_gmd(explicit: str | None) -> Path | None:
    for candidate in (explicit, os.environ.get("GMD_EXECUTABLE"), shutil.which("gmd"), "/opt/gmd/bin/gmd"):
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    return Path(explicit) if explicit else None


def normalise_arch(text: str) -> str:
    lowered = text.lower()
    if "aarch64" in lowered or "arm64" in lowered:
        return "aarch64"
    if "x86-64" in lowered or "x86_64" in lowered:
        return "x86_64"
    return "unknown"


def static_check(gmd: Path | None, expected_arch: str) -> dict[str, Any]:
    exists = bool(gmd and gmd.is_file())
    mode = gmd.stat().st_mode if exists else 0
    result: dict[str, Any] = {
        "gmd_path": str(gmd) if gmd else None,
        "gmd_exists": exists,
        "gmd_executable": bool(exists and mode & stat.S_IXUSR),
        "expected_architecture": expected_arch,
    }
    if not exists:
        result.update({"architecture": "unknown", "linkage": None, "libtorch_dependency": False,
                       "missing_libraries": [], "cuda_driver_library": "unknown"})
        return result
    file_result = run(["file", "-L", str(gmd)]) if shutil.which("file") else None
    result["file"] = file_result
    result["architecture"] = normalise_arch(file_result["stdout"]) if file_result else "unknown"
    if shutil.which("ldd"):
        linkage = run(["ldd", str(gmd)])
    elif shutil.which("otool"):
        linkage = run(["otool", "-L", str(gmd)])
    else:
        linkage = None
    result["linkage"] = linkage
    text = (linkage["stdout"] + linkage["stderr"]) if linkage else ""
    result["libtorch_dependency"] = "libtorch" in text
    missing = sorted({line.split("=>")[0].strip() for line in text.splitlines() if "not found" in line})
    result["missing_libraries"] = missing
    if any(name in missing for name in EXPECTED_DRIVER_GAPS):
        result["cuda_driver_library"] = "not_available_on_local_builder"
    elif "libcuda" in text:
        result["cuda_driver_library"] = "available"
    else:
        result["cuda_driver_library"] = "not_linked"
    return result


def static_passed(result: dict[str, Any]) -> bool:
    unexpected = [lib for lib in result["missing_libraries"] if lib not in EXPECTED_DRIVER_GAPS]
    return all((
        result["gmd_exists"],
        result["gmd_executable"],
        result["architecture"] == result["expected_architecture"],
        result["libtorch_dependency"],
        not unexpected,
    ))


def parse_log(path: Path) -> dict[str, Any]:
    """Parse GMD's output.log: '#'-prefixed header naming the columns, then rows."""
    if not path.is_file():
        return {"exists": False, "rows": [], "steps": [], "finite": False}
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            tokens = stripped.lstrip("#").split()
            if header is None and tokens and tokens[0] == "step":
                header = tokens
            continue
        if header is None:
            continue
        parts = stripped.split()
        if len(parts) < len(header):
            continue
        try:
            rows.append({name: float(value) for name, value in zip(header, parts)})
        except ValueError:
            continue
    finite = bool(rows)
    for row in rows:
        values = [row.get(name, math.nan) for name in FINITE_COLUMNS]
        if row.get("P_valid", 0.0) == 1.0:
            values.append(row.get("P[bar]", math.nan))
        if not all(math.isfinite(value) for value in values):
            finite = False
    return {
        "exists": True,
        "columns": header,
        "rows": len(rows),
        "steps": [int(row["step"]) for row in rows if "step" in row],
        "first": rows[0] if rows else None,
        "last": rows[-1] if rows else None,
        "finite": finite,
    }


def run_case(gmd: Path, workdir: Path, probe_dir: Path, model: Path, run_file: str,
             expected_last_step: int, timeout: float) -> dict[str, Any]:
    case_dir = workdir / Path(run_file).stem
    case_dir.mkdir(parents=True, exist_ok=True)
    for name in PROBE_INPUTS:
        shutil.copy2(probe_dir / name, case_dir / name)
    shutil.copy2(model, case_dir / MODEL_NAME)
    completed = run([str(gmd), "h2.xyz", run_file], cwd=case_dir, timeout=timeout)
    log = parse_log(case_dir / "output.log")
    status_path = case_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else None
    model_sha = sha256(case_dir / MODEL_NAME)
    output = completed["stdout"] + completed["stderr"]
    checks = {
        "returncode_zero": completed["returncode"] == 0,
        "status_completed": bool(status and status.get("state") == "completed"),
        "model_sha256_recorded": bool(status and status.get("model_sha256") == model_sha),
        "expected_steps": log["steps"] == list(range(0, expected_last_step + 1)),
        "energy_finite": log["finite"],
        "no_runtime_errors": not LOG_FAILURE.search(output),
    }
    return {
        "workdir": str(case_dir),
        "command": completed["command"],
        "returncode": completed["returncode"],
        "stdout_tail": completed["stdout"][-2000:],
        "stderr_tail": completed["stderr"][-2000:],
        "status_file": status,
        "log": log,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("static", "runtime"), default="static")
    parser.add_argument("--gmd", help="gmd executable (default: $GMD_EXECUTABLE, PATH, /opt/gmd/bin/gmd)")
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR,
                        help="directory with h2.xyz, run0.run, run10.run (and sgt-probe.pt)")
    parser.add_argument("--model", type=Path, help=f"probe model (default: <probe-dir>/{MODEL_NAME})")
    parser.add_argument("--export-model", action="store_true",
                        help="export a fresh random GMD_SGT probe model into the work directory")
    parser.add_argument("--workdir", type=Path, help="writable directory (default: a new temp dir)")
    parser.add_argument("--expected-arch", default=normalise_arch(platform.machine()))
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per gmd run")
    parser.add_argument("--output", type=Path, help="also write the JSON document here")
    args = parser.parse_args(argv)

    gmd = resolve_gmd(args.gmd)
    result: dict[str, Any] = {
        "mode": args.mode,
        "platform.machine": platform.machine(),
        "python": platform.python_version(),
        "probe_dir": str(args.probe_dir),
        "probe_model_path": None,
        "probe_model_sha256": None,
        "probe_model_metadata": None,
        "gmd_runtime": "not_run",
        "run0": "not_run",
        "run10": "not_run",
        "run0_passed": False,
        "run10_passed": False,
        "energy_finite": False,
        "overall_status": "FAIL",
        "errors": [],
        "traceback": "",
    }
    try:
        result["static"] = static_check(gmd, args.expected_arch)
        result["static_validation"] = "passed" if static_passed(result["static"]) else "failed"

        model = args.model or args.probe_dir / MODEL_NAME
        if args.mode == "runtime":
            workdir = args.workdir or Path(tempfile.mkdtemp(prefix="gmd-probe-"))
            workdir.mkdir(parents=True, exist_ok=True)
            result["workdir"] = str(workdir)
            if args.export_model:
                sys.path.insert(0, str(SCRIPT_DIR))
                from build_gmd_probe_model import build_probe_model

                model = workdir / MODEL_NAME
                with contextlib.redirect_stdout(sys.stderr):
                    build_probe_model(model)
        result["probe_model_path"] = str(model)
        result["probe_model_sha256"] = sha256(model)
        sidecar = model.with_suffix(".json")
        if sidecar.is_file():
            result["probe_model_metadata"] = json.loads(sidecar.read_text(encoding="utf-8"))
        if result["probe_model_sha256"] is None:
            result["errors"].append(f"probe model not found: {model}")

        if args.mode == "static":
            if result["static_validation"] == "passed" and result["probe_model_sha256"]:
                result["overall_status"] = "PASS"
        elif gmd is None or not result["static"]["gmd_exists"]:
            result["errors"].append("gmd executable not found")
        elif result["probe_model_sha256"]:
            result["gmd_runtime"] = "run"
            run0 = run_case(gmd, workdir, args.probe_dir, model, "run0.run", 1, args.timeout)
            run10 = run_case(gmd, workdir, args.probe_dir, model, "run10.run", 10, args.timeout)
            # The single-point gate is the step-0 row (initial configuration).
            run0_point = bool(run0["log"]["steps"][:1] == [0] and run0["log"]["finite"])
            result["run0"] = run0
            result["run10"] = run10
            result["run0_passed"] = run0["passed"] and run0_point
            result["run10_passed"] = run10["passed"]
            result["energy_finite"] = run0["log"]["finite"] and run10["log"]["finite"]
            if result["run0_passed"] and result["run10_passed"] and result["energy_finite"]:
                result["overall_status"] = "PASS"
    except Exception as exc:  # pragma: no cover - container diagnostic path
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()

    rendered = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
