#!/usr/bin/env python3
# Ported from PHAL (containers/scripts/validate_cp2k_ml.py) at 96263b6; MIT License, see NOTICE.
"""Validate a CP2K labeling stack with an H2 single point; emit one JSON document.

Two stages, both executed for real (no fallbacks):

1. ``version``: ``cp2k.psmp --version`` must report the expected CP2K version
   (``--expected-version`` or ``$CP2K_VERSION_EXPECTED``, e.g. ``v2026.2``).
2. ``scf``: the native Quickstep H2 single point in ``probes/cp2k/h2.inp`` must
   exit 0, contain ``SCF run converged`` and ``PROGRAM ENDED``, and report a
   finite total energy.

The input is copied into a writable work directory first (an Apptainer image
is read-only). With ``--mpi-ranks N`` (N > 1) CP2K is launched through
``mpirun -np N``. Exit 0 when overall_status is ``passed``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CONVERGED = "SCF run converged"
ENDED = "PROGRAM ENDED"
ENERGY_RE = re.compile(r"ENERGY\|\s+Total FORCE_EVAL \( QS \) energy \[hartree\]\s+(\S+)")
SCF_STEPS_RE = re.compile(r"SCF run converged in\s+(\d+)\s+steps")
VERSION_RE = re.compile(r"CP2K version\s+(\S+)")
REVISION_RE = re.compile(r"Source code revision\s+(\S+)")
FLAGS_RE = re.compile(r"cp2kflags:\s*(.+)")


def default_input() -> Path:
    candidates = [
        Path(os.environ.get("GMD_AL_HOME", "/opt/gmd_al")) / "probes" / "cp2k" / "h2.inp",
        SCRIPT_DIR.parent / "probes" / "cp2k" / "h2.inp",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def run(command: list[str], cwd: Path | None = None, timeout: float | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                                   check=False, timeout=timeout)
        return {"command": command, "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}"}


def parse_version(text: str) -> dict[str, Any]:
    version = VERSION_RE.search(text)
    revision = REVISION_RE.search(text)
    flags = FLAGS_RE.search(text)
    return {
        "cp2k_version": version.group(1) if version else None,
        "revision": revision.group(1) if revision else None,
        "cp2kflags": flags.group(1).split() if flags else [],
    }


def parse_scf_output(text: str) -> dict[str, Any]:
    energies = ENERGY_RE.findall(text)
    energy: float | None
    try:
        energy = float(energies[-1].replace("D", "E")) if energies else None
    except ValueError:
        energy = None
    steps = SCF_STEPS_RE.findall(text)
    return {
        "scf_converged": CONVERGED in text,
        "program_ended": ENDED in text,
        "scf_steps": int(steps[-1]) if steps else None,
        "total_energy_hartree": energy,
        "energy_finite": energy is not None and math.isfinite(energy),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cp2k", default=os.environ.get("CP2K_EXE", "cp2k.psmp"))
    parser.add_argument("--input", type=Path, default=None, help="CP2K input (default: probes/cp2k/h2.inp)")
    parser.add_argument("--workdir", type=Path, help="writable directory (default: a new temp dir)")
    parser.add_argument("--expected-version", default=os.environ.get("CP2K_VERSION_EXPECTED"))
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, help="also write the JSON document here")
    args = parser.parse_args(argv)

    input_path = args.input or default_input()
    result: dict[str, Any] = {
        "cp2k": args.cp2k,
        "input": str(input_path),
        "expected_version": args.expected_version,
        "mpi_ranks": args.mpi_ranks,
        "version_check": "not_run",
        "scf_test": "not_run",
        "errors": [],
        "overall_status": "failed",
    }

    executable = shutil.which(args.cp2k) or (args.cp2k if Path(args.cp2k).is_file() else None)
    if executable is None:
        result["errors"].append(f"CP2K executable not found: {args.cp2k}")
    elif not input_path.is_file():
        result["errors"].append(f"CP2K input not found: {input_path}")
    else:
        version_run = run([executable, "--version"], timeout=120)
        version = parse_version(version_run["stdout"] + version_run["stderr"])
        result.update(version)
        result["version_returncode"] = version_run["returncode"]
        expected = (args.expected_version or "").lstrip("v")
        version_ok = version_run["returncode"] == 0 and version["cp2k_version"] is not None
        if expected and version["cp2k_version"] != expected:
            version_ok = False
            result["errors"].append(
                f"expected CP2K version {expected}, found {version['cp2k_version']}")
        result["version_check"] = "passed" if version_ok else "failed"

        workdir = args.workdir or Path(tempfile.mkdtemp(prefix="cp2k-probe-"))
        workdir.mkdir(parents=True, exist_ok=True)
        local_input = workdir / input_path.name
        shutil.copy2(input_path, local_input)
        output = workdir / (input_path.stem + ".out")
        command = [executable, "-i", local_input.name, "-o", output.name]
        if args.mpi_ranks > 1:
            command = [args.mpirun, "-np", str(args.mpi_ranks), *command]
        scf_run = run(command, cwd=workdir, timeout=args.timeout)
        text = output.read_text(encoding="utf-8", errors="replace") if output.is_file() else ""
        scf = parse_scf_output(text + scf_run["stdout"])
        result.update(scf)
        result["workdir"] = str(workdir)
        result["scf_command"] = command
        result["scf_returncode"] = scf_run["returncode"]
        result["stderr_tail"] = scf_run["stderr"][-2000:]
        scf_ok = (scf_run["returncode"] == 0 and scf["scf_converged"] and scf["program_ended"]
                  and scf["energy_finite"])
        if not scf["scf_converged"]:
            result["errors"].append(f"'{CONVERGED}' not found in {output}")
        result["scf_test"] = "passed" if scf_ok else "failed"
        if version_ok and scf_ok:
            result["overall_status"] = "passed"

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
