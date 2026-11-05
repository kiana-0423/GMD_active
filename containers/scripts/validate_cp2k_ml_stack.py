#!/usr/bin/env python3
# Ported from PHAL (containers/scripts/validate_cp2k_ml_stack.py) at 96263b6; MIT License, see NOTICE.
"""Validate the experimental CP2K-ML development toolchain without requiring CP2K source."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

COMMANDS = ("gcc", "gfortran", "cmake", "ninja", "mpirun")


def _compile_and_run(
    compiler: str, source_name: str, source: str, libraries: tuple[str, ...]
) -> dict[str, Any]:
    """Compile and execute a tiny dependency probe in an isolated directory."""
    with tempfile.TemporaryDirectory(prefix="cp2k-ml-probe-") as raw_dir:
        directory = Path(raw_dir)
        source_path = directory / source_name
        executable = directory / "probe"
        source_path.write_text(source, encoding="utf-8")
        command = [compiler, str(source_path), *libraries, "-o", str(executable)]
        compiled = subprocess.run(command, check=False, capture_output=True, text=True)
        if compiled.returncode != 0:
            return {
                "available": False,
                "compile_command": command,
                "error": (compiled.stderr or compiled.stdout).strip(),
            }
        executed = subprocess.run(
            [str(executable)], check=False, capture_output=True, text=True
        )
        return {
            "available": executed.returncode == 0,
            "compile_command": command,
            "output": (executed.stdout or executed.stderr).strip(),
        }


def _native_library_probes() -> dict[str, dict[str, Any]]:
    blas_fftw = _compile_and_run(
        "gcc",
        "blas_fftw.c",
        """#include <cblas.h>
#include <fftw3.h>
int main(void) {
  double x[2] = {1.0, 2.0}, y[2] = {3.0, 4.0};
  fftw_complex in[2], out[2];
  fftw_plan p = fftw_plan_dft_1d(2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  if (!p || cblas_ddot(2, x, 1, y, 1) != 11.0) return 1;
  fftw_destroy_plan(p);
  return 0;
}
""",
        ("-lblas", "-lfftw3", "-lm"),
    )
    lapack = _compile_and_run(
        "gfortran",
        "lapack.f90",
        """program probe
  implicit none
  double precision :: a(1,1), b(1,1)
  integer :: ipiv(1), info
  a(1,1) = 2.0d0
  b(1,1) = 4.0d0
  call dgesv(1, 1, a, 1, ipiv, b, 1, info)
  if (info /= 0 .or. abs(b(1,1) - 2.0d0) > 1.0d-12) stop 1
end program probe
""",
        ("-llapack", "-lblas"),
    )
    return {
        "blas": dict(blas_fftw),
        "lapack": lapack,
        "fftw": dict(blas_fftw),
    }


def _first_line(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else "unknown"


def collect() -> dict[str, Any]:
    import ase
    import ase.io
    import numpy
    import scipy
    import torch
    import yaml

    command_paths = {name: shutil.which(name) for name in COMMANDS}
    torch_root = Path(torch.__file__).resolve().parent
    libtorch_dir = torch_root / "lib"
    native_libraries = _native_library_probes()
    payload: dict[str, Any] = {
        "platform": {"machine": platform.machine(), "system": platform.system()},
        "commands": command_paths,
        "versions": {
            "gcc": _first_line(["gcc", "--version"]) if command_paths["gcc"] else None,
            "gfortran": _first_line(["gfortran", "--version"]) if command_paths["gfortran"] else None,
            "cmake": _first_line(["cmake", "--version"]) if command_paths["cmake"] else None,
            "openmpi": _first_line(["mpirun", "--version"]) if command_paths["mpirun"] else None,
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pyyaml": yaml.__version__,
            # Installed at build time so dataset tooling runs offline.
            "ase": ase.__version__,
        },
        "libtorch": {
            "directory": str(libtorch_dir),
            "available": libtorch_dir.is_dir() and any(libtorch_dir.glob("libtorch*")),
        },
        "native_libraries": native_libraries,
        "cp2k_source_in_image": False,
    }
    passed = (
        all(command_paths.values())
        and payload["libtorch"]["available"]
        and all(item["available"] for item in native_libraries.values())
    )
    payload["overall_status"] = "passed" if passed else "failed"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = collect()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
