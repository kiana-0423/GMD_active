#!/usr/bin/env python3
# Ported from PHAL (containers/scripts/validate_platform.py) at 96263b6; MIT License, see NOTICE.
"""Validate the ARM64 PyTorch/CUDA platform of a GMD_active image; emit one JSON document.

Keys kept from PHAL: machine, python, platform, torch, torch_cuda,
cuda_available, cuda_device_count, gpu_name, cuda_device_capability,
cuda_tensor_test, require_cuda, errors, failures, package_versions, status.
The stack-specific keys are re-targeted: PHAL's nequip / allegro / phal_import
become gmd_sgt / e3nn / gmd_sgt_import.

Environment:
    GMD_AL_REQUIRE_CUDA=1     CUDA and a Miyabi-G H100/GH200 device are mandatory
                              (replaces PHAL_REQUIRE_CUDA).
    GMD_AL_PLATFORM_MODULES   comma-separated modules that must import
                              (default "torch,gmd_sgt"); e3nn is always reported
                              but optional.

Exit status: 0 when status == "passed", otherwise 1. A missing or broken torch
is reported in the JSON (torch = null, a failure record), never a crash.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import platform
import sys
import traceback
from typing import Any

DEFAULT_REQUIRED = ("torch", "gmd_sgt")
OPTIONAL = ("e3nn",)
EXPECTED_GPU_TOKENS = ("H100", "GH200")


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _required_modules() -> tuple[str, ...]:
    raw = os.environ.get("GMD_AL_PLATFORM_MODULES")
    if raw is None:
        return DEFAULT_REQUIRED
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def _failure(stage: str, exc: BaseException, device: str | None) -> dict[str, Any]:
    return {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "selected_device": device,
    }


def collect() -> dict[str, Any]:
    require_cuda = os.environ.get("GMD_AL_REQUIRE_CUDA") == "1"
    required = _required_modules()
    result: dict[str, Any] = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": None,
        "torch_cuda": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "gpu_name": None,
        "cuda_device_capability": None,
        "cuda_tensor_test": False,
        "gmd_sgt": None,
        "e3nn": None,
        "gmd_sgt_import": False,
        "required_modules": list(required),
        "require_cuda": require_cuda,
        "errors": [],
        "failures": [],
    }

    modules: dict[str, Any] = {}
    for name in dict.fromkeys((*required, "torch", "gmd_sgt", *OPTIONAL)):
        try:
            # Scientific packages print banners/warnings at import; keep stdout
            # reserved for the machine-readable JSON document.
            with contextlib.redirect_stdout(sys.stderr):
                modules[name] = importlib.import_module(name)
        except Exception as exc:  # binary import failures must be reported too
            if name in required:
                result["errors"].append(f"failed to import {name}: {type(exc).__name__}: {exc}")
            result["failures"].append(_failure(f"import-{name}", exc, None))

    torch = modules.get("torch")
    if torch is not None:
        result["torch"] = _version(torch)
        result["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            result["cuda_available"] = bool(torch.cuda.is_available())
            result["cuda_device_count"] = int(torch.cuda.device_count())
            if result["cuda_available"]:
                result["gpu_name"] = str(torch.cuda.get_device_name(0))
                result["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
                left = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
                product = left @ left.transpose(0, 1)
                torch.cuda.synchronize()
                result["cuda_tensor_test"] = bool(torch.isfinite(product).all().item())
                if not result["cuda_tensor_test"]:
                    result["errors"].append("CUDA matrix multiplication produced NaN or Inf")
                if require_cuda and not any(
                    token in result["gpu_name"].upper() for token in EXPECTED_GPU_TOKENS
                ):
                    result["errors"].append(
                        f"expected a Miyabi-G H100/GH200 device, found {result['gpu_name']}"
                    )
            elif require_cuda:
                result["errors"].append("torch imported but CUDA is unavailable")
        except Exception as exc:
            result["errors"].append(f"CUDA tensor validation failed: {type(exc).__name__}: {exc}")
            result["failures"].append(_failure("cuda-tensor", exc, "cuda:0"))
    elif require_cuda:
        result["errors"].append("CUDA is required but torch could not be imported")

    if "gmd_sgt" in modules:
        result["gmd_sgt"] = _version(modules["gmd_sgt"])
    if "e3nn" in modules:
        result["e3nn"] = _version(modules["e3nn"])
    result["gmd_sgt_import"] = "gmd_sgt" in modules
    result["package_versions"] = {name: _version(module) for name, module in modules.items()}
    for failure in result["failures"]:
        failure["package_versions"] = result["package_versions"]

    if result["machine"] not in {"aarch64", "arm64"}:
        result["errors"].append(f"expected ARM64 machine, found {result['machine']}")
    if sys.version_info < (3, 10):
        result["errors"].append("GMD_SGT requires Python 3.10 or newer")

    cuda_passed = result["cuda_available"] and result["cuda_tensor_test"]
    passed = not result["errors"] and all(name in modules for name in required)
    if require_cuda:
        passed = passed and cuda_passed
    result["status"] = "passed" if passed else "failed"
    return result


def main() -> int:
    result = collect()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
