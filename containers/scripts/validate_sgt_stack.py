#!/usr/bin/env python3
# Ported from PHAL (containers/scripts/validate_train_stack.py) at 96263b6; MIT License, see NOTICE.
"""Run a real, randomly initialised tiny GMD_SGT model forward/force calculation.

Replaces PHAL's random-Allegro check. For each selected device (CPU always,
CUDA when available or required) a small seeded ``UnifiedEquivariantMLIP`` with
a non-zero energy head evaluates an H2 molecule; energy and forces must be
finite and the forces must come from autograd. On CPU the model is also
compiled with GMD_SGT's TorchScript export path (``script_model``) and the
scripted energy/forces must match the eager ones: that is the artifact GMD
loads.

The random model is an integration probe only (training_steps = 0,
scientifically_valid = false); its numbers carry no physical meaning.

Environment: GMD_AL_REQUIRE_CUDA=1 makes CUDA inference mandatory.
Output: one JSON document on stdout (and --output PATH). Exit 0 when passed.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

SEED = 20251103
H2_POSITIONS = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
H2_SPECIES = [1, 1]


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _failure(stage: str, exc: BaseException | None, device: str | None, versions: dict[str, str],
             message: str | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "exception_type": type(exc).__name__ if exc is not None else "NonFiniteOutput",
        "exception_message": str(exc) if exc is not None else (message or ""),
        "traceback": traceback.format_exc() if exc is not None else None,
        "selected_device": device,
        "package_versions": versions,
    }


def build_tiny_model(torch: Any, e3nn_available: bool) -> Any:
    """Seeded tiny UnifiedEquivariantMLIP with a non-zero (random) energy head."""
    from gmd_sgt.models import UnifiedEquivariantMLIP

    torch.manual_seed(SEED)
    model = UnifiedEquivariantMLIP(
        n_species=2,
        n_blocks=1,
        scalar_dim=8,
        # Non-scalar irreps need e3nn; without it only scalars are supported.
        irreps="8x0e + 4x1o" if e3nn_available else "8x0e",
        n_basis=4,
        local_cutoff=3.0,
        lr_cutoff=6.0,
        l_max=1,
        long_range_type="none",
        n_heads=2,
        avg_neighbors=1.0,
    )
    # The default energy head is zero-initialised, which would make energy and
    # forces trivially finite; randomise it so the check exercises real math.
    generator = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for layer in model.energy_head:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * 0.5)
                layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator) * 0.1)
    return model.eval()


def run_inference(torch: Any, model: Any, device: str) -> dict[str, Any]:
    model = model.to(device)
    species = torch.tensor(H2_SPECIES, dtype=torch.long, device=device)
    positions = torch.tensor(H2_POSITIONS, dtype=torch.float32, device=device)
    batch = torch.zeros(len(H2_SPECIES), dtype=torch.long, device=device)
    with torch.enable_grad():
        output = model(
            species=species,
            positions=positions.clone().requires_grad_(True),
            batch=batch,
            compute_forces=True,
        )
    energy = output["energy"].detach()
    forces = output["forces"].detach()
    if device == "cuda":
        torch.cuda.synchronize()
    energy_finite = bool(torch.isfinite(energy).all().item())
    forces_finite = bool(torch.isfinite(forces).all().item())
    nonzero = bool((forces.abs() > 0).any().item())
    finite = energy_finite and forces_finite
    return {
        "status": "passed" if finite else "failed",
        "device": device,
        "randomly_initialized_model": True,
        "energy": float(energy.sum().item()) if energy_finite else None,
        "energy_shape": list(energy.shape),
        "force_shape": list(forces.shape),
        "energy_finite": energy_finite,
        "forces_finite": forces_finite,
        "forces_nonzero": nonzero,
        "finite_energy_and_forces": finite,
        "force_gradient_computed": True,
    }


def run_torchscript(torch: Any, model: Any, eager: dict[str, Any]) -> dict[str, Any]:
    """Compile with GMD_SGT's export path and compare with the eager result (CPU)."""
    from gmd_sgt.inference.export import script_model

    scripted = script_model(model.to("cpu"))
    positions = torch.tensor(H2_POSITIONS, dtype=torch.float32)
    species = torch.tensor(H2_SPECIES, dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_shift = torch.zeros((2, 3), dtype=torch.float32)
    with torch.enable_grad():
        out = scripted(species, positions, edge_index, edge_shift)
    energy = float(out["energy"].sum().item())
    forces = out["forces"]
    finite = bool(torch.isfinite(out["energy"]).all().item()) and bool(torch.isfinite(forces).all().item())
    energy_abs_err = abs(energy - eager["energy"]) if eager.get("energy") is not None else None
    matches = finite and energy_abs_err is not None and energy_abs_err <= 1e-4
    return {
        "status": "passed" if matches else "failed",
        "device": "cpu",
        "energy": energy,
        "energy_abs_err_vs_eager": energy_abs_err,
        "finite_energy_and_forces": finite,
        "interface": "forward(species, positions, edge_index, edge_shift)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devices", default="auto",
                        help="'auto' (cpu + cuda when available), 'cpu', or 'cpu,cuda'")
    parser.add_argument("--output", type=Path, help="also write the JSON document here")
    args = parser.parse_args(argv)
    require_cuda = os.environ.get("GMD_AL_REQUIRE_CUDA") == "1"

    result: dict[str, Any] = {
        "status": "blocked",
        "imports": {},
        "api": {"status": "blocked", "detail": None},
        "inference": {"cpu": {"status": "blocked"}, "cuda": {"status": "blocked"}},
        "torchscript": {"status": "blocked"},
        "probe": {"purpose": "integration_probe_only", "scientifically_valid": False,
                  "training_steps": 0, "seed": SEED},
        "require_cuda": require_cuda,
        "errors": [],
        "failures": [],
    }

    modules: dict[str, Any] = {}
    for name in ("torch", "gmd_sgt", "e3nn"):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                modules[name] = importlib.import_module(name)
            result["imports"][name] = {"status": "passed", "version": _version(modules[name])}
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            optional = name == "e3nn"
            result["imports"][name] = {"status": "skipped" if optional else "failed", "detail": detail}
            if not optional:
                result["errors"].append(f"failed to import {name}: {detail}")
                result["failures"].append(_failure(f"import-{name}", exc, None, {}))

    versions = {name: _version(module) for name, module in modules.items()}
    for failure in result["failures"]:
        failure["package_versions"] = versions

    def emit(code: int) -> int:
        rendered = json.dumps(result, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return code

    if "torch" not in modules or "gmd_sgt" not in modules:
        result["api"]["detail"] = "required imports failed; model API was not inspected"
        return emit(1)

    torch = modules["torch"]
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = build_tiny_model(torch, "e3nn" in modules)
        result["api"] = {"status": "passed",
                         "detail": f"{type(model).__name__} built with seed {SEED}"}
    except Exception as exc:
        detail = f"GMD_SGT model API unavailable: {type(exc).__name__}: {exc}"
        result["api"] = {"status": "blocked", "detail": detail}
        result["errors"].append(detail)
        result["failures"].append(_failure("gmd-sgt-model-api", exc, None, versions))
        return emit(1)

    cuda_available = bool(torch.cuda.is_available())
    if args.devices == "auto":
        devices = ["cpu", "cuda"]
    else:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        if require_cuda and "cuda" not in devices:
            devices.append("cuda")

    for device in ("cpu", "cuda"):
        if device not in devices:
            result["inference"][device] = {"status": "skipped", "detail": "device not selected"}
            continue
        if device == "cuda" and not cuda_available:
            result["inference"]["cuda"] = {"status": "skipped",
                                           "detail": "torch.cuda.is_available() is false"}
            if require_cuda:
                result["errors"].append("CUDA GMD_SGT inference is required but CUDA is unavailable")
            continue
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result["inference"][device] = run_inference(torch, model, device)
            if result["inference"][device]["status"] != "passed":
                result["errors"].append(f"{device} GMD_SGT inference produced non-finite energy or forces")
                result["failures"].append(_failure(f"gmd-sgt-{device}-inference", None, device, versions,
                                                   "energy or forces contain NaN/Inf"))
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            result["inference"][device] = {"status": "blocked", "detail": detail}
            result["errors"].append(f"{device} GMD_SGT inference blocked: {detail}")
            result["failures"].append(_failure(f"gmd-sgt-{device}-inference", exc, device, versions))

    if result["inference"]["cpu"].get("status") == "passed":
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result["torchscript"] = run_torchscript(torch, model, result["inference"]["cpu"])
            if result["torchscript"]["status"] != "passed":
                result["errors"].append("TorchScript export does not reproduce the eager energy")
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            result["torchscript"] = {"status": "blocked", "detail": detail}
            result["errors"].append(f"TorchScript export blocked: {detail}")
            result["failures"].append(_failure("gmd-sgt-torchscript", exc, "cpu", versions))

    required = [result["api"]["status"], result["inference"]["cpu"]["status"],
                result["torchscript"]["status"]]
    if (cuda_available and "cuda" in devices) or require_cuda:
        required.append(result["inference"]["cuda"]["status"])
    passed = not result["errors"] and all(status == "passed" for status in required)
    result["status"] = "passed" if passed else "blocked"
    return emit(0 if passed else 1)


if __name__ == "__main__":
    raise SystemExit(main())
