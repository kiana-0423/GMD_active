#!/usr/bin/env python3
# Ported from PHAL (containers/scripts/build_md_probe_model.py) at 96263b6; MIT License, see NOTICE.
"""Export a random, untrained GMD_SGT model as the GMD integration-probe model.

PHAL compiled a random Allegro model for pair_allegro; GMD instead loads a
GMD_SGT TorchScript export with the interface
``forward(species, positions, edge_index, edge_shift)``. This script builds a
seeded ``AllegroStyleBackbone`` (the same family GMD's own controller test
uses), gives its readout a non-zero random init, exports it with GMD_SGT's
``script_model``, checks the reloaded export against the eager model with
``validate_exported_model`` and writes a sidecar ``<model>.json``:

    purpose = integration_probe_only, scientifically_valid = false,
    training_steps = 0, sha256 of the .pt file.

The model proves only that GMD can load and evaluate a GMD_SGT export.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SEED = 20251103
CUTOFF = 3.0
N_SPECIES = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_probe_model(output: Path, seed: int = SEED) -> dict[str, Any]:
    """Write the scripted probe model and its metadata; return the metadata."""
    import torch
    import gmd_sgt
    from gmd_sgt.inference.export import script_model, validate_exported_model
    from gmd_sgt.models import AllegroStyleBackbone

    torch.manual_seed(seed)
    model = AllegroStyleBackbone(
        n_species=N_SPECIES,
        hidden_channels=16,
        num_layers=2,
        n_basis=4,
        cutoff=CUTOFF,
        l_max=1,
        avg_neighbors=4.0,
    )
    generator = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        for parameter in model.readout.net.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = script_model(model)
    scripted.save(str(output))
    reloaded = torch.jit.load(str(output), map_location="cpu")
    report = validate_exported_model(reloaded, model, device="cpu")

    metadata = {
        "purpose": "integration_probe_only",
        "scientifically_valid": False,
        "training_steps": 0,
        "seed": seed,
        "model_type": type(model).__name__,
        "local_cutoff": float(model.local_cutoff),
        "n_species": N_SPECIES,
        "gmd_sgt_version": str(gmd_sgt.__version__),
        "torch_version": str(torch.__version__),
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "export_validation_cases": len(report),
        "path": str(output),
        "sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="path of the .pt file to write")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        metadata = build_probe_model(args.output, args.seed)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
