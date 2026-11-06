# Ported from PHAL (examples/05_write_allegro_dataset.py) at 96263b6; MIT License, see NOTICE.
"""Turn a labeled region into a versioned GMD_SGT training dataset.

    python examples/05_write_sgt_dataset.py --output-dir dataset_store
    python examples/05_write_sgt_dataset.py region.json cp2k.out --output-dir store --config my_run.yaml

Inputs default to a real pair from ``examples/data``: an H-capped CH3 cluster region and the CP2K
2026.2 single-point output computed for it. The output is validated exactly as the labeling
service does, becomes a ``LabelResult``, then a ``LabeledSample`` under the ``datasets`` energy /
stress policy, and is published by ``VersionedDatasetStore`` as an immutable version holding
GMD_SGT-ready ``train.extxyz`` / ``val.extxyz`` / ``sgt.npz`` with per-atom ``force_weight``
(0 for the H cap). PHAL's example wrote an Allegro file with all-zero forces and no label at all;
an INVALID label is refused here (exit 1) instead of entering a dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from _example_utils import DATA_DIR, load_sections, require_file

from gmd_active_learning.adapters.dft.cp2k.config import ValidationSettings
from gmd_active_learning.adapters.dft.cp2k.parse import parse_cp2k_output_file
from gmd_active_learning.adapters.dft.cp2k.validation import validation_reasons, validation_status
from gmd_active_learning.contracts import ExtractedRegion, LabelResult, ValidationStatus, stable_id
from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.datasets import DatasetsConfig, VersionedDatasetStore
from gmd_active_learning.utils.logging import configure_logging


def label_from_output(region: ExtractedRegion, output_path: Path) -> LabelResult:
    output = parse_cp2k_output_file(output_path)
    reasons = validation_reasons(output, n_atoms=region.n_atoms, settings=ValidationSettings())
    return LabelResult(
        task_id=stable_id("label", region.region_id, file_sha256(output_path)),
        region_id=region.region_id,
        forces=output.forces if output.forces is not None else np.zeros((0, 3)),
        energy=output.energy,
        converged=output.scf_converged,
        validation=validation_status(reasons),
        validation_reasons=reasons,
        stress=output.stress,
        scf_steps=output.scf_steps,
        engine="cp2k",
        engine_version=output.version,
        output_sha256=file_sha256(output_path),
        metadata={"output_file": output_path.name},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", nargs="?", type=Path, default=DATA_DIR / "ch3_capped_region.json", help="region JSON (ExtractedRegion.to_dict)")
    parser.add_argument("output", nargs="?", type=Path, default=DATA_DIR / "ch3_capped_sp.out", help="CP2K output for that region")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_store"), help="VersionedDatasetStore root")
    parser.add_argument("--config", type=Path, default=None, help="YAML with a datasets section (defaults otherwise)")
    args = parser.parse_args(argv)
    require_file(parser, args.region, "region file")
    require_file(parser, args.output, "CP2K output file")
    configure_logging("WARNING")

    region = ExtractedRegion.from_dict(json.loads(args.region.read_text(encoding="utf-8")))
    label = label_from_output(region, args.output)
    if label.validation is not ValidationStatus.VALID:
        print("label INVALID, not added:", "; ".join(label.validation_reasons))
        return 1
    store = VersionedDatasetStore(args.output_dir, DatasetsConfig.from_mapping(load_sections(args.config).get("datasets")))
    sample_id, sample_path = store.add_sample(region, label, provenance={"source": "examples/05_write_sgt_dataset.py"})
    version = store.publish([sample_path], parent=None)
    sample = store.read_samples(version)[0]
    print(f"sample {sample_id} n_atoms={region.n_atoms} force_weight={sample.region.force_weights.tolist()} "
          f"energy_weight={sample.energy_weight}")
    print(f"dataset version {version.version} sha256={version.sha256[:16]}... samples={len(store.read_samples(version))}")
    for key in ("train_file", "val_file", "sgt_npz"):
        print(f"{key}: {version.metadata[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
