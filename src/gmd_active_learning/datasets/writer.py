# Ported from PHAL (hotspot_al.datasets.writer) at 96263b6; MIT License, see NOTICE.
"""Backend-neutral per-sample files: ``<prefix>.extxyz``, ``<prefix>_labels.npz``, ``<prefix>_metadata.json``.

Same three-file layout and keys as PHAL's ``write_dataset_entry`` (``forces``, ``mask``,
``original_indices`` (parent atom IDs, -1 for caps), per-role index lists; metadata with
``region_labels``/``atom_role``/``force_weight``/``energy_weight``/``core_atom_indices``/
``masked_atom_indices``) plus the GMD_active additions (image shifts, region codes, energy and
stress with their weights, provenance). Differences: the mask is the region's own
``force_weights`` (it cannot disagree with the roles, finding F-04), the extxyz carries the GMD_SGT
keys, the region energy is written with an explicit ``energy_weight`` (PHAL wrote none, F-12), and
every file is written atomically.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.contracts import RegionRole
from gmd_active_learning.datasets.extxyz_io import write_samples_extxyz
from gmd_active_learning.datasets.npz_io import _atomic_bytes
from gmd_active_learning.datasets.samples import LabeledSample
from gmd_active_learning.utils.atomic_io import write_json_atomic


def write_dataset_entry(
    sample: LabeledSample,
    *,
    output_dir: str | Path,
    prefix: str,
    float_format: str = "shortest",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    target = Path(output_dir)
    region = sample.region
    structure = write_samples_extxyz([sample], target / f"{prefix}.extxyz", float_format=float_format)
    labels = [role.value for role in region.roles]

    def indices(*roles: RegionRole) -> np.ndarray:
        return region.indices_with(*roles).astype(int)

    buffer = io.BytesIO()
    np.savez(
        buffer,
        forces=sample.forces,
        mask=region.force_weights,
        force_weight=region.force_weights,
        region_code=region.region_codes,
        original_indices=region.parent_atom_ids.astype(int),
        image_shift=region.image_shifts,
        core_indices=indices(RegionRole.CORE),
        inner_buffer_indices=indices(RegionRole.INNER_BUFFER),
        outer_buffer_indices=indices(RegionRole.OUTER_BUFFER),
        boundary_indices=indices(RegionRole.BOUNDARY),
        h_cap_indices=indices(RegionRole.H_CAP),
        energy=np.asarray(np.nan if sample.energy is None else sample.energy),
        stress=np.full((3, 3), np.nan) if sample.stress is None else sample.stress,
    )
    labels_path = _atomic_bytes(target / f"{prefix}_labels.npz", buffer.getvalue())
    metadata = {
        "region_labels": labels,
        "atom_role": labels,
        "force_weight": region.force_weights.tolist(),
        "energy_weight": sample.energy_weight,
        "stress_weight": sample.stress_weight,
        "energy": sample.energy,
        "core_atom_indices": indices(RegionRole.CORE).tolist(),
        "masked_atom_indices": np.where(region.force_weights <= 0.0)[0].astype(int).tolist(),
        "provenance": sample.provenance(),
        "metadata": region.metadata,
        "extra_metadata": {**sample.metadata, **(extra_metadata or {})},
    }
    json.dumps(metadata)
    metadata_path = write_json_atomic(target / f"{prefix}_metadata.json", metadata)
    return {"structure": structure, "labels": labels_path, "metadata": metadata_path}


__all__ = ["write_dataset_entry"]
