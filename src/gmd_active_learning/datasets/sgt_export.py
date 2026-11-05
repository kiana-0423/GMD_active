"""Export labeled samples exactly as GMD_SGT reads them (replaces PHAL's Allegro exporter, TRN-004).

GMD_SGT (``gmd_sgt.data.reader``, v0.4) reads per-atom ``force_weight`` from extxyz arrays or NPZ,
per-structure ``energy_weight`` / ``stress_weight`` / ``split_group`` from ``atoms.info`` or NPZ,
drops frames without forces, and accepts a missing energy only with ``energy_weight == 0``. The
writers in :mod:`gmd_active_learning.datasets.extxyz_io` and :mod:`~gmd_active_learning.datasets.npz_io`
produce those keys; :func:`export_for_gmd_sgt` writes a split set of files outside the store (the
store does the same inside every published version).
"""

from __future__ import annotations

from pathlib import Path

from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.datasets.config import DatasetsConfig
from gmd_active_learning.datasets.extxyz_io import write_samples_extxyz
from gmd_active_learning.datasets.npz_io import write_sgt_npz
from gmd_active_learning.datasets.samples import LabeledSample
from gmd_active_learning.datasets.split import TRAIN, VAL, assign_groups

SGT_ATOM_KEYS = ("force_weight",)
SGT_STRUCTURE_KEYS = ("energy_weight", "stress_weight", "split_group")


def export_for_gmd_sgt(samples: list[LabeledSample], output_dir: str | Path, config: DatasetsConfig | None = None) -> dict[str, Path]:
    """Write ``all.extxyz``, ``train.extxyz``, ``val.extxyz`` and ``sgt.npz`` (whole groups per side)."""

    if not samples:
        raise DataContractError("nothing to export")
    config = config or DatasetsConfig()
    target = Path(output_dir)
    groups = assign_groups([sample.split_group for sample in samples], config.split)
    fmt = config.float_format
    return {
        "all": write_samples_extxyz(samples, target / "all.extxyz", float_format=fmt),
        "train": write_samples_extxyz([s for s in samples if groups[s.split_group] == TRAIN], target / "train.extxyz", float_format=fmt),
        "val": write_samples_extxyz([s for s in samples if groups[s.split_group] == VAL], target / "val.extxyz", float_format=fmt),
        "npz": write_sgt_npz(samples, target / "sgt.npz"),
    }


__all__ = ["SGT_ATOM_KEYS", "SGT_STRUCTURE_KEYS", "export_for_gmd_sgt"]
