"""Extended-XYZ writer/reader for labeled samples, in the layout GMD_SGT reads.

Per-atom columns: ``species pos forces force_weight region_code frozen parent_atom_id image_shift``.
Per-structure info: ``energy`` (region energy, only when present), ``energy_weight``,
``stress`` (only when labeled and periodic), ``stress_weight``, ``split_group``,
``region_label_map`` (PHAL JSON) and ``gmd_provenance`` (JSON: sample/region/event/run/task IDs,
parent step, model version, label engine, convergence, ...).

ASE's own writer formats per-atom floats with ``%16.8f``, which is lossy; this writer uses
``DatasetsConfig.float_format`` (default: the shortest exact repr) for every float, so forces, weights
and positions survive a write/read cycle bit for bit. Files are written atomically.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import chemical_symbols
from ase.io import read
from ase.io.extxyz import key_val_dict_to_str

from gmd_active_learning.contracts import ExtractedRegion
from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.datasets.masks import region_label_map_json, roles_from_codes
from gmd_active_learning.datasets.samples import LabeledSample

PROPERTIES = (
    "species:S:1:pos:R:3:forces:R:3:force_weight:R:1:region_code:I:1:frozen:I:1:parent_atom_id:I:1:image_shift:I:3"
)
PROVENANCE_KEY = "gmd_provenance"


class _Formatter:
    """``shortest`` = Python's shortest exact repr; anything else is a %-format."""

    def __init__(self, spec: str) -> None:
        self.spec = spec

    def __mod__(self, value: Any) -> str:
        if self.spec == "shortest":
            return repr(float(value))
        return self.spec % float(value)


def _frame_text(sample: LabeledSample, float_format: str) -> str:
    region = sample.region
    fmt = _Formatter(float_format)
    lines = [str(region.n_atoms)]
    header: list[str] = []
    if region.cell is not None:
        header.append('Lattice="' + " ".join(fmt % value for value in np.asarray(region.cell, dtype=float).reshape(9)) + '"')
    header.append(f"Properties={PROPERTIES}")
    info: dict[str, Any] = {}
    if sample.energy is not None:
        info["energy"] = float(sample.energy)
    info["energy_weight"] = float(sample.energy_weight)
    if sample.stress is not None:
        info["stress"] = np.asarray(sample.stress, dtype=float)
    info["stress_weight"] = float(sample.stress_weight)
    info["split_group"] = sample.split_group
    info["region_label_map"] = region_label_map_json()
    info[PROVENANCE_KEY] = {**sample.provenance(), "sample_metadata": sample.metadata}
    info["pbc"] = np.asarray(region.pbc, dtype=bool)
    header.append(key_val_dict_to_str(info))
    lines.append(" ".join(header))
    codes = region.region_codes
    for k in range(region.n_atoms):
        values = [chemical_symbols[int(region.numbers[k])]]
        values += [fmt % value for value in region.positions[k]]
        values += [fmt % value for value in sample.forces[k]]
        values.append(fmt % float(region.force_weights[k]))
        values.append(str(int(codes[k])))
        values.append(str(int(region.frozen[k])))
        values.append(str(int(region.parent_atom_ids[k])))
        values += [str(int(value)) for value in region.image_shifts[k]]
        lines.append(" ".join(values))
    return "\n".join(lines) + "\n"


def write_samples_extxyz(samples: Iterable[LabeledSample], path: str | Path, *, float_format: str = "shortest") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_frame_text(sample, float_format) for sample in samples)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)
    return target


def _result(atoms: Any, key: str) -> Any:
    if key in atoms.arrays:
        return atoms.arrays[key]
    if key in atoms.info:
        return atoms.info[key]
    calc = atoms.calc
    if calc is not None and key in getattr(calc, "results", {}):
        if key == "stress":
            return atoms.get_stress(voigt=False)
        return calc.results[key]
    return None


def read_samples_extxyz(path: str | Path) -> list[LabeledSample]:
    """Read files written by :func:`write_samples_extxyz` (region metadata is not stored there)."""

    frames = read(str(path), index=":", format="extxyz")
    samples = []
    for atoms in frames:
        record = atoms.info.get(PROVENANCE_KEY)
        if not isinstance(record, dict):
            raise DataContractError(f"{path}: frame without {PROVENANCE_KEY} provenance")
        cell = atoms.cell.array.copy() if atoms.cell.rank == 3 else None
        region = ExtractedRegion(
            region_id=record["region_id"],
            mode=record["extraction_mode"],
            run_id=str(record["run_id"]),
            parent_step=int(record["parent_step"]),
            event_id=record.get("event_id"),
            numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            cell=cell,
            pbc=tuple(bool(flag) for flag in atoms.pbc),  # type: ignore[arg-type]
            parent_atom_ids=atoms.arrays["parent_atom_id"],
            image_shifts=atoms.arrays["image_shift"],
            roles=roles_from_codes(np.asarray(atoms.arrays["region_code"]).tolist()),
            force_weights=atoms.arrays["force_weight"],
            frozen=np.asarray(atoms.arrays["frozen"]).astype(bool),
            hotspot_atom_ids=list(record.get("hotspot_atom_ids", [])),
            charge=int(record.get("charge", 0)),
            multiplicity=record.get("multiplicity"),
            model_version=record.get("model_version"),
            metadata={},
        )
        energy = _result(atoms, "energy")
        stress = _result(atoms, "stress")
        samples.append(
            LabeledSample(
                sample_id=record["sample_id"],
                region=region,
                forces=np.asarray(_result(atoms, "forces"), dtype=float),
                energy=None if energy is None else float(energy),
                stress=None if stress is None else np.asarray(stress, dtype=float).reshape(3, 3),
                energy_weight=float(atoms.info["energy_weight"]),
                stress_weight=float(atoms.info["stress_weight"]),
                split_group=str(record["split_group"]),
                task_id=record["task_id"],
                label_engine=record["label_engine"],
                engine_version=record.get("engine_version"),
                converged=bool(record.get("converged", True)),
                validation=record.get("validation", "valid"),
                output_sha256=record.get("output_sha256"),
                metadata=dict(record.get("sample_metadata", {})),
            )
        )
    return samples


def provenance_of(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(json.dumps(atoms.info[PROVENANCE_KEY])) for atoms in read(str(path), index=":", format="extxyz")]


__all__ = ["PROPERTIES", "PROVENANCE_KEY", "provenance_of", "read_samples_extxyz", "write_samples_extxyz"]
