# Ported from PHAL (hotspot_al.cp2k_ml_scf.dataset_adapter) at 96263b6; MIT License, see NOTICE.
"""Turn a collected Pulay/trajectory dataset into density-model training data.

The dataset was collected to study mixing. Its converged densities are also
exactly what an initial-density model needs -- geometry in, ``P`` out -- so
rather than run a second generation campaign, this reads what is already on
disk into :class:`SCFSample` objects the existing training pipeline consumes.

Two things it will not do quietly:

* **Skip a sample with no overlap.** ``Tr(P)`` is not the electron count in a
  non-orthogonal AO basis, and a model trained against a target normalised the
  wrong way is wrong everywhere.
* **Skip an unconverged run.** Its final density is not a converged density,
  and a model that learns from those learns to predict a partial answer.

The geometry comes from the original structure file rather than being
reconstructed from the CP2K input, which is why every sample records its
source file and index at generation time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gmd_active_learning.adapters.dft.scf.schema import (
    SCFDensityArtifacts,
    SCFElectronicState,
    SCFIterationRecord,
    SCFSample,
    SCFStructure,
    SCFTrajectory,
)

REPRESENTATION_AO = "ao-nonorthogonal"


class AdapterError(RuntimeError):
    """Raised when a dataset cannot be read as density training data."""


def _structure_from_source(payload: dict[str, Any], root: Path) -> SCFStructure | None:
    detail = payload.get("structure") or {}
    source = detail.get("source")
    if not source:
        return None

    path = Path(source)
    if not path.is_absolute():
        for base in (root, root.parent, Path.cwd()):
            candidate = base / path
            if candidate.is_file():
                path = candidate
                break
    if not path.is_file():
        return None

    from tools.scf.structures import read_structures

    structures = read_structures(path)
    index = int(detail.get("source_index", 0))
    if index >= len(structures):
        return None
    structure = structures[index]

    from ase.data import atomic_numbers as ase_numbers

    cell = structure.cell if structure.cell is not None else np.eye(3) * 20.0
    return SCFStructure(
        atomic_numbers=tuple(int(ase_numbers[symbol]) for symbol in structure.symbols),
        positions=tuple(tuple(float(value) for value in row) for row in structure.positions),
        cell=tuple(tuple(float(value) for value in row) for row in np.asarray(cell, dtype=float)),
        pbc=(structure.periodic,) * 3,
    )


#: Which recorded density a sample should be trained to predict.
#:
#: ``"final"`` is the converged density: the obvious target, and the one the
#: first model used. Measured result: it does not accelerate the SCF. A
#: converged density is a fixed point, not a place the SCF passes through, and a
#: regression toward it lands *near* it without landing *on* the manifold the
#: SCF moves along -- closer in norm, 2.9 Ha higher in energy, error spread into
#: modes the mixer clears slowly (docs/ml_density_failure_analysis.md).
#:
#: An integer ``k`` selects the density after ``k`` SCF iterations instead.
#: That density is one CP2K actually produced, so it is reachable by
#: construction, and the remaining cost from it is a measured number rather
#: than an assumption.
TARGET_FINAL = "final"


def _cost_labels(metadata: dict[str, Any], index: int) -> dict[str, Any]:
    """What the SCF still had to do from a given frame.

    These are facts about the recording, not predictions: how many iterations
    remained, and how fast the residual and energy were falling over them. The
    first is the quantity that matters and the only one a benchmark can check.
    """

    frames = list(metadata.get("frames", ()))
    total = len(frames)
    if total == 0 or index >= total:
        return {}

    residuals = [
        float(frame["residual"])
        for frame in frames[index:]
        if frame.get("residual") is not None and np.isfinite(float(frame["residual"]))
    ]
    energies = [
        float(frame["energy"])
        for frame in frames[index:]
        if frame.get("energy") is not None and np.isfinite(float(frame["energy"]))
    ]

    def geometric_rate(values: list[float]) -> float | None:
        ratios = [
            values[position + 1] / values[position]
            for position in range(len(values) - 1)
            if values[position] > 0.0 and values[position + 1] > 0.0
        ]
        return float(np.exp(np.mean(np.log(ratios)))) if ratios else None

    energy_gaps = [abs(value - energies[-1]) for value in energies] if len(energies) > 1 else []

    return {
        # The primary label: iterations still needed after this density.
        "scf_steps_remaining": total - 1 - index,
        "residual_reduction_rate": geometric_rate(residuals),
        "energy_reduction_rate": geometric_rate([gap for gap in energy_gaps if gap > 0.0]),
        "residual_at_frame": residuals[0] if residuals else None,
        "energy_gap_at_frame": energy_gaps[0] if energy_gaps else None,
        "n_frames": total,
        "frame_index": index,
    }


def sample_from_directory(
    sample_dir: str | Path,
    *,
    root: Path | None = None,
    target_frame: str | int = TARGET_FINAL,
) -> SCFSample | None:
    """Read one collected sample as an :class:`SCFSample`, or ``None``.

    ``target_frame`` selects which recorded density becomes the training
    target: ``"final"`` for the converged one, or ``k`` for the density after
    ``k`` SCF iterations. See :data:`TARGET_FINAL` for why the second exists.

    ``None`` rather than an exception: a dataset is a mixture of what converged
    and what did not, and the caller wants the usable subset plus a count of
    what was dropped, not the first failure.
    """

    directory = Path(sample_dir)
    sample_file = directory / "sample.json"
    trajectory_dir = directory / "trajectory"
    final_density = trajectory_dir / "final" / "density.npy"
    overlap = trajectory_dir / "overlap.npy"
    metadata_file = trajectory_dir / "metadata.json"

    if not (sample_file.is_file() and final_density.is_file() and metadata_file.is_file()):
        return None
    if not overlap.is_file():
        # Tr(P) is not the electron count here; a target normalised that way is
        # wrong by 20-50% and the model would learn the error.
        return None

    payload = json.loads(sample_file.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if not payload.get("scf_converged") or not metadata.get("scf_converged", True):
        return None

    structure = _structure_from_source(payload, root or directory.parent)
    if structure is None:
        return None

    # Every recorded frame is renormalised to the same count, so the run-level
    # value is the right one for any target frame.
    n_electrons = float(metadata.get("n_electrons") or 0.0)
    if n_electrons <= 0:
        return None
    spin_channels = int(metadata.get("spin_channels") or 1)

    iterations = tuple(
        SCFIterationRecord(
            step=int(frame.get("iteration") or index + 1),
            residual=frame.get("residual"),
            energy=frame.get("energy"),
        )
        for index, frame in enumerate(metadata.get("frames", ()))
        if frame.get("iteration") is not None
    )

    # Two different electron counts, and confusing them is a real failure mode.
    # electronic_state.n_electrons is the all-electron count the schema checks
    # against sum(Z) - charge; the recorded density holds only the *valence*
    # electrons, because these are pseudopotential calculations. The valence
    # count goes on the artifacts, which is what the training loop normalises
    # a prediction to.
    charge = int(payload.get("charge") or 0)
    all_electron = int(sum(structure.atomic_numbers) - charge)

    # Resolve the requested target to a file on disk. A frame that was never
    # recorded is a skipped sample, not a silent fall back to the converged
    # density: training half a set against one target and half against another
    # produces a model that fits neither.
    target_path = final_density
    frame_index = len(metadata.get("frames", ())) - 1
    if target_frame != TARGET_FINAL:
        frame_index = int(target_frame)
        target_path = trajectory_dir / f"iter_{frame_index:04d}" / "density.npy"
        if not target_path.is_file():
            return None

    labels = _cost_labels(metadata, frame_index)

    return SCFSample(
        sample_id=str(payload.get("sample_id") or directory.name),
        structure=structure,
        electronic_state=SCFElectronicState(
            charge=charge,
            multiplicity=int(payload.get("multiplicity") or 1),
            n_electrons=all_electron,
            spin_polarized=spin_channels > 1,
            spin_channels=spin_channels,
            spin_source="dataset",
        ),
        trajectory=SCFTrajectory(
            iterations=iterations,
            eps_scf=None,
            max_scf=None,
            converged=True,
            initial_guess="ATOMIC",
            mixing_method="PULAY_MIXING",
        ),
        artifacts=SCFDensityArtifacts(
            density_matrix_path=str(target_path),
            overlap_path=str(overlap),
            n_ao=int(metadata.get("n_ao") or 0),
            basis_set=metadata.get("basis_set"),
            density_electron_count=n_electrons,
            representation=REPRESENTATION_AO,
            electron_count_method="trace_ps",
            spin_channels=spin_channels,
            trajectory_path=str(trajectory_dir),
            trajectory_frames=int(metadata.get("n_frames") or 0),
        ),
        metadata={
            "structure": payload.get("structure", {}),
            "target_frame": target_frame,
            "scf_cost": labels,
        },
    )


def samples_from_dataset(
    root: str | Path, *, target_frame: str | int = TARGET_FINAL
) -> tuple[list[SCFSample], dict[str, int]]:
    """Read a whole dataset directory; return the usable samples and a tally."""

    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise AdapterError(f"{dataset_root}: not a dataset directory")

    samples: list[SCFSample] = []
    skipped = 0
    for sample_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        sample = sample_from_directory(sample_dir, root=dataset_root, target_frame=target_frame)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    return samples, {"n_samples": len(samples), "n_skipped": skipped}


def chemistry_key(sample: SCFSample) -> str:
    """The family a sample belongs to, for holdout purposes.

    The reduced formula, so H2O and a stretched H2O are one family and cannot
    land on opposite sides of a split. Splitting them apart makes validation
    loss a measure of interpolation between two geometries of the same molecule
    and says nothing about a molecule the model has not seen -- which is the
    only thing the first model turned out to be bad at.
    """

    counts: dict[str, int] = {}
    from ase.data import chemical_symbols

    for number in sample.structure.atomic_numbers:
        symbol = chemical_symbols[int(number)]
        counts[symbol] = counts.get(symbol, 0) + 1

    from math import gcd
    from functools import reduce

    divisor = reduce(gcd, counts.values()) if counts else 1
    return "".join(f"{element}{counts[element] // divisor}" for element in sorted(counts))


def split_by_chemistry(
    samples: Sequence[SCFSample], *, holdout_fraction: float = 0.3, seed: int = 0
) -> tuple[list[SCFSample], list[SCFSample]]:
    """Hold out whole chemistries, never individual systems.

    Groups are assigned entire to one side or the other, so no formula appears
    in both. The alternative -- a random split over samples -- is what made the
    first model look fine in validation and fail on anything new: it was scored
    on other geometries of molecules it had been fitted on.
    """

    groups: dict[str, list[SCFSample]] = {}
    for sample in samples:
        groups.setdefault(chemistry_key(sample), []).append(sample)

    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    n_holdout = max(1, int(round(len(keys) * holdout_fraction))) if len(keys) > 2 else 0

    held_keys = {keys[int(index)] for index in order[:n_holdout]}
    train = [sample for key in keys if key not in held_keys for sample in groups[key]]
    holdout = [sample for key in keys if key in held_keys for sample in groups[key]]

    overlap = {chemistry_key(item) for item in train} & {chemistry_key(item) for item in holdout}
    if overlap:  # pragma: no cover - the construction above prevents it
        raise AdapterError(f"chemistry leaked across the split: {sorted(overlap)}")
    return train, holdout


def split_samples(
    samples: Sequence[SCFSample], *, validation_fraction: float = 0.25, seed: int = 0
) -> tuple[list[SCFSample], list[SCFSample]]:
    """Split by system, deterministically.

    Random over *samples* would be wrong if a dataset ever holds several
    geometries of one system: near-duplicates on both sides of the split make
    validation loss a measure of memorisation.
    """

    ordered = sorted(samples, key=lambda item: item.sample_id)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ordered))
    n_validation = max(1, int(round(len(ordered) * validation_fraction))) if len(ordered) > 3 else 0

    validation = {int(index) for index in indices[:n_validation]}
    train = [item for index, item in enumerate(ordered) if index not in validation]
    held = [item for index, item in enumerate(ordered) if index in validation]
    return train, held
