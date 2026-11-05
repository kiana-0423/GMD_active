# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.dataset_loader) at 96263b6; MIT License, see NOTICE.
"""Load SCF samples into arrays a PyTorch model can consume.

The loader is numpy-first: it produces plain arrays that can be inspected and
tested without torch, and only wraps them in a ``torch.utils.data.Dataset`` at
the point of use.  That keeps the dataset layer testable on machines without a
GPU stack, which is where most of this code is actually debugged.

Two supervision targets are supported.

``atomic_population``
    Per-atom valence populations, derivable from any Stage 1 sample.  A model
    trained on this reproduces the atomic guess; it validates the pipeline but
    cannot be expected to beat ``SCF_GUESS ATOMIC``.

``density_matrix``
    The real target, read from the ``artifacts.density_matrix_path`` of each
    sample.  Samples without one are rejected rather than silently substituted,
    because quietly falling back would produce a model that claims to predict
    densities while having only ever seen atomic guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from gmd_active_learning.adapters.dft.scf.dataset import SCFDataset
from gmd_active_learning.adapters.dft.scf.density.predictor import VALENCE_ELECTRONS
from gmd_active_learning.adapters.dft.scf.density.prediction import (
    DENSITY_FORMAT_ATOMIC_POPULATION,
    DENSITY_FORMAT_DENSITY_MATRIX,
)
from gmd_active_learning.adapters.dft.scf.export import read_jsonl
from gmd_active_learning.adapters.dft.scf.features import PER_ATOM_FEATURE_NAMES, featurize_sample
from gmd_active_learning.adapters.dft.scf.schema import SCFSample
from gmd_active_learning.adapters.dft.scf.validation import validate_sample

TARGET_ATOMIC_POPULATION = "atomic_population"
TARGET_DENSITY_MATRIX = "density_matrix"
SUPPORTED_TARGETS = (TARGET_ATOMIC_POPULATION, TARGET_DENSITY_MATRIX)


class DataError(RuntimeError):
    """Raised when samples cannot be turned into a training batch."""


@dataclass(frozen=True, slots=True)
class SCFExample:
    """One training example: features in, density representation out."""

    sample_id: str
    structure_hash: str
    system_features: np.ndarray
    per_atom_features: np.ndarray
    density_target: np.ndarray
    electron_count: float
    spin_channels: int
    n_scf_iterations: int
    converged: bool
    density_format: str
    #: AO overlap, present only for density-matrix targets that recorded one.
    #: Without it the electron count is Tr(P), which is only correct in an
    #: orthonormal basis.
    overlap: np.ndarray | None = None
    #: Raw geometry, kept alongside the derived features because the Stage 10
    #: graph model conditions on coordinates directly rather than on the
    #: fixed-length descriptors the Stage 3 model uses.
    atomic_numbers: tuple[int, ...] = ()
    positions: np.ndarray | None = None
    cell: np.ndarray | None = None
    charge: int = 0
    multiplicity: int = 1

    @property
    def n_atoms(self) -> int:
        return int(self.per_atom_features.shape[0])

    @property
    def n_ao(self) -> int | None:
        if self.density_format != DENSITY_FORMAT_DENSITY_MATRIX:
            return None
        return int(self.density_target.shape[1])


@dataclass(slots=True)
class Normalization:
    """Per-feature mean and scale, fitted on the training split only."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "Normalization":
        values = np.asarray(matrix, dtype=float)
        if values.size == 0:
            width = values.shape[-1] if values.ndim >= 1 else 0
            return cls(mean=np.zeros(width), scale=np.ones(width))
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        # A constant feature has zero variance; dividing by it would produce
        # NaNs that only surface as a silently dead model.
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        return (np.asarray(matrix, dtype=float) - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Normalization":
        return cls(
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
        )


@dataclass(slots=True)
class SCFExampleSet:
    """A split's worth of examples, plus the metadata a trainer needs."""

    examples: list[SCFExample] = field(default_factory=list)
    target: str = TARGET_ATOMIC_POPULATION
    rejected: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SCFExample:
        return self.examples[index]

    def __iter__(self):
        return iter(self.examples)

    @property
    def n_system_features(self) -> int:
        return int(self.examples[0].system_features.shape[0]) if self.examples else 0

    @property
    def n_per_atom_features(self) -> int:
        return len(PER_ATOM_FEATURE_NAMES)

    def system_matrix(self) -> np.ndarray:
        if not self.examples:
            return np.zeros((0, 0), dtype=float)
        return np.vstack([example.system_features for example in self.examples])

    def summary(self) -> dict[str, Any]:
        return {
            "n_examples": len(self.examples),
            "target": self.target,
            "n_rejected": len(self.rejected),
            "rejected": list(self.rejected),
            "n_atoms": {
                "min": min((e.n_atoms for e in self.examples), default=None),
                "max": max((e.n_atoms for e in self.examples), default=None),
            },
        }


def build_examples(
    samples: Iterable[SCFSample],
    *,
    target: str = TARGET_ATOMIC_POPULATION,
    require_converged: bool = True,
    skip_invalid: bool = True,
) -> SCFExampleSet:
    """Turn samples into examples, reporting what was dropped and why."""

    if target not in SUPPORTED_TARGETS:
        raise DataError(f"unsupported target {target!r}; expected one of {', '.join(SUPPORTED_TARGETS)}")

    wants_density = target == TARGET_DENSITY_MATRIX
    result = SCFExampleSet(target=target)
    for sample in samples:
        # A density-matrix target opens and checks the recorded density; the
        # atomic-population target has nothing on disk to check.
        report = validate_sample(sample, check_density=wants_density)
        if not report.ok:
            if not skip_invalid:
                raise DataError(f"{sample.sample_id} failed validation: {report.errors[0].message}")
            result.rejected.append(f"{sample.sample_id}: {report.errors[0].message}")
            continue
        if require_converged and not sample.trajectory.converged:
            result.rejected.append(f"{sample.sample_id}: SCF did not converge")
            continue
        try:
            if wants_density:
                density, overlap = load_density_matrix(sample)
                density_format = DENSITY_FORMAT_DENSITY_MATRIX
            else:
                density, density_format = density_target(sample, target=target)
                overlap = None
        except DataError as exc:
            if not skip_invalid:
                raise
            result.rejected.append(f"{sample.sample_id}: {exc}")
            continue

        features = featurize_sample(sample)
        result.examples.append(
            SCFExample(
                sample_id=sample.sample_id,
                structure_hash=sample.structure_hash,
                system_features=features.flat(),
                per_atom_features=features.per_atom,
                density_target=density,
                electron_count=_reference_electron_count(sample, density, overlap, wants_density),
                spin_channels=int(sample.electronic_state.spin_channels),
                n_scf_iterations=int(sample.trajectory.n_iterations),
                converged=bool(sample.trajectory.converged),
                density_format=density_format,
                overlap=overlap,
                atomic_numbers=tuple(int(number) for number in sample.structure.atomic_numbers),
                positions=np.asarray(sample.structure.positions, dtype=float).reshape(-1, 3),
                cell=np.asarray(sample.structure.cell, dtype=float).reshape(3, 3),
                charge=int(sample.electronic_state.charge),
                multiplicity=int(sample.electronic_state.multiplicity),
            )
        )
    return result


def _reference_electron_count(
    sample: SCFSample,
    density: np.ndarray,
    overlap: np.ndarray | None,
    wants_density: bool,
) -> float:
    """The electron count a prediction for this sample must reproduce.

    For a density-matrix target this is the *valence* count the recorded
    density actually holds, which differs from the all-electron count in the
    electronic state whenever pseudopotentials are used. Using the wrong one
    would make every physically correct prediction look like it lost electrons.
    """

    if not wants_density:
        # Deliberate fix (finding F-23): populations hold the GTH valence count too.
        return _valence_electrons(sample)
    recorded = sample.artifacts.density_electron_count
    if recorded is not None:
        return float(recorded)
    return density_electron_count(density, overlap)


def density_target(sample: SCFSample, *, target: str) -> tuple[np.ndarray, str]:
    """Return the supervision target for one sample."""

    if target == TARGET_ATOMIC_POPULATION:
        return atomic_population_target(sample), DENSITY_FORMAT_ATOMIC_POPULATION

    values, _overlap = load_density_matrix(sample)
    return values, DENSITY_FORMAT_DENSITY_MATRIX


def load_density_matrix(sample: SCFSample) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a sample's recorded density matrix and its AO overlap.

    Both ``.npy`` (what :mod:`gmd_active_learning.adapters.dft.scf.recording.density` writes) and
    ``.npz`` (what a prediction export writes) are accepted, because the same
    training loop is used to fit a model and to score its own output.
    """

    path = sample.artifacts.density_matrix_path
    if not path:
        raise DataError(
            "target 'density_matrix' requires artifacts.density_matrix_path, which this sample does not have"
        )
    source = Path(path)
    if not source.is_file():
        raise DataError(f"density matrix file is missing: {source}")

    values = _load_array(source, "density matrix")
    if values.ndim == 2:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise DataError(f"density matrix {source} has shape {values.shape}, expected (spin, nao, nao)")
    if not np.isfinite(values).all():
        raise DataError(f"density matrix {source} contains non-finite values")

    expected_channels = sample.electronic_state.spin_channels
    if values.shape[0] != expected_channels:
        raise DataError(
            f"density matrix {source} has {values.shape[0]} spin channel(s), "
            f"the electronic state declares {expected_channels}"
        )

    overlap: np.ndarray | None = None
    if sample.artifacts.overlap_path:
        overlap_source = Path(sample.artifacts.overlap_path)
        if not overlap_source.is_file():
            raise DataError(f"overlap file is missing: {overlap_source}")
        overlap = _load_array(overlap_source, "overlap matrix")
        if overlap.shape != values.shape[1:]:
            raise DataError(
                f"overlap {overlap_source} has shape {overlap.shape}, expected {values.shape[1:]}"
            )
        if not np.isfinite(overlap).all():
            raise DataError(f"overlap {overlap_source} contains non-finite values")

    return values, overlap


def density_electron_count(density: np.ndarray, overlap: np.ndarray | None) -> float:
    """Return ``Tr(P S)``.

    Deliberate deviation from PHAL: without an overlap this raises instead of
    returning ``Tr(P)`` (an unverifiable count is refused, not approximated).
    """

    matrices = np.asarray(density, dtype=float)
    if matrices.ndim == 2:
        matrices = matrices[None, :, :]
    if overlap is None:
        raise DataError("no overlap matrix: the electron count Tr(P S) cannot be computed")
    return float(sum(np.sum(channel * np.asarray(overlap, dtype=float)) for channel in matrices))


def _load_array(source: Path, label: str) -> np.ndarray:
    from gmd_active_learning.adapters.dft.scf.recording.density import load_matrix_file

    try:
        return load_matrix_file(source)
    except (OSError, ValueError, IndexError, RuntimeError) as exc:
        raise DataError(f"could not read {label} {source}: {exc}") from exc


def atomic_population_target(sample: SCFSample) -> np.ndarray:
    """Per-atom valence populations, shape ``(spin_channels, n_atoms)``.

    This mirrors what CP2K's ``SCF_GUESS ATOMIC`` produces, so it is a
    reference target rather than an improvement over it.
    """

    weights = np.asarray(
        [float(VALENCE_ELECTRONS.get(int(number), int(number))) for number in sample.structure.atomic_numbers],
        dtype=float,
    )
    if weights.sum() <= 0.0:
        weights = np.ones(sample.structure.n_atoms, dtype=float)
    share = weights / weights.sum()
    # Deliberate fix (finding F-23): the shares are valence shares, and so is the
    # total; PHAL scaled them to the all-electron sum(Z) - charge.
    electrons = _valence_electrons(sample)

    if sample.electronic_state.spin_channels == 1:
        return (share * electrons)[None, :]
    unpaired = float(sample.electronic_state.multiplicity - 1)
    alpha = share * (electrons + unpaired) / 2.0
    beta = share * (electrons - unpaired) / 2.0
    return np.vstack([alpha, beta])


def _valence_electrons(sample: SCFSample) -> float:
    from gmd_active_learning.adapters.dft.scf.density.prediction import PredictionError
    from gmd_active_learning.adapters.dft.scf.density.predictor import expected_density_electrons

    try:
        return expected_density_electrons(sample.structure, sample.electronic_state)
    except PredictionError as exc:
        raise DataError(f"{sample.sample_id}: {exc}") from exc


def load_split(
    source: str | Path | SCFDataset,
    split: str,
    *,
    target: str = TARGET_ATOMIC_POPULATION,
    require_converged: bool = True,
) -> SCFExampleSet:
    """Load one split from a dataset directory or from a ``.jsonl`` export."""

    return build_examples(
        _read_samples(source, split),
        target=target,
        require_converged=require_converged,
    )


def _read_samples(source: str | Path | SCFDataset, split: str) -> list[SCFSample]:
    if isinstance(source, SCFDataset):
        return source.samples(split)
    path = Path(source)
    if path.is_dir():
        jsonl = path / f"{split}.jsonl"
        if jsonl.is_file():
            return read_jsonl(jsonl)
        return SCFDataset.open(path).samples(split)
    if path.is_file():
        return read_jsonl(path)
    raise DataError(f"no dataset or export found at {path}")


def pad_batch(examples: Sequence[SCFExample]) -> dict[str, np.ndarray]:
    """Pad a batch of variable-size structures to a common atom count.

    ``mask`` marks the real atoms, so padded rows contribute nothing to the
    loss or to the predicted electron count.
    """

    if not examples:
        raise DataError("cannot build a batch from zero examples")
    max_atoms = max(example.n_atoms for example in examples)
    max_channels = max(example.spin_channels for example in examples)
    batch = len(examples)
    n_per_atom = examples[0].per_atom_features.shape[1]

    per_atom = np.zeros((batch, max_atoms, n_per_atom), dtype=np.float32)
    mask = np.zeros((batch, max_atoms), dtype=np.float32)
    density = np.zeros((batch, max_channels, max_atoms), dtype=np.float32)
    channel_mask = np.zeros((batch, max_channels), dtype=np.float32)

    for index, example in enumerate(examples):
        if example.density_format != DENSITY_FORMAT_ATOMIC_POPULATION:
            raise DataError(
                "pad_batch only supports atomic-population targets; "
                f"{example.sample_id} carries {example.density_format!r}"
            )
        n_atoms = example.n_atoms
        per_atom[index, :n_atoms] = example.per_atom_features
        mask[index, :n_atoms] = 1.0
        channels = example.spin_channels
        density[index, :channels, :n_atoms] = example.density_target
        channel_mask[index, :channels] = 1.0

    return {
        "system_features": np.vstack([example.system_features for example in examples]).astype(np.float32),
        "per_atom_features": per_atom,
        "atom_mask": mask,
        "density_target": density,
        "channel_mask": channel_mask,
        "electron_count": np.asarray([example.electron_count for example in examples], dtype=np.float32),
        "n_scf_iterations": np.asarray([example.n_scf_iterations for example in examples], dtype=np.float32),
    }


def iterate_batches(
    examples: SCFExampleSet,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> Iterable[dict[str, np.ndarray]]:
    """Yield padded batches. Shuffling is seeded so runs are reproducible."""

    if batch_size <= 0:
        raise DataError(f"batch_size must be positive, got {batch_size}")
    order = np.arange(len(examples))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [examples[int(index)] for index in order[start : start + batch_size]]
        if chunk:
            yield pad_batch(chunk)


__all__ = [
    "SUPPORTED_TARGETS",
    "TARGET_ATOMIC_POPULATION",
    "TARGET_DENSITY_MATRIX",
    "DataError",
    "Normalization",
    "SCFExample",
    "SCFExampleSet",
    "atomic_population_target",
    "build_examples",
    "density_electron_count",
    "density_target",
    "load_density_matrix",
    "iterate_batches",
    "load_split",
    "pad_batch",
]
