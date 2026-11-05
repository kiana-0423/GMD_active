# Ported from PHAL (hotspot_al.cp2k_ml_scf.schema) at 96263b6; MIT License, see NOTICE.
"""Serialisable schema for one ML-SCF training sample.

A sample pairs the *inputs* an ML model is allowed to see (geometry plus the
resolved electronic state) with the *outcome* CP2K produced for that system
(the SCF trajectory).  Nothing here predicts anything; the schema exists so the
dataset on disk has one authoritative shape that both the exporter and the
training loader agree on.

Density, wavefunction and Hamiltonian artifacts are declared but optional.  The
first models are trained on geometry alone, and requiring binary restart files
up front would make the whole dataset uncollectable.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "phal-scf-sample-v1"

#: Energies are stored exactly as CP2K prints them.
ENERGY_UNIT = "hartree"
#: Coordinates and cell vectors are stored in Angstrom (ASE's convention).
LENGTH_UNIT = "angstrom"


class SchemaError(ValueError):
    """Raised when a payload cannot be interpreted as a sample."""


@dataclass(frozen=True, slots=True)
class SCFStructure:
    """The atomic configuration a prediction is conditioned on."""

    atomic_numbers: tuple[int, ...]
    positions: tuple[tuple[float, float, float], ...]
    cell: tuple[tuple[float, float, float], ...]
    pbc: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        if len(self.atomic_numbers) != len(self.positions):
            raise SchemaError(
                f"atomic_numbers ({len(self.atomic_numbers)}) and positions "
                f"({len(self.positions)}) must have the same length"
            )
        if len(self.cell) != 3 or any(len(row) != 3 for row in self.cell):
            raise SchemaError("cell must be a 3x3 matrix")

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atomic_numbers": list(self.atomic_numbers),
            "positions": [list(row) for row in self.positions],
            "cell": [list(row) for row in self.cell],
            "pbc": list(self.pbc),
            "length_unit": LENGTH_UNIT,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SCFStructure":
        return cls(
            atomic_numbers=tuple(int(value) for value in _require(payload, "atomic_numbers")),
            positions=tuple(_row3(row) for row in _require(payload, "positions")),
            cell=tuple(_row3(row) for row in _require(payload, "cell")),
            pbc=_pbc(payload.get("pbc", (True, True, True))),
        )

    @classmethod
    def from_atoms(cls, atoms: Any) -> "SCFStructure":
        """Build from an :class:`ase.Atoms` object without importing ASE here."""

        return cls(
            atomic_numbers=tuple(int(value) for value in atoms.get_atomic_numbers()),
            positions=tuple(_row3(row) for row in atoms.get_positions()),
            cell=tuple(_row3(row) for row in atoms.get_cell()),
            pbc=_pbc(atoms.get_pbc()),
        )


@dataclass(frozen=True, slots=True)
class SCFElectronicState:
    """Charge and spin, i.e. everything CP2K needs beyond the geometry."""

    charge: int
    multiplicity: int
    n_electrons: int
    spin_polarized: bool
    spin_channels: int = 1
    spin_source: str | None = None

    def __post_init__(self) -> None:
        if self.multiplicity < 1:
            raise SchemaError(f"multiplicity must be >= 1, got {self.multiplicity}")
        if self.n_electrons < 0:
            raise SchemaError(f"n_electrons must be >= 0, got {self.n_electrons}")
        if self.spin_channels not in (1, 2):
            raise SchemaError(f"spin_channels must be 1 or 2, got {self.spin_channels}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "n_electrons": self.n_electrons,
            "spin_polarized": self.spin_polarized,
            "spin_channels": self.spin_channels,
            "spin_source": self.spin_source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SCFElectronicState":
        multiplicity = int(_require(payload, "multiplicity"))
        spin_polarized = bool(payload.get("spin_polarized", multiplicity > 1))
        return cls(
            charge=int(_require(payload, "charge")),
            multiplicity=multiplicity,
            n_electrons=int(_require(payload, "n_electrons")),
            spin_polarized=spin_polarized,
            spin_channels=int(payload.get("spin_channels", 2 if spin_polarized else 1)),
            spin_source=_optional_str(payload.get("spin_source")),
        )

    @classmethod
    def from_electronic_state(cls, state: Any) -> "SCFElectronicState":
        """Adapt :class:`gmd_active_learning.utils.electronic_state.ElectronicState`."""

        return cls(
            charge=int(state.charge),
            multiplicity=int(state.multiplicity),
            n_electrons=int(state.n_electrons),
            spin_polarized=bool(state.is_open_shell),
            spin_channels=2 if state.is_open_shell else 1,
            spin_source=getattr(state, "source", None),
        )


@dataclass(frozen=True, slots=True)
class SCFIterationRecord:
    """One SCF iteration as recorded by CP2K."""

    step: int
    residual: float | None
    energy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "residual": self.residual, "energy": self.energy}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SCFIterationRecord":
        return cls(
            step=int(_require(payload, "step")),
            residual=_optional_float(payload.get("residual")),
            energy=_optional_float(payload.get("energy")),
        )


@dataclass(frozen=True, slots=True)
class SCFTrajectory:
    """The SCF convergence history plus the settings that produced it."""

    iterations: tuple[SCFIterationRecord, ...]
    eps_scf: float | None
    max_scf: int | None
    converged: bool
    initial_guess: str | None
    mixing_method: str | None = None
    cp2k_version: str | None = None

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    @property
    def residual_history(self) -> tuple[float | None, ...]:
        return tuple(iteration.residual for iteration in self.iterations)

    @property
    def energy_history(self) -> tuple[float | None, ...]:
        return tuple(iteration.energy for iteration in self.iterations)

    @property
    def final_energy(self) -> float | None:
        for iteration in reversed(self.iterations):
            if iteration.energy is not None:
                return iteration.energy
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "n_iterations": self.n_iterations,
            "eps_scf": self.eps_scf,
            "max_scf": self.max_scf,
            "converged": self.converged,
            "initial_guess": self.initial_guess,
            "mixing_method": self.mixing_method,
            "cp2k_version": self.cp2k_version,
            "energy_unit": ENERGY_UNIT,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SCFTrajectory":
        iterations = _require(payload, "iterations")
        if not isinstance(iterations, Sequence) or isinstance(iterations, (str, bytes)):
            raise SchemaError("iterations must be a list")
        return cls(
            iterations=tuple(SCFIterationRecord.from_dict(item) for item in iterations),
            eps_scf=_optional_float(payload.get("eps_scf")),
            max_scf=_optional_int(payload.get("max_scf")),
            converged=bool(_require(payload, "converged")),
            initial_guess=_optional_str(payload.get("initial_guess")),
            mixing_method=_optional_str(payload.get("mixing_method")),
            cp2k_version=_optional_str(payload.get("cp2k_version")),
        )


@dataclass(frozen=True, slots=True)
class SCFDensityArtifacts:
    """Paths to electronic-structure restart files, when they were kept.

    Every field is optional on purpose: the Stage 1 dataset is collectable from
    text output alone, and the paths become populated only once CP2K is asked
    to retain the corresponding binaries.
    """

    density_matrix_path: str | None = None
    wavefunction_path: str | None = None
    hamiltonian_path: str | None = None
    density_format: str | None = None
    density_digest: str | None = None
    # Populated once a real CP2K density is recorded. The overlap is what makes
    # the electron count checkable as Tr(P S) rather than Tr(P).
    overlap_path: str | None = None
    n_ao: int | None = None
    basis_set: str | None = None
    density_electron_count: float | None = None
    #: Which basis the matrix is expressed in. "ao-nonorthogonal" means the
    #: electron count is Tr(P S) and the overlap is required to evaluate it.
    representation: str | None = None
    #: "trace_ps" when the count above was computed with the overlap,
    #: "trace_p" when it is the orthonormal-basis approximation. A consumer
    #: that cannot tell these apart cannot tell a correct count from one that
    #: is 20-50% low; see docs/density_representation.md.
    electron_count_method: str | None = None
    #: 1 for restricted, 2 for spin-polarised.
    spin_channels: int | None = None
    #: Directory holding the per-iteration densities, when the SCF trajectory
    #: was recorded. The trajectory is what a neural mixer trains on: it needs
    #: the path the SCF took, not only where it ended up.
    trajectory_path: str | None = None
    #: Number of recorded frames, including the final converged density, so a
    #: consumer can size a batch without opening the directory.
    trajectory_frames: int | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.density_matrix_path, self.wavefunction_path, self.hamiltonian_path)
        )

    @property
    def has_density_matrix(self) -> bool:
        return bool(self.density_matrix_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "density_matrix_path": self.density_matrix_path,
            "wavefunction_path": self.wavefunction_path,
            "hamiltonian_path": self.hamiltonian_path,
            "density_format": self.density_format,
            "density_digest": self.density_digest,
            "overlap_path": self.overlap_path,
            "n_ao": self.n_ao,
            "basis_set": self.basis_set,
            "density_electron_count": self.density_electron_count,
            # Deliberate fix (finding F-23): PHAL persisted only the nine keys
            # above and silently dropped these five on every dataset write. They
            # are written when set, so samples without them stay byte-identical.
            **{
                key: value
                for key, value in (
                    ("representation", self.representation),
                    ("electron_count_method", self.electron_count_method),
                    ("spin_channels", self.spin_channels),
                    ("trajectory_path", self.trajectory_path),
                    ("trajectory_frames", self.trajectory_frames),
                )
                if value is not None
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SCFDensityArtifacts":
        if not payload:
            return cls()
        return cls(
            density_matrix_path=_optional_str(payload.get("density_matrix_path")),
            wavefunction_path=_optional_str(payload.get("wavefunction_path")),
            hamiltonian_path=_optional_str(payload.get("hamiltonian_path")),
            density_format=_optional_str(payload.get("density_format")),
            density_digest=_optional_str(payload.get("density_digest")),
            overlap_path=_optional_str(payload.get("overlap_path")),
            n_ao=_optional_int(payload.get("n_ao")),
            basis_set=_optional_str(payload.get("basis_set")),
            density_electron_count=_optional_float(payload.get("density_electron_count")),
            representation=_optional_str(payload.get("representation")),
            electron_count_method=_optional_str(payload.get("electron_count_method")),
            spin_channels=_optional_int(payload.get("spin_channels")),
            trajectory_path=_optional_str(payload.get("trajectory_path")),
            trajectory_frames=_optional_int(payload.get("trajectory_frames")),
        )


@dataclass(frozen=True, slots=True)
class SCFSample:
    """One (structure, electronic state) -> SCF trajectory training record."""

    sample_id: str
    structure: SCFStructure
    electronic_state: SCFElectronicState
    trajectory: SCFTrajectory
    artifacts: SCFDensityArtifacts = field(default_factory=SCFDensityArtifacts)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def structure_hash(self) -> str:
        return structure_hash(self.structure, self.electronic_state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "structure_hash": self.structure_hash,
            "structure": self.structure.to_dict(),
            "electronic_state": self.electronic_state.to_dict(),
            "trajectory": self.trajectory.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SCFSample":
        if not isinstance(payload, Mapping):
            raise SchemaError(f"sample must be a mapping, got {type(payload).__name__}")
        version = str(payload.get("schema_version", SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise SchemaError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise SchemaError("metadata must be a mapping")
        return cls(
            sample_id=str(_require(payload, "sample_id")),
            structure=SCFStructure.from_dict(_mapping(payload, "structure")),
            electronic_state=SCFElectronicState.from_dict(_mapping(payload, "electronic_state")),
            trajectory=SCFTrajectory.from_dict(_mapping(payload, "trajectory")),
            artifacts=SCFDensityArtifacts.from_dict(payload.get("artifacts")),
            metadata=dict(metadata),
            schema_version=version,
        )


def structure_hash(structure: SCFStructure, electronic_state: SCFElectronicState | None = None) -> str:
    """Return a stable digest of the ML model's input.

    The hash covers everything a predictor may condition on, so two samples
    with the same hash are interchangeable as far as a prediction is concerned.
    Coordinates are rounded to 1e-6 Angstrom so that bit-level float noise from
    a round trip through JSON does not produce a different identity.
    """

    digest = hashlib.sha256()
    digest.update(SCHEMA_VERSION.encode("utf-8"))
    for number in structure.atomic_numbers:
        digest.update(f"|Z{int(number)}".encode("utf-8"))
    for row in structure.positions:
        digest.update(("|R" + ",".join(_canonical_float(value) for value in row)).encode("utf-8"))
    for row in structure.cell:
        digest.update(("|C" + ",".join(_canonical_float(value) for value in row)).encode("utf-8"))
    digest.update(("|P" + ",".join("1" if flag else "0" for flag in structure.pbc)).encode("utf-8"))
    if electronic_state is not None:
        digest.update(
            (
                f"|Q{electronic_state.charge}"
                f"|M{electronic_state.multiplicity}"
                f"|N{electronic_state.n_electrons}"
                f"|S{electronic_state.spin_channels}"
            ).encode("utf-8")
        )
    return f"sha256:{digest.hexdigest()}"


def _canonical_float(value: float) -> str:
    rounded = round(float(value), 6)
    # Collapse -0.0 so the sign of a zero cannot change a structure's identity.
    return f"{rounded + 0.0:.6f}"


def _require(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise SchemaError(f"missing required field: {key}")
    value = payload[key]
    if value is None:
        raise SchemaError(f"required field is null: {key}")
    return value


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _require(payload, key)
    if not isinstance(value, Mapping):
        raise SchemaError(f"{key} must be a mapping, got {type(value).__name__}")
    return value


def _row3(row: Any) -> tuple[float, float, float]:
    values = [float(value) for value in row]
    if len(values) != 3:
        raise SchemaError(f"expected 3 components, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise SchemaError("coordinate components must be finite")
    return (values[0], values[1], values[2])


def _pbc(value: Any) -> tuple[bool, bool, bool]:
    if isinstance(value, bool):
        return (value, value, value)
    flags = [bool(item) for item in value]
    if len(flags) != 3:
        raise SchemaError(f"pbc must have 3 components, got {len(flags)}")
    return (flags[0], flags[1], flags[2])


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise SchemaError(f"expected a finite number, got {value!r}")
    return number


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "ENERGY_UNIT",
    "LENGTH_UNIT",
    "SCHEMA_VERSION",
    "SCFDensityArtifacts",
    "SCFElectronicState",
    "SCFIterationRecord",
    "SCFSample",
    "SCFStructure",
    "SCFTrajectory",
    "SchemaError",
    "structure_hash",
]
