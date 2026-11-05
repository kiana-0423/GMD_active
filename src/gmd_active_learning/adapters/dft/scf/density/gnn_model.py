# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.density_model) at 96263b6; MIT License, see NOTICE.
"""A graph neural network that predicts an AO density matrix.

The model maps ``(atomic numbers, coordinates, cell, charge, multiplicity)`` to
a density matrix ``P`` in CP2K's atomic-orbital basis.

Two of the three physical constraints are satisfied *by construction* rather
than by a penalty term, because a violated constraint is not a slightly worse
prediction -- it is a density CP2K will reject outright:

``P = P†``
    The pair-block head is explicitly symmetrised, so the output is Hermitian
    to machine precision regardless of what the network learned.

``Tr(P S) = Ne``
    The assembled matrix is rescaled to the requested electron count.  The
    network therefore only has to learn *where* the electrons are, never how
    many there are.

Finiteness is not something an architecture can guarantee -- a diverged weight
produces NaN whatever the design -- so it is checked at inference and the
prediction is rejected.

The structure of the prediction follows the physics: a density matrix is
naturally blocked by atom pairs, ``P_ij`` having shape ``(nao_i, nao_j)``.  The
network produces those blocks from a pair of atom embeddings and the distance
between them, and they are scattered into the full matrix.  This is what makes
the model transferable across system sizes: it never sees ``nao`` as a
dimension it must learn.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.adapters.dft.scf._torch import require_torch

MODEL_ARCHITECTURE = "phal-scf-density-gnn-v1"

#: AO counts per element for CP2K's DZVP-MOLOPT-SR-GTH valence basis. Only a
#: starting point: any other basis, or any element not listed, must come from
#: :meth:`AOBasis.fit`, which derives the counts from the dataset itself.
DZVP_MOLOPT_SR_GTH_AO: Mapping[int, int] = {
    1: 5, 2: 5,
    3: 13, 4: 13, 5: 13, 6: 13, 7: 13, 8: 13, 9: 13, 10: 13,
    11: 13, 12: 13, 13: 13, 14: 13, 15: 13, 16: 13, 17: 13, 18: 13,
}

KNOWN_BASIS_SETS: Mapping[str, Mapping[int, int]] = {
    "DZVP-MOLOPT-SR-GTH": DZVP_MOLOPT_SR_GTH_AO,
    "DZVP-MOLOPT-GTH": DZVP_MOLOPT_SR_GTH_AO,
}


class DensityModelError(RuntimeError):
    """Raised when a density model cannot be built or applied."""


@dataclass(frozen=True, slots=True)
class AOBasis:
    """How many atomic orbitals each element contributes.

    This is basis-set information, not a hyper-parameter: getting it wrong
    produces a matrix of the wrong size, which every consumer then rejects.
    It is therefore verified against the dataset before use.
    """

    ao_per_element: Mapping[int, int]
    basis_set: str | None = None

    def __post_init__(self) -> None:
        for number, count in self.ao_per_element.items():
            if int(count) <= 0:
                raise DensityModelError(f"element {number} has a non-positive AO count {count}")

    @property
    def max_ao_per_atom(self) -> int:
        return max(self.ao_per_element.values()) if self.ao_per_element else 0

    def n_ao(self, atomic_numbers: Sequence[int]) -> int:
        return sum(self.count(int(number)) for number in atomic_numbers)

    def count(self, atomic_number: int) -> int:
        try:
            return int(self.ao_per_element[int(atomic_number)])
        except KeyError as exc:
            raise DensityModelError(
                f"no AO count known for element {atomic_number}; "
                "supply one explicitly or use AOBasis.fit on a dataset that contains it"
            ) from exc

    def ao_layout(self, atomic_numbers: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(atom_of_ao, position_within_atom)`` for one system."""

        atom_of: list[int] = []
        position: list[int] = []
        for index, number in enumerate(atomic_numbers):
            for slot in range(self.count(int(number))):
                atom_of.append(index)
                position.append(slot)
        return np.asarray(atom_of, dtype=np.int64), np.asarray(position, dtype=np.int64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ao_per_element": {str(key): int(value) for key, value in sorted(self.ao_per_element.items())},
            "basis_set": self.basis_set,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AOBasis":
        return cls(
            ao_per_element={int(key): int(value) for key, value in payload["ao_per_element"].items()},
            basis_set=payload.get("basis_set"),
        )

    @classmethod
    def for_basis_set(cls, basis_set: str) -> "AOBasis":
        key = str(basis_set).strip().upper()
        if key not in KNOWN_BASIS_SETS:
            raise DensityModelError(
                f"no AO table for basis set {basis_set!r}; known: {', '.join(sorted(KNOWN_BASIS_SETS))}. "
                "Use AOBasis.fit to derive the counts from recorded densities instead."
            )
        return cls(ao_per_element=dict(KNOWN_BASIS_SETS[key]), basis_set=key)

    @classmethod
    def fit(cls, compositions: Sequence[tuple[Sequence[int], int]], *, basis_set: str | None = None) -> "AOBasis":
        """Derive per-element AO counts from ``(atomic_numbers, n_ao)`` pairs.

        Recorded densities already state their dimension, so the counts can be
        solved for rather than looked up -- which is the only way to support a
        basis set this module has never heard of.  The solution is verified to
        reproduce every input exactly; an approximate fit is rejected, because
        an off-by-one AO count silently corrupts every block boundary.
        """

        if not compositions:
            raise DensityModelError("cannot fit an AO basis from zero systems")

        elements = sorted({int(number) for numbers, _ in compositions for number in numbers})
        design = np.zeros((len(compositions), len(elements)), dtype=float)
        target = np.zeros(len(compositions), dtype=float)
        index_of = {element: position for position, element in enumerate(elements)}

        for row, (numbers, n_ao) in enumerate(compositions):
            for number in numbers:
                design[row, index_of[int(number)]] += 1.0
            target[row] = float(n_ao)

        # An under-determined system still reproduces every total exactly while
        # assigning the wrong count to each element -- H2O alone admits both
        # (H=5, O=13) and (H=9, O=5). Both give nao=23 and both pass the
        # verification below, but only one puts the block boundaries in the
        # right place. Rank deficiency must therefore be refused, not solved.
        rank = int(np.linalg.matrix_rank(design))
        if rank < len(elements):
            raise DensityModelError(
                f"the dataset does not determine per-element AO counts: {len(elements)} elements "
                f"({elements}) but only rank {rank} of independent compositions. Add systems with "
                "different stoichiometries, or pass an explicit AOBasis."
            )

        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        rounded = np.rint(solution).astype(int)

        if not np.allclose(design @ rounded, target, atol=1e-9):
            raise DensityModelError(
                "no consistent per-element AO count reproduces every recorded density dimension; "
                "the dataset probably mixes basis sets"
            )
        if (rounded <= 0).any():
            bad = [elements[position] for position, value in enumerate(rounded) if value <= 0]
            raise DensityModelError(
                f"fitted AO counts are non-positive for elements {bad}; "
                "the dataset does not constrain them (add systems containing them alone)"
            )

        return cls(
            ao_per_element={element: int(rounded[index_of[element]]) for element in elements},
            basis_set=basis_set,
        )


@dataclass(frozen=True, slots=True)
class ValenceTable:
    """How many electrons each element actually contributes.

    Not ``Z``. Every calculation here uses GTH pseudopotentials, so the density
    CP2K works with holds *valence* electrons only: water is 8, not 10; Cu2O is
    28, not 66. A model normalised to the nuclear charge is 25%-140% wrong
    depending on the chemistry, and CP2K rejects it -- correctly -- against a
    10^-3 tolerance.

    The counts are fitted from the dataset rather than transcribed, exactly as
    :class:`AOBasis` fits AO counts: every recorded density states the electron
    count it integrates to, so the per-element valences are solved for and then
    verified to reproduce every sample. A table nobody typed is a table nobody
    can typo.
    """

    valence_per_element: Mapping[int, int]
    #: What produced these counts, e.g. "GTH-PBE". Recorded so a model fitted
    #: against one pseudopotential family is identifiable as such.
    potential_family: str | None = None

    def __post_init__(self) -> None:
        for number, count in self.valence_per_element.items():
            if int(count) <= 0:
                raise DensityModelError(
                    f"element {number} has a non-positive valence count {count}"
                )
            if int(count) > int(number):
                raise DensityModelError(
                    f"element {number} has valence {count}, more than its nuclear charge; "
                    "these counts are electrons the pseudopotential keeps, never more than Z"
                )

    def count(self, atomic_number: int) -> int:
        try:
            return int(self.valence_per_element[int(atomic_number)])
        except KeyError as exc:
            raise DensityModelError(
                f"no valence electron count known for element {atomic_number}; "
                "fit the table on a dataset that contains it"
            ) from exc

    def electrons(self, atomic_numbers: Sequence[int], charge: float = 0.0) -> float:
        """``Ne`` for one system: valence electrons minus the total charge."""

        return float(sum(self.count(int(number)) for number in atomic_numbers) - float(charge))

    def is_nuclear_charge(self) -> bool:
        """Whether this table is really just ``Z`` -- i.e. no pseudopotential.

        An all-electron calculation is legitimate, but a table that equals ``Z``
        for a GTH run is the bug this class exists to prevent, so callers get to
        ask.
        """

        return all(int(count) == int(number) for number, count in self.valence_per_element.items())

    def to_dict(self) -> dict[str, Any]:
        return {
            "valence_per_element": {
                str(key): int(value) for key, value in sorted(self.valence_per_element.items())
            },
            "potential_family": self.potential_family,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValenceTable":
        return cls(
            valence_per_element={
                int(key): int(value) for key, value in payload["valence_per_element"].items()
            },
            potential_family=payload.get("potential_family"),
        )

    @classmethod
    def fit(
        cls,
        compositions: Sequence[tuple[Sequence[int], float]],
        *,
        charges: Sequence[float] | None = None,
        potential_family: str | None = None,
    ) -> "ValenceTable":
        """Derive per-element valences from ``(atomic_numbers, electrons)`` pairs.

        ``electrons`` is the count the recorded density integrates to,
        ``Tr(P S)``, with the system's charge already removed by ``charges``.

        Rank deficiency is refused for the same reason as in :meth:`AOBasis.fit`:
        a dataset of only H2O admits (H=1, O=6) and (H=3, O=2) equally, and both
        reproduce 8. Guessing between them puts the wrong number of electrons on
        every atom of every other system.
        """

        if not compositions:
            raise DensityModelError("cannot fit valence counts from zero systems")

        offsets = list(charges) if charges is not None else [0.0] * len(compositions)
        if len(offsets) != len(compositions):
            raise DensityModelError("charges must have one entry per composition")

        elements = sorted({int(number) for numbers, _ in compositions for number in numbers})
        design = np.zeros((len(compositions), len(elements)), dtype=float)
        target = np.zeros(len(compositions), dtype=float)
        index_of = {element: position for position, element in enumerate(elements)}

        for row, (numbers, electrons) in enumerate(compositions):
            for number in numbers:
                design[row, index_of[int(number)]] += 1.0
            # Ne = sum(valence) - charge, so the charge moves to the target.
            target[row] = float(electrons) + float(offsets[row])

        rank = int(np.linalg.matrix_rank(design))
        if rank < len(elements):
            raise DensityModelError(
                f"the dataset does not determine per-element valence counts: {len(elements)} "
                f"elements ({elements}) but only rank {rank} of independent compositions. Add "
                "systems with different stoichiometries, or pass an explicit ValenceTable."
            )

        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        rounded = np.rint(solution).astype(int)

        # Electron counts are integers. A fit that needs fractions is a fit of
        # something else -- a mixed pseudopotential set, or a charged system
        # whose charge was not passed in.
        if not np.allclose(solution, rounded, atol=1e-3):
            raise DensityModelError(
                f"fitted valence counts are not integral ({solution.round(3).tolist()}); "
                "the dataset probably mixes pseudopotentials or omits a system charge"
            )
        if not np.allclose(design @ rounded, target, atol=1e-2):
            raise DensityModelError(
                "no consistent per-element valence count reproduces every recorded electron "
                "count; the dataset probably mixes pseudopotentials"
            )

        return cls(
            valence_per_element={element: int(rounded[index_of[element]]) for element in elements},
            potential_family=potential_family,
        )


@dataclass(frozen=True, slots=True)
class DensityGNNConfig:
    """Architecture of the density graph network."""

    hidden_size: int = 64
    n_interactions: int = 3
    n_radial_basis: int = 24
    cutoff: float = 6.0
    max_atomic_number: int = 100
    max_ao_per_atom: int = 13
    max_spin_channels: int = 2
    n_global_features: int = 4
    architecture: str = MODEL_ARCHITECTURE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DensityGNNConfig":
        known = {key: payload[key] for key in payload if key in cls.__slots__}
        return cls(**known)


@dataclass(frozen=True, slots=True)
class LossWeights:
    """Relative weights of the three loss terms.

    ``electron`` is a *diagnostic* weight in normal use: the output is already
    rescaled to conserve electrons, so this term is near zero unless the
    rescaling was skipped (no overlap available). Keeping it costs nothing and
    catches a broken normalisation immediately.
    """

    density: float = 1.0
    electron: float = 1.0
    energy: float = 0.0
    idempotency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DensityBatch:
    """A padded batch of systems, ready for the model."""

    atomic_numbers: np.ndarray  # (B, A)
    positions: np.ndarray  # (B, A, 3)
    atom_mask: np.ndarray  # (B, A)
    global_features: np.ndarray  # (B, G)
    ao_atom_index: np.ndarray  # (B, N) atom each AO belongs to
    ao_slot_index: np.ndarray  # (B, N) position within that atom
    ao_mask: np.ndarray  # (B, N)
    density_target: np.ndarray  # (B, S, N, N)
    overlap: np.ndarray  # (B, N, N)
    has_overlap: np.ndarray  # (B,)
    channel_electrons: np.ndarray  # (B, S)
    channel_mask: np.ndarray  # (B, S)
    sample_ids: tuple[str, ...] = ()

    @property
    def n_ao(self) -> int:
        return int(self.ao_mask.shape[1])


def channel_electron_counts(electron_count: float, multiplicity: int, spin_channels: int) -> tuple[float, ...]:
    """Split an electron count across spin channels."""

    if spin_channels == 1:
        return (float(electron_count),)
    unpaired = float(multiplicity - 1)
    return ((electron_count + unpaired) / 2.0, (electron_count - unpaired) / 2.0)


def collate_density_batch(examples: Sequence[Any], basis: AOBasis) -> DensityBatch:
    """Pad a list of :class:`SCFExample` into a :class:`DensityBatch`.

    Padded atoms and padded AOs are masked out everywhere, so a short system
    batched with a long one contributes nothing spurious to the loss or to the
    predicted electron count.
    """

    if not examples:
        raise DensityModelError("cannot collate an empty batch")

    layouts = []
    for example in examples:
        if example.positions is None:
            raise DensityModelError(f"{example.sample_id} carries no coordinates")
        if example.density_target.ndim != 3:
            raise DensityModelError(
                f"{example.sample_id} density target has shape {example.density_target.shape}, "
                "expected (spin, nao, nao)"
            )
        atom_of, slot = basis.ao_layout(example.atomic_numbers)
        if atom_of.size != example.density_target.shape[1]:
            raise DensityModelError(
                f"{example.sample_id}: the AO basis implies {atom_of.size} orbitals but the recorded "
                f"density has {example.density_target.shape[1]}"
            )
        layouts.append((atom_of, slot))

    batch = len(examples)
    max_atoms = max(example.n_atoms for example in examples)
    max_ao = max(layout[0].size for layout in layouts)
    max_channels = max(example.spin_channels for example in examples)

    atomic_numbers = np.zeros((batch, max_atoms), dtype=np.int64)
    positions = np.zeros((batch, max_atoms, 3), dtype=np.float32)
    atom_mask = np.zeros((batch, max_atoms), dtype=np.float32)
    global_features = np.zeros((batch, 4), dtype=np.float32)
    ao_atom_index = np.zeros((batch, max_ao), dtype=np.int64)
    ao_slot_index = np.zeros((batch, max_ao), dtype=np.int64)
    ao_mask = np.zeros((batch, max_ao), dtype=np.float32)
    density = np.zeros((batch, max_channels, max_ao, max_ao), dtype=np.float32)
    overlap = np.zeros((batch, max_ao, max_ao), dtype=np.float32)
    has_overlap = np.zeros(batch, dtype=np.float32)
    channel_electrons = np.zeros((batch, max_channels), dtype=np.float32)
    channel_mask = np.zeros((batch, max_channels), dtype=np.float32)

    for index, (example, (atom_of, slot)) in enumerate(zip(examples, layouts, strict=True)):
        n_atoms = example.n_atoms
        n_ao = atom_of.size
        channels = example.spin_channels

        atomic_numbers[index, :n_atoms] = np.asarray(example.atomic_numbers, dtype=np.int64)
        positions[index, :n_atoms] = example.positions
        atom_mask[index, :n_atoms] = 1.0
        global_features[index] = (
            float(example.charge),
            float(example.multiplicity - 1),
            float(example.electron_count) / max(n_atoms, 1),
            float(channels),
        )
        ao_atom_index[index, :n_ao] = atom_of
        ao_slot_index[index, :n_ao] = slot
        ao_mask[index, :n_ao] = 1.0
        density[index, :channels, :n_ao, :n_ao] = example.density_target

        if example.overlap is not None:
            overlap[index, :n_ao, :n_ao] = example.overlap
            has_overlap[index] = 1.0
        else:
            # Without a recorded overlap the identity makes Tr(P S) fall back
            # to Tr(P); `has_overlap` records that the count is approximate.
            overlap[index, :n_ao, :n_ao] = np.eye(n_ao)

        counts = channel_electron_counts(example.electron_count, example.multiplicity, channels)
        channel_electrons[index, :channels] = counts
        channel_mask[index, :channels] = 1.0

    return DensityBatch(
        atomic_numbers=atomic_numbers,
        positions=positions,
        atom_mask=atom_mask,
        global_features=global_features,
        ao_atom_index=ao_atom_index,
        ao_slot_index=ao_slot_index,
        ao_mask=ao_mask,
        density_target=density,
        overlap=overlap,
        has_overlap=has_overlap,
        channel_electrons=channel_electrons,
        channel_mask=channel_mask,
        sample_ids=tuple(example.sample_id for example in examples),
    )


def build_density_model(config: DensityGNNConfig) -> Any:
    """Construct the density GNN described by ``config``."""

    torch = require_torch()
    nn = torch.nn
    # TorchScript resolves annotations from the enclosing scope, so tensor
    # arguments need a real type name rather than the module-level `Any` alias.
    # Without this the model trains fine but cannot be deployed to CP2K.
    from torch import Tensor

    class GaussianExpansion(nn.Module):
        """Expand interatomic distances onto a fixed Gaussian radial basis."""

        def __init__(self, n_basis: int, cutoff: float) -> None:
            super().__init__()
            self.register_buffer("centers", torch.linspace(0.0, cutoff, n_basis))
            self.width = float(cutoff) / max(n_basis - 1, 1)
            self.cutoff = float(cutoff)

        def forward(self, distances: Tensor) -> Tensor:
            difference = distances.unsqueeze(-1) - self.centers
            expanded = torch.exp(-0.5 * (difference / self.width) ** 2)
            # Cosine envelope: contributions must go smoothly to zero at the
            # cutoff, otherwise an atom crossing it produces a discontinuity.
            envelope = 0.5 * (torch.cos(math.pi * distances.clamp(max=self.cutoff) / self.cutoff) + 1.0)
            return expanded * envelope.unsqueeze(-1)

    class Interaction(nn.Module):
        """One continuous-filter message-passing step."""

        def __init__(self, hidden: int, n_basis: int) -> None:
            super().__init__()
            self.filter_network = nn.Sequential(
                nn.Linear(n_basis, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.message = nn.Linear(hidden, hidden)
            self.update = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

        def forward(self, features: Tensor, radial: Tensor, pair_mask: Tensor) -> Tensor:
            filters = self.filter_network(radial)
            neighbors = self.message(features).unsqueeze(1)
            messages = (filters * neighbors * pair_mask.unsqueeze(-1)).sum(dim=2)
            return features + self.update(messages)

    class DensityGNN(nn.Module):
        """Atom embeddings -> per-pair AO blocks -> Hermitian, normalised P."""

        def __init__(self, model_config: DensityGNNConfig) -> None:
            super().__init__()
            # Scalars rather than the dataclass: TorchScript cannot carry an
            # arbitrary Python object as a module attribute, and the model has
            # to be scriptable to be deployable into CP2K.
            self.max_spin_channels = int(model_config.max_spin_channels)
            hidden = model_config.hidden_size
            block = model_config.max_ao_per_atom

            self.embedding = nn.Embedding(model_config.max_atomic_number + 1, hidden, padding_idx=0)
            self.global_encoder = nn.Sequential(
                nn.Linear(model_config.n_global_features, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.radial = GaussianExpansion(model_config.n_radial_basis, model_config.cutoff)
            self.interactions = nn.ModuleList(
                Interaction(hidden, model_config.n_radial_basis) for _ in range(model_config.n_interactions)
            )
            self.pair_encoder = nn.Sequential(
                nn.Linear(2 * hidden + model_config.n_radial_basis, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.block_head = nn.Linear(hidden, model_config.max_spin_channels * block * block)
            self.block_size = block

        def forward(
            self,
            atomic_numbers: Tensor,
            positions: Tensor,
            atom_mask: Tensor,
            global_features: Tensor,
            ao_atom_index: Tensor,
            ao_slot_index: Tensor,
            ao_mask: Tensor,
        ) -> Tensor:
            batch, n_atoms = atomic_numbers.shape

            deltas = positions.unsqueeze(2) - positions.unsqueeze(1)
            distances = torch.linalg.norm(deltas, dim=-1)
            pair_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
            # Self-interaction is excluded from messages but kept for blocks:
            # P_ii is the on-site density and must still be predicted.
            eye = torch.eye(n_atoms, device=atomic_numbers.device).unsqueeze(0)
            message_mask = pair_mask * (1.0 - eye)

            radial = self.radial(distances)
            features = self.embedding(atomic_numbers) + self.global_encoder(global_features).unsqueeze(1)
            features = features * atom_mask.unsqueeze(-1)
            for interaction in self.interactions:
                features = interaction(features, radial, message_mask)
                features = features * atom_mask.unsqueeze(-1)

            pair_input = torch.cat(
                [
                    features.unsqueeze(2).expand(-1, -1, n_atoms, -1),
                    features.unsqueeze(1).expand(-1, n_atoms, -1, -1),
                    radial,
                ],
                dim=-1,
            )
            blocks = self.block_head(self.pair_encoder(pair_input))
            blocks = blocks.view(batch, n_atoms, n_atoms, self.max_spin_channels, self.block_size, self.block_size)
            blocks = blocks * pair_mask[:, :, :, None, None, None]

            # Scatter atom-pair blocks into the full AO matrix by advanced
            # indexing: P[b, p, q] = block[b, atom(p), atom(q), :, slot(p), slot(q)].
            batch_index = torch.arange(batch, device=blocks.device).view(batch, 1, 1)
            row_atom = ao_atom_index.unsqueeze(2)
            column_atom = ao_atom_index.unsqueeze(1)
            row_slot = ao_slot_index.unsqueeze(2)
            column_slot = ao_slot_index.unsqueeze(1)
            matrix = blocks[batch_index, row_atom, column_atom, :, row_slot, column_slot]
            matrix = matrix.permute(0, 3, 1, 2)

            ao_pair_mask = (ao_mask.unsqueeze(1) * ao_mask.unsqueeze(2)).unsqueeze(1)
            return matrix * ao_pair_mask

    return DensityGNN(config)


def symmetrize(matrix: Any) -> Any:
    """Return ``(P + P^T) / 2``, making the output Hermitian exactly."""

    return 0.5 * (matrix + matrix.transpose(-1, -2))


def normalize_electrons(
    matrix: Any,
    overlap: Any,
    channel_electrons: Any,
    channel_mask: Any,
    *,
    epsilon: float = 1.0e-8,
) -> Any:
    """Rescale each spin channel so that ``Tr(P S)`` equals its electron count.

    A channel whose trace is numerically zero is left alone rather than being
    divided by it: producing an infinity here would turn a merely bad guess
    into a NaN that propagates through the whole batch.
    """

    torch = require_torch()

    traces = (matrix * overlap.unsqueeze(1)).sum(dim=(-1, -2))
    safe = torch.where(traces.abs() > epsilon, traces, torch.ones_like(traces))
    scale = torch.where(traces.abs() > epsilon, channel_electrons / safe, torch.ones_like(traces))
    scale = scale * channel_mask + (1.0 - channel_mask)
    return matrix * scale[:, :, None, None]


def predicted_electron_counts(matrix: Any, overlap: Any) -> Any:
    """Return ``Tr(P S)`` per spin channel."""

    return (matrix * overlap.unsqueeze(1)).sum(dim=(-1, -2))


def density_loss(
    predicted: Any,
    batch: Any,
    weights: LossWeights,
    *,
    hamiltonian: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    """``L = L_density + w1 L_electron + w2 L_energy (+ w3 L_idempotency)``.

    Returns the total and a JSON-safe breakdown, so a training run can show
    which term is actually moving.
    """

    torch = require_torch()

    target = batch["density_target"]
    ao_pair_mask = (batch["ao_mask"].unsqueeze(1) * batch["ao_mask"].unsqueeze(2)).unsqueeze(1)
    mask = ao_pair_mask * batch["channel_mask"][:, :, None, None]
    n_entries = mask.sum().clamp(min=1.0)

    l_density = (((predicted - target) ** 2) * mask).sum() / n_entries

    counts = predicted_electron_counts(predicted, batch["overlap"])
    electron_error = (counts - batch["channel_electrons"]) * batch["channel_mask"]
    l_electron = (electron_error**2).sum() / batch["channel_mask"].sum().clamp(min=1.0)

    total = weights.density * l_density + weights.electron * l_electron
    breakdown = {
        "density": float(l_density.detach()),
        "electron": float(l_electron.detach()),
        "energy": 0.0,
        "idempotency": 0.0,
    }

    if weights.energy > 0.0:
        if hamiltonian is None:
            # Silently contributing zero would let a run report an energy term
            # it never actually applied.
            raise DensityModelError(
                "the energy-consistency term is weighted but no core Hamiltonian was supplied; "
                "record one, or set LossWeights.energy to 0"
            )
        predicted_energy = (predicted * hamiltonian.unsqueeze(1) * mask).sum(dim=(-1, -2, -3))
        reference_energy = (target * hamiltonian.unsqueeze(1) * mask).sum(dim=(-1, -2, -3))
        l_energy = ((predicted_energy - reference_energy) ** 2).mean()
        total = total + weights.energy * l_energy
        breakdown["energy"] = float(l_energy.detach())

    if weights.idempotency > 0.0:
        # For a converged single-determinant density, P S P = (2/n_spin) P.
        # Useful as a regulariser and computable from P and S alone.
        factor = 2.0 / predicted.shape[1]
        overlap = batch["overlap"].unsqueeze(1)
        product = torch.matmul(torch.matmul(predicted, overlap), predicted)
        residual = (product - factor * predicted) * mask
        l_idempotency = (residual**2).sum() / n_entries
        total = total + weights.idempotency * l_idempotency
        breakdown["idempotency"] = float(l_idempotency.detach())

    breakdown["total"] = float(total.detach())
    return total, breakdown


def batch_to_tensors(batch: DensityBatch, torch: Any, device: Any) -> dict[str, Any]:
    """Move a collated batch onto ``device`` as tensors."""

    def to(array: np.ndarray, dtype: Any) -> Any:
        return torch.as_tensor(array, dtype=dtype, device=device)

    return {
        "atomic_numbers": to(batch.atomic_numbers, torch.long),
        "positions": to(batch.positions, torch.float32),
        "atom_mask": to(batch.atom_mask, torch.float32),
        "global_features": to(batch.global_features, torch.float32),
        "ao_atom_index": to(batch.ao_atom_index, torch.long),
        "ao_slot_index": to(batch.ao_slot_index, torch.long),
        "ao_mask": to(batch.ao_mask, torch.float32),
        "density_target": to(batch.density_target, torch.float32),
        "overlap": to(batch.overlap, torch.float32),
        "channel_electrons": to(batch.channel_electrons, torch.float32),
        "channel_mask": to(batch.channel_mask, torch.float32),
    }


def forward_density(model: Any, tensors: Mapping[str, Any]) -> Any:
    """Run the model and apply both hard physical constraints."""

    raw = model(
        tensors["atomic_numbers"],
        tensors["positions"],
        tensors["atom_mask"],
        tensors["global_features"],
        tensors["ao_atom_index"],
        tensors["ao_slot_index"],
        tensors["ao_mask"],
    )
    # The head always emits `max_spin_channels`; a closed-shell batch uses one.
    # Broadcasting the mismatch instead of slicing would normalise a channel
    # that does not exist against another channel's electron count.
    n_channels = int(tensors["channel_mask"].shape[1])
    if raw.shape[1] < n_channels:
        raise DensityModelError(
            f"the model emits {raw.shape[1]} spin channel(s) but the batch needs {n_channels}; "
            "increase DensityGNNConfig.max_spin_channels"
        )
    hermitian = symmetrize(raw[:, :n_channels])
    return normalize_electrons(
        hermitian, tensors["overlap"], tensors["channel_electrons"], tensors["channel_mask"]
    )



DENSITY_CHECKPOINT_VERSION = "phal-scf-density-checkpoint-v1"


def load_density_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a density checkpoint, refusing an unsupported version."""

    torch = require_torch()
    source = Path(path)
    if not source.is_file():
        raise DensityModelError(f"density checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise DensityModelError(f"{source}: checkpoint is not a mapping")

    version = str(payload.get("checkpoint_version", ""))
    if version != DENSITY_CHECKPOINT_VERSION:
        raise DensityModelError(
            f"{source}: checkpoint_version {version!r} is not supported "
            f"(expected {DENSITY_CHECKPOINT_VERSION!r})"
        )
    return dict(payload)


__all__ = [
    "DENSITY_CHECKPOINT_VERSION",
    "load_density_checkpoint",
    "DZVP_MOLOPT_SR_GTH_AO",
    "KNOWN_BASIS_SETS",
    "MODEL_ARCHITECTURE",
    "AOBasis",
    "DensityBatch",
    "DensityGNNConfig",
    "DensityModelError",
    "LossWeights",
    "ValenceTable",
    "batch_to_tensors",
    "build_density_model",
    "channel_electron_counts",
    "collate_density_batch",
    "density_loss",
    "forward_density",
    "normalize_electrons",
    "predicted_electron_counts",
    "symmetrize",
]
