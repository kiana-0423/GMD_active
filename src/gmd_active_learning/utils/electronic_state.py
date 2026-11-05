# Ported from PHAL (hotspot_al.utils.electronic_state) at 96263b6; MIT License, see NOTICE.
"""Per-region charge and spin multiplicity resolution for DFT labeling.

Carving a local region out of a reactive MD frame routinely produces fragments
that are *not* neutral closed-shell singlets: homolytic bond cleavage yields
radicals, and oxide or transition-metal fragments are frequently open-shell.
Emitting ``CHARGE 0 / MULTIPLICITY 1`` for every extracted region therefore
produces reference forces that are silently wrong, and those forces become
training labels.

Spin multiplicity is not free to choose: the number of unpaired electrons must
have the same parity as the total electron count.  That makes the common failure
mode *provable* rather than a matter of judgement, so it is checked here.

Pseudopotentials do not affect this check.  GTH-style potentials replace closed
core shells, which always hold an even number of electrons, so the valence
electron count has the same parity as the all-electron count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ase import Atoms

from gmd_active_learning.core.exceptions import ConfigurationError as ConfigError

DEFAULT_SPIN_POLICY = "auto"


@dataclass(frozen=True, slots=True)
class ElectronicState:
    """Resolved charge and spin state for one labeling task."""

    charge: int
    multiplicity: int
    n_electrons: int
    source: str
    corrected: bool = False

    @property
    def is_open_shell(self) -> bool:
        """Whether the calculation needs a spin-polarized (UKS/LSD) treatment."""

        return self.multiplicity > 1

    def as_metadata(self) -> dict[str, Any]:
        """Provenance recorded alongside the generated input."""

        return {
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "n_electrons": self.n_electrons,
            "spin_source": self.source,
            "spin_corrected": self.corrected,
            "open_shell": self.is_open_shell,
        }


def total_electron_count(atoms: Atoms, charge: int) -> int:
    """Return the total electron count of ``atoms`` at the given net charge."""

    return int(sum(int(number) for number in atoms.get_atomic_numbers())) - int(charge)


def minimum_multiplicity(n_electrons: int) -> int:
    """Return the lowest spin multiplicity compatible with ``n_electrons``.

    Even electron counts admit a singlet; odd counts require at least a doublet.
    """

    return 1 + (int(n_electrons) % 2)


def multiplicity_is_consistent(n_electrons: int, multiplicity: int) -> bool:
    """Whether ``multiplicity`` is achievable for ``n_electrons`` electrons."""

    if multiplicity < 1:
        return False
    unpaired = multiplicity - 1
    if unpaired > n_electrons:
        return False
    return unpaired % 2 == int(n_electrons) % 2


def resolve_electronic_state(
    atoms: Atoms,
    dft_cfg: dict[str, Any],
    *,
    region_metadata: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> ElectronicState:
    """Resolve the charge and multiplicity to use for one extracted region.

    Precedence is region metadata, then configuration, then derivation from the
    electron count.  ``spin_policy`` controls what happens when the requested
    multiplicity is impossible for the fragment:

    ``auto`` (default)
        Replace it with the lowest consistent multiplicity and warn.  This
        turns a provably wrong singlet on a radical into a doublet.
    ``strict``
        Raise :class:`ConfigError` so the pipeline stops instead of writing a
        label that cannot be right.
    """

    metadata = region_metadata or {}
    charge = int(metadata.get("charge", dft_cfg.get("charge", 0)))
    n_electrons = total_electron_count(atoms, charge)
    if n_electrons < 0:
        msg = f"Region has a negative electron count ({n_electrons}) at charge {charge:+d}."
        raise ConfigError(msg)

    requested = metadata.get("multiplicity", dft_cfg.get("multiplicity"))
    source = "region" if "multiplicity" in metadata else ("config" if requested is not None else "derived")
    if requested is None:
        return ElectronicState(
            charge=charge,
            multiplicity=minimum_multiplicity(n_electrons),
            n_electrons=n_electrons,
            source="derived",
        )

    multiplicity = int(requested)
    if multiplicity_is_consistent(n_electrons, multiplicity):
        return ElectronicState(charge=charge, multiplicity=multiplicity, n_electrons=n_electrons, source=source)

    corrected = minimum_multiplicity(n_electrons)
    policy = str(dft_cfg.get("spin_policy", DEFAULT_SPIN_POLICY)).strip().lower()
    detail = (
        f"multiplicity {multiplicity} is impossible for a fragment with {n_electrons} electrons "
        f"at charge {charge:+d} ({len(atoms)} atoms, formula {atoms.get_chemical_formula()}); "
        f"the lowest consistent value is {corrected}"
    )
    if policy == "strict":
        msg = (
            f"Refusing to generate a DFT input: {detail}. "
            "Set cp2k.spin_policy=auto to derive it automatically, or provide "
            "an explicit per-region 'multiplicity' in the region metadata."
        )
        raise ConfigError(msg)
    if policy != "auto":
        msg = f"Unsupported spin_policy: {policy!r}. Use 'auto' or 'strict'."
        raise ConfigError(msg)

    if logger is not None:
        logger.warning("spin state corrected for extracted region: %s", detail)
    return ElectronicState(
        charge=charge,
        multiplicity=corrected,
        n_electrons=n_electrons,
        source="derived",
        corrected=True,
    )


__all__ = [
    "ElectronicState",
    "minimum_multiplicity",
    "multiplicity_is_consistent",
    "resolve_electronic_state",
    "total_electron_count",
]
