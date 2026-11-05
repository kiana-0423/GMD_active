# Ported from PHAL (hotspot_al.cp2k_ml_scf.density.density_predictor) at 96263b6; MIT License, see NOTICE.
"""The SCF density prediction contract, plus a dummy implementation.

The abstract base class fixes the four operations CP2K's ML initial-guess hook
needs: load a model, predict from a structure, persist the prediction, and
validate it before it is allowed anywhere near the SCF.  Validation is not
optional politeness -- an initial density with the wrong electron count or a
NaN in it would either make CP2K diverge or, worse, converge somewhere quietly
wrong, so a predictor that cannot prove its output is sane is rejected and CP2K
falls back to ``SCF_GUESS ATOMIC``.
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gmd_active_learning.adapters.dft.scf.density.prediction import (
    DEFAULT_ELECTRON_TOLERANCE,
    DENSITY_FORMAT_ATOMIC_POPULATION,
    DENSITY_FORMAT_DENSITY_MATRIX,
    PredictionError,
    SCFDensityPrediction,
    write_prediction,
)
from gmd_active_learning.adapters.dft.scf.schema import SCFElectronicState, SCFStructure
from gmd_active_learning.adapters.dft.scf.schema import structure_hash as compute_structure_hash

#: Valence electron counts for GTH-style pseudopotentials, used by the dummy
#: predictor to distribute charge in a way that at least has the right total.
VALENCE_ELECTRONS: Mapping[int, int] = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 9, 12: 10, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
    19: 9, 20: 10, 22: 12, 24: 14, 25: 15, 26: 16, 27: 17, 28: 18,
    29: 11, 30: 12, 31: 13, 32: 4, 33: 5, 34: 6, 35: 7,
}


def expected_density_electrons(
    structure: SCFStructure,
    electronic_state: SCFElectronicState,
    *,
    valence: Mapping[int, int] = VALENCE_ELECTRONS,
) -> float:
    """Electrons a CP2K density holds under GTH pseudopotentials: sum(valence) - charge.

    ``SCFElectronicState.n_electrons`` is the *all-electron* count (sum(Z) -
    charge); every density CP2K holds under GTH is valence-only, so comparing a
    density against it rejects every correct prediction (finding F-23). An
    element without a known valence raises rather than being approximated.
    """

    missing = sorted({int(z) for z in structure.atomic_numbers if int(z) not in valence})
    if missing:
        raise PredictionError(f"no GTH valence electron count known for element(s) {missing}")
    return float(sum(valence[int(z)] for z in structure.atomic_numbers) - electronic_state.charge)


@dataclass(frozen=True, slots=True)
class PredictionValidation:
    """Outcome of validating a prediction before CP2K consumes it."""

    valid: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "reasons": list(self.reasons)}


class SCFDensityPredictor(abc.ABC):
    """Predict an initial electronic state for one structure.

    Implementations must be side-effect free with respect to CP2K: producing a
    prediction never commits CP2K to using it.
    """

    #: Human-readable model identity, recorded in every prediction.
    model_name: str = "unnamed"
    model_version: str = "0"
    #: Registry key, e.g. ``"dummy"`` or ``"torch"``.
    predictor_type: str = "abstract"
    #: What :meth:`predict` emits.
    density_format: str = DENSITY_FORMAT_ATOMIC_POPULATION

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self._loaded = False

    # -- required contract ----------------------------------------------

    @abc.abstractmethod
    def load_model(self) -> "SCFDensityPredictor":
        """Load weights into memory. Idempotent; returns ``self``."""

    @abc.abstractmethod
    def predict(
        self,
        structure: SCFStructure,
        electronic_state: SCFElectronicState,
    ) -> SCFDensityPrediction:
        """Return a predicted initial density for ``structure``."""

    @property
    @abc.abstractmethod
    def model_digest(self) -> str:
        """A digest identifying the exact weights used for a prediction."""

    # -- shared behaviour ------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def save_prediction(self, prediction: SCFDensityPrediction, path: str | Path) -> Path:
        """Persist a prediction and its provenance."""

        return write_prediction(prediction, path)

    def validate_prediction(
        self,
        prediction: SCFDensityPrediction,
        electronic_state: SCFElectronicState,
        *,
        structure: SCFStructure | None = None,
        electron_tolerance: float = DEFAULT_ELECTRON_TOLERANCE,
        expected_electrons: float | None = None,
    ) -> PredictionValidation:
        """Check a prediction against the physics CP2K will assume.

        The same checks are mirrored on the Fortran side before the density is
        accepted; duplicating them here means a broken predictor is caught in
        Python, where the error message is useful.
        """

        reasons: list[str] = []
        values = prediction.values

        if values is None:
            reasons.append("prediction carries no density values")
            return PredictionValidation(valid=False, reasons=tuple(reasons))

        array = np.asarray(values, dtype=float)
        if not np.isfinite(array).all():
            reasons.append("density contains non-finite values")

        expected_channels = electronic_state.spin_channels
        if prediction.spin_channels != expected_channels:
            reasons.append(
                f"prediction declares {prediction.spin_channels} spin channel(s), "
                f"expected {expected_channels}"
            )
        if array.ndim == 0 or array.shape[0] != prediction.spin_channels:
            reasons.append(
                f"density array leading dimension {array.shape} does not match "
                f"{prediction.spin_channels} spin channel(s)"
            )

        if prediction.density_format == DENSITY_FORMAT_ATOMIC_POPULATION:
            if array.ndim != 2:
                reasons.append(f"atomic populations must be 2-D, got shape {array.shape}")
            elif structure is not None and array.shape[1] != structure.n_atoms:
                reasons.append(
                    f"atomic populations cover {array.shape[1]} atoms, structure has {structure.n_atoms}"
                )
            if np.isfinite(array).all() and (array < 0.0).any():
                reasons.append("atomic populations must be non-negative")
            total = float(array.sum()) if np.isfinite(array).all() else float("nan")
        elif prediction.density_format == DENSITY_FORMAT_DENSITY_MATRIX:
            if array.ndim != 3 or array.shape[1] != array.shape[2]:
                reasons.append(f"density matrices must be square per spin, got shape {array.shape}")
                total = float("nan")
            else:
                asymmetry = float(np.abs(array - np.swapaxes(array, 1, 2)).max()) if np.isfinite(array).all() else float("nan")
                if not np.isfinite(asymmetry) or asymmetry > 1.0e-8:
                    reasons.append(f"density matrices are not symmetric (max asymmetry {asymmetry:g})")
                # Ne = Tr(P S), not Tr(P): the AO basis is non-orthogonal, and
                # Tr(P) runs 20-50% low on real systems. Checking a v2
                # prediction (whose declared count is Tr(P S)) against Tr(P)
                # would reject every correct density.
                # See docs/density_representation.md.
                overlap = getattr(prediction, "overlap", None)
                if overlap is None:
                    reasons.append(
                        "density matrix carries no overlap, so its electron count Tr(P S) "
                        "cannot be verified"
                    )
                    total = float("nan")
                else:
                    weights = np.asarray(overlap, dtype=float)
                    if weights.shape != array.shape[1:]:
                        reasons.append(
                            f"overlap shape {weights.shape} does not match density {array.shape[1:]}"
                        )
                        total = float("nan")
                    else:
                        total = float(sum(np.sum(channel * weights) for channel in array))
        else:  # pragma: no cover - constructor already rejects unknown formats
            reasons.append(f"unsupported density_format {prediction.density_format!r}")
            total = float("nan")

        declared = float(prediction.electron_count)
        # Deliberate fix (finding F-23): the reference is the *valence* count
        # the density lives in. PHAL compared against the all-electron count,
        # so a correct valence-normalised matrix was always rejected.
        if expected_electrons is not None:
            expected = float(expected_electrons)
        elif structure is not None and not self._all_electron(prediction):
            try:
                expected = expected_density_electrons(structure, electronic_state)
            except PredictionError as exc:
                reasons.append(str(exc))
                expected = float("nan")
        else:
            # Without the structure only the all-electron count is known; correct
            # only for all-electron calculations or predictors that declare it.
            expected = float(electronic_state.n_electrons)
        if np.isfinite(total) and abs(total - declared) > max(electron_tolerance, 1e-6 * max(abs(declared), 1.0)):
            reasons.append(
                f"density integrates to {total:.6f} electrons but the prediction declares {declared:.6f}"
            )
        if np.isfinite(expected) and abs(declared - expected) > electron_tolerance * max(1.0, abs(expected)):
            reasons.append(
                f"prediction declares {declared:.6f} electrons but the electronic state has {expected:.6f}"
            )

        if structure is not None and prediction.structure_hash != compute_structure_hash(
            structure, electronic_state
        ):
            reasons.append("prediction structure_hash does not match the structure it is being applied to")

        return PredictionValidation(valid=not reasons, reasons=tuple(reasons))

    @staticmethod
    def _all_electron(prediction: SCFDensityPrediction) -> bool:
        return str(prediction.metadata.get("electron_count_basis", "valence")) == "all_electron"


class DummyDensityPredictor(SCFDensityPredictor):
    """A deterministic, model-free predictor used to exercise the pipeline.

    It distributes the requested electron count over the atoms in proportion to
    their pseudopotential valence, which is roughly what ``SCF_GUESS ATOMIC``
    already does.  It is therefore *not* expected to accelerate anything; its
    job is to prove the plumbing -- prediction, provenance, persistence,
    validation, and CP2K's fallback path -- before a real model exists.
    """

    model_name = "dummy-atomic-population"
    model_version = "1"
    predictor_type = "dummy"
    density_format = DENSITY_FORMAT_ATOMIC_POPULATION

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        settings = dict(self.config.get("options", {})) if isinstance(self.config.get("options"), Mapping) else {}
        self.use_valence = bool(settings.get("use_valence_electrons", True))

    def load_model(self) -> "DummyDensityPredictor":
        # Nothing to load: the "model" is the valence table above.
        self._loaded = True
        return self

    @property
    def model_digest(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"{self.model_name}:{self.model_version}:{self.use_valence}".encode("utf-8"))
        for number, valence in sorted(VALENCE_ELECTRONS.items()):
            hasher.update(f"|{number}:{valence}".encode("utf-8"))
        return f"sha256:{hasher.hexdigest()}"

    def predict(
        self,
        structure: SCFStructure,
        electronic_state: SCFElectronicState,
    ) -> SCFDensityPrediction:
        if not self._loaded:
            self.load_model()
        if structure.n_atoms == 0:
            raise PredictionError("cannot predict a density for a structure with no atoms")

        weights = np.asarray(
            [self._weight(int(number)) for number in structure.atomic_numbers], dtype=float
        )
        if weights.sum() <= 0.0:  # pragma: no cover - defensive
            weights = np.ones(structure.n_atoms, dtype=float)

        # Deliberate fix (finding F-23): normalise to the GTH valence count, not
        # to sum(Z) - charge (the nuclear-charge error CP2K rejects).
        if self.use_valence:
            electron_count = expected_density_electrons(structure, electronic_state)
        else:
            electron_count = float(electronic_state.n_electrons)
        populations = weights / weights.sum() * electron_count

        channels = electronic_state.spin_channels
        if channels == 1:
            values = populations[None, :]
        else:
            # Split by the requested multiplicity: the excess unpaired
            # electrons all go into the alpha channel, spread over the atoms.
            unpaired = float(electronic_state.multiplicity - 1)
            share = weights / weights.sum()
            alpha = (electron_count + unpaired) / 2.0 * share
            beta = (electron_count - unpaired) / 2.0 * share
            values = np.vstack([alpha, beta])

        return SCFDensityPrediction(
            model_name=self.model_name,
            model_version=self.model_version,
            model_digest=self.model_digest,
            structure_hash=compute_structure_hash(structure, electronic_state),
            electron_count=electron_count,
            spin_channels=channels,
            density_format=self.density_format,
            values=values,
            predictor_type=self.predictor_type,
            metadata={
                "n_atoms": structure.n_atoms,
                "use_valence_electrons": self.use_valence,
                "electron_count_basis": "valence" if self.use_valence else "all_electron",
            },
        )

    def _weight(self, atomic_number: int) -> float:
        if not self.use_valence:
            return float(atomic_number)
        return float(VALENCE_ELECTRONS.get(atomic_number, atomic_number))


__all__ = [
    "VALENCE_ELECTRONS",
    "expected_density_electrons",
    "DummyDensityPredictor",
    "PredictionValidation",
    "SCFDensityPredictor",
]
