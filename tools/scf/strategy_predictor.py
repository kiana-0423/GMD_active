# Ported from PHAL (hotspot_al.cp2k_ml_scf.scf_strategy: StrategyPredictor, structure_features,
# leave_one_out) at 96263b6; MIT License, see NOTICE.
"""Structure-based SCF strategy predictor (k-NN) -- experimental, negative result.

PHAL's measurement, quoted verbatim from ``docs/scf/scf_strategy_prediction.md``:
"The predictor does not work at this size" and "It is not recommended for use as
it stands, and its own default threshold makes it a no-op rather than a risk."
At a permissive threshold it is worse than doing nothing (250 vs 196 SCF steps).

The model-free result that *does* work -- the deterministic retry ladder
``default -> nbuffer4`` (12/12 converged against 9/12) -- lives in the runtime
module :mod:`gmd_active_learning.adapters.dft.scf.strategy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from gmd_active_learning.adapters.dft.scf.strategy import (  # noqa: F401 - re-exported
    DEFAULT_SMEARED,
    DEFAULT_STRATEGY,
    FEATURE_NAMES,
    KNOWN_MIXERS,
    MIXING_BROYDEN,
    MIXING_DIRECT,
    MIXING_PULAY,
    STRATEGY_FORMAT,
    SCFStrategy,
    StrategyDataset,
    StrategyError,
    StrategyOutcome,
    apply_strategy,
    retry_ladder,
    strategy_grid,
)

#: Z ranges that make an element a d-block metal for feature purposes.
_TRANSITION_METALS = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81))

#: Pauling electronegativities for the elements this project uses. A wide
#: spread means ionic bonding, which is what correlates with the hard,
#: badly-conditioned SCFs.
_ELECTRONEGATIVITY = {
    1: 2.20, 3: 0.98, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 11: 0.93, 12: 1.31,
    13: 1.61, 14: 1.90, 17: 3.16, 22: 1.54, 25: 1.55, 28: 1.91, 29: 1.90, 30: 1.65,
}


def structure_features(
    atomic_numbers: Sequence[int],
    *,
    cell: np.ndarray | None = None,
    periodic: bool = False,
    charge: float = 0.0,
    multiplicity: int = 1,
    valence_electrons: float | None = None,
    min_distance: float | None = None,
) -> dict[str, float]:
    """Everything the predictor sees. Structure only -- no SCF has run yet."""

    numbers = [int(value) for value in atomic_numbers]
    if not numbers:
        raise StrategyError("a structure with no atoms has no features")

    volume = 0.0
    if cell is not None and periodic:
        volume = abs(float(np.linalg.det(np.asarray(cell, dtype=float))))

    electronegativities = [_ELECTRONEGATIVITY.get(number) for number in set(numbers)]
    known = [value for value in electronegativities if value is not None]
    spread = float(max(known) - min(known)) if len(known) > 1 else 0.0

    return {
        "n_atoms": float(len(numbers)),
        "log_n_atoms": float(np.log(len(numbers))),
        "periodic": 1.0 if periodic else 0.0,
        "cell_volume": volume,
        "number_density": (len(numbers) / volume) if volume > 0 else 0.0,
        "min_distance": float(min_distance) if min_distance else 0.0,
        "mean_atomic_number": float(np.mean(numbers)),
        "max_atomic_number": float(max(numbers)),
        "n_elements": float(len(set(numbers))),
        "has_transition_metal": 1.0 if set(numbers) & _TRANSITION_METALS else 0.0,
        "has_oxygen": 1.0 if 8 in numbers else 0.0,
        "electronegativity_spread": spread,
        "valence_electrons": float(valence_electrons or 0.0),
        "charge": float(charge),
        "multiplicity": float(multiplicity),
    }


def feature_vector(features: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """What to run, how sure the predictor is, and why."""

    strategy: str
    confidence: float
    reason: str
    neighbours: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "confidence": self.confidence,
            "reason": self.reason,
            "neighbours": list(self.neighbours),
        }


class StrategyPredictor:
    """Nearest neighbours in feature space, weighted by distance.

    A network on a dozen systems would be a way of hiding how little data there
    is. k-NN says the same thing honestly: it recommends what worked on the
    most similar systems it has seen, reports how similar those were, and can
    be inspected by name.

    Features are standardised on the training split only. Fitting the scaler on
    everything leaks the held-out system into its own prediction, which for a
    dataset this small is most of the answer.
    """

    #: Below this the predictor was measured to *degrade* results: at 0.5 it
    #: recommended alpha0.2 for NiO on a 52% vote and cost 86 SCF steps against
    #: 33. At 0.7 and above it declines to guess often enough that it never
    #: hurts -- and, on a 12-system dataset, often enough that it never helps
    #: either. Both facts are in docs/scf_strategy_prediction.md.
    DEFAULT_MIN_CONFIDENCE = 0.7

    def __init__(self, k: int = 3, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> None:
        self.k = int(k)
        self.min_confidence = float(min_confidence)
        self._names: list[str] = []
        self._matrix: np.ndarray | None = None
        self._labels: list[str] = []
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, dataset: StrategyDataset, systems: Iterable[str] | None = None) -> "StrategyPredictor":
        chosen = sorted(systems) if systems is not None else dataset.systems()
        rows, labels, names = [], [], []
        for system in chosen:
            label = dataset.best_strategy(system)
            if label is None:
                # A system nothing converged on teaches nothing about which
                # configuration to pick; it belongs in the robustness metric.
                continue
            rows.append(feature_vector(dataset.features[system]))
            labels.append(label)
            names.append(system)

        if not rows:
            raise StrategyError("no system in the training split has a best configuration")

        matrix = np.vstack(rows)
        self._mean = matrix.mean(axis=0)
        spread = matrix.std(axis=0)
        # A feature that never varies carries no information; dividing by its
        # zero spread would turn it into the only one that matters.
        spread[spread < 1e-12] = 1.0
        self._scale = spread
        self._matrix = (matrix - self._mean) / self._scale
        self._labels = labels
        self._names = names
        return self

    def recommend(self, features: Mapping[str, float]) -> Recommendation:
        if self._matrix is None or self._mean is None or self._scale is None:
            raise StrategyError("the predictor has not been fitted")

        query = (feature_vector(features) - self._mean) / self._scale
        distances = np.linalg.norm(self._matrix - query, axis=1)
        order = np.argsort(distances)[: max(1, min(self.k, len(distances)))]

        # Inverse-distance votes, so a very close neighbour outweighs several
        # distant ones rather than being outvoted by them.
        votes: dict[str, float] = {}
        for index in order:
            weight = 1.0 / (1.0 + float(distances[index]))
            votes[self._labels[index]] = votes.get(self._labels[index], 0.0) + weight

        total = sum(votes.values())
        winner = max(votes, key=lambda key: votes[key])
        confidence = votes[winner] / total if total > 0 else 0.0
        neighbours = tuple(self._names[index] for index in order)

        if confidence < self.min_confidence:
            # Safety: an unsure recommendation is worth less than the default,
            # which is known to work on most systems and never surprises.
            return Recommendation(
                strategy="default",
                confidence=confidence,
                reason=(
                    f"nearest systems disagree ({votes}); falling back to CP2K's defaults "
                    f"rather than acting on a {confidence:.0%} vote"
                ),
                neighbours=neighbours,
            )

        return Recommendation(
            strategy=winner,
            confidence=confidence,
            reason=f"{confidence:.0%} of the weighted vote from {', '.join(neighbours)}",
            neighbours=neighbours,
        )


def leave_one_out(
    dataset: StrategyDataset, *, k: int = 3, min_confidence: float = StrategyPredictor.DEFAULT_MIN_CONFIDENCE
) -> dict[str, Any]:
    """Evaluate the predictor the only way a dozen systems allow.

    Each system is predicted by a model that never saw it. With a dataset this
    small any other split reports memorisation -- and the comparison that
    matters is not accuracy but *measured cost*: whether following the
    recommendation would have used fewer SCF steps than the default.
    """

    systems = [system for system in dataset.systems() if dataset.best_strategy(system)]
    records: list[dict[str, Any]] = []

    for system in systems:
        others = [name for name in systems if name != system]
        if not others:
            continue
        predictor = StrategyPredictor(k=k, min_confidence=min_confidence).fit(dataset, others)
        recommendation = predictor.recommend(dataset.features[system])

        outcomes = dataset.by_system(system)
        default = outcomes.get("default")
        chosen = outcomes.get(recommendation.strategy)
        best = dataset.best_strategy(system)
        oracle = outcomes.get(best) if best else None

        records.append(
            {
                "system": system,
                "recommended": recommendation.strategy,
                "confidence": recommendation.confidence,
                "reason": recommendation.reason,
                "best": best,
                "correct": recommendation.strategy == best,
                "default_steps": default.scf_steps if default and default.converged else None,
                "recommended_steps": chosen.scf_steps if chosen and chosen.converged else None,
                "oracle_steps": oracle.scf_steps if oracle and oracle.converged else None,
                "default_converged": bool(default and default.converged),
                "recommended_converged": bool(chosen and chosen.converged),
            }
        )

    paired = [
        item
        for item in records
        if item["default_steps"] is not None and item["recommended_steps"] is not None
    ]
    totals = {
        "n_systems": len(records),
        "accuracy": (
            sum(1 for item in records if item["correct"]) / len(records) if records else 0.0
        ),
        "default_total_steps": sum(item["default_steps"] for item in paired),
        "recommended_total_steps": sum(item["recommended_steps"] for item in paired),
        "oracle_total_steps": sum(
            item["oracle_steps"] for item in records if item["oracle_steps"] is not None
        ),
        "n_improved": sum(
            1 for item in paired if item["recommended_steps"] < item["default_steps"]
        ),
        "n_degraded": sum(
            1 for item in paired if item["recommended_steps"] > item["default_steps"]
        ),
        "n_default_failed_recommendation_converged": sum(
            1 for item in records if not item["default_converged"] and item["recommended_converged"]
        ),
        "n_default_converged_recommendation_failed": sum(
            1 for item in records if item["default_converged"] and not item["recommended_converged"]
        ),
    }
    if totals["default_total_steps"]:
        totals["step_saving"] = (
            totals["default_total_steps"] - totals["recommended_total_steps"]
        ) / totals["default_total_steps"]
    return {"per_system": records, "summary": totals}
