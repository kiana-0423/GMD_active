# Ported from PHAL (hotspot_al.cp2k_ml_scf.guess_diagnostics) at 96263b6; MIT License, see NOTICE.
"""Why an initial guess helps or hurts, measured rather than argued.

The ML density guess costs 63 SCF steps where the atomic guess costs 53. The
pipeline is correct -- the model runs in process, the density is installed, the
electron count is right -- so the deficit is algorithmic, and there are four
candidate explanations. This module measures the quantities that tell them
apart.

**A. Poor prediction.** The predicted density is simply farther from the
converged one than CP2K's atomic superposition is. Measured directly:
``||P_guess - P_final||``.

**B. Bad trajectory initialisation.** The prediction is *closer* in norm but
starts the SCF worse. A density can be near the answer and still be a bad place
to begin: what the first iteration responds to is the residual, and a density
that is off the manifold reachable by the SCF -- not idempotent, not consistent
with any Hamiltonian in the sequence -- can be close in distance and far in
residual. Distinguished by comparing the first residual against the distance.

**C. Generalisation failure.** The prediction is good on systems the model was
fitted on and bad on ones it was not. Needs held-out systems, so the caller
marks which is which.

**D. Dataset bias.** The deficit tracks a property of the training set rather
than of the model -- one chemistry, one difficulty class, one size.

The distances are computed in the metric the basis actually has. ``P`` lives in
a non-orthogonal AO basis, so a plain Frobenius norm weights diffuse functions
by however large their overlap happens to be. ``sqrt(Tr(dP S dP S))`` is the
basis-independent version and is reported beside it; where the two disagree,
the second is the one to believe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPORT_NAME = "ml_density_failure_report.json"
REPORT_FORMAT = "phal-ml-guess-diagnosis-v1"

CAUSE_PREDICTION = "A_poor_prediction"
CAUSE_INITIALISATION = "B_bad_trajectory_initialisation"
CAUSE_GENERALISATION = "C_generalisation_failure"
CAUSE_BIAS = "D_dataset_bias"
CAUSE_NONE = "none_ml_guess_helped"
#: The ML arm never ran a model (fallback), or a run did not converge. Neither
#: is evidence about the model, and counting them as such is how a diagnosis
#: acquires a majority it did not earn.
NOT_APPLICABLE = "not_applicable"

#: Below this relative difference the two guesses are the same distance away
#: and neither explanation applies.
DISTANCE_TIE = 0.05


class DiagnosticsError(RuntimeError):
    """Raised when a diagnosis cannot be computed from what was recorded."""


@dataclass(frozen=True, slots=True)
class GuessDistance:
    """How far a starting density is from the converged one."""

    frobenius: float
    #: sqrt(Tr(dP S dP S)) -- the same difference measured in the metric the
    #: overlap defines, which is what "distance" means in a non-orthogonal
    #: basis.
    metric: float
    #: Electron-count error of the guess itself, Tr(P S) - Ne.
    electron_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frobenius": self.frobenius,
            "metric": self.metric,
            "electron_error": self.electron_error,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    """The shape of one SCF descent."""

    n_iterations: int
    first_residual: float | None
    second_residual: float | None
    final_residual: float | None
    #: Geometric mean residual ratio per iteration; below 1 is convergence.
    reduction_rate: float | None
    #: Energy of the first iteration minus the converged energy, in hartree.
    initial_energy_error: float | None
    energy_path: tuple[float, ...] = ()
    residual_path: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_iterations": self.n_iterations,
            "first_residual": self.first_residual,
            "second_residual": self.second_residual,
            "final_residual": self.final_residual,
            "reduction_rate": self.reduction_rate,
            "initial_energy_error": self.initial_energy_error,
            "energy_path": list(self.energy_path),
            "residual_path": list(self.residual_path),
        }


@dataclass(frozen=True, slots=True)
class SystemDiagnosis:
    """One system, both guesses, and what the comparison implies."""

    system: str
    difficulty: str | None
    in_training_set: bool | None
    atomic_distance: GuessDistance | None
    ml_distance: GuessDistance | None
    scf_steps_atomic: int | None
    scf_steps_ml: int | None
    atomic_trajectory: TrajectoryMetrics | None
    ml_trajectory: TrajectoryMetrics | None
    classification: str
    evidence: tuple[str, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def step_delta(self) -> int | None:
        if self.scf_steps_atomic is None or self.scf_steps_ml is None:
            return None
        return self.scf_steps_ml - self.scf_steps_atomic

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "difficulty": self.difficulty,
            "in_training_set": self.in_training_set,
            "atomic_distance": self.atomic_distance.to_dict() if self.atomic_distance else None,
            "ml_distance": self.ml_distance.to_dict() if self.ml_distance else None,
            "scf_steps_atomic": self.scf_steps_atomic,
            "scf_steps_ml": self.scf_steps_ml,
            "step_delta": self.step_delta,
            "atomic_trajectory": self.atomic_trajectory.to_dict() if self.atomic_trajectory else None,
            "ml_trajectory": self.ml_trajectory.to_dict() if self.ml_trajectory else None,
            "classification": self.classification,
            "evidence": list(self.evidence),
            "notes": dict(self.notes),
        }


# -- measurement --------------------------------------------------------


def density_distance(
    guess: np.ndarray, final: np.ndarray, overlap: np.ndarray, *, n_electrons: float | None = None
) -> GuessDistance:
    """``||P_guess - P_final||`` in both the plain and the overlap metric."""

    start = np.asarray(guess, dtype=float)
    end = np.asarray(final, dtype=float)
    s = np.asarray(overlap, dtype=float)

    if start.ndim == 2:
        start = start[None, :, :]
    if end.ndim == 2:
        end = end[None, :, :]
    if start.shape != end.shape:
        raise DiagnosticsError(f"shapes differ: guess {start.shape}, final {end.shape}")
    if s.shape != start.shape[1:]:
        raise DiagnosticsError(f"overlap {s.shape} does not match density {start.shape[1:]}")

    delta = start - end
    frobenius = float(np.sqrt((delta**2).sum()))

    # Tr(dP S dP S), summed over spin. Non-negative for symmetric dP because
    # S is positive definite, so the square root is real.
    metric_sq = 0.0
    for channel in delta:
        product = channel @ s
        metric_sq += float(np.trace(product @ product))
    metric = float(np.sqrt(max(metric_sq, 0.0)))

    electron_error = 0.0
    if n_electrons is not None:
        electron_error = float((start * s).sum() - float(n_electrons))

    return GuessDistance(frobenius=frobenius, metric=metric, electron_error=electron_error)


def trajectory_metrics(
    residuals: Sequence[float | None], energies: Sequence[float | None]
) -> TrajectoryMetrics:
    """Summarise a descent: where it started, how fast it fell, where it ended.

    ``reduction_rate`` is the geometric mean of the per-iteration residual
    ratio. The geometric mean rather than the arithmetic one because the
    residual falls multiplicatively; averaging ratios arithmetically lets a
    single bad iteration dominate a run that was otherwise fine.
    """

    values = [float(item) for item in residuals if item is not None and np.isfinite(float(item))]
    energy_values = [float(item) for item in energies if item is not None and np.isfinite(float(item))]

    ratios = [
        values[index + 1] / values[index]
        for index in range(len(values) - 1)
        if values[index] > 0.0 and values[index + 1] > 0.0
    ]
    rate = float(np.exp(np.mean(np.log(ratios)))) if ratios else None

    initial_energy_error = None
    if len(energy_values) >= 2:
        initial_energy_error = float(energy_values[0] - energy_values[-1])

    return TrajectoryMetrics(
        n_iterations=len(values),
        first_residual=values[0] if values else None,
        second_residual=values[1] if len(values) > 1 else None,
        final_residual=values[-1] if values else None,
        reduction_rate=rate,
        initial_energy_error=initial_energy_error,
        energy_path=tuple(energy_values),
        residual_path=tuple(values),
    )


# -- interpretation -----------------------------------------------------


def classify(
    *,
    atomic_distance: GuessDistance | None,
    ml_distance: GuessDistance | None,
    atomic_trajectory: TrajectoryMetrics | None,
    ml_trajectory: TrajectoryMetrics | None,
    scf_steps_atomic: int | None,
    scf_steps_ml: int | None,
    ml_guess_used: bool = True,
    both_converged: bool = True,
) -> tuple[str, list[str]]:
    """Decide, for one system, whether the guess or the start is at fault.

    Only A and B are decidable from a single system. C and D are statements
    about a *set* -- whether the deficit tracks training membership, or a
    property of the training distribution -- and are left to
    :func:`diagnose_set`.
    """

    evidence: list[str] = []

    if not ml_guess_used:
        return NOT_APPLICABLE, [
            "the ML arm fell back to the atomic guess, so both runs are the same run"
        ]
    if not both_converged:
        return NOT_APPLICABLE, [
            "a run did not converge, so there is no converged density to measure against"
        ]
    if atomic_distance is None or ml_distance is None:
        return "undetermined", ["one of the two guesses was not measured"]

    atomic_metric = atomic_distance.metric
    ml_metric = ml_distance.metric
    if atomic_metric <= 0.0:
        return "undetermined", ["the atomic guess is already the converged density"]

    relative = (ml_metric - atomic_metric) / atomic_metric
    evidence.append(
        f"distance to P_final: ML {ml_metric:.4f} vs atomic {atomic_metric:.4f} "
        f"({relative:+.1%})"
    )

    if scf_steps_atomic is not None and scf_steps_ml is not None:
        evidence.append(f"SCF steps: ML {scf_steps_ml} vs atomic {scf_steps_atomic}")
        if scf_steps_ml < scf_steps_atomic:
            return CAUSE_NONE, evidence + ["the ML guess converged faster on this system"]

    if relative > DISTANCE_TIE:
        return CAUSE_PREDICTION, evidence + [
            "the predicted density is farther from the answer than the atomic "
            "superposition, so no amount of SCF machinery can profit from it"
        ]

    # Closer (or no worse) yet slower: the density is a bad starting *point*
    # rather than a bad answer. Three quantities say how.
    detail: list[str] = []
    if atomic_trajectory and ml_trajectory:
        first_atomic = atomic_trajectory.first_residual
        first_ml = ml_trajectory.first_residual
        if first_atomic and first_ml:
            evidence.append(
                f"first residual: ML {first_ml:.3e} vs atomic {first_atomic:.3e} "
                f"({(first_ml / first_atomic):.2f}x)"
            )
            if first_ml > first_atomic:
                detail.append("it opens with a larger residual despite being closer")

        energy_atomic = atomic_trajectory.initial_energy_error
        energy_ml = ml_trajectory.initial_energy_error
        if energy_atomic is not None and energy_ml is not None:
            evidence.append(
                f"initial energy error: ML {energy_ml:+.3f} Ha vs atomic {energy_atomic:+.3f} Ha"
            )
            if abs(energy_ml) > abs(energy_atomic):
                detail.append(
                    "it starts at a higher energy than the atomic superposition, so the "
                    "density is closer in norm without being a more physical state"
                )

        rate_atomic = atomic_trajectory.reduction_rate
        rate_ml = ml_trajectory.reduction_rate
        if rate_atomic and rate_ml:
            evidence.append(
                f"reduction rate per iteration: ML {rate_ml:.3f} vs atomic {rate_atomic:.3f}"
            )
            if rate_ml > rate_atomic:
                detail.append(
                    "the descent itself is slower, so the error it leaves is in modes the "
                    "mixer clears less efficiently than the ones an atomic guess leaves"
                )

    return CAUSE_INITIALISATION, evidence + (
        detail or ["no closer and no faster: the guess does not start the SCF anywhere useful"]
    )


def diagnose_set(diagnoses: Sequence[SystemDiagnosis]) -> dict[str, Any]:
    """Look across systems for the two causes a single system cannot show.

    C and D are comparisons: fitted-on against held-out, and one slice of the
    training distribution against another. Neither is visible from one run, and
    both are easy to assert without measuring -- which is why they are computed
    here from the per-system numbers rather than concluded from them.
    """

    usable = [
        item
        for item in diagnoses
        if item.atomic_distance
        and item.ml_distance
        and item.classification != NOT_APPLICABLE
    ]
    if not usable:
        return {"verdict": "undetermined", "reason": "no system produced both distances"}

    def excess(item: SystemDiagnosis) -> float:
        atomic = item.atomic_distance.metric  # type: ignore[union-attr]
        ml = item.ml_distance.metric  # type: ignore[union-attr]
        return (ml - atomic) / atomic if atomic > 0 else float("nan")

    seen = [item for item in usable if item.in_training_set]
    unseen = [item for item in usable if item.in_training_set is False]

    generalisation: dict[str, Any] = {"n_fitted_on": len(seen), "n_held_out": len(unseen)}
    if seen and unseen:
        seen_excess = float(np.mean([excess(item) for item in seen]))
        unseen_excess = float(np.mean([excess(item) for item in unseen]))
        generalisation.update(
            mean_excess_distance_fitted=seen_excess,
            mean_excess_distance_held_out=unseen_excess,
            gap=unseen_excess - seen_excess,
        )
        # A model that predicts what it was fitted on and not what it was not
        # is a generalisation failure. One that is equally bad on both is not:
        # it is simply a bad model, which is cause A.
        generalisation["indicated"] = bool(
            unseen_excess > seen_excess + 0.10 and seen_excess < 0.05
        )
    else:
        generalisation["indicated"] = None
        generalisation["note"] = "needs both fitted-on and held-out systems"

    by_difficulty: dict[str, dict[str, Any]] = {}
    for item in usable:
        key = item.difficulty or "unknown"
        bucket = by_difficulty.setdefault(key, {"systems": [], "excess": [], "step_delta": []})
        bucket["systems"].append(item.system)
        bucket["excess"].append(excess(item))
        if item.step_delta is not None:
            bucket["step_delta"].append(item.step_delta)
    for bucket in by_difficulty.values():
        bucket["mean_excess_distance"] = float(np.mean(bucket["excess"]))
        bucket["mean_step_delta"] = (
            float(np.mean(bucket["step_delta"])) if bucket["step_delta"] else None
        )
        bucket.pop("excess")
        bucket.pop("step_delta")

    counts: dict[str, int] = {}
    for item in diagnoses:
        counts[item.classification] = counts.get(item.classification, 0) + 1
    decided = {key: value for key, value in counts.items() if key != NOT_APPLICABLE}

    dominant = max(decided, key=lambda key: decided[key]) if decided else "undetermined"
    verdict = dominant
    reasons: list[str] = []
    if generalisation.get("indicated"):
        verdict = CAUSE_GENERALISATION
        reasons.append(
            "the excess distance is small on systems the model was fitted on and large on "
            "held-out ones"
        )
    else:
        reasons.append(
            f"{decided.get(dominant, 0)} of {sum(decided.values())} usable systems show {dominant}"
        )
        if dominant == CAUSE_PREDICTION and generalisation.get("n_fitted_on"):
            fitted = generalisation.get("mean_excess_distance_fitted")
            if fitted is not None and fitted > DISTANCE_TIE:
                reasons.append(
                    f"the model is {fitted:+.0%} worse than the atomic guess even on systems "
                    "it was fitted on, so this is not a generalisation gap"
                )

    # The question behind the four causes: is the quantity the model is trained
    # to minimise the quantity that decides SCF cost? If distance and step count
    # do not move together, closing the distance further buys nothing, and the
    # next model is a better answer to the wrong question.
    paired = [
        (excess(item), float(item.step_delta))
        for item in usable
        if item.step_delta is not None and np.isfinite(excess(item))
    ]
    correlation: dict[str, Any] = {"n": len(paired)}
    if len(paired) >= 3:
        distances = np.asarray([value for value, _ in paired])
        deltas = np.asarray([value for _, value in paired])
        if distances.std() > 0 and deltas.std() > 0:
            correlation["pearson_excess_distance_vs_step_delta"] = float(
                np.corrcoef(distances, deltas)[0, 1]
            )
            ranked = (np.argsort(np.argsort(distances)), np.argsort(np.argsort(deltas)))
            correlation["spearman_excess_distance_vs_step_delta"] = float(
                np.corrcoef(ranked[0], ranked[1])[0, 1]
            )
    correlation["note"] = (
        "Correlation between how much farther the ML guess starts and how many extra SCF "
        "steps it costs. A weak correlation means density distance is not what governs "
        "convergence, and a better-fitting model would not follow."
    )

    energy_pairs = [
        (
            abs(item.ml_trajectory.initial_energy_error),
            abs(item.atomic_trajectory.initial_energy_error),
        )
        for item in usable
        if item.ml_trajectory
        and item.atomic_trajectory
        and item.ml_trajectory.initial_energy_error is not None
        and item.atomic_trajectory.initial_energy_error is not None
    ]
    energy: dict[str, Any] = {"n": len(energy_pairs)}
    if energy_pairs:
        energy["mean_initial_energy_error_ml"] = float(np.mean([ml for ml, _ in energy_pairs]))
        energy["mean_initial_energy_error_atomic"] = float(
            np.mean([atomic for _, atomic in energy_pairs])
        )
        energy["n_ml_starts_higher"] = int(sum(1 for ml, atomic in energy_pairs if ml > atomic))
        energy["note"] = (
            "|E_first - E_converged|. A prediction that is closer in density but starts at a "
            "much higher energy is not a more physical state, it is a closer one."
        )

    # One letter is rarely the whole answer, and reporting only the dominant
    # one hides the findings that decide what to do next. These are the checks
    # that a fix for the primary cause would *not* be sufficient.
    secondary: list[dict[str, Any]] = []

    closer_but_slower = [
        item
        for item in usable
        if excess(item) < -DISTANCE_TIE and (item.step_delta or 0) > 0
    ]
    if closer_but_slower:
        secondary.append(
            {
                "cause": CAUSE_INITIALISATION,
                "systems": [item.system for item in closer_but_slower],
                "finding": (
                    "these systems start measurably closer to the converged density and still "
                    "take more SCF steps, so a better-fitting model would not, by itself, "
                    "make the guess useful"
                ),
            }
        )

    spearman = correlation.get("spearman_excess_distance_vs_step_delta")
    if spearman is not None and abs(spearman) < 0.5:
        secondary.append(
            {
                "cause": CAUSE_INITIALISATION,
                "finding": (
                    f"distance to P_final and SCF cost are barely related (Spearman "
                    f"{spearman:+.2f} over {correlation['n']} systems): the quantity the model "
                    "minimises is not the quantity that governs convergence"
                ),
            }
        )

    if energy.get("n") and energy["n_ml_starts_higher"] >= 0.75 * energy["n"]:
        secondary.append(
            {
                "cause": CAUSE_INITIALISATION,
                "finding": (
                    f"the ML guess starts at a higher energy on {energy['n_ml_starts_higher']} of "
                    f"{energy['n']} systems ({energy['mean_initial_energy_error_ml']:.2f} Ha mean "
                    f"against {energy['mean_initial_energy_error_atomic']:.2f} Ha): it is closer "
                    "to the answer without being a more physical state"
                ),
            }
        )

    slices = [
        (name, bucket["mean_excess_distance"])
        for name, bucket in by_difficulty.items()
        if name != "unknown"
    ]
    if len(slices) >= 2:
        worst = max(slices, key=lambda item: item[1])
        best = min(slices, key=lambda item: item[1])
        if worst[1] - best[1] > 0.25:
            secondary.append(
                {
                    "cause": CAUSE_BIAS,
                    "finding": (
                        f"the deficit tracks difficulty: {worst[0]} systems are "
                        f"{worst[1]:+.0%} worse than atomic against {best[1]:+.0%} for "
                        f"{best[0]}, which is a property of what the training set contains"
                    ),
                }
            )

    return {
        "verdict": verdict,
        "reasons": reasons,
        "secondary_findings": secondary,
        "classification_counts": dict(sorted(counts.items())),
        "n_usable_systems": len(usable),
        "distance_vs_cost": correlation,
        "initial_energy": energy,
        "generalisation": generalisation,
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "mean_excess_distance": float(np.mean([excess(item) for item in usable])),
        "note": (
            "Excess distance is (ML - atomic) / atomic in the overlap metric. Positive means "
            "the prediction starts farther from the converged density than CP2K's atomic "
            "superposition does."
        ),
    }


def write_report(
    diagnoses: Sequence[SystemDiagnosis], path: str | Path, *, extra: Mapping[str, Any] | None = None
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report_format": REPORT_FORMAT,
        "summary": diagnose_set(diagnoses),
        "systems": [item.to_dict() for item in diagnoses],
    }
    if extra:
        payload.update(dict(extra))
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
