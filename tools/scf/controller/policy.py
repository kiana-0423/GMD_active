# Ported from PHAL (hotspot_al.cp2k_ml_scf.controller) at 96263b6; MIT License, see NOTICE.
"""Adaptive SCF control: decide the *strategy*, never the physics.

CP2K keeps every part of the calculation -- Hamiltonian, KS solve, density
update, energy, forces, stress, and the final ``EPS_SCF`` test. The controller
only answers a question CP2K currently answers with a fixed setting: *given how
this SCF is going, should anything change?*

That question is where the remaining signal is. Native DIIS already extrapolates
optimally from a given history (``docs/ml_pulay_objective.md``), so predicting
its coefficients is a tautology. What DIIS cannot do is notice that its history
has gone stale, that the residual has been rising for four iterations, or that
its own ``B`` has become numerically singular -- and decide to start over.

Five actions, deliberately few:

``keep``
    Change nothing. The default, and correct for most iterations.
``reset_diis``
    Clear the mixing history. For an SCF stuck extrapolating from a stale or
    ill-conditioned history.
``switch_mixing``
    Move to another native CP2K mixer (Broyden, direct). Still CP2K's.
``increase_stability``
    More conservative settings: smaller ALPHA, shorter history.
``fallback``
    Give up on control, run CP2K's own strategy to the end. Terminal.

Every action is validated *after* it takes effect, against the same physical
checks CP2K itself applies (``docs/density_representation.md``). A rejected
action is rolled back and the run continues natively -- the controller can cost
an iteration, but it cannot make a calculation wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

#: Bump when the feature vector changes; a policy trained on one version must
#: not be fed another.
FEATURE_VERSION = "phal-scf-controller-features-v1"

ACTION_KEEP = "keep"
ACTION_RESET_DIIS = "reset_diis"
ACTION_SWITCH_MIXING = "switch_mixing"
ACTION_INCREASE_STABILITY = "increase_stability"
ACTION_FALLBACK = "fallback"

ACTIONS: tuple[str, ...] = (
    ACTION_KEEP,
    ACTION_RESET_DIIS,
    ACTION_SWITCH_MIXING,
    ACTION_INCREASE_STABILITY,
    ACTION_FALLBACK,
)

#: Mixers the controller may switch to. All are CP2K's own; the controller
#: cannot invent one, and cannot select the experimental ML mixer.
SWITCHABLE_MIXERS: tuple[str, ...] = ("PULAY_MIXING", "BROYDEN_MIXING", "DIRECT_P_MIXING")

FEATURE_NAMES: tuple[str, ...] = (
    # Numerical
    "iteration",
    "log_residual",
    "log_residual_ratio",
    "mean_log_ratio_3",
    "n_recent_increases",
    "log_energy_change",
    "log_density_change",
    "iterations_since_best",
    # DIIS
    "log_condition_number",
    "rank_deficit",
    "coefficient_l1",
    "history_depth",
    # Electronic
    "electron_error",
    "spin_channels",
    "magnetization_change",
)

#: An SCF is not in trouble on iteration one; it has no history to judge.
MIN_ITERATION_FOR_ACTION = 4
#: Two resets in a row is thrashing, not control.
MIN_ITERATIONS_BETWEEN_ACTIONS = 3
#: After this many rejected actions the controller stops proposing and lets
#: CP2K finish its own way. A controller that keeps costing iterations is
#: worse than no controller.
MAX_REJECTIONS = 2

#: Same tolerances as the exchange contract, so a state the controller accepts
#: is a state CP2K would also have accepted.
HERMITIAN_TOLERANCE = 1.0e-8
ELECTRON_TOLERANCE = 1.0e-3
#: A residual allowed to grow by more than this after an action is divergence,
#: not a transient.
MAX_RESIDUAL_GROWTH = 10.0


class ControllerError(RuntimeError):
    """Raised when a controller is configured in a way that cannot be safe."""


@dataclass(frozen=True, slots=True)
class SCFState:
    """One SCF iteration, as the controller sees it.

    Everything here is something CP2K already computes or the recorder already
    writes. The controller introduces no new physics and no new cost.
    """

    iteration: int
    residual: float
    energy: float | None = None
    #: Residuals of previous iterations, oldest first.
    residual_history: tuple[float, ...] = ()
    energy_change: float | None = None
    density_change: float | None = None
    #: Tr(P S), which must equal the electron count in a non-orthogonal basis.
    electron_count: float | None = None
    n_electrons: float | None = None
    spin_channels: int = 1
    magnetization: float | None = None
    previous_magnetization: float | None = None
    #: From ``b_matrix_diagnostics``: condition_number, rank, depth, l1.
    pulay: Mapping[str, Any] = field(default_factory=dict)

    @property
    def best_residual(self) -> float:
        history = [*self.residual_history, self.residual]
        finite = [value for value in history if value is not None and np.isfinite(value)]
        return min(finite) if finite else float("nan")


@dataclass(frozen=True, slots=True)
class Decision:
    """What the controller chose, and why."""

    action: str
    reason: str
    confidence: float = 1.0
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether an action's outcome may stand."""

    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reason": self.reason}


# -- features -----------------------------------------------------------


def _safe_log10(value: float | None, floor: float = 1e-30) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(np.log10(max(abs(float(value)), floor)))


def extract_features(state: SCFState) -> np.ndarray:
    """Turn one SCF iteration into a fixed-length vector.

    Residuals span many decades over a run, so they enter as logarithms and as
    *ratios*: whether this iteration improved on the last is scale-free, and it
    is the thing that distinguishes a healthy SCF from a stuck one. Absolute
    residual alone would make every early iteration look like trouble.
    """

    history = [*state.residual_history, state.residual]
    ratios = [
        history[index + 1] / history[index]
        for index in range(len(history) - 1)
        if history[index] > 0.0 and np.isfinite(history[index + 1])
    ]
    recent = ratios[-3:]
    increases = sum(1 for ratio in ratios[-5:] if ratio > 1.0)

    best = state.best_residual
    since_best = 0
    for value in reversed(history):
        if np.isfinite(value) and value <= best * (1.0 + 1e-12):
            break
        since_best += 1

    pulay = state.pulay or {}
    condition = pulay.get("condition_number")
    depth = float(pulay.get("depth") or 0)
    rank = float(pulay.get("rank") or depth)

    electron_error = 0.0
    if state.electron_count is not None and state.n_electrons:
        electron_error = abs(float(state.electron_count) - float(state.n_electrons))

    magnetization_change = 0.0
    if state.magnetization is not None and state.previous_magnetization is not None:
        magnetization_change = abs(float(state.magnetization) - float(state.previous_magnetization))

    values = {
        "iteration": float(state.iteration),
        "log_residual": _safe_log10(state.residual),
        "log_residual_ratio": _safe_log10(ratios[-1], floor=1e-12) if ratios else 0.0,
        "mean_log_ratio_3": (
            float(np.mean([_safe_log10(ratio, floor=1e-12) for ratio in recent])) if recent else 0.0
        ),
        "n_recent_increases": float(increases),
        "log_energy_change": _safe_log10(state.energy_change),
        "log_density_change": _safe_log10(state.density_change),
        "iterations_since_best": float(since_best),
        "log_condition_number": _safe_log10(condition, floor=1.0),
        "rank_deficit": float(depth - rank),
        "coefficient_l1": float(pulay.get("coefficient_l1") or 0.0),
        "history_depth": depth,
        "electron_error": float(electron_error),
        "spin_channels": float(state.spin_channels),
        "magnetization_change": float(magnetization_change),
    }
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)


def feature_dict(state: SCFState) -> dict[str, float]:
    """The same features, named -- for datasets, reports and debugging."""

    return dict(zip(FEATURE_NAMES, extract_features(state).tolist()))


# -- policies -----------------------------------------------------------


class Policy(Protocol):
    """Anything that can choose an action from an SCF state."""

    def decide(self, state: SCFState) -> Decision: ...


@dataclass(frozen=True, slots=True)
class HeuristicPolicy:
    """A rule-based policy, and the baseline any model has to beat.

    The rules encode what an experienced user does by hand: reset a stale DIIS
    history, switch mixer when resetting did not help, and give up on control
    rather than keep interfering. It is deliberately shipped as the default --
    a heuristic whose behaviour can be read off the source is a better starting
    point than an untrained network, and it makes "did the model help?" a
    question with a real baseline rather than a comparison against nothing.
    """

    #: Residual rising this many times in the recent window means trouble.
    increase_threshold: int = 3
    #: log10 cond(B) past this and the history is numerically exhausted.
    condition_threshold: float = 12.0
    #: Iterations without improving on the best residual before acting.
    stall_threshold: int = 5

    def decide(self, state: SCFState) -> Decision:
        features = feature_dict(state)

        if state.iteration < MIN_ITERATION_FOR_ACTION:
            return Decision(ACTION_KEEP, "too early to judge")

        if features["n_recent_increases"] >= self.increase_threshold:
            if features["log_condition_number"] >= self.condition_threshold:
                return Decision(
                    ACTION_RESET_DIIS,
                    f"residual rose {int(features['n_recent_increases'])} times with "
                    f"log10 cond(B) = {features['log_condition_number']:.1f}: the history "
                    "is both stale and numerically exhausted",
                )
            return Decision(
                ACTION_INCREASE_STABILITY,
                f"residual rose {int(features['n_recent_increases'])} times: mix more "
                "conservatively before changing method",
                parameters={"alpha_scale": 0.5},
            )

        if features["log_condition_number"] >= self.condition_threshold:
            return Decision(
                ACTION_RESET_DIIS,
                f"log10 cond(B) = {features['log_condition_number']:.1f}: the DIIS history "
                "is linearly dependent and the solve is at the edge of float64",
            )

        if features["iterations_since_best"] >= self.stall_threshold:
            return Decision(
                ACTION_SWITCH_MIXING,
                f"no improvement for {int(features['iterations_since_best'])} iterations",
                parameters={"mixing_method": "BROYDEN_MIXING"},
            )

        return Decision(ACTION_KEEP, "residual is descending")


@dataclass(frozen=True, slots=True)
class NativePolicy:
    """Never intervenes. The control experiment.

    Method 2 in the benchmark is exactly this policy, so the controller's cost
    when it does nothing is measured rather than assumed.
    """

    def decide(self, state: SCFState) -> Decision:  # noqa: ARG002 - protocol
        return Decision(ACTION_KEEP, "native strategy")


# -- safety -------------------------------------------------------------


def validate_action(action: str, parameters: Mapping[str, Any] | None = None) -> None:
    """Refuse an action that is not one of the five, or is not CP2K's.

    A policy is allowed to be wrong about *when* to act. It is not allowed to
    name a mixer CP2K does not have, or to select the experimental ML mixer,
    because neither failure would surface as anything but a broken run.
    """

    if action not in ACTIONS:
        raise ControllerError(f"unknown action {action!r}; known: {list(ACTIONS)}")

    detail = dict(parameters or {})
    if action == ACTION_SWITCH_MIXING:
        method = str(detail.get("mixing_method", "")).upper()
        if method not in SWITCHABLE_MIXERS:
            raise ControllerError(
                f"cannot switch to {method!r}; the controller may select only "
                f"{list(SWITCHABLE_MIXERS)} (ML_PULAY is experimental and excluded)"
            )
    if action == ACTION_INCREASE_STABILITY:
        scale = float(detail.get("alpha_scale", 0.5))
        if not (0.0 < scale < 1.0):
            raise ControllerError(
                f"increase_stability must reduce ALPHA (0 < alpha_scale < 1), got {scale}"
            )


def validate_state(
    state: SCFState,
    *,
    density: np.ndarray | None = None,
    overlap: np.ndarray | None = None,
) -> Verdict:
    """Check that the electronic state is still physical.

    The same three properties CP2K enforces: finite, Hermitian, and
    ``Tr(P S) = Ne`` -- not ``Tr(P)``, which is only the electron count in an
    orthonormal basis and is wrong in every AO basis CP2K uses.
    """

    if not np.isfinite(state.residual):
        return Verdict(False, "residual is not finite")
    if state.energy is not None and not np.isfinite(state.energy):
        return Verdict(False, "energy is not finite")

    if density is not None:
        matrices = np.asarray(density, dtype=float)
        if not np.isfinite(matrices).all():
            return Verdict(False, "density contains non-finite values")
        if matrices.ndim == 3:
            asymmetry = float(np.abs(matrices - np.swapaxes(matrices, 1, 2)).max())
        else:
            asymmetry = float(np.abs(matrices - matrices.T).max())
        if asymmetry > HERMITIAN_TOLERANCE:
            return Verdict(False, f"density is not Hermitian (max asymmetry {asymmetry:.2e})")

        if overlap is not None:
            counted = float((matrices * np.asarray(overlap, dtype=float)).sum())
            expected = state.n_electrons
            if expected is not None:
                tolerance = ELECTRON_TOLERANCE * max(1.0, abs(float(expected)))
                if abs(counted - float(expected)) > tolerance:
                    return Verdict(
                        False,
                        f"Tr(P S) = {counted:.6f}, expected {float(expected):.6f}",
                    )

    if state.electron_count is not None and state.n_electrons is not None:
        tolerance = ELECTRON_TOLERANCE * max(1.0, abs(float(state.n_electrons)))
        if abs(float(state.electron_count) - float(state.n_electrons)) > tolerance:
            return Verdict(
                False,
                f"Tr(P S) = {float(state.electron_count):.6f}, "
                f"expected {float(state.n_electrons):.6f}",
            )

    return Verdict(True, "state is physical")


def validate_transition(before: SCFState, after: SCFState, **kwargs: Any) -> Verdict:
    """Judge an action by what it did, not by what it intended.

    A reset legitimately costs one iteration -- the history is gone, so the next
    step is a plain mix -- which is why the test is against divergence rather
    than against any increase. What is not allowed is the residual running away.
    """

    physical = validate_state(after, **kwargs)
    if not physical.accepted:
        return physical

    if before.residual > 0.0 and np.isfinite(after.residual):
        growth = after.residual / before.residual
        if growth > MAX_RESIDUAL_GROWTH:
            return Verdict(
                False,
                f"residual grew {growth:.1f}x (limit {MAX_RESIDUAL_GROWTH:.0f}x): diverging",
            )
    return Verdict(True, "state is physical and the residual is under control")


# -- the controller -----------------------------------------------------


@dataclass
class ControllerTally:
    """What the controller did over one SCF, for the benchmark report."""

    n_iterations: int = 0
    n_actions: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    fell_back: bool = False
    actions: list[str] = field(default_factory=list)

    @property
    def rejection_rate(self) -> float | None:
        return (self.n_rejected / self.n_actions) if self.n_actions else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_iterations": self.n_iterations,
            "n_actions": self.n_actions,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "rejection_rate": self.rejection_rate,
            "fell_back": self.fell_back,
            "actions": list(self.actions),
        }


class AdaptiveController:
    """Applies a policy under rate limits, and rolls back what it breaks.

    The rate limits are not tuning: an SCF that resets its history every
    iteration never accumulates one, so a policy that proposes resets
    enthusiastically would prevent convergence rather than accelerate it. The
    controller enforces the spacing so a policy cannot.
    """

    def __init__(
        self,
        policy: Policy | None = None,
        *,
        min_iteration: int = MIN_ITERATION_FOR_ACTION,
        min_spacing: int = MIN_ITERATIONS_BETWEEN_ACTIONS,
        max_rejections: int = MAX_REJECTIONS,
    ) -> None:
        self.policy = policy or HeuristicPolicy()
        self.min_iteration = int(min_iteration)
        self.min_spacing = int(min_spacing)
        self.max_rejections = int(max_rejections)
        self.tally = ControllerTally()
        self._last_action_iteration: int | None = None
        self._fallback = False

    @property
    def fell_back(self) -> bool:
        return self._fallback

    def decide(self, state: SCFState) -> Decision:
        """Choose an action, subject to the limits the policy cannot see."""

        self.tally.n_iterations += 1

        if self._fallback:
            return Decision(ACTION_KEEP, "controller has fallen back to the native strategy")
        if state.iteration < self.min_iteration:
            return Decision(ACTION_KEEP, "too early to judge")
        if (
            self._last_action_iteration is not None
            and state.iteration - self._last_action_iteration < self.min_spacing
        ):
            return Decision(
                ACTION_KEEP,
                f"an action was taken {state.iteration - self._last_action_iteration} "
                "iterations ago; giving it time to show an effect",
            )

        decision = self.policy.decide(state)
        if decision.action == ACTION_KEEP:
            return decision

        try:
            validate_action(decision.action, decision.parameters)
        except ControllerError as exc:
            # An unsafe proposal is not a crash: the run continues natively and
            # the proposal is counted as a rejection.
            self.tally.n_rejected += 1
            self._maybe_fall_back()
            return Decision(ACTION_KEEP, f"proposal refused: {exc}")

        if decision.action == ACTION_FALLBACK:
            self._fallback = True
            self.tally.fell_back = True

        self.tally.n_actions += 1
        self.tally.actions.append(decision.action)
        self._last_action_iteration = state.iteration
        return decision

    def observe(self, before: SCFState, after: SCFState, **kwargs: Any) -> Verdict:
        """Judge the outcome of the last action; roll back if it went wrong.

        Returning a rejecting verdict is the signal to restore the previous
        electronic state and continue with CP2K's own strategy. The controller
        cannot make a result wrong -- at worst it costs iterations, and after
        ``max_rejections`` of those it stops entirely.
        """

        verdict = validate_transition(before, after, **kwargs)
        if verdict.accepted:
            self.tally.n_accepted += 1
        else:
            self.tally.n_rejected += 1
            self._maybe_fall_back()
        return verdict

    def _maybe_fall_back(self) -> None:
        if self.tally.n_rejected >= self.max_rejections and not self._fallback:
            self._fallback = True
            self.tally.fell_back = True


def states_from_trajectory(metadata: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]) -> list[SCFState]:
    """Rebuild the controller's view of a recorded SCF.

    This is what makes offline evaluation possible: a recorded trajectory has
    every input the controller would have had live, so a policy can be replayed
    against real runs before it is ever allowed to touch one.
    """

    ordered = sorted(frames, key=lambda item: int(item.get("index", 0)))
    states: list[SCFState] = []
    residuals: list[float] = []
    previous_energy: float | None = None

    for frame in ordered:
        residual = frame.get("residual")
        if residual is None or not np.isfinite(float(residual)):
            continue
        residual = float(residual)
        energy = frame.get("energy")
        energy = float(energy) if energy is not None else None
        change = None
        if energy is not None and previous_energy is not None:
            change = abs(energy - previous_energy)

        states.append(
            SCFState(
                iteration=int(frame.get("iteration") or len(states) + 1),
                residual=residual,
                energy=energy,
                residual_history=tuple(residuals),
                energy_change=change,
                electron_count=frame.get("electron_count"),
                n_electrons=metadata.get("n_electrons"),
                spin_channels=int(frame.get("spin_channels") or metadata.get("spin_channels") or 1),
                pulay=frame.get("pulay_diagnostics") or {},
            )
        )
        residuals.append(residual)
        previous_energy = energy
    return states
