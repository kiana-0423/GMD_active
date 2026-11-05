# Ported from PHAL (tests.scf.test_scf_controller) at 96263b6; MIT License, see NOTICE.
"""Tests for the adaptive SCF controller.

The controller may cost iterations; it must never make a result wrong. Most of
what follows is about that boundary: which actions are refusable, what happens
to a rejected one, and when the controller stops interfering altogether.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.scf.controller.policy import (
    ACTION_FALLBACK,
    ACTION_INCREASE_STABILITY,
    ACTION_KEEP,
    ACTION_RESET_DIIS,
    ACTION_SWITCH_MIXING,
    FEATURE_NAMES,
    AdaptiveController,
    ControllerError,
    Decision,
    HeuristicPolicy,
    NativePolicy,
    SCFState,
    extract_features,
    feature_dict,
    states_from_trajectory,
    validate_action,
    validate_state,
    validate_transition,
)


def state(iteration: int = 6, residual: float = 1e-3, **kwargs) -> SCFState:
    kwargs.setdefault("residual_history", (1.0, 0.3, 0.1, 0.03, 0.01))
    kwargs.setdefault("n_electrons", 8.0)
    return SCFState(iteration=iteration, residual=residual, **kwargs)


# -- features -----------------------------------------------------------


def test_the_feature_vector_matches_its_names():
    values = extract_features(state())

    assert values.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(values).all()


def test_residuals_enter_as_ratios_not_only_magnitudes():
    """A tiny residual early is not trouble; a rising one is, at any scale."""

    descending = feature_dict(state(residual=1e-6, residual_history=(1e-4, 1e-5)))
    rising = feature_dict(state(residual=1e-4, residual_history=(1e-6, 1e-5)))

    assert descending["log_residual_ratio"] < 0.0
    assert rising["log_residual_ratio"] > 0.0


def test_a_non_finite_residual_does_not_produce_a_non_finite_feature():
    """Features feed a model; a NaN there is silent, an action is not."""

    values = extract_features(state(residual=float("nan")))

    assert np.isfinite(values).all()


def test_iterations_since_best_counts_the_stall():
    detail = feature_dict(state(residual=0.02, residual_history=(1.0, 0.01, 0.02, 0.03)))

    assert detail["iterations_since_best"] == 3


def test_conditioning_enters_from_the_recorded_diagnostics():
    detail = feature_dict(
        state(pulay={"condition_number": 1e12, "depth": 8, "rank": 6, "coefficient_l1": 40.0})
    )

    assert detail["log_condition_number"] == pytest.approx(12.0)
    assert detail["rank_deficit"] == 2.0
    assert detail["coefficient_l1"] == 40.0


# -- safety -------------------------------------------------------------


def test_an_unknown_action_is_refused():
    with pytest.raises(ControllerError, match="unknown action"):
        validate_action("delete_everything")


def test_the_controller_cannot_select_a_mixer_cp2k_does_not_have():
    with pytest.raises(ControllerError, match="may select only"):
        validate_action(ACTION_SWITCH_MIXING, {"mixing_method": "MAGIC_MIXING"})


def test_the_controller_cannot_select_the_experimental_ml_mixer():
    """ML_PULAY is a documented negative result; control must not reach for it."""

    with pytest.raises(ControllerError, match="ML_PULAY is experimental"):
        validate_action(ACTION_SWITCH_MIXING, {"mixing_method": "ML_PULAY"})


def test_increase_stability_must_actually_be_more_stable():
    with pytest.raises(ControllerError, match="must reduce ALPHA"):
        validate_action(ACTION_INCREASE_STABILITY, {"alpha_scale": 2.0})


def test_a_non_hermitian_density_is_refused():
    density = np.array([[[1.0, 0.2], [0.0, 1.0]]])

    verdict = validate_state(state(), density=density)

    assert not verdict.accepted
    assert "Hermitian" in verdict.reason


def test_a_density_with_the_wrong_electron_count_is_refused():
    """Tr(P S), not Tr(P): in an AO basis the second is not the electron count."""

    density = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    overlap = np.array([[1.0, 0.5], [0.5, 1.0]])

    verdict = validate_state(state(n_electrons=8.0), density=density, overlap=overlap)

    assert not verdict.accepted
    assert "Tr(P S)" in verdict.reason


def test_a_physical_density_is_accepted():
    density = np.array([[[2.0, 0.0], [0.0, 2.0]]])
    overlap = np.array([[1.0, 0.0], [0.0, 1.0]])

    verdict = validate_state(state(n_electrons=4.0), density=density, overlap=overlap)

    assert verdict.accepted


def test_a_nan_residual_is_refused():
    assert not validate_state(state(residual=float("nan"))).accepted


def test_one_bad_iteration_after_a_reset_is_tolerated():
    """A reset legitimately costs an iteration; that is not a failure."""

    before = state(residual=1e-3)
    after = state(iteration=7, residual=3e-3)

    assert validate_transition(before, after).accepted


def test_a_diverging_residual_is_refused():
    before = state(residual=1e-3)
    after = state(iteration=7, residual=1.0)

    verdict = validate_transition(before, after)

    assert not verdict.accepted
    assert "diverging" in verdict.reason


# -- policy -------------------------------------------------------------


def test_the_heuristic_keeps_quiet_while_the_residual_descends():
    decision = HeuristicPolicy().decide(state())

    assert decision.action == ACTION_KEEP


def test_the_heuristic_does_not_act_before_it_has_evidence():
    early = state(iteration=1, residual=10.0, residual_history=())

    assert HeuristicPolicy().decide(early).action == ACTION_KEEP


def test_a_singular_history_gets_reset():
    decision = HeuristicPolicy().decide(state(pulay={"condition_number": 1e14, "depth": 8}))

    assert decision.action == ACTION_RESET_DIIS
    assert "float64" in decision.reason


def test_an_oscillating_residual_gets_more_conservative_mixing_first():
    """Try mixing more gently before changing method: it is the smaller change."""

    oscillating = state(residual=0.5, residual_history=(0.1, 0.4, 0.15, 0.45, 0.2))

    decision = HeuristicPolicy().decide(oscillating)

    assert decision.action == ACTION_INCREASE_STABILITY
    assert decision.parameters["alpha_scale"] < 1.0


def test_a_long_stall_switches_mixer():
    # One excursion, then a crawl that never regains the best residual: the
    # mixer is descending, just not toward anything.
    stalled = state(residual=0.016, residual_history=(1.0, 0.01, 0.02, 0.019, 0.018, 0.017))

    decision = HeuristicPolicy().decide(stalled)

    assert decision.action == ACTION_SWITCH_MIXING
    assert decision.parameters["mixing_method"] in {"BROYDEN_MIXING", "DIRECT_P_MIXING"}


def test_the_native_policy_never_acts():
    assert NativePolicy().decide(state(pulay={"condition_number": 1e18})).action == ACTION_KEEP


# -- the controller -----------------------------------------------------


class AlwaysReset:
    def decide(self, state: SCFState) -> Decision:  # noqa: ARG002
        return Decision(ACTION_RESET_DIIS, "always")


class ProposesGarbage:
    def decide(self, state: SCFState) -> Decision:  # noqa: ARG002
        return Decision(ACTION_SWITCH_MIXING, "nonsense", parameters={"mixing_method": "NOPE"})


def test_actions_are_spaced_so_a_history_can_accumulate():
    """Resetting every iteration means never having a history at all."""

    controller = AdaptiveController(AlwaysReset())

    actions = [controller.decide(state(iteration=index)).action for index in range(4, 12)]

    assert actions[0] == ACTION_RESET_DIIS
    assert actions[1] == ACTION_KEEP
    assert actions[2] == ACTION_KEEP
    assert actions[3] == ACTION_RESET_DIIS


def test_an_unsafe_proposal_becomes_a_keep_rather_than_a_crash():
    controller = AdaptiveController(ProposesGarbage())

    decision = controller.decide(state(iteration=6))

    assert decision.action == ACTION_KEEP
    assert "refused" in decision.reason
    assert controller.tally.n_rejected == 1


def test_the_controller_stops_interfering_after_repeated_rejections():
    """A controller that keeps costing iterations is worse than none."""

    controller = AdaptiveController(AlwaysReset(), max_rejections=2)
    before = state(iteration=6, residual=1e-3)
    diverged = state(iteration=7, residual=10.0)

    controller.decide(before)
    assert not controller.observe(before, diverged).accepted
    controller.decide(state(iteration=9))
    assert not controller.observe(before, diverged).accepted

    assert controller.fell_back
    assert controller.decide(state(iteration=20)).action == ACTION_KEEP
    assert controller.tally.fell_back


def test_an_accepted_transition_is_tallied():
    controller = AdaptiveController(AlwaysReset())
    before = state(iteration=6, residual=1e-3)
    after = state(iteration=7, residual=5e-4)

    controller.decide(before)
    verdict = controller.observe(before, after)

    assert verdict.accepted
    assert controller.tally.n_accepted == 1
    assert controller.tally.n_actions == 1


def test_explicit_fallback_is_terminal():
    class Surrenders:
        def decide(self, state: SCFState) -> Decision:  # noqa: ARG002
            return Decision(ACTION_FALLBACK, "give up")

    controller = AdaptiveController(Surrenders())

    assert controller.decide(state(iteration=6)).action == ACTION_FALLBACK
    assert controller.decide(state(iteration=20)).action == ACTION_KEEP
    assert controller.fell_back


# -- reconstruction -----------------------------------------------------


def test_states_are_rebuilt_from_a_recorded_trajectory():
    metadata = {"n_electrons": 8.0, "spin_channels": 1}
    frames = [
        {"index": 0, "iteration": 1, "residual": 1.0, "energy": -1.0},
        {"index": 1, "iteration": 2, "residual": 0.1, "energy": -1.5},
        {"index": 2, "iteration": 3, "residual": None, "energy": -1.6, "is_final": True},
    ]

    states = states_from_trajectory(metadata, frames)

    assert [item.iteration for item in states] == [1, 2]
    assert states[1].residual_history == (1.0,)
    assert states[1].energy_change == pytest.approx(0.5)
    assert states[1].n_electrons == 8.0
