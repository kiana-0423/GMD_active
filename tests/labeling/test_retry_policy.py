"""Retry decisions: bounded budget, per-class ladders, native ML-SCF fallback, walltime parsing."""

from __future__ import annotations

import pytest

from gmd_active_learning.adapters.dft.cp2k.config import FailureClass, RetrySettings, RetryStep
from gmd_active_learning.adapters.schedulers.config import walltime_seconds
from gmd_active_learning.labeling.retry_policy import RetryPolicy


def _retry(trigger: FailureClass) -> dict:
    return {"kind": "retry", "trigger_class": trigger.value}


def test_scf_ladder_is_walked_in_order_then_exhausted() -> None:
    policy = RetryPolicy(RetrySettings())
    first = policy.decide(FailureClass.SCF_NOT_CONVERGED, [{"kind": "initial"}])
    assert first.retry and first.step == RetryStep(use_ot=False, mixing_method="BROYDEN_MIXING", mixing_nbuffer=4)
    second = policy.decide(FailureClass.SCF_NOT_CONVERGED, [{"kind": "initial"}, _retry(FailureClass.SCF_NOT_CONVERGED)])
    assert second.retry and second.step == RetryStep(max_scf_multiplier=3.0)
    history = [{"kind": "initial"}, _retry(FailureClass.SCF_NOT_CONVERGED), _retry(FailureClass.SCF_NOT_CONVERGED)]
    third = policy.decide(FailureClass.SCF_NOT_CONVERGED, history)
    assert not third.retry and "no further retry step" in third.reason


def test_ladders_are_counted_per_failure_class() -> None:
    policy = RetryPolicy(RetrySettings())
    history = [{"kind": "initial"}, _retry(FailureClass.EXECUTION_FAILED)]
    decision = policy.decide(FailureClass.SCF_NOT_CONVERGED, history)
    assert decision.retry and decision.ladder_index == 0
    assert not policy.decide(FailureClass.EXECUTION_FAILED, history).retry


def test_total_retry_budget_bounds_everything() -> None:
    policy = RetryPolicy(RetrySettings(max_retries=1))
    history = [{"kind": "initial"}, _retry(FailureClass.WALLTIME)]
    decision = policy.decide(FailureClass.SCF_NOT_CONVERGED, history)
    assert not decision.retry and "budget exhausted" in decision.reason
    stage_transitions_are_free = [{"kind": "initial"}, {"kind": "stage"}]
    assert policy.decide(FailureClass.SCF_NOT_CONVERGED, stage_transitions_are_free).retry


@pytest.mark.parametrize("failure", [FailureClass.CANCELLED, FailureClass.INVALID_RESULT])
def test_classes_without_retry(failure: FailureClass) -> None:
    assert not RetryPolicy(RetrySettings()).decide(failure, [{"kind": "initial"}]).retry


def test_disabled_retries() -> None:
    decision = RetryPolicy(RetrySettings(enabled=False)).decide(FailureClass.WALLTIME, [{"kind": "initial"}])
    assert not decision.retry and "disabled" in decision.reason


def test_ml_scf_failure_falls_back_natively_first_keeping_the_current_step() -> None:
    policy = RetryPolicy(RetrySettings())
    step = RetryStep(max_scf=500)
    decision = policy.decide(FailureClass.SCF_NOT_CONVERGED, [{"kind": "initial"}], ml_active=True, current_step=step)
    assert decision.retry and decision.native_fallback and decision.step == step
    assert not RetryPolicy(RetrySettings(ml_scf_native_fallback=False)).decide(
        FailureClass.INVALID_RESULT, [{"kind": "initial"}], ml_active=True
    ).retry


def test_configured_ladders_merge_with_defaults() -> None:
    settings = RetrySettings.model_validate({"ladders": {"walltime": [{"max_scf": 50}, {}]}})
    assert len(settings.ladders[FailureClass.WALLTIME]) == 2
    assert len(settings.ladders[FailureClass.SCF_NOT_CONVERGED]) == 2  # default kept
    with pytest.raises(ValueError):
        RetrySettings.model_validate({"ladders": {"walltime": [{"max_scf_typo": 1}]}})


@pytest.mark.parametrize(
    "raw, seconds",
    [("01:00:00", 3600), ("30:00", 1800), ("1-02:00:00", 93600), (90, 90), ("45", 45), (None, None), (1.4, 1)],
)
def test_walltime_parsing_table(raw, seconds) -> None:
    assert walltime_seconds(raw) == seconds


@pytest.mark.parametrize("raw", ["1h", "-5", 0, True, "1:2:3:4"])
def test_walltime_parsing_rejects(raw) -> None:
    with pytest.raises(ValueError):
        walltime_seconds(raw)
