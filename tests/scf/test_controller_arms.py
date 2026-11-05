# Ported from PHAL (tests.scf.test_controller_arms) at 96263b6; MIT License, see NOTICE.
"""Tests for driving and reading the in-CP2K adaptive controller.

The controller is a strategy layer inside CP2K's SCF loop; PHAL's side of it is
switching it on, forcing an action for counterfactual labelling, and reading
back what it did. What these protect is that a controlled run is
distinguishable from an uncontrolled one, and that a label is never derived
from a run that did not converge.
"""

from __future__ import annotations

import pytest

from tools.scf.benchmark.guess import (
    CONTROLLER_ENV,
    CONTROLLER_FORCE_ENV,
    GUESS_ATOMIC,
    GUESS_ML,
    GuessArm,
    parse_run,
)

CONTROLLED = """
 ML-CTRL| iteration    7: increase_stability  alpha   0.2000
 ML-CTRL| iteration   10: reset_diis  alpha   0.2000
 ML-CTRL| iteration   11: action was wrong and cannot be undone, residual   1.6551E-01
 ML-CTRL| too many rejected actions; the controller is off for the rest of this SCF
  *** SCF run converged in    26 steps ***
 ML-CTRL| summary: actions=2 accepted=1 rejected=1 fell_back=T
 ML-CTRL| tally: reset_diis=1 switch_mixing=0 increase_stability=1
"""

UNCONTROLLED = """
  *** SCF run converged in    17 steps ***
"""


# -- arms ---------------------------------------------------------------


def test_the_controller_arm_sets_only_the_controller_switch():
    """Orthogonal to the guess: a difference must be attributable to one."""

    environment = GuessArm(label="controller", guess=GUESS_ATOMIC, controller=True).environment()

    assert environment == {CONTROLLER_ENV: "1"}


def test_the_native_arm_sets_nothing():
    assert GuessArm(label="native", guess=GUESS_ATOMIC).environment() == {}


def test_a_forced_action_is_passed_through():
    arm = GuessArm(label="i5-reset", guess=GUESS_ATOMIC, controller=True, force="5:reset_diis")

    assert arm.environment()[CONTROLLER_FORCE_ENV] == "5:reset_diis"


def test_forcing_without_the_controller_does_nothing():
    """The override is read by the controller; without it there is no reader."""

    arm = GuessArm(label="broken", guess=GUESS_ATOMIC, controller=False, force="5:reset_diis")

    assert CONTROLLER_FORCE_ENV not in arm.environment()


def test_a_guess_and_the_controller_can_be_combined():
    arm = GuessArm(
        label="ml+ctrl", guess=GUESS_ML, model="/m.pt", alpha=0.5, controller=True
    )

    environment = arm.environment()
    assert environment[CONTROLLER_ENV] == "1"
    assert environment["PHAL_ML_SCF_GUESS_MODEL"] == "/m.pt"


# -- reading a run ------------------------------------------------------


def test_a_controlled_run_reports_its_actions():
    result = parse_run(CONTROLLED, system="mno", initial_guess=GUESS_ATOMIC, runtime_s=1.0)

    assert result.controller is not None
    assert result.controller["n_actions"] == 2
    assert result.controller["n_rejected"] == 1
    assert result.controller["fell_back"] is True
    assert [item["action"] for item in result.controller["actions"]] == [
        "increase_stability",
        "reset_diis",
    ]


def test_the_rollback_line_is_not_read_as_an_action():
    """It also starts "iteration N:", and counting it would inflate the tally."""

    result = parse_run(CONTROLLED, system="mno", initial_guess=GUESS_ATOMIC, runtime_s=1.0)

    assert len(result.controller["actions"]) == 2


def test_an_uncontrolled_run_has_no_controller_record():
    """Absent, not zeroed: a run with no controller is not a run that did nothing."""

    result = parse_run(UNCONTROLLED, system="h2o", initial_guess=GUESS_ATOMIC, runtime_s=1.0)

    assert result.controller is None


# -- labels -------------------------------------------------------------


def make_run(system, steps, converged=True):
    from tools.scf.benchmark.guess import GuessRun

    return GuessRun(
        system=system,
        initial_guess=GUESS_ATOMIC,
        converged=converged,
        scf_steps=steps,
        runtime_s=1.0,
        energy=-1.0,
    )


def label_for(results, iterations):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "tools" / "scf" / "scripts" / "benchmark_controller.py"
    spec = importlib.util.spec_from_file_location("benchmark_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.counterfactual_labels(results, iterations)


def test_the_label_is_the_action_that_left_least_to_do():
    results = {
        "native": [make_run("mno", 30)],
        "i5-keep": [make_run("mno", 30)],
        "i5-reset_diis": [make_run("mno", 22)],
        "i5-switch_mixing": [make_run("mno", 34)],
        "i5-increase_stability": [make_run("mno", 28)],
    }

    (label,) = label_for(results, [5])

    assert label["label"] == "reset_diis"
    assert label["remaining_by_action"]["reset_diis"] == 17
    assert label["saving_vs_keep"] == 8


def test_a_failed_counterfactual_gets_no_cost_rather_than_a_good_one():
    """Otherwise the action that broke the run looks like the cheapest one."""

    results = {
        "native": [make_run("mno", 30)],
        "i5-keep": [make_run("mno", 30)],
        "i5-reset_diis": [make_run("mno", None, converged=False)],
        "i5-switch_mixing": [make_run("mno", 34)],
        "i5-increase_stability": [make_run("mno", 28)],
    }

    (label,) = label_for(results, [5])

    assert label["remaining_by_action"]["reset_diis"] is None
    assert label["label"] == "increase_stability"


def test_no_label_is_produced_after_the_run_had_already_finished():
    """There was no decision to make at iteration 12 of an 8-step SCF."""

    results = {
        "native": [make_run("si", 8)],
        "i12-keep": [make_run("si", 8)],
        "i12-reset_diis": [make_run("si", 8)],
        "i12-switch_mixing": [make_run("si", 8)],
        "i12-increase_stability": [make_run("si", 8)],
    }

    assert label_for(results, [12]) == []


def test_a_system_whose_baseline_failed_is_skipped():
    """Remaining steps are counted against the native path; without one there
    is nothing to compare to."""

    results = {
        "native": [make_run("mgo", None, converged=False)],
        "i5-keep": [make_run("mgo", 20)],
        "i5-reset_diis": [make_run("mgo", 18)],
        "i5-switch_mixing": [make_run("mgo", 25)],
        "i5-increase_stability": [make_run("mgo", 22)],
    }

    assert label_for(results, [5]) == []
