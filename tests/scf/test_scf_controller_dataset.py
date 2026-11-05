# Ported from PHAL (tests.scf.test_scf_controller_dataset) at 96263b6; MIT License, see NOTICE.
"""Tests for controller training data and offline policy replay.

The distinction these tests protect is between what a recording *measures* and
what a labeller *guesses*. Mixing the two is how a model ends up scoring well
against its own supervision and badly against reality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.scf.controller.policy import (
    ACTION_KEEP,
    ACTION_RESET_DIIS,
    HeuristicPolicy,
    NativePolicy,
    SCFState,
)
from tools.scf.controller.dataset import (
    DATASET_FORMAT,
    LABEL_HEURISTIC,
    build_dataset,
    dataset_summary,
    evaluate_policy,
    heuristic_action,
    load_states,
    measured_targets,
    replay,
    samples_from_sample_dir,
    write_dataset,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "scf" / "pulay_sample_periodic"


def states(residuals: list[float]) -> list[SCFState]:
    return [
        SCFState(
            iteration=index + 1,
            residual=value,
            residual_history=tuple(residuals[:index]),
            n_electrons=8.0,
        )
        for index, value in enumerate(residuals)
    ]


# -- measured targets ---------------------------------------------------


def test_steps_remaining_counts_to_the_end_of_the_recording():
    targets = measured_targets(states([1.0, 0.1, 0.01, 0.001]))

    assert [item["steps_remaining"] for item in targets] == [3, 2, 1, 0]


def test_a_clean_descent_is_not_stalled():
    targets = measured_targets(states([1.0, 0.1, 0.01, 0.001, 1e-4]))

    assert targets[0]["stalled"] is False
    assert targets[0]["rose_next"] is False


def test_a_flat_stretch_is_stalled():
    targets = measured_targets(states([1.0, 0.9, 0.85, 0.84, 0.83]))

    assert targets[0]["stalled"] is True


def test_a_rise_is_recorded_on_the_iteration_before_it():
    targets = measured_targets(states([1.0, 2.0, 0.5]))

    assert targets[0]["rose_next"] is True
    assert targets[1]["rose_next"] is False


# -- heuristic labels ---------------------------------------------------


def test_a_healthy_iteration_is_labelled_keep():
    sequence = states([1.0, 0.1, 0.01, 0.001])

    assert heuristic_action(sequence[0], {"stalled": False, "rose_next": False}) == ACTION_KEEP


def test_trouble_on_a_singular_history_is_labelled_reset():
    troubled = SCFState(
        iteration=8,
        residual=0.1,
        residual_history=(1.0, 0.5, 0.2, 0.15, 0.12, 0.11, 0.105),
        pulay={"condition_number": 1e14, "depth": 8},
    )

    assert heuristic_action(troubled, {"stalled": True, "rose_next": False}) == ACTION_RESET_DIIS


def test_the_label_source_is_carried_in_the_data():
    """The guess must be identifiable as one after it leaves this module."""

    samples = samples_from_sample_dir(FIXTURE)

    assert samples
    assert all(sample.label_source == LABEL_HEURISTIC for sample in samples)


# -- real sample --------------------------------------------------------


def test_states_are_loaded_from_a_real_recorded_sample():
    sequence, metadata = load_states(FIXTURE)

    assert len(sequence) >= 4
    assert metadata["n_ao"] == 26
    # Conditioning comes from the per-frame diagnostics recorded at run time.
    assert any(item.pulay.get("condition_number") for item in sequence)


def test_samples_from_a_real_sample_carry_features_and_measured_targets():
    samples = samples_from_sample_dir(FIXTURE, difficulty="easy")

    assert samples
    first = samples[0]
    assert first.features["log_residual"] < 10.0
    assert first.steps_remaining == len(samples) - 1
    assert first.difficulty == "easy"


def test_the_dataset_round_trips_to_disk(tmp_path: Path):
    samples = samples_from_sample_dir(FIXTURE)

    path = write_dataset(samples, tmp_path / "controller.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["dataset_format"] == DATASET_FORMAT
    assert payload["n_samples"] == len(samples)
    assert "counterfactual" in payload["note"] or "guess" in payload["note"]


def test_the_summary_reports_class_balance(tmp_path: Path):
    samples = samples_from_sample_dir(FIXTURE, difficulty="easy")

    summary = dataset_summary(samples)

    assert summary["n_systems"] == 1
    assert sum(summary["heuristic_action_counts"].values()) == len(samples)
    assert summary["difficulty_counts"] == {"easy": len(samples)}


# -- replay -------------------------------------------------------------


def test_replay_produces_one_decision_per_iteration():
    sequence = states([1.0, 0.5, 0.2, 0.1, 0.05, 0.01])

    result = replay(HeuristicPolicy(), sequence, sample_id="x")

    assert len(result.decisions) == len(sequence)
    assert result.tally["n_iterations"] == len(sequence)


def test_the_native_policy_never_fires_in_replay():
    """Method 2 in the benchmark is this: the controller's null cost."""

    result = replay(NativePolicy(), states([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    assert result.n_actions == 0


def test_evaluating_a_policy_on_a_real_dataset(tmp_path: Path):
    import shutil

    root = tmp_path / "dataset"
    root.mkdir()
    shutil.copytree(FIXTURE, root / "si-diamond")

    report = evaluate_policy(HeuristicPolicy(), root)

    assert report["n_samples"] == 1
    assert report["n_iterations"] > 0
    # Precision and recall are against measured trouble, not acceleration --
    # the note must say so, because the numbers invite the other reading.
    assert "not acceleration" in report["note"]


def test_the_real_dataset_concentrates_actions_on_hard_systems():
    """A controller that fires on easy runs is pure overhead."""

    root = Path("runtime/cp2k/pulay_dataset")
    if not root.is_dir():
        pytest.skip("the generated dataset is not present")

    report = evaluate_policy(HeuristicPolicy(), root)
    by_difficulty = report["by_difficulty"]
    if "hard" not in by_difficulty or "easy" not in by_difficulty:
        pytest.skip("the dataset does not span both classes")

    hard_rate = by_difficulty["hard"]["actions"] / by_difficulty["hard"]["iterations"]
    easy_rate = by_difficulty["easy"]["actions"] / max(1, by_difficulty["easy"]["iterations"])

    assert hard_rate > easy_rate


def test_a_dataset_built_from_the_real_run_covers_every_difficulty():
    root = Path("runtime/cp2k/pulay_dataset")
    if not root.is_dir():
        pytest.skip("the generated dataset is not present")

    summary = dataset_summary(build_dataset(root))

    assert summary["n_systems"] > 20
    assert set(summary["difficulty_counts"]) >= {"easy", "medium", "hard"}
