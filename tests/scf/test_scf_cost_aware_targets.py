# Ported from PHAL (tests.scf.test_scf_cost_aware_targets) at 96263b6; MIT License, see NOTICE.
"""Tests for SCF-cost-aware density targets and the guess arms that use them.

Three things are protected here: a trajectory target resolves to the frame it
names (and skips rather than silently falling back), the chemistry split does
not leak a formula family across it, and a benchmark arm passes the mixing
weight CP2K actually reads.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase", reason="the adapter reads structures through ASE")

from tools.scf.training.dataset_adapter import (  # noqa: E402
    TARGET_FINAL,
    chemistry_key,
    sample_from_directory,
    samples_from_dataset,
    split_by_chemistry,
)
from tools.scf.benchmark.guess import (  # noqa: E402
    GUESS_ALPHA_ENV,
    GUESS_ATOMIC,
    GUESS_ML,
    GUESS_MODEL_ENV,
    ML_ENABLED_ENV,
    GuessArm,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "scf" / "pulay_sample_periodic"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    target = root / "si-diamond"
    shutil.copytree(FIXTURE, target)

    payload = json.loads((target / "sample.json").read_text(encoding="utf-8"))
    source = REPO / payload["structure"]["source"]
    mirrored = root / payload["structure"]["source"]
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, mirrored)
    return root


# -- trajectory targets -------------------------------------------------


def test_the_final_target_is_the_converged_density(dataset: Path):
    sample = sample_from_directory(dataset / "si-diamond", root=dataset)

    assert sample is not None
    assert sample.artifacts.density_matrix_path.endswith("final/density.npy")
    assert sample.metadata["target_frame"] == TARGET_FINAL
    # Nothing is left to do from the converged density, by definition.
    assert sample.metadata["scf_cost"]["scf_steps_remaining"] == 0


def test_an_early_frame_target_selects_that_frame(dataset: Path):
    """P after one SCF iteration: a density CP2K actually produced.

    The converged density is a fixed point the SCF never passes through, which
    is why a regression toward it can be close in norm and useless as a start
    (docs/ml_density_failure_analysis.md).
    """

    sample = sample_from_directory(dataset / "si-diamond", root=dataset, target_frame=1)

    assert sample is not None
    assert sample.artifacts.density_matrix_path.endswith("iter_0001/density.npy")
    assert sample.metadata["scf_cost"]["scf_steps_remaining"] > 0


def test_the_cost_labels_are_measured_from_the_target_frame(dataset: Path):
    early = sample_from_directory(dataset / "si-diamond", root=dataset, target_frame=1)
    later = sample_from_directory(dataset / "si-diamond", root=dataset, target_frame=3)

    assert early and later
    remaining_early = early.metadata["scf_cost"]["scf_steps_remaining"]
    remaining_later = later.metadata["scf_cost"]["scf_steps_remaining"]
    assert remaining_later == remaining_early - 2

    # A converging SCF reduces its residual, so the rate is below one.
    assert 0.0 < early.metadata["scf_cost"]["residual_reduction_rate"] < 1.0


def test_a_frame_that_was_never_recorded_is_skipped_not_substituted(dataset: Path):
    """Half a set trained on one target and half on another fits neither."""

    assert sample_from_directory(dataset / "si-diamond", root=dataset, target_frame=99) is None


def test_the_whole_dataset_can_be_read_at_a_trajectory_target(dataset: Path):
    samples, tally = samples_from_dataset(dataset, target_frame=1)

    assert tally["n_samples"] == 1
    assert samples[0].metadata["target_frame"] == 1


# -- chemistry split ----------------------------------------------------


class FakeStructure:
    def __init__(self, numbers):
        self.atomic_numbers = tuple(numbers)


class FakeSample:
    def __init__(self, sample_id, numbers):
        self.sample_id = sample_id
        self.structure = FakeStructure(numbers)


def test_the_chemistry_key_reduces_the_formula():
    """H2O and (H2O)2 are the same chemistry; H2O and H2O2 are not."""

    water = FakeSample("a", (8, 1, 1))
    double = FakeSample("b", (8, 1, 1, 8, 1, 1))
    peroxide = FakeSample("c", (8, 8, 1, 1))

    assert chemistry_key(water) == chemistry_key(double)
    assert chemistry_key(water) != chemistry_key(peroxide)


def test_a_stretched_geometry_stays_with_its_family():
    """Different geometry, same formula: it must not cross the split."""

    relaxed = FakeSample("h2o", (8, 1, 1))
    stretched = FakeSample("h2o-x110", (8, 1, 1))

    assert chemistry_key(relaxed) == chemistry_key(stretched)


def test_the_split_never_puts_one_family_on_both_sides():
    samples = [FakeSample(f"h2o-{index}", (8, 1, 1)) for index in range(4)]
    samples += [FakeSample(f"nh3-{index}", (7, 1, 1, 1)) for index in range(3)]
    samples += [FakeSample(f"ch4-{index}", (6, 1, 1, 1, 1)) for index in range(3)]
    samples += [FakeSample("si", (14, 14))]

    train, holdout = split_by_chemistry(samples, holdout_fraction=0.5)

    train_families = {chemistry_key(item) for item in train}
    holdout_families = {chemistry_key(item) for item in holdout}
    assert train_families & holdout_families == set()
    assert train and holdout


def test_the_split_is_deterministic():
    samples = [FakeSample(f"s{index}", (index % 6 + 1,) * 2) for index in range(12)]

    first = split_by_chemistry(samples, seed=3)
    second = split_by_chemistry(samples, seed=3)

    assert [item.sample_id for item in first[0]] == [item.sample_id for item in second[0]]


def test_the_real_dataset_splits_without_leakage():
    root = Path("runtime/cp2k/pulay_dataset")
    if not root.is_dir():
        pytest.skip("the generated dataset is not present")

    samples, _ = samples_from_dataset(root, target_frame=1)
    train, holdout = split_by_chemistry(samples, holdout_fraction=0.3)

    assert train and holdout
    assert {chemistry_key(item) for item in train} & {chemistry_key(item) for item in holdout} == set()


# -- benchmark arms -----------------------------------------------------


def test_the_atomic_arm_carries_no_ml_environment():
    """The control must not be able to take the ML path."""

    assert GuessArm(label="atomic", guess=GUESS_ATOMIC).environment() == {}


def test_an_ml_arm_passes_its_model_and_alpha():
    arm = GuessArm(label="early-a0.5", guess=GUESS_ML, model="/models/early.ts.pt", alpha=0.5)

    environment = arm.environment()

    assert environment[GUESS_MODEL_ENV] == "/models/early.ts.pt"
    assert environment[ML_ENABLED_ENV] == "1"
    assert float(environment[GUESS_ALPHA_ENV]) == pytest.approx(0.5)


def test_an_ml_arm_without_a_model_gets_nothing():
    """Otherwise it would enable ML with nothing to run, i.e. a fallback."""

    assert GuessArm(label="broken", guess=GUESS_ML, model=None).environment() == {}


def test_alpha_one_is_the_pure_prediction():
    arm = GuessArm(label="final-a1", guess=GUESS_ML, model="/m.pt", alpha=1.0)

    assert float(arm.environment()[GUESS_ALPHA_ENV]) == 1.0
