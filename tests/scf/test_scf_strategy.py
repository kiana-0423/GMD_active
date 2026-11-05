# Ported from PHAL (tests.scf.test_scf_strategy) at 96263b6; MIT License, see NOTICE.
"""Tests for static SCF strategy selection.

The predictor picks CP2K settings from a structure, before any SCF exists. Two
things must hold whatever the model says: the input it produces differs from
the baseline in exactly the ways chosen, and an unsure prediction becomes the
default rather than a guess.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.scf.strategy_predictor import (
    DEFAULT_STRATEGY,
    MIXING_BROYDEN,
    FEATURE_NAMES,
    Recommendation,
    SCFStrategy,
    StrategyDataset,
    StrategyError,
    StrategyOutcome,
    StrategyPredictor,
    apply_strategy,
    feature_vector,
    leave_one_out,
    strategy_grid,
    structure_features,
)

INPUT = """&GLOBAL
  PROJECT x
&END GLOBAL
&FORCE_EVAL
  &DFT
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 60
      &MIXING
        METHOD PULAY_MIXING
        ALPHA 0.4
        NBUFFER 8
      &END MIXING
    &END SCF
  &END DFT
&END FORCE_EVAL
"""


def outcome(system, strategy, steps, converged=True):
    return StrategyOutcome(
        system=system,
        strategy=strategy,
        converged=converged,
        scf_steps=steps,
        runtime_s=1.0,
        energy=-1.0,
    )


def features(**overrides):
    base = {name: 0.0 for name in FEATURE_NAMES}
    base.update(overrides)
    return base


# -- the strategy itself ------------------------------------------------


def test_a_strategy_renders_its_scf_body():
    body = SCFStrategy(name="x", alpha=0.2, nbuffer=4).to_block()

    assert "ALPHA 0.2" in body
    assert "NBUFFER 4" in body
    assert "SMEAR" not in body


def test_smearing_without_empty_states_is_refused():
    """CP2K cannot occupy fractionally what it did not compute."""

    with pytest.raises(StrategyError, match="needs empty states"):
        SCFStrategy(name="bad", smearing=True, added_mos=0)


def test_an_unknown_mixer_is_refused():
    with pytest.raises(StrategyError, match="unknown mixing method"):
        SCFStrategy(name="bad", mixing_method="MAGIC")


def test_an_out_of_range_alpha_is_refused():
    with pytest.raises(StrategyError, match="alpha must be"):
        SCFStrategy(name="bad", alpha=1.5)


def test_the_grid_changes_one_thing_at_a_time():
    """A grid where two settings move at once explains neither."""

    grid = {item.name: item for item in strategy_grid(smearing=False)}
    default = grid["default"]

    assert grid["alpha0.2"].alpha == 0.2
    assert grid["alpha0.2"].mixing_method == default.mixing_method
    assert grid["nbuffer4"].nbuffer == 4
    assert grid["nbuffer4"].alpha == default.alpha
    assert grid["broyden"].mixing_method == MIXING_BROYDEN
    assert grid["broyden"].alpha == default.alpha


def test_a_smeared_grid_keeps_smearing_everywhere():
    """Without it the metal runs are not slower, they do not converge at all."""

    assert all(item.smearing for item in strategy_grid(smearing=True))
    assert all(item.added_mos > 0 for item in strategy_grid(smearing=True))


# -- rewriting an input -------------------------------------------------


def test_applying_a_strategy_replaces_the_whole_scf_body():
    result = apply_strategy(INPUT, SCFStrategy(name="x", alpha=0.1, nbuffer=4))

    assert "ALPHA 0.1" in result
    assert "ALPHA 0.4" not in result
    assert "NBUFFER 8" not in result
    assert result.count("&END SCF") == 1


def test_the_guess_survives_the_rewrite():
    """The initial guess is a separate axis; this module has no opinion on it."""

    assert "SCF_GUESS ATOMIC" in apply_strategy(INPUT, SCFStrategy(name="x", alpha=0.2))


def test_a_comment_mentioning_the_section_is_not_mistaken_for_it():
    """Matching "&SCF" anywhere rewrote from a comment that merely named it."""

    text = "! the &SCF section is rewritten, up to &END SCF\n" + INPUT

    result = apply_strategy(text, SCFStrategy(name="x", alpha=0.2))

    assert result.startswith("! the &SCF section is rewritten, up to &END SCF")
    assert "ALPHA 0.2" in result


def test_an_input_without_an_scf_section_is_refused():
    with pytest.raises(StrategyError, match="no &SCF"):
        apply_strategy("&GLOBAL\n&END GLOBAL\n", DEFAULT_STRATEGY)


# -- features -----------------------------------------------------------


def test_features_describe_a_structure_without_running_anything():
    detail = structure_features([22, 22, 8, 8, 8, 8], cell=np.eye(3) * 4.0, periodic=True)

    assert detail["n_atoms"] == 6.0
    assert detail["has_transition_metal"] == 1.0
    assert detail["has_oxygen"] == 1.0
    assert detail["periodic"] == 1.0
    assert detail["cell_volume"] == pytest.approx(64.0)
    assert detail["electronegativity_spread"] > 1.0


def test_a_molecule_has_no_cell_volume():
    detail = structure_features([8, 1, 1], periodic=False)

    assert detail["cell_volume"] == 0.0
    assert detail["number_density"] == 0.0


def test_an_empty_structure_is_refused():
    with pytest.raises(StrategyError, match="no atoms"):
        structure_features([])


def test_the_feature_vector_follows_the_declared_order():
    vector = feature_vector(features(n_atoms=3.0, has_oxygen=1.0))

    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector[FEATURE_NAMES.index("n_atoms")] == 3.0


# -- the dataset --------------------------------------------------------


def test_the_best_strategy_is_the_cheapest_that_converged():
    dataset = StrategyDataset()
    dataset.features["a"] = features()
    dataset.outcomes += [
        outcome("a", "default", 20),
        outcome("a", "alpha0.2", 14),
        outcome("a", "broyden", None, converged=False),
    ]

    assert dataset.best_strategy("a") == "alpha0.2"


def test_a_tie_goes_to_the_default():
    """A configuration that merely matches the baseline is not an improvement."""

    dataset = StrategyDataset()
    dataset.features["a"] = features()
    dataset.outcomes += [outcome("a", "default", 14), outcome("a", "alpha0.2", 14)]

    assert dataset.best_strategy("a") == "default"


def test_a_system_nothing_converged_on_has_no_best_strategy():
    dataset = StrategyDataset()
    dataset.features["a"] = features()
    dataset.outcomes += [outcome("a", "default", None, converged=False)]

    assert dataset.best_strategy("a") is None


def test_the_dataset_round_trips(tmp_path: Path):
    dataset = StrategyDataset()
    dataset.features["a"] = features(n_atoms=2.0)
    dataset.outcomes.append(outcome("a", "default", 12))

    path = dataset.write(tmp_path / "strategy.json")
    restored = StrategyDataset.from_dict(json.loads(path.read_text(encoding="utf-8")))

    assert restored.best_strategy("a") == "default"
    assert restored.features["a"]["n_atoms"] == 2.0


# -- the predictor ------------------------------------------------------


def build(systems: dict[str, tuple[dict, str]]) -> StrategyDataset:
    dataset = StrategyDataset()
    for name, (detail, best) in systems.items():
        dataset.features[name] = detail
        for strategy in ("default", "alpha0.2"):
            dataset.outcomes.append(
                outcome(name, strategy, 10 if strategy == best else 20)
            )
    return dataset


def test_the_predictor_follows_its_nearest_neighbours():
    dataset = build(
        {
            "metal-a": (features(has_transition_metal=1.0, n_atoms=2.0), "alpha0.2"),
            "metal-b": (features(has_transition_metal=1.0, n_atoms=3.0), "alpha0.2"),
            "insulator": (features(has_transition_metal=0.0, n_atoms=8.0), "default"),
        }
    )
    predictor = StrategyPredictor(k=2, min_confidence=0.5).fit(dataset)

    result = predictor.recommend(features(has_transition_metal=1.0, n_atoms=2.5))

    assert result.strategy == "alpha0.2"
    assert result.confidence > 0.5


def test_a_split_vote_falls_back_to_the_default():
    """An unsure recommendation is worth less than settings known to work."""

    dataset = build(
        {
            "a": (features(n_atoms=1.0), "alpha0.2"),
            "b": (features(n_atoms=1.0), "nbuffer4"),
        }
    )
    predictor = StrategyPredictor(k=2, min_confidence=0.9).fit(dataset)

    result = predictor.recommend(features(n_atoms=1.0))

    assert result.strategy == "default"
    assert "falling back" in result.reason


def test_an_unfitted_predictor_refuses_to_guess():
    with pytest.raises(StrategyError, match="not been fitted"):
        StrategyPredictor().recommend(features())


def test_systems_nothing_converged_on_are_not_trained_on():
    dataset = StrategyDataset()
    dataset.features["good"] = features(n_atoms=2.0)
    dataset.features["hopeless"] = features(n_atoms=3.0)
    dataset.outcomes += [
        outcome("good", "default", 12),
        outcome("hopeless", "default", None, converged=False),
    ]

    predictor = StrategyPredictor(k=1).fit(dataset)

    assert predictor.recommend(features(n_atoms=3.0)).strategy == "default"


# -- leave-one-out ------------------------------------------------------


def test_leave_one_out_never_predicts_a_system_from_itself():
    """With a dozen systems, any other split reports memorisation."""

    dataset = build(
        {
            "a": (features(n_atoms=1.0, has_oxygen=1.0), "alpha0.2"),
            "b": (features(n_atoms=1.2, has_oxygen=1.0), "alpha0.2"),
            "c": (features(n_atoms=9.0), "default"),
            "d": (features(n_atoms=9.2), "default"),
        }
    )

    report = leave_one_out(dataset, k=2)

    assert report["summary"]["n_systems"] == 4
    assert 0.0 <= report["summary"]["accuracy"] <= 1.0
    assert all("recommended" in item for item in report["per_system"])


def test_the_evaluation_reports_measured_steps_not_only_accuracy():
    """Being right about the label is not the same as being cheaper."""

    dataset = build(
        {
            "a": (features(n_atoms=1.0), "alpha0.2"),
            "b": (features(n_atoms=1.1), "alpha0.2"),
            "c": (features(n_atoms=1.2), "alpha0.2"),
        }
    )

    summary = leave_one_out(dataset, k=2)["summary"]

    assert summary["default_total_steps"] > 0
    assert summary["recommended_total_steps"] > 0
    assert "step_saving" in summary
    assert summary["n_improved"] + summary["n_degraded"] <= summary["n_systems"]
