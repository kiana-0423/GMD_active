"""The deterministic retry ladder (``scf.retry``) -- PHAL's one positive, model-free SCF result."""

from __future__ import annotations

import pytest

from gmd_active_learning.adapters.dft.scf.strategy import (
    RetryLadder,
    documented_strategy_dataset,
    retry_ladder,
)
from gmd_active_learning.core.exceptions import ConfigurationError

INPUT = """&FORCE_EVAL
  &DFT
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 60
    &END SCF
  &END DFT
&END FORCE_EVAL
"""


def test_the_documented_ladder_numbers_are_reproduced() -> None:
    """docs/scf/scf_strategy_prediction.md: default 9/12, 556 steps; default -> nbuffer4 12/12, 582."""

    dataset = documented_strategy_dataset()

    default = retry_ladder(dataset, ["default"])
    ladder = retry_ladder(dataset, ["default", "nbuffer4"])
    longer = retry_ladder(dataset, ["default", "nbuffer4", "broyden"])

    assert (default["n_converged"], default["total_steps_including_failures"]) == (9, 556)
    assert default["total_steps"] == 196
    assert (ladder["n_converged"], ladder["total_steps_including_failures"]) == (12, 582)
    assert ladder["total_steps_including_failures"] / default["total_steps_including_failures"] == pytest.approx(
        1.047, abs=1e-3
    )
    assert (longer["n_converged"], longer["total_steps_including_failures"]) == (12, 582)
    rescued = {item["system"] for item in ladder["per_system"] if item["resolved_with"] == "nbuffer4"}
    assert rescued == {"lih-rocksalt", "mgo-rocksalt", "nacl-rocksalt"}


def test_a_disabled_ladder_changes_nothing() -> None:
    ladder = RetryLadder({"scf": {}})

    assert ladder.enabled is False
    assert ladder.strategy_for_attempt(0) is None
    assert ladder.apply(INPUT, 0) == INPUT


def test_the_enabled_ladder_walks_default_then_nbuffer4() -> None:
    ladder = RetryLadder({"scf": {"retry": {"enabled": True}}})

    first, second = ladder.apply(INPUT, 0), ladder.apply(INPUT, 1)

    assert "NBUFFER 8" in first and "NBUFFER 4" in second
    assert "SCF_GUESS ATOMIC" in first and "SCF_GUESS ATOMIC" in second  # the guess axis is untouched
    assert "MAX_SCF 120" in first
    assert ladder.has_next(0) and not ladder.has_next(1)
    assert ladder.strategy_for_attempt(2) is None


def test_the_ladder_base_and_order_come_from_config() -> None:
    ladder = RetryLadder(
        {"scf": {"retry": {"enabled": True, "order": ["nbuffer4", "broyden"], "base": {"nbuffer": 6, "max_scf": 80}}}}
    )

    assert [strategy.name for strategy in ladder.strategies] == ["nbuffer4", "broyden"]
    assert ladder.strategies[0].nbuffer == 4
    assert ladder.strategies[1].nbuffer == 6 and ladder.strategies[1].max_scf == 80
    assert ladder.evaluate(documented_strategy_dataset())["order"] == ["nbuffer4", "broyden"]


def test_an_unknown_rung_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown strategies"):
        RetryLadder({"scf": {"retry": {"enabled": True, "order": ["default", "magic"]}}})
