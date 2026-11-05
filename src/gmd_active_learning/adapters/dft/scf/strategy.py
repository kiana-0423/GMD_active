# Ported from PHAL (hotspot_al.cp2k_ml_scf.scf_strategy: SCFStrategy, strategy_grid, apply_strategy,
# StrategyOutcome, StrategyDataset, retry_ladder) at 96263b6; MIT License, see NOTICE.
"""Static SCF strategies and the deterministic retry ladder (runtime part).

The one *positive*, model-free result of PHAL's SCF work lives here: trying
``default -> nbuffer4`` in order converged 12 of 12 solids against 9 of 12 for
the default alone, for 4.7% more SCF iterations (docs/scf/scf_strategy_prediction.md).
:class:`RetryLadder` turns that into a labeling-path component driven by
``scf.retry`` (off by default). Caveat preserved from the ledger: PHAL's
"default" uses ``NBUFFER 8``; CP2K v2026.2's own default is 4, so the
``nbuffer4`` rung returns to upstream's default.

The k-NN strategy *predictor* (a documented negative result) is tooling:
:mod:`tools.scf.strategy_predictor`.

PHAL's module docstring follows.

Choose the SCF settings *before* the calculation starts.

Three ML-inside-the-SCF ideas were measured and none of them paid:

* predicting the density -- no acceleration at any mixing weight, because the
  payoff curve is flat until a prediction is nearly exact
  (``docs/ml_density_v2.md``);
* predicting DIIS coefficients -- native DIIS already solves that minimisation
  exactly (``docs/ml_pulay_objective.md``);
* deciding strategy mid-run -- twelve of twelve counterfactual labels say the
  optimal action is to do nothing (``docs/adaptive_scf_controller.md``).

The same experiments turned up one thing that *does* move: the settings chosen
before any SCF state exists. Measured on transition-metal oxides, changing
``ALPHA`` from 0.4 to 0.2 takes TiO2 from 19 SCF steps to 14 and NiO from 31 to
65. A factor of two, in both directions, decided by one number picked from the
structure.

That makes this a static prediction problem, and unlike density prediction it
has no accuracy threshold: recommending 0.2 rather than 0.4 is a discrete
choice that is either right or wrong for a system, and being *approximately*
right still lands on one of them.

CP2K remains the authority. This module writes an input; nothing here inspects
or overrides a result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

STRATEGY_FORMAT = "phal-scf-strategy-v1"

MIXING_PULAY = "PULAY_MIXING"
MIXING_BROYDEN = "BROYDEN_MIXING"
MIXING_DIRECT = "DIRECT_P_MIXING"
KNOWN_MIXERS = (MIXING_PULAY, MIXING_BROYDEN, MIXING_DIRECT)


class StrategyError(RuntimeError):
    """Raised when a strategy cannot be built or applied."""


@dataclass(frozen=True, slots=True)
class SCFStrategy:
    """One set of SCF settings, and how to write it into a CP2K input."""

    name: str
    mixing_method: str = MIXING_PULAY
    alpha: float = 0.4
    nbuffer: int = 8
    #: Smearing and empty states travel together: CP2K rejects one without the
    #: other, and a metal needs both or does not converge at all.
    smearing: bool = False
    added_mos: int = 0
    electronic_temperature: float = 500.0
    max_scf: int = 120
    eps_scf: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.mixing_method not in KNOWN_MIXERS:
            raise StrategyError(
                f"unknown mixing method {self.mixing_method!r}; known: {list(KNOWN_MIXERS)}"
            )
        if not 0.0 < self.alpha <= 1.0:
            raise StrategyError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.nbuffer < 1:
            raise StrategyError(f"nbuffer must be positive, got {self.nbuffer}")
        if self.smearing and self.added_mos <= 0:
            raise StrategyError(
                "smearing needs empty states; CP2K cannot occupy fractionally what it "
                "did not compute"
            )

    def to_block(self) -> str:
        """The ``&SCF`` body this strategy corresponds to."""

        lines = [
            f"      EPS_SCF {self.eps_scf:.1E}",
            f"      MAX_SCF {self.max_scf}",
        ]
        if self.smearing:
            lines += [
                f"      ADDED_MOS {self.added_mos}",
                "      &SMEAR ON",
                "        METHOD FERMI_DIRAC",
                f"        ELECTRONIC_TEMPERATURE {self.electronic_temperature:.0f}",
                "      &END SMEAR",
            ]
        lines += [
            "      &MIXING",
            f"        METHOD {self.mixing_method}",
            f"        ALPHA {self.alpha}",
            f"        NBUFFER {self.nbuffer}",
            "      &END MIXING",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mixing_method": self.mixing_method,
            "alpha": self.alpha,
            "nbuffer": self.nbuffer,
            "smearing": self.smearing,
            "added_mos": self.added_mos,
            "electronic_temperature": self.electronic_temperature,
            "max_scf": self.max_scf,
            "eps_scf": self.eps_scf,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SCFStrategy":
        known = {field_name for field_name in cls.__slots__}
        return cls(**{key: value for key, value in payload.items() if key in known})


#: CP2K's own defaults for this template, and the baseline every comparison is
#: against. Recommending it is always allowed; it is never a regression.
DEFAULT_STRATEGY = SCFStrategy(name="default")
DEFAULT_SMEARED = SCFStrategy(name="default", smearing=True, added_mos=80)


def strategy_grid(*, smearing: bool, added_mos: int = 80) -> list[SCFStrategy]:
    """The configurations to try for one system.

    Deliberately small and interpretable. Each entry changes one thing from the
    default, so a result attributes to that thing: a grid where two settings
    move at once measures their combination and explains neither.
    """

    def make(name: str, **changes: Any) -> SCFStrategy:
        base = DEFAULT_SMEARED if smearing else DEFAULT_STRATEGY
        base = replace(base, added_mos=added_mos) if smearing else base
        return replace(base, name=name, **changes)

    return [
        make("default"),
        make("alpha0.2", alpha=0.2),
        make("alpha0.1", alpha=0.1),
        make("nbuffer4", nbuffer=4),
        make("broyden", mixing_method=MIXING_BROYDEN),
        make("broyden-alpha0.2", mixing_method=MIXING_BROYDEN, alpha=0.2),
    ]


# -- rendering ----------------------------------------------------------

#: The section header must be the whole line. Matching "&SCF" anywhere caught
#: a comment that merely mentioned it, and rewrote from there -- an input whose
#: documentation names a section is not an input with two of them.
_SCF_BODY = re.compile(r"(^[ \t]*&SCF[ \t]*$\n)(.*?)(^[ \t]*&END SCF)", re.DOTALL | re.MULTILINE)


def apply_strategy(text: str, strategy: SCFStrategy, *, keep_guess: bool = True) -> str:
    """Rewrite the ``&SCF`` section of a CP2K input to match ``strategy``.

    The section is replaced wholesale rather than patched line by line: a
    partial edit leaves whatever the template happened to contain, and two
    inputs that differ in ways nobody chose are not a controlled comparison.

    ``SCF_GUESS`` is carried across, because the guess is a separate axis and
    this module has no opinion about it.
    """

    match = _SCF_BODY.search(text)
    if not match:
        raise StrategyError("the input has no &SCF ... &END SCF section to rewrite")

    body = strategy.to_block()
    if keep_guess:
        guess = re.search(r"^\s*SCF_GUESS\s+(\S+)", match.group(2), re.MULTILINE)
        if guess:
            body = f"      SCF_GUESS {guess.group(1)}\n" + body

    return text[: match.start(2)] + body + "\n" + text[match.end(2) :]


# -- features -----------------------------------------------------------

FEATURE_NAMES: tuple[str, ...] = (
    "n_atoms",
    "log_n_atoms",
    "periodic",
    "cell_volume",
    "number_density",
    "min_distance",
    "mean_atomic_number",
    "max_atomic_number",
    "n_elements",
    "has_transition_metal",
    "has_oxygen",
    "electronegativity_spread",
    "valence_electrons",
    "charge",
    "multiplicity",
)

# -- the predictor ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyOutcome:
    """What one configuration did on one system."""

    system: str
    strategy: str
    converged: bool
    scf_steps: int | None
    runtime_s: float | None
    energy: float | None
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "strategy": self.strategy,
            "converged": self.converged,
            "scf_steps": self.scf_steps,
            "runtime_s": self.runtime_s,
            "energy": self.energy,
            "failure": self.failure,
        }


@dataclass
class StrategyDataset:
    """Systems, their features, and how every configuration fared on them."""

    features: dict[str, dict[str, float]] = field(default_factory=dict)
    outcomes: list[StrategyOutcome] = field(default_factory=list)

    def systems(self) -> list[str]:
        return sorted(self.features)

    def by_system(self, system: str) -> dict[str, StrategyOutcome]:
        return {item.strategy: item for item in self.outcomes if item.system == system}

    def best_strategy(self, system: str) -> str | None:
        """Fewest SCF steps among the configurations that converged.

        Ties go to ``default``: a configuration that merely matches the
        baseline is not an improvement, and preferring it would inflate every
        later accuracy number with choices that changed nothing.
        """

        results = [
            item
            for item in self.outcomes
            if item.system == system and item.converged and item.scf_steps is not None
        ]
        if not results:
            return None
        best = min(item.scf_steps for item in results)
        winners = {item.strategy for item in results if item.scf_steps == best}
        return "default" if "default" in winners else sorted(winners)[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_format": STRATEGY_FORMAT,
            "feature_names": list(FEATURE_NAMES),
            "systems": {
                system: {
                    "features": self.features[system],
                    "best_strategy": self.best_strategy(system),
                    "outcomes": [item.to_dict() for item in self.by_system(system).values()],
                }
                for system in self.systems()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StrategyDataset":
        dataset = cls()
        for system, entry in payload.get("systems", {}).items():
            dataset.features[system] = dict(entry["features"])
            for outcome in entry.get("outcomes", []):
                dataset.outcomes.append(StrategyOutcome(**outcome))
        return dataset

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", "utf-8")
        return target


def retry_ladder(dataset: StrategyDataset, order: Sequence[str], *, failure_cost: int = 120) -> dict[str, Any]:
    """What a fixed retry order costs, computed from the recorded outcomes.

    The alternative to predicting a configuration is trying them in a fixed
    order until one converges. It needs no model and no features, so it is the
    baseline a predictor has to beat -- and on a dataset where the label is not
    predictable, it is also the better product.

    A failed attempt is charged ``failure_cost`` iterations, which is what
    ``MAX_SCF`` actually costs before CP2K gives up. Charging nothing for a
    failure would make the ladder look free.
    """

    records = []
    for system in dataset.systems():
        outcomes = dataset.by_system(system)
        spent = 0
        resolved = None
        for name in order:
            item = outcomes.get(name)
            if item is None:
                continue
            if item.converged and item.scf_steps is not None:
                spent += item.scf_steps
                resolved = name
                break
            spent += failure_cost
        records.append({"system": system, "resolved_with": resolved, "steps": spent})

    converged = [item for item in records if item["resolved_with"]]
    return {
        "order": list(order),
        "failure_cost": failure_cost,
        "n_systems": len(records),
        "n_converged": len(converged),
        "total_steps": sum(item["steps"] for item in converged),
        "total_steps_including_failures": sum(item["steps"] for item in records),
        "per_system": records,
    }


class RetryLadder:
    """Deterministic SCF retry ladder for labeling runs (``scf.retry``).

    ``strategy_for_attempt(0)`` is the first rung, ``(1)`` the second, ...;
    ``None`` once the ladder is exhausted. :meth:`apply` rewrites a CP2K input's
    ``&SCF`` body for an attempt. Disabled ladders return the unchanged input
    and no further attempts, so the caller's behaviour is unchanged.
    """

    def __init__(self, config: Any = None) -> None:
        from gmd_active_learning.adapters.dft.scf.config import scf_config

        settings = scf_config(config).retry
        self.enabled = settings.enabled
        self.failure_cost = settings.failure_cost
        base = settings.base
        base_strategy = SCFStrategy(
            name="default",
            mixing_method=base.mixing_method,
            alpha=base.alpha,
            nbuffer=base.nbuffer,
            smearing=base.smearing,
            added_mos=base.added_mos,
            electronic_temperature=base.electronic_temperature,
            max_scf=base.max_scf,
            eps_scf=base.eps_scf,
        )
        grid = {item.name: item for item in _grid_from(base_strategy)}
        unknown = [name for name in settings.order if name not in grid]
        if unknown:
            from gmd_active_learning.core.exceptions import ConfigurationError

            raise ConfigurationError(f"scf.retry.order names unknown strategies {unknown}; known: {sorted(grid)}")
        self.order = tuple(settings.order)
        self.strategies = tuple(grid[name] for name in self.order)

    def strategy_for_attempt(self, attempt: int) -> SCFStrategy | None:
        if not self.enabled:
            return None
        return self.strategies[attempt] if 0 <= attempt < len(self.strategies) else None

    def has_next(self, attempt: int) -> bool:
        return self.strategy_for_attempt(attempt + 1) is not None

    def apply(self, input_text: str, attempt: int) -> str:
        strategy = self.strategy_for_attempt(attempt)
        return input_text if strategy is None else apply_strategy(input_text, strategy)

    def evaluate(self, dataset: StrategyDataset) -> dict[str, Any]:
        """Replay this ladder over recorded outcomes (see :func:`retry_ladder`)."""

        return retry_ladder(dataset, self.order, failure_cost=self.failure_cost)


def _grid_from(base: SCFStrategy) -> list[SCFStrategy]:
    return [
        replace(base, name="default"),
        replace(base, name="alpha0.2", alpha=0.2),
        replace(base, name="alpha0.1", alpha=0.1),
        replace(base, name="nbuffer4", nbuffer=4),
        replace(base, name="broyden", mixing_method=MIXING_BROYDEN),
        replace(base, name="broyden-alpha0.2", mixing_method=MIXING_BROYDEN, alpha=0.2),
    ]


#: PHAL's measured 12-solid strategy table (docs/scf/scf_strategy_prediction.md);
#: ``None`` is a failed run. The acceptance data for :func:`retry_ladder`.
DOCUMENTED_STRATEGY_TABLE: dict[str, dict[str, int | None]] = {
    "aln-wurtzite": {"default": 10, "alpha0.2": 10, "alpha0.1": 11, "nbuffer4": 11, "broyden": 11, "broyden-alpha0.2": 14},
    "c-diamond": {"default": 7, "alpha0.2": 7, "alpha0.1": 7, "nbuffer4": 7, "broyden": 9, "broyden-alpha0.2": 9},
    "cu2o-cuprite": {"default": 19, "alpha0.2": 26, "alpha0.1": 22, "nbuffer4": 86, "broyden": 19, "broyden-alpha0.2": 19},
    "lih-rocksalt": {"default": None, "alpha0.2": None, "alpha0.1": None, "nbuffer4": 7, "broyden": 10, "broyden-alpha0.2": 13},
    "mgo-rocksalt": {"default": None, "alpha0.2": None, "alpha0.1": None, "nbuffer4": 10, "broyden": 13, "broyden-alpha0.2": 14},
    "mno-rocksalt": {"default": 71, "alpha0.2": 30, "alpha0.1": 118, "nbuffer4": 59, "broyden": 65, "broyden-alpha0.2": 39},
    "nacl-rocksalt": {"default": None, "alpha0.2": None, "alpha0.1": None, "nbuffer4": 9, "broyden": 12, "broyden-alpha0.2": 14},
    "nio-rocksalt": {"default": 33, "alpha0.2": 86, "alpha0.1": 66, "nbuffer4": None, "broyden": 39, "broyden-alpha0.2": 51},
    "si-diamond": {"default": 8, "alpha0.2": 8, "alpha0.1": 8, "nbuffer4": 8, "broyden": 8, "broyden-alpha0.2": 8},
    "si-diamond-strained": {"default": 8, "alpha0.2": 8, "alpha0.1": 8, "nbuffer4": 8, "broyden": 8, "broyden-alpha0.2": 8},
    "tio2-rutile": {"default": 19, "alpha0.2": 14, "alpha0.1": 14, "nbuffer4": 18, "broyden": 16, "broyden-alpha0.2": 14},
    "zno-wurtzite": {"default": 21, "alpha0.2": 22, "alpha0.1": 21, "nbuffer4": None, "broyden": 19, "broyden-alpha0.2": 21},
}


def documented_strategy_dataset() -> StrategyDataset:
    """:data:`DOCUMENTED_STRATEGY_TABLE` as a :class:`StrategyDataset` (no features recorded)."""

    dataset = StrategyDataset()
    for system, row in DOCUMENTED_STRATEGY_TABLE.items():
        dataset.features[system] = {}
        for strategy, steps in row.items():
            dataset.outcomes.append(
                StrategyOutcome(
                    system=system,
                    strategy=strategy,
                    converged=steps is not None,
                    scf_steps=steps,
                    runtime_s=None,
                    energy=None,
                    failure=None if steps is not None else "not converged",
                )
            )
    return dataset


__all__ = [
    "DEFAULT_SMEARED",
    "DEFAULT_STRATEGY",
    "DOCUMENTED_STRATEGY_TABLE",
    "FEATURE_NAMES",
    "KNOWN_MIXERS",
    "MIXING_BROYDEN",
    "MIXING_DIRECT",
    "MIXING_PULAY",
    "STRATEGY_FORMAT",
    "RetryLadder",
    "SCFStrategy",
    "StrategyDataset",
    "StrategyError",
    "StrategyOutcome",
    "apply_strategy",
    "documented_strategy_dataset",
    "retry_ladder",
    "strategy_grid",
]
