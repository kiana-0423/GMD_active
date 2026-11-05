# Ported from PHAL (hotspot_al.cp2k_ml_scf.controller_dataset) at 96263b6; MIT License, see NOTICE.
"""Controller training data, and offline evaluation of a policy.

Two kinds of target come out of a recorded SCF, and they are not equally
trustworthy:

**Measured targets.** From iteration *n*, how many more iterations did this run
actually need, and did the residual stall or rise in the next few steps? These
are facts about the recording. A model can be trained on them, and its error can
be quantified.

**Heuristic action labels.** "The right action at iteration *n* was
``reset_diis``." This is a *guess*. Nothing in a recorded run says what would
have happened had the history been reset, because it wasn't. Training on these
labels distils whatever heuristic wrote them and cannot exceed it.

The distinction is kept in the data (``label_source``) rather than in a comment,
because it is the difference between a model that learned something and a model
that learned the labeller. Real action labels need counterfactual runs -- the
same system restarted from iteration *n* under each action, labelled by measured
steps-to-convergence -- which is a generation campaign, not a parsing exercise.

Replay is what makes a policy testable today: a recorded trajectory contains
every input the controller would have had live, so a policy can be run against
real SCFs and scored on whether it fires where trouble measurably was.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.scf.controller.policy import (
    ACTION_INCREASE_STABILITY,
    ACTION_KEEP,
    ACTION_RESET_DIIS,
    ACTION_SWITCH_MIXING,
    FEATURE_NAMES,
    FEATURE_VERSION,
    AdaptiveController,
    Policy,
    SCFState,
    feature_dict,
    states_from_trajectory,
)

DATASET_FORMAT = "phal-scf-controller-dataset-v1"

#: How far ahead a "did this stall?" label looks.
STALL_WINDOW = 3
#: Residual reduction over that window below which the SCF counts as stalled.
STALL_REDUCTION = 2.0

LABEL_MEASURED = "measured"
LABEL_HEURISTIC = "hindsight_heuristic"


@dataclass(frozen=True, slots=True)
class ControllerSample:
    """One SCF iteration as a training example."""

    sample_id: str
    iteration: int
    features: Mapping[str, float]
    #: Measured: iterations from here to convergence in the recorded run.
    steps_remaining: int | None
    #: Measured: did the residual fail to fall by STALL_REDUCTION over the
    #: next STALL_WINDOW iterations?
    stalled: bool
    #: Measured: did the residual rise on the very next iteration?
    rose_next: bool
    #: Guessed. See the module docstring before training on this.
    heuristic_action: str
    label_source: str = LABEL_HEURISTIC
    difficulty: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "iteration": self.iteration,
            "features": dict(self.features),
            "steps_remaining": self.steps_remaining,
            "stalled": self.stalled,
            "rose_next": self.rose_next,
            "heuristic_action": self.heuristic_action,
            "label_source": self.label_source,
            "difficulty": self.difficulty,
        }


def load_states(sample_dir: str | Path) -> tuple[list[SCFState], dict[str, Any]]:
    """Rebuild the controller's per-iteration view of one recorded sample."""

    directory = Path(sample_dir)
    trajectory = directory / "trajectory" if (directory / "trajectory").is_dir() else directory
    metadata_path = trajectory / "metadata.json"
    if not metadata_path.is_file():
        return [], {}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames: list[dict[str, Any]] = []
    for frame in metadata.get("frames", ()):
        label = "final" if frame.get("is_final") else f"iter_{int(frame['index']):04d}"
        detail_path = trajectory / label / "metadata.json"
        detail = dict(frame)
        if detail_path.is_file():
            try:
                detail.update(json.loads(detail_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        frames.append(detail)

    return states_from_trajectory(metadata, frames), metadata


def measured_targets(states: Sequence[SCFState]) -> list[dict[str, Any]]:
    """Facts about the recorded run, per iteration.

    ``steps_remaining`` is counted to the last recorded iteration, which for an
    accepted sample is convergence. For a run that never converged it is a lower
    bound, and is marked as such by the sample's difficulty rather than being
    silently treated as a completion.
    """

    residuals = [state.residual for state in states]
    total = len(states)
    targets: list[dict[str, Any]] = []

    for index, _ in enumerate(states):
        window = residuals[index : index + STALL_WINDOW + 1]
        reduction = (window[0] / window[-1]) if len(window) > 1 and window[-1] > 0 else float("inf")
        rose = bool(len(residuals) > index + 1 and residuals[index + 1] > residuals[index])
        targets.append(
            {
                "steps_remaining": total - 1 - index,
                "stalled": bool(reduction < STALL_REDUCTION),
                "rose_next": rose,
            }
        )
    return targets


def heuristic_action(state: SCFState, target: Mapping[str, Any]) -> str:
    """A guessed action label. Weak supervision -- read the module docstring.

    It is derived from what the run went on to do, so it is at least *aimed* at
    trouble: an iteration followed by a stall is labelled with an intervention,
    one followed by clean descent with ``keep``. What it cannot claim is that
    the named action would have helped.
    """

    if not target.get("stalled") and not target.get("rose_next"):
        return ACTION_KEEP

    features = feature_dict(state)
    if features["log_condition_number"] >= 12.0:
        return ACTION_RESET_DIIS
    if features["n_recent_increases"] >= 3:
        return ACTION_INCREASE_STABILITY
    if features["iterations_since_best"] >= 5:
        return ACTION_SWITCH_MIXING
    return ACTION_KEEP


def samples_from_sample_dir(
    sample_dir: str | Path, *, difficulty: str | None = None
) -> list[ControllerSample]:
    """Build controller training samples from one recorded SCF."""

    directory = Path(sample_dir)
    states, _ = load_states(directory)
    if not states:
        return []

    targets = measured_targets(states)
    return [
        ControllerSample(
            sample_id=directory.name,
            iteration=state.iteration,
            features=feature_dict(state),
            steps_remaining=int(target["steps_remaining"]),
            stalled=bool(target["stalled"]),
            rose_next=bool(target["rose_next"]),
            heuristic_action=heuristic_action(state, target),
            difficulty=difficulty,
        )
        for state, target in zip(states, targets)
    ]


def build_dataset(root: str | Path) -> list[ControllerSample]:
    """Every recorded SCF under ``root``, as controller samples."""

    from tools.scf.difficulty import assess_sample

    dataset_root = Path(root)
    samples: list[ControllerSample] = []
    for sample_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if not (sample_dir / "trajectory").is_dir():
            continue
        assessment = assess_sample(sample_dir)
        samples.extend(samples_from_sample_dir(sample_dir, difficulty=assessment.difficulty))
    return samples


def write_dataset(samples: Sequence[ControllerSample], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "dataset_format": DATASET_FORMAT,
                "feature_version": FEATURE_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "n_samples": len(samples),
                "targets": {
                    "measured": ["steps_remaining", "stalled", "rose_next"],
                    "heuristic": ["heuristic_action"],
                },
                "note": (
                    "steps_remaining/stalled/rose_next are facts about the recorded run. "
                    "heuristic_action is a guess: nothing in a recording says what a "
                    "different action would have done. Training on it distils the "
                    "labeller. Real action labels need counterfactual runs."
                ),
                "samples": [sample.to_dict() for sample in samples],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


# -- replay -------------------------------------------------------------


@dataclass
class ReplayResult:
    """What a policy would have done to one recorded SCF."""

    sample_id: str
    n_iterations: int
    decisions: list[dict[str, Any]] = field(default_factory=list)
    tally: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_actions(self) -> int:
        return sum(1 for item in self.decisions if item["action"] != ACTION_KEEP)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "n_iterations": self.n_iterations,
            "n_actions": self.n_actions,
            "decisions": list(self.decisions),
            "tally": dict(self.tally),
        }


def replay(policy: Policy, states: Sequence[SCFState], *, sample_id: str = "") -> ReplayResult:
    """Run a policy against a recorded SCF without touching CP2K.

    Replay cannot show that an action *would have helped* -- the recording only
    contains the path actually taken. What it does show, and what it is for, is
    where a policy fires, how often, and whether that coincides with the
    iterations where the run measurably got into trouble.
    """

    controller = AdaptiveController(policy)
    decisions: list[dict[str, Any]] = []
    for state in states:
        decision = controller.decide(state)
        decisions.append({"iteration": state.iteration, **decision.to_dict()})
    return ReplayResult(
        sample_id=sample_id,
        n_iterations=len(states),
        decisions=decisions,
        tally=controller.tally.to_dict(),
    )


def evaluate_policy(policy: Policy, root: str | Path) -> dict[str, Any]:
    """Score a policy against every recorded SCF under ``root``.

    The score is agreement with *measured* trouble: an iteration is "in
    trouble" if the residual stalled or rose next. Precision is how often the
    policy fired into real trouble; recall is how much of the trouble it
    noticed.

    Neither is a claim about acceleration. A policy could score perfectly and
    still slow every run down, because firing at the right moment is necessary
    and not sufficient. Acceleration can only be measured by running CP2K under
    the controller, which needs the actions actually wired into the SCF loop.
    """

    dataset_root = Path(root)
    from tools.scf.difficulty import assess_sample

    fired_in_trouble = 0
    fired_total = 0
    trouble_total = 0
    by_difficulty: dict[str, dict[str, int]] = {}
    replays: list[ReplayResult] = []

    for sample_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if not (sample_dir / "trajectory").is_dir():
            continue
        states, _ = load_states(sample_dir)
        if not states:
            continue
        difficulty = assess_sample(sample_dir).difficulty
        targets = measured_targets(states)
        result = replay(policy, states, sample_id=sample_dir.name)
        replays.append(result)

        bucket = by_difficulty.setdefault(difficulty, {"samples": 0, "actions": 0, "iterations": 0})
        bucket["samples"] += 1
        bucket["actions"] += result.n_actions
        bucket["iterations"] += result.n_iterations

        for decision, target in zip(result.decisions, targets):
            in_trouble = bool(target["stalled"] or target["rose_next"])
            trouble_total += int(in_trouble)
            if decision["action"] != ACTION_KEEP:
                fired_total += 1
                fired_in_trouble += int(in_trouble)

    return {
        "n_samples": len(replays),
        "n_iterations": sum(item.n_iterations for item in replays),
        "n_actions": fired_total,
        "n_troubled_iterations": trouble_total,
        "precision": (fired_in_trouble / fired_total) if fired_total else None,
        "recall": (fired_in_trouble / trouble_total) if trouble_total else None,
        "by_difficulty": by_difficulty,
        "action_counts": _action_counts(replays),
        "note": (
            "Agreement with measured trouble, not acceleration. Firing at the right "
            "iteration is necessary and not sufficient; whether an action helps can only "
            "be measured by running CP2K under the controller."
        ),
        "replays": [item.to_dict() for item in replays],
    }


def _action_counts(replays: Sequence[ReplayResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in replays:
        for decision in result.decisions:
            action = decision["action"]
            if action == ACTION_KEEP:
                continue
            counts[action] = counts.get(action, 0) + 1
    return dict(sorted(counts.items()))


def dataset_summary(samples: Sequence[ControllerSample]) -> dict[str, Any]:
    """Class balance and target spread, for the report."""

    actions: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    for sample in samples:
        actions[sample.heuristic_action] = actions.get(sample.heuristic_action, 0) + 1
        key = sample.difficulty or "unknown"
        difficulties[key] = difficulties.get(key, 0) + 1

    remaining = [s.steps_remaining for s in samples if s.steps_remaining is not None]
    return {
        "n_samples": len(samples),
        "n_systems": len({sample.sample_id for sample in samples}),
        "heuristic_action_counts": dict(sorted(actions.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "n_stalled": sum(1 for sample in samples if sample.stalled),
        "n_rose_next": sum(1 for sample in samples if sample.rose_next),
        "steps_remaining": {
            "min": int(np.min(remaining)) if remaining else None,
            "max": int(np.max(remaining)) if remaining else None,
            "mean": float(np.mean(remaining)) if remaining else None,
        },
        "feature_version": FEATURE_VERSION,
    }
