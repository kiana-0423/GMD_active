# Ported from PHAL (hotspot_al.cp2k_ml_scf.difficulty) at 96263b6; MIT License, see NOTICE.
"""Classify how hard each SCF in a dataset was.

Speed is the obvious metric and the less useful one. A dataset of systems that
all converge in twelve steps says nothing about the case that matters -- the
run that oscillates for forty iterations, or the one CP2K abandons. Robustness
is what an adaptive controller is for, so the dataset has to be able to say
which samples are hard and *why*.

Three classes, from explicit thresholds rather than quantiles, so the label of
one sample does not change when another is added:

``easy``
    converged quickly and monotonically.
``medium``
    converged, but slowly or non-monotonically.
``hard``
    did not converge, or converged only after a long struggle.

Conditioning is reported **separately**, as ``numerical_risk``, not folded into
the difficulty label. Si in diamond converges in eight monotonic steps on a
``B`` whose condition number reaches 10^14.4: it is an easy SCF running on a
numerically dangerous solve, and collapsing those two facts into one word loses
the more interesting half. Difficulty answers "did this converge, and how
painfully"; numerical risk answers "how close was the linear algebra to the
edge".

The signals are kept alongside both labels. "This run is hard" is not
actionable; "the residual rose on twenty-one iterations and cond(B) reached
10^14" is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

DIFFICULTY_EASY = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD = "hard"
DIFFICULTY_ORDER = (DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD)

#: Converged within this many steps, with nothing else wrong: easy.
EASY_MAX_STEPS = 15
#: Beyond this, a converged run is still a hard one.
HARD_MIN_STEPS = 35

#: A residual that rises this many times is fighting the mixer, not descending.
MEDIUM_RESIDUAL_INCREASES = 2
HARD_RESIDUAL_INCREASES = 6

RISK_LOW = "low"
RISK_ELEVATED = "elevated"
RISK_SEVERE = "severe"
RISK_ORDER = (RISK_LOW, RISK_ELEVATED, RISK_SEVERE)

#: log10 cond(B). Past the first the DIIS solve has lost half its digits; past
#: the second it is solving a numerically singular system.
ELEVATED_LOG_CONDITION = 10.0
SEVERE_LOG_CONDITION = 13.0

#: ||c||_1 for the native solve. sum(c) = 1, so a large value means the
#: extrapolation is a difference of large opposite-signed terms.
ELEVATED_COEFFICIENT_L1 = 10.0
SEVERE_COEFFICIENT_L1 = 50.0

_STEPS = re.compile(r"SCF run converged in\s+(\d+)\s+steps")
_GUESS = re.compile(r"Density guess:\s*(\S+)")
_MIXING = re.compile(r"Mixing method:\s*(\S+)")
_EPS_SCF = re.compile(r"eps_scf:\s*([0-9.E+-]+)")
_MAX_SCF = re.compile(r"max_scf:\s*(\d+)")
#: CP2K prints its abort message inside a banner, one boxed line at a time,
#: which is why this reads lines rather than matching a single expression.
_BANNER = re.compile(r"^\s*\*(.*)\*\s*$")
_BANNER_NOISE = re.compile(r"^[\s*_/\\|]*$")


def parse_abort(text: str) -> str | None:
    """Pull the message out of CP2K's ABORT banner.

    The banner interleaves the message with ASCII art, so the art is dropped
    and the remaining boxed lines are joined. Worth doing properly: the message
    is the difference between "this system is hard" and "this input is wrong".
    """

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "[ABORT]" not in line:
            continue
        parts: list[str] = []
        for follower in lines[index : index + 12]:
            match = _BANNER.match(follower)
            if not match:
                break
            body = match.group(1).replace("[ABORT]", " ")
            if _BANNER_NOISE.match(body):
                continue
            # Strip the art column that precedes the text on each line.
            parts.append(" ".join(body.split()))
        message = " ".join(part for part in parts if part).strip()
        if message:
            # The art characters survive as isolated tokens; drop them.
            tokens = [token for token in message.split() if not _BANNER_NOISE.match(token)]
            return " ".join(tokens)[:200]
    return None


@dataclass(frozen=True, slots=True)
class SCFSettings:
    """What CP2K was told to do, read back from its own output."""

    initial_guess: str | None = None
    mixing_method: str | None = None
    eps_scf: float | None = None
    max_scf: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_guess": self.initial_guess,
            "mixing_method": self.mixing_method,
            "eps_scf": self.eps_scf,
            "max_scf": self.max_scf,
        }


@dataclass(frozen=True, slots=True)
class DifficultyAssessment:
    """How hard one SCF was, and what made it so."""

    sample_id: str
    difficulty: str
    converged: bool
    scf_steps: int | None
    #: Conditioning of the DIIS solve, deliberately not folded into difficulty.
    numerical_risk: str = RISK_LOW
    reasons: tuple[str, ...] = ()
    risk_reasons: tuple[str, ...] = ()
    settings: SCFSettings = field(default_factory=SCFSettings)
    signals: Mapping[str, Any] = field(default_factory=dict)
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "difficulty": self.difficulty,
            "converged": self.converged,
            "scf_steps": self.scf_steps,
            "numerical_risk": self.numerical_risk,
            "reasons": list(self.reasons),
            "risk_reasons": list(self.risk_reasons),
            "settings": self.settings.to_dict(),
            "signals": dict(self.signals),
            "failure": self.failure,
        }


def parse_scf_settings(text: str) -> SCFSettings:
    """Read the guess and mixing method CP2K reported for a run.

    Taken from the output rather than the input file: what CP2K resolved is
    what the sample actually is, and the two can differ.
    """

    guess = _GUESS.search(text)
    mixing = _MIXING.search(text)
    eps = _EPS_SCF.search(text)
    max_scf = _MAX_SCF.search(text)
    return SCFSettings(
        initial_guess=guess.group(1) if guess else None,
        mixing_method=mixing.group(1) if mixing else None,
        eps_scf=float(eps.group(1)) if eps else None,
        max_scf=int(max_scf.group(1)) if max_scf else None,
    )


def residual_signals(residuals: Sequence[float | None]) -> dict[str, Any]:
    """Describe the shape of the residual history.

    A converging SCF descends. Counting where it does not is a better
    difficulty signal than the step count alone: a run that reaches the
    tolerance in twenty smooth steps is easier than one that reaches it in
    fifteen after three excursions.
    """

    values = [float(value) for value in residuals if value is not None and np.isfinite(value)]
    if len(values) < 2:
        return {"n_residuals": len(values), "n_increases": 0, "monotonic": True}

    increases = [
        (index, values[index + 1] / values[index])
        for index in range(len(values) - 1)
        if values[index] > 0.0 and values[index + 1] > values[index]
    ]
    reduction = values[-1] / values[0] if values[0] > 0.0 else float("nan")
    return {
        "n_residuals": len(values),
        "n_increases": len(increases),
        "monotonic": not increases,
        "max_increase_ratio": max((ratio for _, ratio in increases), default=1.0),
        "first_increase_at": increases[0][0] if increases else None,
        "total_reduction": float(reduction),
        "log10_total_reduction": float(np.log10(reduction)) if reduction > 0 else None,
    }


def _conditioning_signals(frames: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    conditions: list[float] = []
    l1: list[float] = []
    n_regularise = 0
    for frame in frames:
        detail = frame.get("pulay_diagnostics")
        if not isinstance(detail, Mapping) or not detail.get("finite"):
            continue
        value = detail.get("condition_number")
        if value is not None and np.isfinite(value) and value > 0:
            conditions.append(float(np.log10(value)))
        norm = detail.get("coefficient_l1")
        if norm is not None and np.isfinite(norm):
            l1.append(float(norm))
        if detail.get("regularization_required"):
            n_regularise += 1

    return {
        "n_pulay_matrices": len(conditions),
        "max_log10_condition": max(conditions) if conditions else None,
        "median_log10_condition": float(np.median(conditions)) if conditions else None,
        "max_coefficient_l1": max(l1) if l1 else None,
        "n_regularization_required": n_regularise,
    }


def assess_sample(sample_dir: str | Path, *, sample_id: str | None = None) -> DifficultyAssessment:
    """Classify one sample directory, converged or not.

    Deliberately tolerant: a failed run has no trajectory and often no step
    count, and it is exactly the sample the difficulty classification exists to
    describe. Refusing to assess it would drop the hard class entirely.
    """

    directory = Path(sample_dir)
    name = sample_id or directory.name

    text = ""
    output_path = directory / "cp2k.out"
    if output_path.is_file():
        text = output_path.read_text(encoding="utf-8", errors="replace")

    settings = parse_scf_settings(text)
    match = _STEPS.search(text)
    steps = int(match.group(1)) if match else None
    converged = match is not None
    failure = None
    if not converged and text:
        abort = parse_abort(text)
        if abort:
            failure = abort
        elif "SCF run NOT converged" in text:
            failure = "SCF run NOT converged"
        else:
            failure = "no convergence line in the output"

    frames: list[Mapping[str, Any]] = []
    metadata_path = directory / "trajectory" / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        frames = list(metadata.get("frames", ()))

    frame_details: list[Mapping[str, Any]] = []
    for frame in frames:
        label = "final" if frame.get("is_final") else f"iter_{int(frame['index']):04d}"
        path = directory / "trajectory" / label / "metadata.json"
        if path.is_file():
            try:
                frame_details.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue

    signals: dict[str, Any] = {
        "residual": residual_signals([frame.get("residual") for frame in frames]),
        "conditioning": _conditioning_signals(frame_details),
    }

    difficulty, reasons = _classify(converged, steps, signals, failure)
    risk, risk_reasons = _numerical_risk(signals, failure)
    return DifficultyAssessment(
        sample_id=name,
        difficulty=difficulty,
        converged=converged,
        scf_steps=steps,
        numerical_risk=risk,
        reasons=tuple(reasons),
        risk_reasons=tuple(risk_reasons),
        settings=settings,
        signals=signals,
        failure=failure,
    )


def _classify(
    converged: bool,
    steps: int | None,
    signals: Mapping[str, Any],
    failure: str | None,
) -> tuple[str, list[str]]:
    """Worst signal wins.

    A run is as hard as its hardest symptom: one that converges in ten steps on
    a numerically singular ``B`` got lucky, and calling it easy would teach a
    controller to expect luck.
    """

    reasons: list[str] = []
    level = DIFFICULTY_EASY

    def escalate(to: str, reason: str) -> None:
        nonlocal level
        reasons.append(reason)
        if DIFFICULTY_ORDER.index(to) > DIFFICULTY_ORDER.index(level):
            level = to

    if not converged:
        escalate(DIFFICULTY_HARD, f"did not converge ({failure or 'reason unknown'})")
        return level, reasons

    if steps is not None:
        if steps >= HARD_MIN_STEPS:
            escalate(DIFFICULTY_HARD, f"{steps} SCF steps (>= {HARD_MIN_STEPS})")
        elif steps > EASY_MAX_STEPS:
            escalate(DIFFICULTY_MEDIUM, f"{steps} SCF steps (> {EASY_MAX_STEPS})")

    residual = signals.get("residual", {})
    increases = int(residual.get("n_increases", 0))
    if increases >= HARD_RESIDUAL_INCREASES:
        escalate(DIFFICULTY_HARD, f"residual rose on {increases} iterations")
    elif increases >= MEDIUM_RESIDUAL_INCREASES:
        escalate(DIFFICULTY_MEDIUM, f"residual rose on {increases} iterations")

    if not reasons:
        reasons.append(f"converged in {steps} steps with a monotonic residual")
    return level, reasons


def _numerical_risk(signals: Mapping[str, Any], failure: str | None = None) -> tuple[str, list[str]]:
    """How close the DIIS solve ran to the edge of float64.

    Separate from difficulty on purpose. A fast, smooth SCF on a numerically
    singular ``B`` is not a hard system -- it is a lucky one, and a controller
    that learns to expect that luck will be wrong on the run where the same
    conditioning ends in the abort CP2K raises on MgO.
    """

    conditioning = signals.get("conditioning", {})
    reasons: list[str] = []
    level = RISK_LOW

    def escalate(to: str, reason: str) -> None:
        nonlocal level
        reasons.append(reason)
        if RISK_ORDER.index(to) > RISK_ORDER.index(level):
            level = to

    # A run that aborted inside the DIIS solve recorded no diagnostics, because
    # it never got to write one. Calling that "low risk" would be exactly
    # backwards, so the abort message itself is the signal.
    if failure and "condition number" in failure.lower():
        escalate(RISK_SEVERE, f"CP2K aborted on the DIIS solve: {failure[:80]}")

    condition = conditioning.get("max_log10_condition")
    if condition is not None:
        if condition >= SEVERE_LOG_CONDITION:
            escalate(RISK_SEVERE, f"log10 cond(B) reached {condition:.1f}")
        elif condition >= ELEVATED_LOG_CONDITION:
            escalate(RISK_ELEVATED, f"log10 cond(B) reached {condition:.1f}")

    norm = conditioning.get("max_coefficient_l1")
    if norm is not None:
        if norm >= SEVERE_COEFFICIENT_L1:
            escalate(RISK_SEVERE, f"||c||_1 reached {norm:.1f}")
        elif norm >= ELEVATED_COEFFICIENT_L1:
            escalate(RISK_ELEVATED, f"||c||_1 reached {norm:.1f}")

    if not reasons:
        reasons.append("no conditioning signal above threshold")
    return level, reasons


def assess_dataset(root: str | Path) -> list[DifficultyAssessment]:
    """Assess every sample under a dataset root, failures included."""

    dataset_root = Path(root)
    assessments: list[DifficultyAssessment] = []
    for sample_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if not any(
            (sample_dir / name).exists()
            for name in ("sample.json", "cp2k.out", "trajectory")
        ):
            continue
        assessments.append(assess_sample(sample_dir))
    return assessments


def difficulty_summary(assessments: Sequence[DifficultyAssessment]) -> dict[str, Any]:
    """Aggregate for ``dataset_report.json``.

    Reports the failed samples too. A summary computed over converged runs only
    would show a dataset getting easier exactly as it acquired the systems
    worth having.
    """

    counts = {level: 0 for level in DIFFICULTY_ORDER}
    risks = {level: 0 for level in RISK_ORDER}
    steps_by_level: dict[str, list[float]] = {level: [] for level in DIFFICULTY_ORDER}
    guesses: dict[str, int] = {}
    mixers: dict[str, int] = {}
    failures: dict[str, int] = {}

    for item in assessments:
        counts[item.difficulty] += 1
        risks[item.numerical_risk] += 1
        if item.scf_steps is not None:
            steps_by_level[item.difficulty].append(float(item.scf_steps))
        guess = item.settings.initial_guess or "unknown"
        guesses[guess] = guesses.get(guess, 0) + 1
        mixing = item.settings.mixing_method or "unknown"
        mixers[mixing] = mixers.get(mixing, 0) + 1
        if item.failure:
            key = item.failure.split("(")[0].strip()[:80]
            failures[key] = failures.get(key, 0) + 1

    total = len(assessments) or 1
    return {
        "n_samples": len(assessments),
        "counts": counts,
        "fractions": {level: counts[level] / total for level in DIFFICULTY_ORDER},
        "numerical_risk_counts": risks,
        "scf_steps_by_difficulty": {
            level: _summary(values) for level, values in steps_by_level.items()
        },
        "initial_guess": dict(sorted(guesses.items())),
        "mixing_method": dict(sorted(mixers.items())),
        "failures": dict(sorted(failures.items())),
        "n_converged": sum(1 for item in assessments if item.converged),
        "thresholds": {
            "easy_max_steps": EASY_MAX_STEPS,
            "hard_min_steps": HARD_MIN_STEPS,
            "medium_residual_increases": MEDIUM_RESIDUAL_INCREASES,
            "hard_residual_increases": HARD_RESIDUAL_INCREASES,
            "elevated_log10_condition": ELEVATED_LOG_CONDITION,
            "severe_log10_condition": SEVERE_LOG_CONDITION,
            "elevated_coefficient_l1": ELEVATED_COEFFICIENT_L1,
            "severe_coefficient_l1": SEVERE_COEFFICIENT_L1,
        },
        "note": (
            "Absolute thresholds, not quantiles: a sample's label must not change "
            "because another sample was added. Failed runs are included -- they are "
            "the hard class, and a summary over converged runs only would show the "
            "dataset getting easier as it acquired the systems worth having."
        ),
        "samples": [item.to_dict() for item in assessments],
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }
