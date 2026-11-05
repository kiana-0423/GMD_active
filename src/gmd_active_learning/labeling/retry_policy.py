# Ported from PHAL (hotspot_al.cp2k.cp2k_task_submitter: _retry_or_return, _retry_natively,
# _input_with_adjusted_scf) at 96263b6; MIT License, see NOTICE.
"""Bounded, explicit retry decisions per failure class (findings F-25).

PHAL multiplied ``MAX_SCF`` of the *previous retry input* for every failure cause (100 -> 200 -> 600),
and a failed ML-SCF run that was converted to native CP2K was never resubmitted. Here:

- each failure class has its own ladder of :class:`RetryStep` overrides, applied to the *base*
  settings (no compounding) and capped by ``max_scf_cap``;
- the total number of retries is bounded by ``retry.max_retries``;
- a failed ML-SCF attempt is retried natively first (and the caller actually submits it);
- ``UNKNOWN`` scheduler state is not a failure class: it never reaches this policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gmd_active_learning.adapters.dft.cp2k.config import FailureClass, RetrySettings, RetryStep

KIND_INITIAL = "initial"
KIND_RETRY = "retry"
KIND_STAGE = "stage"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    retry: bool
    reason: str
    step: RetryStep | None = None
    native_fallback: bool = False
    ladder_index: int | None = None


class RetryPolicy:
    def __init__(self, settings: RetrySettings) -> None:
        self.settings = settings

    def decide(
        self,
        failure: FailureClass,
        attempts: Sequence[Mapping[str, Any]],
        *,
        ml_active: bool = False,
        current_step: RetryStep | None = None,
    ) -> RetryDecision:
        """Whether and how to retry after an attempt failed with ``failure``.

        ``attempts`` is the task's attempt history (dicts with ``kind`` and ``trigger_class``).
        """

        if failure is FailureClass.CANCELLED:
            return RetryDecision(False, "cancelled jobs are never retried automatically")
        if not self.settings.enabled:
            return RetryDecision(False, "retries are disabled (cp2k.retry.enabled = false)")
        used = sum(1 for attempt in attempts if attempt.get("kind") == KIND_RETRY)
        if used >= self.settings.max_retries:
            return RetryDecision(False, f"retry budget exhausted ({used}/{self.settings.max_retries})")
        if ml_active and self.settings.ml_scf_native_fallback:
            return RetryDecision(
                True,
                f"ML-SCF attempt failed ({failure.value}); resubmitting with native CP2K SCF",
                step=current_step,
                native_fallback=True,
            )
        ladder = self.settings.ladders.get(failure, [])
        index = sum(1 for attempt in attempts if attempt.get("kind") == KIND_RETRY and attempt.get("trigger_class") == failure.value)
        if index >= len(ladder):
            return RetryDecision(False, f"no further retry step for {failure.value} ({index}/{len(ladder)} used)")
        step = ladder[index]
        return RetryDecision(True, f"{failure.value}: retry step {index + 1}/{len(ladder)}", step=step, ladder_index=index)


__all__ = ["KIND_INITIAL", "KIND_RETRY", "KIND_STAGE", "RetryDecision", "RetryPolicy"]
