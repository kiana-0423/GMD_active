# Ported from PHAL (hotspot_al.active_learning.candidate_pool / deduplication) at 96263b6; MIT License, see NOTICE.
"""Candidate ranking, diversity selection and labeling budgets.

Changes from PHAL:
- fingerprints are finite for any region (PHAL produced NaN when no pair fell under the
  cutoff, which silently rejected the candidate and could block later ones; F-30);
- periodic regions use minimum-image pair distances (PHAL used plain ``pdist``);
- diversity is checked against previously labeled fingerprints as well as within the batch;
- budgets per cycle, per event and in total are explicit, and every rejection is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gmd_active_learning.contracts import ExtractedRegion
from gmd_active_learning.utils.logging import get_logger
from gmd_active_learning.utils.periodic import minimum_image_vectors

LOGGER = get_logger(__name__)
REFERENCE_ATOMIC_NUMBER = 100.0


class CandidateConfig(BaseModel):
    """``candidates`` config section."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    fingerprint: str = Field("type_weighted_pair_distance_histogram", pattern="^(pair_distance_histogram|type_weighted_pair_distance_histogram)$")
    fingerprint_bins: int = Field(16, ge=2)
    fingerprint_cutoff: float = Field(8.0, gt=0.0)
    deduplicate: bool = True
    diversity_threshold: float = Field(0.1, ge=0.0)
    compare_with_labeled: bool = True
    max_per_cycle: int = Field(20, ge=0)
    max_per_event: int = Field(4, ge=0)
    max_total: int | None = Field(None, ge=0)
    min_score: float = 0.0
    core_atoms_only_for_ranking: bool = False


def _pair_distances(region: ExtractedRegion) -> np.ndarray:
    positions = region.positions
    if len(positions) < 2:
        return np.zeros(0)
    left, right = np.triu_indices(len(positions), k=1)
    vectors = positions[right] - positions[left]
    if any(region.pbc) and region.cell is not None:
        vectors = minimum_image_vectors(vectors, cell=region.cell, pbc=region.pbc)
    return np.linalg.norm(vectors, axis=1)


def _normalised_histogram(distances: np.ndarray, *, bins: int, cutoff: float, weights: np.ndarray | None = None) -> np.ndarray:
    counts, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff), weights=weights)
    total = float(np.sum(counts))
    return np.zeros(bins) if total <= 0.0 else counts.astype(float) / total


def fingerprint_region(region: ExtractedRegion, config: CandidateConfig) -> np.ndarray:
    """Intensive, always-finite geometric fingerprint of a region."""

    bins, cutoff = config.fingerprint_bins, config.fingerprint_cutoff
    distances = _pair_distances(region)
    plain = _normalised_histogram(distances, bins=bins, cutoff=cutoff)
    if config.fingerprint == "pair_distance_histogram":
        return plain
    if len(distances) == 0:
        return np.zeros(3 * bins)
    numbers = region.numbers.astype(float)
    left, right = np.triu_indices(len(numbers), k=1)
    weights = 0.5 * (numbers[left] + numbers[right]) / REFERENCE_ATOMIC_NUMBER
    weighted_sum, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff), weights=weights)
    counts, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff))
    weighted_mean = np.divide(weighted_sum, counts, out=np.zeros(bins), where=counts > 0)
    weighted_density = weighted_sum / float(len(distances))
    fingerprint = np.concatenate([plain, weighted_density, weighted_mean])
    if not np.all(np.isfinite(fingerprint)):  # defensive: must never happen
        raise ValueError(f"non-finite fingerprint for region {region.region_id}")
    return fingerprint


@dataclass(slots=True)
class Candidate:
    region: ExtractedRegion
    score: float
    fingerprint: np.ndarray
    event_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_id(self) -> str:
        return self.region.region_id


@dataclass(slots=True)
class SelectionDecision:
    selected: list[Candidate]
    rejected: dict[str, str]  # candidate_id -> reason


class CandidateSelector:
    """Greedy score-ordered selection with diversity and budget constraints.

    The unique set is recomputed from scratch for every call (PHAL's rationale: admitting a
    high-scoring candidate can change which others survive, so incremental maintenance is wrong).
    """

    def __init__(self, config: CandidateConfig) -> None:
        self.config = config

    def make_candidate(self, region: ExtractedRegion, *, score: float, metadata: dict[str, Any] | None = None) -> Candidate:
        return Candidate(
            region=region,
            score=float(score),
            fingerprint=fingerprint_region(region, self.config),
            event_id=region.event_id,
            metadata=dict(metadata or {}),
        )

    def select(
        self,
        candidates: Iterable[Candidate],
        *,
        labeled_fingerprints: Iterable[np.ndarray] = (),
        already_labeled_total: int = 0,
    ) -> SelectionDecision:
        config = self.config
        rejected: dict[str, str] = {}
        pool = sorted(candidates, key=lambda item: (-item.score, item.candidate_id))
        if not config.enabled:
            return SelectionDecision([], {item.candidate_id: "selection_disabled" for item in pool})
        budget = config.max_per_cycle
        if config.max_total is not None:
            budget = min(budget, max(0, config.max_total - already_labeled_total))
        history = [np.asarray(item, dtype=float) for item in labeled_fingerprints] if config.compare_with_labeled else []
        kept: list[Candidate] = []
        per_event: dict[str | None, int] = {}
        for candidate in pool:
            if candidate.score < config.min_score:
                rejected[candidate.candidate_id] = "below_min_score"
                continue
            if config.deduplicate:
                references = [*history, *(item.fingerprint for item in kept)]
                if any(
                    reference.shape == candidate.fingerprint.shape
                    and float(np.linalg.norm(candidate.fingerprint - reference)) < config.diversity_threshold
                    for reference in references
                ):
                    rejected[candidate.candidate_id] = "duplicate"
                    continue
            if per_event.get(candidate.event_id, 0) >= config.max_per_event:
                rejected[candidate.candidate_id] = "event_budget"
                continue
            if len(kept) >= budget:
                rejected[candidate.candidate_id] = "cycle_budget" if budget == config.max_per_cycle else "total_budget"
                continue
            kept.append(candidate)
            per_event[candidate.event_id] = per_event.get(candidate.event_id, 0) + 1
        LOGGER.info("selected %d of %d candidates (%d rejected)", len(kept), len(pool), len(rejected))
        return SelectionDecision(kept, rejected)
