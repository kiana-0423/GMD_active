# Ported from PHAL (hotspot_al.active_learning.candidate_pool / deduplication) at 96263b6; MIT License, see NOTICE.
"""Candidate ranking, diversity selection and labeling budgets.

PHAL parity (ledger AL-002..004): fingerprints are pair-distance histograms over ``[0, cutoff)``
with ``histogram_normalisation: density`` (``np.histogram(..., density=True)``, so
``diversity_threshold`` keeps PHAL's meaning), compared with the L2 distance; a candidate is kept
iff its distance to every kept fingerprint is ``>= diversity_threshold`` (boundary kept); ties in
score keep insertion order (``tie_break: insertion``); the unique set is recomputed from scratch on
every selection, so admitting a high-scoring candidate that suppresses a suppressor re-admits the
third (PHAL's eviction scenario).

Deliberate changes from PHAL:
- fingerprints are finite for any region: PHAL's density histogram is NaN when no pair lies inside
  the cutoff, which silently rejected the candidate and, if it was kept first, every later one
  (finding F-30); here such a histogram is all zeros;
- ``pair_distances: minimum_image`` (default) uses minimum-image distances for periodic regions;
  PHAL used plain ``pdist`` (``pair_distances: plain``), which is identical for the non-periodic
  regions of the default cluster mode and wrong for periodic slab/block regions;
- diversity is also checked against previously labeled fingerprints (``compare_with_labeled``);
- budgets per cycle, per event (``max_per_event``, ``None`` = PHAL's unlimited) and in total are
  explicit, an optional ``min_score`` filter exists, and every rejection is recorded with a reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gmd_active_learning.contracts import ExtractedRegion
from gmd_active_learning.utils.logging import get_logger
from gmd_active_learning.utils.periodic import minimum_image_vectors

LOGGER = get_logger(__name__)
REFERENCE_ATOMIC_NUMBER = 100.0

#: PHAL's accepted fingerprint names (deduplication.fingerprint_region) -> canonical name.
FINGERPRINT_ALIASES: dict[str, str] = {
    "pair_distance_histogram": "pair_distance_histogram",
    "pair_distance": "pair_distance_histogram",
    "distance": "pair_distance_histogram",
    "type_weighted_pair_distance_histogram": "type_weighted_pair_distance_histogram",
    "type_weighted_pair_distance": "type_weighted_pair_distance_histogram",
    "weighted_pair": "type_weighted_pair_distance_histogram",
}


def canonical_fingerprint(mode: str) -> str:
    try:
        return FINGERPRINT_ALIASES[str(mode).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported candidate fingerprint mode: {mode}") from exc


class CandidateConfig(BaseModel):
    """``workflow.candidates`` config section."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    fingerprint: str = Field("type_weighted_pair_distance_histogram", description="pair_distance_histogram | type_weighted_pair_distance_histogram (PHAL aliases accepted)")
    fingerprint_bins: int = Field(16, ge=2)
    fingerprint_cutoff: float = Field(8.0, gt=0.0)
    pair_distances: Literal["minimum_image", "plain"] = Field("minimum_image", description="plain = PHAL pdist (ignores periodicity)")
    histogram_normalisation: Literal["density", "probability"] = Field("density", description="density = PHAL (threshold meaning); probability sums to 1")
    deduplicate: bool = True
    diversity_threshold: float = Field(0.1, ge=0.0)
    compare_with_labeled: bool = True
    tie_break: Literal["insertion", "candidate_id"] = Field("insertion", description="order of equal scores; insertion = PHAL stable sort")
    max_per_cycle: int = Field(20, ge=0)
    max_per_event: int | None = Field(4, ge=0, description="None = no per-event budget (PHAL)")
    max_total: int | None = Field(None, ge=0)
    min_score: float | None = Field(0.0, description="None disables the score filter (PHAL had none)")
    core_atoms_only_for_ranking: bool = False

    @field_validator("fingerprint")
    @classmethod
    def _canonical(cls, value: str) -> str:
        return canonical_fingerprint(value)


def pair_distances(region: ExtractedRegion, *, minimum_image: bool = True) -> np.ndarray:
    """Upper-triangle pair distances in ``scipy.spatial.distance.pdist`` order."""

    positions = np.asarray(region.positions, dtype=float)
    if len(positions) < 2:
        return np.zeros(0)
    left, right = np.triu_indices(len(positions), k=1)
    vectors = positions[right] - positions[left]
    if minimum_image and any(region.pbc) and region.cell is not None:
        vectors = minimum_image_vectors(vectors, cell=region.cell, pbc=region.pbc)
    return np.linalg.norm(vectors, axis=1)


def _histogram(distances: np.ndarray, *, bins: int, cutoff: float, normalisation: str) -> np.ndarray:
    counts, edges = np.histogram(distances, bins=bins, range=(0.0, cutoff))
    total = float(np.sum(counts))
    if total <= 0.0:  # PHAL: density=True gives NaN here (F-30)
        return np.zeros(bins)
    probability = counts.astype(float) / total
    return probability / np.diff(edges) if normalisation == "density" else probability


def fingerprint_region(region: ExtractedRegion, config: CandidateConfig) -> np.ndarray:
    """Intensive, always-finite geometric fingerprint of a region."""

    bins, cutoff, normalisation = config.fingerprint_bins, config.fingerprint_cutoff, config.histogram_normalisation
    mode = canonical_fingerprint(config.fingerprint)
    distances = pair_distances(region, minimum_image=config.pair_distances == "minimum_image")
    plain = _histogram(distances, bins=bins, cutoff=cutoff, normalisation=normalisation)
    if mode == "pair_distance_histogram":
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


def fingerprint_distance(left: np.ndarray, right: np.ndarray) -> float:
    """L2 distance between fingerprints (PHAL's metric)."""

    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


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


def deduplicate_candidates(candidates: Iterable[Candidate], *, diversity_threshold: float) -> list[Candidate]:
    """PHAL's greedy pass: by descending score (stable), keep iff distance >= threshold to all kept."""

    unique: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: -item.score):
        if all(fingerprint_distance(candidate.fingerprint, kept.fingerprint) >= diversity_threshold for kept in unique):
            unique.append(candidate)
    return unique


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

    def _ordered(self, candidates: Iterable[Candidate]) -> list[Candidate]:
        if self.config.tie_break == "insertion":
            return sorted(candidates, key=lambda item: -item.score)  # stable: ties keep insertion order
        return sorted(candidates, key=lambda item: (-item.score, item.candidate_id))

    def select(
        self,
        candidates: Iterable[Candidate],
        *,
        labeled_fingerprints: Iterable[np.ndarray] = (),
        already_labeled_total: int = 0,
    ) -> SelectionDecision:
        config = self.config
        rejected: dict[str, str] = {}
        pool = self._ordered(candidates)
        if not config.enabled:
            return SelectionDecision([], {item.candidate_id: "selection_disabled" for item in pool})
        budget = config.max_per_cycle
        if config.max_total is not None:
            budget = min(budget, max(0, config.max_total - already_labeled_total))
        history = [np.asarray(item, dtype=float) for item in labeled_fingerprints] if config.compare_with_labeled else []
        unique: list[Candidate] = []  # PHAL's dedup set (before budgets)
        kept: list[Candidate] = []
        per_event: dict[str | None, int] = {}
        for candidate in pool:
            if config.min_score is not None and candidate.score < config.min_score:
                rejected[candidate.candidate_id] = "below_min_score"
                continue
            if config.deduplicate:
                references = [*history, *(item.fingerprint for item in unique)]
                if any(
                    reference.shape == candidate.fingerprint.shape
                    and fingerprint_distance(candidate.fingerprint, reference) < config.diversity_threshold
                    for reference in references
                ):
                    rejected[candidate.candidate_id] = "duplicate"
                    continue
            if config.max_per_event is not None and per_event.get(candidate.event_id, 0) >= config.max_per_event:
                rejected[candidate.candidate_id] = "event_budget"  # does not suppress candidates of other events
                continue
            unique.append(candidate)
            if len(kept) >= budget:
                rejected[candidate.candidate_id] = "cycle_budget" if budget == config.max_per_cycle else "total_budget"
                continue
            kept.append(candidate)
            per_event[candidate.event_id] = per_event.get(candidate.event_id, 0) + 1
        LOGGER.info("selected %d of %d candidates (%d rejected)", len(kept), len(pool), len(rejected))
        return SelectionDecision(kept, rejected)
