# ADR 004: Candidate deduplication uses pair-distance histograms

Ported from PHAL `docs/adr/004-distance-histogram-dedup.md` (commit 96263b6, MIT License) and
amended for GMD_active (ledger DOC-004, AL-004).

## Status

Accepted (PHAL). Amended 2026-11 for GMD_active.

## Context

The candidate pool needs a lightweight diversity filter before expensive DFT labeling. Graph
isomorphism and SOAP-like descriptors can be more expressive, but they add dependencies and cost.

## Decision (PHAL, unchanged)

The default fingerprint is a pair-distance histogram over extracted local regions. A
type-weighted variant is available for systems where atom identities should influence
deduplication.

## Amendment: GMD_active decisions

Implemented in `active_learning/candidates.py`, configured under `workflow.candidates`.

1. **Distance: L2, not cosine.** Fingerprints are compared with the Euclidean distance, as in
   PHAL; a candidate is kept iff its distance to every kept fingerprint is
   `>= diversity_threshold` (the boundary is kept). Cosine distance was rejected: it ignores the
   histogram magnitude, so two regions with the same *shape* of distance distribution but very
   different pair counts would be duplicates, and the existing thresholds would lose their
   meaning. PHAL parity is tested on the golden regions
   (`tests/active_learning/test_phal_candidate_parity.py`).
2. **Periodic boundaries: minimum image by default.** `pair_distances: minimum_image` uses exact
   minimum-image distances (`utils.periodic`, including skewed and partially periodic cells) for
   periodic regions. PHAL used plain `pdist`, identical for the non-periodic regions of the default
   cluster mode but wrong for periodic slab and block regions. `pair_distances: plain` reproduces
   PHAL.
3. **Normalisation: density, finite.** `histogram_normalisation: density` keeps PHAL's
   `np.histogram(..., density=True)` scale, so `diversity_threshold` keeps its meaning. PHAL's
   density histogram is NaN when no pair lies within `fingerprint_cutoff`; such a candidate was
   silently rejected and, if kept first, blocked every later candidate (F-30). The histogram of
   such a region is now all zeros. `probability` (sums to one) is available.
4. **Labeled history.** Diversity is also checked against fingerprints of already labeled regions
   (`compare_with_labeled`), so a new cycle does not re-label what earlier cycles labeled.
5. **Selection is recomputed** from scratch on every selection, preserving PHAL's eviction
   behaviour; ties in score keep insertion order (`tie_break: insertion`).

## Consequences

Fast, dependency-light, adequate for coarse geometric screening. Known blind spots:
stereochemistry, topology-preserving rearrangements, and species changes in the untyped variant.
The default fingerprint is now the type-weighted one. Richer descriptors can be added behind the
same `fingerprint` setting.
