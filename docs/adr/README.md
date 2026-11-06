# Architecture decision records

| ADR | Decision | Origin |
|---|---|---|
| [004](004-distance-histogram-dedup.md) | Candidate deduplication by pair-distance histograms (amended: L2, minimum image, finite density) | PHAL, amended |
| [005](005-segmented-gmd-control.md) | GMD controlled in segments through files, cooperatively | GMD_active |
| [006](006-persistent-sqlite-store.md) | Workflow state in one SQLite file with write-ahead intents | GMD_active |
| [007](007-deterministic-ids.md) | Content-derived identifiers | GMD_active |

PHAL ADRs 001 (atom-wise OOD), 002 (block spatial grid) and 003 (linear LJ residual) describe
behaviour that is documented in [monitoring.md](../monitoring.md) and
[extraction_and_datasets.md](../extraction_and_datasets.md); 003's 2x2 Gram description was wrong
(F-20) and is superseded by monitoring.md's 3x2 least-squares description.
