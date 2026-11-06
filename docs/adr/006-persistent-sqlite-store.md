# ADR 006: Workflow state lives in one SQLite file with write-ahead intents

## Status

Accepted (GMD_active 0.4).

## Context

PHAL persisted no workflow state: its job table, task queues and block cooldowns lived in memory,
task IDs had random suffixes and task directories were overwritten, so a restart could neither
resume nor avoid duplicate DFT submissions (F-24). The loop runs for days on shared clusters and
will be interrupted.

## Decision

- All workflow state of a run is stored in `<work-dir>/workflow.sqlite` (`workflow/store.py`,
  standard-library `sqlite3`, schema version recorded): runs, stage transitions, segments, events,
  candidates, label tasks, samples, dataset versions, models, promotions and round summaries.
- The current stage is written **before** its work starts. Every external effect follows
  **intent → action → result** inside transactions; on restart an intent without a result is
  reconciled against the backend (`Labeler.find_submission`, `MDEngine.completed_segment`) instead
  of being repeated.
- Records are keyed by deterministic IDs (ADR 007) with uniqueness constraints, so every insert
  is idempotent and a replayed stage cannot duplicate an event, candidate, job, sample, dataset
  version or model.
- Large artifacts (frames, regions, labels, datasets, models) live in files next to the database,
  written atomically and referenced by checksum; the database holds metadata only.

## Alternatives considered

- JSON files per object (PHAL style): no transactions across objects, no uniqueness constraints.
- An external database or workflow engine: extra services on HPC login nodes, against the
  project's no-large-dependency rule.

## Consequences

- `gmd-al run` can be killed at any point and re-run; `gmd-al status` reads the same database.
- One controller per work directory: SQLite serialises writers; concurrent registry writers are
  guarded, but two controllers on one run are not supported.
- Records currently store absolute artifact paths, so moving a run directory needs care (see
  `docs/known-issues.md`).
