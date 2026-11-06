# ADR 007: Identifiers are derived from content, never random

## Status

Accepted (GMD_active 0.4).

## Context

PHAL event IDs embedded `uuid4` (F-15), task IDs a random suffix (F-24), and the model registry
silently overwrote a version registered twice (F-14). Random identity makes reruns and crash
recovery create duplicates instead of recognising existing work.

## Decision

- `contracts.ids.stable_id(prefix, *parts)` = prefix + truncated SHA-256 of the canonical JSON of
  the parts; `content_hash` hashes canonical JSON with arrays encoded bit-exactly; `file_sha256`
  and `tree_sha256` hash files and directories.
- Identity is a function of what defines the object:
  - event: `stable_id("evt", run_id, first_trigger_step)`;
  - region: run, step, mode, event, hotspots, parent IDs, image shifts and the mode's settings;
  - label task: `stable_id("label", region_id, region_fingerprint, settings_hash)`, where the
    settings hash covers only settings that change the calculation (not the executable, launcher,
    resources or retry policy, so a campaign can move machines);
  - sample: `stable_id("sample", region_id, task_id)`;
  - dataset version: parent, ordered sample IDs, split groups and export settings; the stored
    `sha256` is the tree hash of the version directory and is re-verified on read;
  - model: `stable_id("model", checkpoint_sha256, export_sha256, dataset_version, parent_model_id,
    extra file hashes)`; registered artifacts are copied into the registry and re-verified against
    their checksums on use, so the same files register once and different files never share an ID.
- Timestamps are recorded as data, never used as identity.

## Consequences

- Re-running extraction or labeling preparation on the same input yields the same IDs, so the
  store's uniqueness constraints make replays no-ops (ADR 006).
- A relabel of the same region needs a new task (different settings), never a silent overwrite.
- Changing which settings enter a hash changes identities; such changes are migrations and must be
  recorded in the CHANGELOG.
- `tools/audit/audit_artifacts.py` re-verifies registry checksums, dataset tree hashes and task
  integrity offline.
