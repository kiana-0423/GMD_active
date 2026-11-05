# PHAL → GMD_active migration: Phase 0 artifacts

This directory holds the Phase 0 deliverables for the plan to rebuild GMD_active as the
complete successor to PHAL (`ref/lammps_monitor`, package `hotspot_al`):

- **GMD** provides molecular dynamics, replacing LAMMPS.
- **GMD_SGT** provides training, inference, and export, replacing Allegro/NequIP.
- **CP2K** remains the labeling engine.

| File | Purpose |
|---|---|
| `feature_ledger.yaml` | **Source of truth.** One entry per PHAL capability: source files, behavior, config keys, PHAL tests, destination, disposition, acceptance test, phase, and migration status. |
| `feature_ledger.md` | Generated overview of the ledger. Do not edit by hand. |
| `phal_inventory.json` | Frozen list of all 438 tracked PHAL files at commit `96263b6`, with SHA-256 hashes. Ledger completeness is checked against this file, so the check keeps working after `ref/` is deleted. |
| `baseline.md` | Phase 0 test baseline for GMD_active, PHAL, and GMD_SGT, with environment and revisions. |
| `integration_gaps.md` | What GMD and GMD_SGT support today, verified gaps, and the minimal upstream extensions. |
| `findings.md` | Defects and parity hazards found in PHAL and GMD_active during Phase 0. |

Related code and data:

| Path | Purpose |
|---|---|
| `tools/migration/build_phal_inventory.py` | Regenerates `phal_inventory.json` from `ref/lammps_monitor`. |
| `tools/migration/check_ledger.py` | Validates the ledger: schema, unique IDs, every reference resolves, every inventoried file covered. |
| `tools/migration/render_ledger.py` | Regenerates `feature_ledger.md`. |
| `tools/migration/capture_phal_reference.py` | Runs PHAL on deterministic synthetic trajectories and writes golden outputs. |
| `tests/fixtures/phal_reference/` | Golden PHAL outputs for parity tests: monitor scores and triggers, events, regions in four extraction modes (raw and H-capped), masks, region codes, and datasets. |
| `tests/migration/` | Ledger, tooling, and fixture-integrity tests. |

## Commands

From the GMD_active root, in the `gmd-active` conda environment (see `baseline.md`):

```bash
python tools/migration/check_ledger.py          # validate ledger against the inventory
python tools/migration/render_ledger.py         # regenerate feature_ledger.md after editing the YAML
python -m pytest tests/migration                # ledger, tooling and golden-fixture tests
# While ref/lammps_monitor still exists:
python tools/migration/build_phal_inventory.py  # only if PHAL itself changes (then re-check the ledger)
python tools/migration/capture_phal_reference.py  # regenerate golden outputs (deterministic)
```

## Working rules

- A ledger entry moves through `not-started → in-progress → migrated → verified`, or ends as
  `replaced` / `dropped` with a justification. `verified` requires the entry's acceptance test to pass
  without importing PHAL.
- Parity tests replay `tests/fixtures/phal_reference/<scenario>/input_frames.extxyz` with
  `config.json` and compare against the recorded outputs. Committee forces are stored in the input
  file, so no MLIP backend is needed.
- Where PHAL behavior is a defect (see `findings.md`), the migrated code fixes it deliberately, and the
  ledger entry and parity test record the intended deviation.

## Removing `ref/lammps_monitor`

This is the final milestone. It requires every ledger entry to be `verified`, `replaced`, or `dropped`,
and the acceptance checklist in the migration plan to be complete.

The PHAL source is already preserved off-tree: commit `96263b6` is `main` at
`github.com/kiana-0423/lammps_monitor`. Two files PHAL's own configs need were never committed
(see `baseline.md`).
