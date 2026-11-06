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

## Reference tree removed (2026-11-06)

`ref/lammps_monitor` was deleted after these gates passed:

| Gate | Evidence |
|---|---|
| Every ledger entry has an implementation and evidence | `feature_ledger.yaml`: 170 of 187 resolved (verified, replaced or dropped). The 17 `migrated` entries are implemented and tested, but their remaining validation needs infrastructure not available here (Slurm/PBS clusters, container builds, CI runner, mkdocs) or data that PHAL itself lost; none needs the reference tree. |
| Attribution, licenses, provenance | `NOTICE` (PHAL MIT license); a header on every ported file; `phal_inventory.json`; `phal_release_baseline/` |
| Reference-derived tests and fixtures are local | `tests/fixtures/phal_reference/` plus other captured PHAL outputs under `tests/`. Live-PHAL comparison tests skip with a reason. |
| Exact revision recoverable off-tree | `96263b6` is `main` at `github.com/kiana-0423/lammps_monitor` (checked with `git ls-remote`), and `../_archive/lammps_monitor-96263b6.bundle` (sha256 `b5b76e7d…`) clones to a file-identical tree |
| No unique uncommitted reference files | `git status --porcelain --ignored` in the reference repository was empty; an empty `lammps_run/` directory created by this session's capture runs was removed |
| No runtime/test/packaging/config dependency | layering test (`tests/architecture/`); precise scan for `hotspot_al` imports and `ref/` paths; only `tools/migration/` capture tools mention the tree |
| Clean installation without the tree | `tools/migration/clean_install_check.sh`: wheel built from the committed tree, installed in a fresh venv, `src/` deleted. 2146 passed, 47 skipped; all 17 real GMD, GMD_SGT and CP2K tests passed from the installed wheel. |
| Integration, recovery and rollback acceptance | `tests/integration/` (real closed loop, rollback, GMD); `evidence/` |

After removal, the ledger checks (`check_ledger.py`, `apply_status.py --check`) and the full suite were rerun.
`build_phal_inventory.py` and `capture_phal_reference.py` now need a fresh checkout of the PHAL
revision above (`git clone <bundle> ref/lammps_monitor`).
