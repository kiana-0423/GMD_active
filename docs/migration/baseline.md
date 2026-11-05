# Phase 0 test baseline

Recorded 2026-11-06 before any migration code was written. Re-run these commands
to compare against later phases.

## Environment

| Item | Value |
|---|---|
| Host | macOS 27.2 (arm64) |
| Python env | conda env `gmd-active` (Python 3.11.16) |
| Packages | numpy 2.2.6, scipy 1.17.1, torch 2.15.0, ase 3.29.0, e3nn 0.6.0, pydantic 2.14.3, PyYAML 6.0.3, pytest 9.2.0 |
| External programs | none (no GMD binary, CP2K, LAMMPS, Allegro, or scheduler on the test path) |

Create the environment:

```bash
conda create -n gmd-active python=3.11 pip
conda run -n gmd-active pip install "numpy<2.3" scipy pyyaml torch pydantic ase pytest pytest-timeout pytest-cov tqdm e3nn
```

## Repository revisions

| Repository | Revision | Notes |
|---|---|---|
| GMD_active | `69e301c` | `ref/` is untracked in this repository |
| PHAL (`ref/lammps_monitor`) | `96263b6` | Its own Git repository; identical to `origin/main` at `github.com/kiana-0423/lammps_monitor` (verified with `git ls-remote`) |
| GMD_SGT | `5c9bba5` | clean |
| GMD | `9d8eede` | clean |
| cp2k-ml-scf | `20759c8` | clean |

## Results

All runs used `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider` so that no
files were written into the repositories.

| Suite | Command | Result |
|---|---|---|
| GMD_active | `python -m pytest` | **10 passed** |
| PHAL syntax | compile every `.py` in `src tests examples` | 230 files, 0 failures |
| PHAL imports | `PYTHONPATH=src python tests/scripts/check_imports.py` | pass |
| PHAL offline | `python -m pytest tests/unit tests/e2e tests/architecture -m "not integration and not slow"` | **973 passed, 1 failed, 12 errors, 7 skipped** |
| PHAL integration | `python -m pytest tests/integration` (no `RUN_EXTERNAL`) | 4 skipped (by design) |
| GMD_SGT | `python -m pytest -o addopts=""` | **234 passed, 4 skipped** (skips are the "e3nn missing" branch tests) |

PHAL's ruff, mypy, coverage, and benchmark CI steps were not run.

### PHAL failures: missing data files that were never committed

All 13 PHAL failures and errors have one cause. They are **not environment problems**:

- `configs/cp2k/pulay_sampling_plan.json`, `docs/pulay_dataset.md`,
  `scripts/generate_pulay_dataset.py`, and `tests/fixtures/pulay_sample_periodic/sample.json`
  reference `data/structures/pulay_sampling/small_molecules.extxyz` and
  `data/structures/pulay_sampling/periodic.extxyz`.
- PHAL's `.gitignore` line 49 (`*.extxyz`) excludes them, and they were never committed.
  Only `tio2_anatase.cif` is tracked. No copy exists anywhere under the workspace.
- Affected: `test_scf_cost_aware_targets.py` (5 errors), `test_scf_dataset_adapter.py`
  (7 errors), and `test_structure_tasks.py::test_the_shipped_plan_and_templates_agree` (1 failure).
  Six of the seven skips also need generated SCF datasets that are not in the tree.

Migration consequence: these structure sets (`small_molecules`, 17 closed-shell
molecules, plus `periodic`) cannot be migrated from the reference tree. They must
be recovered from the original author's machine or regenerated, and documented
as such. GMD_active must not ignore `*.extxyz` globally for its own data directories.

## Reference outputs (golden fixtures)

PHAL's own MD fixture is a single 8-atom frame, which cannot exercise temporal metrics,
event buffering, or cooldowns. `tools/migration/capture_phal_reference.py` therefore drives
PHAL's `OnlineMonitor` over three deterministic synthetic trajectories. Committee forces are
precomputed and stored in the input file. The tool then records monitor results, events, regions
in all four extraction modes (raw and after PHAL's labeling-time H capping), masks, region codes,
and datasets under `tests/fixtures/phal_reference/` (124 files, 5.2 MB).

| Scenario | System | Exercises |
|---|---|---|
| `ethane_box` | 216 atoms, periodic | force bursts, close contact, overlapping events, a boundary-crossing jump, scheduled full frames |
| `polyethylene_chains` | 324 atoms, periodic chains | H capping across cut C–C bonds, an event truncated at end of run |
| `pt111_co_slab` | 66 atoms, `pbc=(T,T,F)` | adsorbate events, a metal surface (capping skipped as `metal_system`) |

The capture is deterministic: a second run in a different directory produced byte-identical files.
The only patch is PHAL's `uuid4` event-ID suffix, replaced by a counter.
`tests/migration/test_phal_reference_fixtures.py` checks the files against the manifest and for
internal consistency, and re-captures while `ref/` exists. The fixtures intentionally record PHAL defects
F-01 (LJ sign), F-04 (block + caps dataset crash) and F-07 (unmarked truncated events). Parity
tests must assert the corrected behavior for those (see `findings.md`).
