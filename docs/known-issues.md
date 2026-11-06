# Known issues

Ported from PHAL `docs/known-issues.md` (commit 96263b6) and updated for GMD_active. Each PHAL
issue keeps its number and states what is true now. Defect IDs (F-NN) refer to
[migration/findings.md](migration/findings.md); its resolution log lists the evidence for every
fixed finding.

Status words: **open** (needs work or a scientific decision), **fail-closed** (unsafe use is
refused by code and tests), **fixed**, **obsolete** (the PHAL component no longer exists).

---

## PHAL issues

### 1. Uncapped metal/oxide fragments require a validated treatment

**Status:** fail-closed; the scientific decision is still open.

H capping is skipped for metals and oxides (`capping` section; recorded as
`h_capping_skipped` / `h_capping_skipped_reason`). An isolated (non-periodic) calculation of such a
fragment would put dangling bonds into the reference forces. Two independent gates refuse it by
default: `extraction.check_labelable` (`capping.uncapped_policy: error`) and the CP2K input
generator (`cp2k.uncapped_fragment_policy: error`). Partial capping is now tagged
(`metadata.h_capping.partial`) and follows `capping.partial_policy`. A periodic region (slab, block)
passes. Example 03 exercises the gate (`tests/examples`).

**To close:** choose a treatment per system class (embedding, periodic cell, or a capping scheme
valid for metals/oxides) and validate the forces against a reference calculation.

### 2. CP2K QM/MM point-charge embedding is not implemented

**Status:** fail-closed.

`extraction.build_embedding` prepares point charges in the region frame, and
`adapters/dft/cp2k/point_charges.py` can write the charge table, but the CP2K input generator
writes pure Quickstep. With `embedding.enabled: true` (or charges on the region) task preparation
raises `UnsupportedCombinationError` (`require_embedding_backend`, `check_embedding`).
`EMBEDDING_CAPABILITIES` declares `qmmm_execution: {cp2k: false}`, no periodic images and no
charge neutralisation.

**To close:** a validated QM/MM template that consumes the charges, checked against a reference.

### 3. Allegro `atom_types` ordering was reasoned, not measured

**Status:** obsolete (Allegro/NequIP and LAMMPS `pair_allegro` are gone).

The equivalent risk for GMD_SGT is closed by construction and by test: GMD_SGT exports carry
their species list in metadata; the GMD_SGT adapter checks species before every evaluation; the
GMD adapter refuses to launch when the initial structure contains a species the model does not
know (GMD would silently map it to Z=0, the padding index); every GMD frame carries the model hash
and is checked against the requested model. The export is validated against the checkpoint before
promotion, and a GMD smoke run is a promotion gate. Real-program evidence:
`tests/integration/test_gmd_species_real.py`, `tests/integration/test_sgt_inference_real.py`.

### 4. CP2K's own Pulay solve fails on small ionic crystals

**Status:** open (upstream CP2K behaviour); unchanged.

MgO, NaCl and LiH two-atom rocksalt cells abort in `pulay_mixing` with
`Bad condition number R_COND = 7.1E-17` under `NBUFFER 8`, with unmodified CP2K as well. They are
excluded from the Pulay dataset with the reason recorded. Only the experimental SCF tooling is
affected; conventional labeling of extracted regions does not use these cells. See
[scf/pulay_dataset.md](scf/pulay_dataset.md).

### 5. The Pulay export check loses digits on near-singular B

**Status:** open, measured, within tolerance; unchanged (worst 1.25e-7 on MnO against a 1e-6
tolerance). If a harder system crosses the tolerance, compare against CP2K's own solve instead of
relaxing it. See [scf/pulay_dataset.md](scf/pulay_dataset.md).

### 6. `SCF_GUESS ML` could not reach the deployed model

**Status:** fixed in PHAL and carried over; the ML-SCF subsystem remains experimental and has shown
no acceleration benefit (F-21, [scf/negative_results.md](scf/negative_results.md)). The CP2K fork
implements only part of PHAL's contract (F-22); `adapters/dft/scf/capabilities.py` probes and
reports it honestly instead of assuming it.

---

## Found during the migration

### F-33: LJ fit applied an unscaled ridge penalty

**Status:** fixed (`monitors/lj_projection.py`; ridge on column-normalised parameters, so
`ridge_lambda` is dimensionless). Regression test
`tests/test_lj_projection.py::test_configured_ridge_does_not_bias_a_well_posed_fit`.

### F-34: Minimum image wrong for skewed cells with partial PBC

**Status:** fixed in GMD_active (`utils/periodic.py` projects onto the periodic sub-lattice). PHAL
itself is unchanged; its golden fixtures do not contain such cells. Regression test
`tests/utils/test_periodic.py`. Audit consequence: displacement and contact metrics of historical
PHAL runs on skewed, partially periodic cells (e.g. tilted slabs) may be wrong; re-monitor them by
importing the dumps (`gmd_active_learning.importers`, example 01).

### F-35: CP2K 2026.2 aborts WAVELET on non-cubic cells

**Status:** fixed. Found during labeling with the real CP2K 2026.2: the WAVELET Poisson solver
aborts with "Poisson solver for non cubic cells not yet implemented", and PHAL's padded cluster
cells were not cubic, so isolated-region labels could not run. Isolated regions now get a cubic
box padded by `cp2k.cluster_padding` when WAVELET is selected. Evidence: `tests/adapters/dft/`,
real CP2K fixtures in `tests/fixtures/cp2k/real_2026.2/`, and `tests/examples` (example 03 writes a
cubic WAVELET box).

---

## Open items in GMD_active

- **Container images were never built or run.** Docker is not available in the development
  environment and there is no access to the target site (Miyabi, GH200, PBS Pro, Apptainer).
  `containers/versions.yaml` records every stack as `not_built` / `not_validated` with null digests
  and hashes; the Dockerfiles, `build.sh --dry-run`, the validators and the PBS templates are
  tested statically and the validators ran on the host only. See [containers/README.md](containers/README.md).
- **PBS validation jobs were never submitted.** `tools/hpc/pbs/*.pbs` compute the SIF digest at job
  time (the hard-coded hash of F-32 is gone), but they ran only with stub `module`/`apptainer`.
- **ML-SCF is not selectable through the backend registry.** `containers/config/cp2k_ml_scf.yaml`
  validates and configures `scf.ml_scf`, but no `dft` backend named `cp2k_ml_scf` is registered in
  `adapters/registry.py`, so `backends.dft.engine: cp2k_ml_scf` is rejected. Conventional CP2K
  labeling is unaffected.
- **Relative paths in programmatic `GMDConfig`.** `load_config` resolves `md.executable` and
  `md.initial_structure` against the config file, but a `GMDConfig` built in code with a relative
  executable fails once GMD runs inside its work directory (example 06 resolves its paths first).
- **`Frame.from_atoms` falls back to row numbers as atom IDs** when an `Atoms` object has neither an
  `atom_id` nor an `id` array, so IDs are stable across frames only if the file order is. GMD frames
  and imported LAMMPS dumps always carry IDs.
- **Absolute paths in registry and dataset records.** `ModelRegistry` and `VersionedDatasetStore`
  record absolute paths, so moving a run directory breaks their own verification;
  `tools/audit/audit_artifacts.py` falls back to paths relative to the record file.
- **Trigger reason vocabulary.** `MonitorResult.trigger_reasons` uses PHAL's reason names
  (`force_large`, `force_burst`, `coordination_change`, `lj_residual`, ...), not metric names as
  `docs/monitoring.md` suggests.
- **Lint and type baselines.** CI runs a conservative ruff gate (syntax errors and undefined names)
  and an advisory mypy step on `contracts` and `utils`; no project-wide ruff or mypy configuration
  exists yet (pyproject is shared; see the operations ledger record OPS-005/006).
- **F-20 (documentation):** PHAL's ADR-003 described a 2×2 Gram solve where the code does a 3×2
  least squares; GMD_active's [monitoring.md](monitoring.md) documents the 3×2 solve. ADR-003 itself
  was not ported.

## Deliberate behaviour (not defects)

- **Graph extraction adds a radial shell around the subgraph** by default; set
  `extraction.graph.radial_shell: false` for exactly the bonded subgraph.
- **Block extraction raises on `max_atoms` but not on `min_atoms`**: `extraction.block.min_atoms_policy`
  (`warn` default, `error`, `allow`).
- **An `UNKNOWN` labeling job is never resubmitted automatically.** Lost jobs are declared failed
  only by an explicit `gmd-al resolve-unknown` after manual reconciliation.
- **A promotion rejection keeps the active model**; a rollback is an explicit, recorded action.
- **No fake or dry-run backends in production.** A missing GMD, GMD_SGT or CP2K raises
  `CapabilityUnavailableError`; fakes exist only under `tests/fakes/`.
