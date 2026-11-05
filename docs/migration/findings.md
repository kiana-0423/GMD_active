# Phase 0 findings: defects and parity hazards

Each finding is labeled with how it was established:

- **Reproduced:** demonstrated by running code, with the numbers given.
- **Code:** established by reading the cited lines.
- **Reported:** found by a Phase 0 inventory pass and not yet independently re-checked.

Paths are relative to `ref/lammps_monitor/` for PHAL and to `src/gmd_active_learning/` for GMD_active,
unless stated otherwise.

The migrated implementation **fixes** defects deliberately. Where that changes PHAL's observable
output, the corresponding parity test must assert the intended deviation rather than reproduce the
defect. The ledger entry's `notes` should reference the finding ID.

## Critical

### F-01: LJ force basis has the wrong sign in both PHAL and GMD_active (Reproduced)

Both implementations fit the 12-6 Lennard-Jones basis to the **negated** forces.

For V(r) = A/r¹² − B/r⁶ and d = rⱼ − rᵢ, the force on atom i is −12A·d/r¹⁴ + 6B·d/r⁸.
- GMD_active `monitors/lj_projection.py:80-85` adds +12·d/r¹⁴ and −6·d/r⁸ to atom i's row.
- PHAL `monitor/lj_residual.py:48-53` and `lj_residual_fast.py` use 24ε(2σ¹²/r¹⁴ − σ⁶/r⁸) times a displacement that points from i to its neighbor.

Test: an 18-atom Ar cluster with ASE `LennardJones` forces (verified as −dE/dx by finite difference):

| Implementation | Exact LJ forces | Negated forces |
|---|---|---|
| GMD_active `LJProjectionFitter` | A, B = −1.000 × truth; `is_physical=False` | exact; physical |
| PHAL `compute_lj_residuals` | residual 1.0; 0/18 atoms valid | residual 1e-15; 18/18 valid |
| PHAL `compute_lj_residuals_fast` | residual 1.0; 0/18 atoms valid | residual 1e-15; 18/18 valid |

Both test suites construct their synthetic forces with the same inverted sign, so they pass.
Consequences:
- The LJ monitor flags physically correct LJ-like interactions and accepts their mirror image.
- In GMD_active, every frame with LJ-like forces is non-physical, so it becomes a candidate.
- In PHAL's online path, every fitted atom on an escalated frame gains the LJ weight (2.0), so any light trigger that escalates also crosses the label threshold (Reported).

**Migration:** fix the sign. The PHAL golden fixtures record the inverted behavior. Parity tests for `lj_residual` should compare the migrated code on **negated** input forces against the fixtures (an equivalence check of the math), plus a correctness test on ASE LJ forces.

### F-02: GMD ML restart loses atomic numbers (Reproduced by the GMD survey)

`GMD/src/io/checkpoint.cpp` does not store Z, and `System::resize` zeroes it. After `restart_from`, every atom is fed to the model as species 0, which is GMD_SGT's padding index.
- Same model, step 10: −0.840180 eV in the continuous run vs −1.223712 eV after restart.
- Python re-evaluation reproduces both values (correct Z vs Z = 0).

Restart also does not validate the force field or model.

**Migration:** this blocks segmented ML runs and model promotion via restart. Fix in GMD (see `integration_gaps.md`) before Phase 6.

### F-03: GMD_SGT ignores per-atom force weights (Reproduced by the GMD_SGT survey)

`force_weight` written to extxyz is dropped by the reader and collate, and the loss is unweighted.
Perturbing an H-cap atom's target force changes the loss by +6.0e-2.
A four-function extension gives an exactly zero response at weight 0 (see `integration_gaps.md`).

**Migration:** this is the Phase 1 feasibility gate. PHAL's own `training/masked_loss.py` / `loss_mask.py` is an unused numpy reference (Reported). PHAL never actually trained with masks.

### F-24: PHAL never collects labels in its own pipeline (Code)

`CP2KTaskSubmitter.poll_job` (`cp2k/cp2k_task_submitter.py:173`) is the only path that turns a
finished CP2K job into a dataset entry (`:280-281`). Nothing in `src/`, `scripts/`, or `examples/`
calls it. The online pipeline submits labeling tasks but never polls, ingests, merges, or retrains.
The label → dataset → retrain loop is closed only in tests.

PHAL also persists no workflow state. The job table, task queues and block cooldown live in memory,
and `task.json` carries no job ID or status. Task IDs embed a random suffix, and task directories
are overwritten, so a restart can neither resume nor avoid duplicate submissions (Reported).

**Migration:** PHAL is a feature reference for components, not for the end-to-end loop. The
persistent state machine (submission, reconciliation, ingestion, retraining) is new work in
GMD_active, not a port.

## High

### F-04: PHAL block regions with H caps crash the dataset write (Reproduced)

`add_h_caps` (`extraction/h_capping.py:160-175`) extends `region_labels` and `mask_weights` but not block mode's `metadata["atom_role"]`. `generate_region_labels` prefers `atom_role` for block regions (`training/mask_generator.py:45-49`). As a result, `region_code` has the pre-cap length and ASE's extxyz writer raises `ValueError: could not broadcast ...`.
- Reproduced on all 7 capped block regions in the golden fixtures (`dataset_error` recorded).
- In PHAL's labeling path, `CP2KTaskSubmitter` caps block regions (`cp2k/cp2k_task_submitter.py:369`) and writes them at ingestion (`:280-281` → `datasets/writer.py:43`). The failure therefore occurs **after** the CP2K calculation has run (Code).

### F-05: PHAL's labeling path ignores slab and graph extraction modes (Code)

`cp2k/cp2k_task_submitter.py:352-365` runs block extraction for `mode: block` and cluster extraction for anything else. `active_learning/workflow.py:extract_regions_for_result` supports all four modes. So the configured `slab`/`graph` mode governs candidate extraction but not what is actually labeled.

### F-06: An unavailable committee is scored as zero deviation (Code)

`monitor/ood_score.py:279` substitutes `np.zeros_like(force)` when no committee result exists. A single-model run is indistinguishable from perfect committee agreement.

**Migration:** the result contract must distinguish not-evaluated from measured. The proposal requires this.

### F-07: Truncated events are not marked (Reproduced)

When the run ends inside an event's post-trigger window, the flushed event simply has fewer post frames.
- Golden fixture `polyethylene_chains`, event `evt-90-*`: 3 post frames configured, 2 recorded.
- There is no completeness flag. `RollingBuffer.events` also grows without bound for the whole run (Reported).

### F-08: Slab extraction builds a multi-period cell holding one copy of the atoms (Reported, reproduced by the inventory pass)

When the requested window is at least 1.5 cell periods, the output cell spans several periods but contains a single image (`extraction/slab_extractor.py:41-44,80-85`). Example: a 15 Å periodic cell holding a 4 Å strip.

### F-09: No region records periodic image shifts (Reported)

All extraction modes place each atom once at its nearest image relative to a centroid. No image shifts are recorded, and nothing prevents a region wider than half the cell. LAMMPS atom IDs are not carried; indices are 0-based frame positions.

**Migration:** the region contract must record image shifts and stable atom IDs (proposal requirement).

### F-10: GMD_active workflow and adapters are placeholders (Code)

| Defect | Location |
|---|---|
| Subprocess MD returns no frames, and monitoring runs after MD with no callback | `adapters/gmd_adapter.py:30-32`; `active_learning/workflow.py:62` |
| Resume restores the iteration counter but not the model: `current_model` comes from config, although it is saved | `active_learning/workflow.py:58,104` |
| `predict()` returns zeros unconditionally; trained/exported paths are assumed (`model.pt`, `exported_model.pt`), but GMD_SGT writes `ckpt_best.pt` and `model.pt` | `adapters/gmd_se3gnn_adapter.py:23,31-34,39` |
| Labels are never ingested; the "merge" writes a two-line manifest including the literal string `existing_dataset` | `active_learning/workflow.py:88-90`; `retraining/dataset_merger.py` |
| `gmd-al run --config` is parsed but never used; `import-labels` is a print placeholder; `train-reliability-model` trains on random data without `--dataset` | `cli.py:35,79-81,87-93` |
| Model registry stores metadata only (a path reference), with no copy, checksum, or promotion state | `retraining/model_registry.py` |

## Medium

| ID | Finding | Basis |
|---|---|---|
| F-11 | PHAL data files `data/structures/pulay_sampling/{small_molecules,periodic}.extxyz` are referenced by configs, docs, scripts and tests but were never committed (`.gitignore:49` `*.extxyz`). This causes 13 failing tests. No copy exists in the workspace. | Reproduced |
| F-12 | PHAL writes no energy to training files. `energy_ev` survives only in metadata JSON, and `energy_weight` is always 0. The CP2K energy regex may not match CP2K v2026.2 output (`cp2k/cp2k_result.py:17`). | Reported |
| F-13 | `RetrainTrigger` fires on the first labeled sample regardless of `min_new_samples` (reason `initial_samples`, `training/retrain_trigger.py:149-150`). It advances its state even in dry-run (`:94`). | Code |
| F-14 | PHAL's model registry stores no checksums and silently overwrites a version registered twice. `RetrainTrigger` and the registry are not wired into any CLI. | Reported |
| F-15 | Event IDs embed `uuid4` (`buffer/rolling_buffer.py:92`), so event identity is not reproducible across reruns. This conflicts with idempotent ingestion. | Code |
| F-16 | LJ residual is computed twice on every escalated frame. `examples/06_online_monitor.py` never calls `start()` and hangs with real LAMMPS. `rmin_threshold` defaults to null, so close contacts contribute nothing. `coordination_scale` / `coordination_smoothing_power` are validated but never read. For the first 5 frames (warmup), raw values are compared against z-thresholds. Two hotspot thresholds coexist (`ood_score.label_threshold`, `hotspot.threshold`). | Reported |
| F-17 | The H-only cap relaxation input is written but never submitted (`cp2k/cp2k_task_submitter.py:144`), so caps are never relaxed. | Reported |
| F-18 | Block metadata's default `inner_buffer` weight is 0.2 (`extraction/block.py:543`), while the mask generator defaults to 0.3. | Reported |
| F-19 | PHAL's online coupling is one-way. `LAMMPSController` tails the dump with one frame of lag, and `None` means both "timeout" and "LAMMPS exited". Nothing stops, pauses or checkpoints MD. | Reported |
| F-20 | ADR-003 describes a 2×2 Gram solve; the code does a 3×2 least squares. | Reported |

### Orchestration, labeling and operations (Reported unless noted)

| ID | Finding |
|---|---|
| F-25 | CP2K retry ladder. A failed ML-accelerated run that falls back to native CP2K is never resubmitted (`cp2k/cp2k_task_submitter.py:409-410`). Each retry multiplies MAX_SCF again (100 → 200 → 600), whatever the failure cause. An unknown scheduler state while output is still being written can trigger a resubmission while the original job is running. |
| F-26 | Schedulers. Local, Slurm and PBS submit/poll/cancel are real code but tested only with mocked commands. There are no resource fields; only raw directive text, taken from `cp2k.slurm_directives` even for PBS. Failed `sbatch`/`qsub` raises an unhandled subprocess error, and no scheduler command has a timeout. PBS state `C` counts as completed whatever the exit status. `pbs.history_query_enabled` cannot be set, because the config schema rejects a `pbs` section. |
| F-27 | A label is valid when the scheduler did not report failure, the output exists, a convergence marker is present with no failure/ABORT marker, and energy, forces and stress parse as finite. A run that merely exits 0 is rejected. **Keep this rule.** |
| F-28 | Allegro inference has never run against a real model. It passes no neighbor list, calls a `nequip.deploy` API that does not match the validated nequip 0.18 stack, and turns any failure into NaN forces. Training and export are user command templates whose outputs are never located or checked. |
| F-29 | LAMMPS online control is file tailing plus signals (SIGSTOP/SIGCONT, SIGTERM→SIGKILL), with a one-frame completeness lag. The launch command has no MPI/Kokkos flags, but the validated GH200 run needs `-k on g 1 -sf kk`. Ten LAMMPS assumptions are listed in ledger entry MD-001. |
| F-30 | The candidate fingerprint becomes NaN when no atom pair is within 8 Å. Such a candidate is always dropped, and if it ranks first it blocks every later one. |
| F-31 | The plugin registry reports some plugin import errors as "unknown backend", and plugins can never replace a built-in backend. |
| F-32 | The train-probe Dockerfile copies a `config/` directory that no longer exists, so that image cannot build at `96263b6`. The PBS md-probe job writes a hard-coded SIF hash into its summary. `check_runtime()` is never called. |

## ML-SCF subsystem (experimental; in scope as optional tooling)

### F-21: No ML-SCF component has shown an acceleration benefit (Code: PHAL's own measurements)

| Component | PHAL's measured result | Source |
|---|---|---|
| ML density guess | 63 SCF steps vs 53 for the atomic guess (speedup 0.85); "worse than the atomic guess on every system" | `docs/ml_scf_architecture.md:130-141` |
| Density model v2 | "it did not help" (2 of 48 runs better) | `docs/ml_density_v2.md:26,64` |
| ML Pulay mixer | "Status: negative result. Experimental only — not a production backend." | `docs/ml_pulay_objective.md:3` |
| Adaptive controller | 172 = 172 steps; 12/12 labels "keep" | Reported |
| Strategy predictor | "does not work". The positive result needs no model: retry `default → nbuffer4` converges 12/12 vs 9/12 (+4.7% iterations). | `docs/scf_strategy_prediction.md:84-90` |

**Migration:** port the interfaces and benchmarks as optional tooling labeled experimental, carrying
these results verbatim. Do not describe ML-SCF as an acceleration feature. The deterministic NBUFFER
retry ladder is the one result worth promoting into the labeling path. PHAL's CP2K templates use NBUFFER 8;
CP2K's default is 4.

### F-22: The reconstructed CP2K fork implements only part of PHAL's contract (Reported)

`cp2k-ml-scf` (pinned v2026.2 @ `67b5da87`) is a clean-room rebuild of a lost fork:
- It implements only mode A: the `<input>.ml_density` exchange file, protocol `phal-ml-scf-v2`.
- Mode B (in-process TorchScript) is a placeholder.
- There is no SCF controller, Pulay recorder, or ML mixing.

PHAL's ML benchmark arms, controller runs, Pulay-dataset generation and diagnostics therefore cannot
run against it. The capability probe ignores `cp2k_ml_scf.environment` (`cp2k_ml_scf/cp2k_capabilities.py:246`).

### F-23: ML-SCF defects (Reported)

- **No overlap for `density_gnn`:** it can never deliver a guess, because the backend passes no overlap matrix (`backends/cp2k_ml_scf.py:209`) and the prediction never carries one.
- **Electron count mismatch:** validation compares against the all-electron count (`cp2k_ml_scf/density/density_predictor.py:190`).
- **Lossy save:** `SCFDensityArtifacts` loses five fields on save (`cp2k_ml_scf/schema.py:269-297`).
- **Missing checks and gates:** ML-mixing support is never checked, and the documented "retry once with plain CP2K" is not implemented. `pulay_mixer.export_mixer` bypasses the experimental gate.
- **Unread config keys:** `electron_tolerance`, `dataset_dir`, `fallback_method`, `runtime_dir`, `trajectory_strict`.
- **Coupling:** `backends/__init__.py:19` imports the SCF backend eagerly. Core labeling depends on `cp2k/scf_history.py` (`cp2k_task_submitter.py:268`). Everything else is separable as an optional extra, with torch loaded lazily.

## Found during implementation

### F-33: GMD_active LJ fit applies an unscaled ridge penalty (Reproduced)

`monitors/lj_projection.py` penalised the raw (A, B) vector. For Ar, A ≈ 9.9e4 and B ≈ 64, so at the
shipped `ridge_lambda: 1e-6` exact LJ forces fitted to A = 8.3e-4, B = −1.24 (non-physical, residual 0.23).
Fixed by applying the ridge to column-normalised parameters; λ is now dimensionless.
Regression test: `tests/test_lj_projection.py::test_configured_ridge_does_not_bias_a_well_posed_fit`.

### F-34: PHAL minimum image is wrong for skewed cells with partial PBC (Reproduced)

`utils/periodic.py:_triclinic_mic` rounds coordinates in the full 3-D (reduced) basis, so the
non-periodic cell vector leaks into the periodic coordinates. For the skewed cell
[[6,0,0],[5,3,0],[4,2.5,2]] with `pbc=(T,T,F)`, 155 of 200 random vectors returned a non-minimal
image (e.g. 14.70 A where 7.19 A exists). Full 3-D PBC was correct. Fixed in GMD_active by projecting
onto the periodic sub-lattice; regression test `tests/utils/test_periodic.py`.

### F-35: CP2K 2026.2 aborts the WAVELET Poisson solver on non-cubic cells (Reproduced, real CP2K)

PHAL padded isolated clusters into non-cubic boxes, so its default cluster inputs cannot run on
CP2K 2026.2 at all. GMD_active places isolated regions in a cubic box (labeling area;
`tests/adapters/dft/`, real-CP2K tests).

### F-36: GMD used stale periodic edge shifts between neighbour-list rebuilds (Reproduced, GMD)

An atom that crossed a box face between rebuilds kept the old image offset, so ML forces were
silently wrong until the next rebuild (1.3e-2 eV on a water molecule straddling a face). Fixed in
GMD commit `f333537` with a regression test; ML results shift at round-off level relative to earlier
GMD builds.

### F-37: Radial clusters cut covalent bonds in small molecular boxes (Reproduced, real run)

In the first real closed-loop run (4 water molecules, 7.5 A box) radial cluster extraction cut O-H
bonds that cannot be H-capped (the kept atom is H), yielding radical fragments whose SCF did not
converge. The bounded retry ladder made 3 attempts per task and marked every task INVALID; nothing
was ingested. This was a configuration/science issue, not a controller defect, but two real gaps
were fixed: hotspot atoms are now grouped per hotspot cluster (one region each, PHAL parity) and the
example uses bonded-graph extraction, which yields whole neutral molecules. The interrupted run was
also used as a real crash-recovery test (see the resolution log).

## Resolution log

| Finding | Resolution | Evidence |
|---|---|---|
| F-01 (GMD_active part) | Force basis sign corrected in `monitors/lj_projection.py`; self-confirming tests replaced by ASE-based tests with finite-difference checks. PHAL golden fixtures left unchanged as historical evidence. | `tests/test_lj_projection.py`, `tests/test_reliability_monitor.py` |
| F-33 | Ridge applied in normalised coordinates. | `tests/test_lj_projection.py` |
| F-34 | Sub-lattice MIC; GMD_active's legacy `minimum_image` now uses it. | `tests/utils/test_periodic.py` |
| F-01 (monitoring) | Local LJ residual sign-corrected; parity on negated forces, correctness on ASE LJ | `tests/monitors/` |
| F-02 | GMD checkpoint v3 stores Z; continuous vs segmented ML runs bitwise identical (jit optimise off) | GMD `tests/controller_integration.py`; `tests/integration/test_gmd_real.py` |
| F-03 | GMD_SGT per-atom `force_weight`; zero-weight perturbation changes neither loss nor gradients (exact) | GMD_SGT `tests/test_force_weights.py` |
| F-04 | Caps extend every per-atom array (`ExtractedRegion.with_caps`); capped block regions write datasets | `tests/extraction/`, `tests/contracts/` |
| F-05 | The labeler receives the region produced by the configured extraction mode; it never re-extracts | `tests/workflow/test_controller.py`, `tests/integration/test_closed_loop_real.py` |
| F-06 | Missing committee is `MetricStatus.UNAVAILABLE`, never zero | `tests/contracts/`, `tests/monitors/` |
| F-07 | `EventRecord.complete` / `truncation_reason`; bounded event memory | `tests/events/` |
| F-08 | `slab.multi_period` replicate/single_period/error | `tests/extraction/` |
| F-09 | Regions carry parent atom IDs and integer image shifts; periodic guard refuses torn regions by default | `tests/extraction/` |
| F-11 | Structures regenerated reproducibly with provenance (`docs/scf/structure_provenance.md`) | `tests/scf/` |
| F-12 | CP2K 2026.2 energy/stress parsing fixed on real output | `tests/adapters/dft/` |
| F-13 | Trigger no longer fires on the first sample; never while labels are pending | `tests/workflow/test_store_registry.py` |
| F-14 | Registry copies, hashes, verifies, gates promotion, records rollback | `tests/workflow/`, `tests/integration/test_rollback_real.py` |
| F-15 | Deterministic event IDs (`stable_id`) | `tests/events/` |
| F-16 | LJ fitted once per frame; warmup policy; `rmin_threshold` default; one hotspot threshold | `tests/monitors/` |
| F-17 | H-only cap relaxation actually runs before the single point | `tests/adapters/dft/`, `tests/labeling/` |
| F-21..F-23 | Negative results preserved verbatim; ML-SCF defects fixed; capability probe honest | `tests/scf/`, `docs/scf/` |
| F-24 | Persistent closed loop polls, validates, ingests, trains and promotes | `tests/workflow/`, `tests/integration/test_closed_loop_real.py` |
| F-25 | Native fallback resubmitted; retries bounded per failure class | `tests/labeling/`, real CP2K tests |
| F-26 | Typed resources, command timeouts, PBS exit status, per-scheduler directives | `tests/adapters/schedulers/` |
| F-30 | Finite fingerprints for any region | `tests/active_learning/test_candidates.py` |
| F-31 | Plugin import errors reported; explicit `override_builtins` | `tests/adapters/test_registry.py` |
| F-35 | Cubic box for isolated regions | `tests/adapters/dft/` |
| F-36 | Fixed in GMD | GMD CTest suite (51/51, Torch build) |
| F-37 | Per-hotspot regions; graph extraction for molecular systems in the example | `tests/workflow/test_assembly.py`, `tests/integration/test_closed_loop_real.py` |

