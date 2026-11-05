# DFT labeling (CP2K), schedulers and backend registry

This page describes how GMD_active turns an `ExtractedRegion` into a validated `LabelResult`.

## Layers

| Layer | Module | Responsibility |
|---|---|---|
| Registry | `adapters/registry.py` | Builds the backend for each role (`md`, `mlip`, `dft`, `scheduler`) from the `backends` section. Also handles plugins and capability checks. |
| CP2K adapter | `adapters/dft/cp2k/` | Writes inputs, builds the command line, parses output, validates labels and classifies failures. |
| Schedulers | `adapters/schedulers/` | `local`, `slurm` and `pbs`, all behind one protocol. |
| Labeling service | `labeling/service.py` | Owns the persistent task state machine. The workflow calls it. |
| Retry policy | `labeling/retry_policy.py` | Decides whether a failed attempt is retried, with bounded steps for each failure class. |

## Workflow API

```python
from gmd_active_learning.labeling.service import LabelingService

service = LabelingService.from_config(config, base_dir=run_dir)  # backends.dft / backends.scheduler
task = service.prepare(region)      # -> LabelTask; idempotent (same region + settings -> same task)
task = service.submit(task)         # no-op once a job exists
task = service.poll(task)           # advances stages and retries; never resubmits on UNKNOWN
result = service.collect(task)      # -> LabelResult (VALID or INVALID with reasons) once finished
service.reconcile(task)             # scheduler + work-dir evidence, never resubmits
service.resolve_unknown(task, reason=...)  # workflow's explicit decision that a job is lost
service.cancel(task)
```

The task ID is `stable_id("label", region_id, region_fingerprint, settings_hash)`.

- The region fingerprint covers numbers, positions, cell, PBC, roles, charge, multiplicity and the
  capping-skipped flag.
- The settings hash covers only the settings that change the calculation. Moving a campaign to
  another machine keeps its task IDs, because the executable, launcher, data directory, resources
  and retry settings are not part of the hash.

Each task lives in `<cp2k.task_dir>/<task_id>/`:

```
task.json            atomic; LabelTask + scheduler job + attempts + failure reasons
region.json          the region as prepared
region_labeled.json  the region with relaxed H caps (when a cap GEO_OPT ran)
result.json          final values (eV, eV/A, eV/A^3) and provenance
attempts/NN-<stage>/ one directory per job: input, output, job wrapper, .gmd_exit_code
```

A `.submit_intent` marker is created before each job submission. If a submission was interrupted,
or its outcome is unknown (for example the submit command timed out), the task becomes `UNKNOWN`
and is never submitted a second time.

## Execution status and validation status

`LabelTask.execution` records what the scheduler or process did. `LabelTask.validation` records
whether the output is a valid label. The two are independent: a job that exits 0 can still produce
an `INVALID` label.

A label is `VALID` when all of the following hold:

- the job completed;
- an SCF convergence marker is present;
- there is no `SCF run NOT converged` or `ABORT` marker;
- the energy is present;
- energy, forces and stress are finite;
- the number of force rows equals the region's atom count;
- CP2K printed `PROGRAM ENDED`.

This is PHAL's rule (finding F-27), extended with the atom-count, energy and termination checks.
`cp2k.validation` can relax the GMD additions.

## Stages

1. **Cap-only GEO_OPT (`hopt`).** This stage runs when `cp2k.h_only_opt.enabled` is true and the
   region contains H caps. Every non-cap atom is fixed with `FIXED_ATOMS`. The service reads the last
   frame of `hopt-pos-1.xyz` and maps it back to the region frame. It then checks that the fixed
   atoms did not move (`fixed_atom_tolerance`) and writes `region_labeled.json`. This fixes finding
   F-17: PHAL wrote this input but never ran it.
2. **Single point (`sp`).** `ENERGY_FORCE` runs on the relaxed geometry. The positions used for the
   label are returned in `LabelResult.metadata["positions"]`.

## Boundary conditions

The boundary conditions always come from `region.pbc`.

| Region | `&CELL` / `&POISSON` | Default solver |
|---|---|---|
| Cluster (no PBC) | `PERIODIC NONE` | WAVELET; the region gets a cubic box padded by `cluster_padding` |
| Slab (e.g. TTF) | `PERIODIC XY` | MT |
| Block (TTT) | `PERIODIC XYZ` | PERIODIC, with the analytic stress tensor (`stress: auto`) |

- If the configured solver does not support the region's periodicity, input generation raises
  `UnsupportedCombinationError`. The boundary conditions are never changed to fit the solver.
- CP2K 2026.2 aborts the WAVELET solver on non-cubic cells. PHAL's padded cluster cells were not
  cubic, so isolated regions now get a cubic box when WAVELET is used.
- Non-periodic directions of a slab must keep at least `min_vacuum` to the cell faces.

## Fail-closed checks

These checks run when the task is prepared, before anything is submitted:

- **Point-charge embedding.** If `embedding.enabled` is true, or the region carries embedding
  charges, input generation raises `UnsupportedCombinationError`. This is a known issue: the
  generator writes a pure Quickstep input. `adapters/dft/cp2k/point_charges.py` can write the charge
  table for a separately validated QM/MM template.
- **Uncapped isolated fragment.** If a region has `metadata.h_capping_skipped` and no PBC, the
  outcome depends on `cp2k.uncapped_fragment_policy`:
  - `error` (default) raises;
  - `warn` logs a warning;
  - `allow` proceeds.
- **Charge and multiplicity.** These come from the region and are resolved with
  `utils.electronic_state`. An open-shell region gets `UKS`. With `spin_policy: strict`, an
  impossible multiplicity raises.

## Retry ladder

Retries are configured in `cp2k.retry`. Each failure class has its own list of `RetryStep`
overrides. Every step is applied to the base settings, never to a previous retry input, so retries
cannot compound. The total number of retries is bounded by `max_retries`, and `max_scf_cap` limits
`MAX_SCF`.

| Failure class | Default ladder |
|---|---|
| `scf_not_converged` | 1. Diagonalization with Broyden mixing `NBUFFER 4`. CP2K ignores `&MIXING` under OT, so this step switches OT off. 2. OT with `MAX_SCF` × 3. |
| `geo_opt_not_converged` | `max_iter` × 2 |
| `walltime`, `execution_failed` | One plain resubmission |
| `invalid_result` | None; the failure is deterministic (for example a bad basis name) |
| `cancelled` | None |

When an ML-SCF attempt fails, the next attempt is resubmitted with native CP2K. This fixes finding
F-25: PHAL converted the input but never resubmitted it. The resubmission is recorded as
`trigger_class: ml_scf_failed`, and `LabelResult.metadata["ml_fallback_used"]` is set.

ML-SCF variants subclass `CP2KBackend` and override these members:

- `ml_scf_enabled`
- `variant_id()`
- `decorate_input()`
- `job_environment()`
- `native_backend()`

## Schedulers

Every scheduler implements the same protocol:

- `submit(work_dir, command, resources) -> SchedulerJob`
- `poll(job) -> ExecutionStatus`, which also fills `job.exit_code`
- `cancel(job)`
- `reconcile(job)`

Resources are typed by `SchedulerResources`: `nodes`, `ntasks`, `ntasks_per_node`, `cpus_per_task`,
`walltime`, `memory`, `partition`/`queue`, `account`, `job_name` and `extra_directives`. Slurm
renders them as `#SBATCH` lines. PBS renders them as `-l select=...` (PBS Pro) or
`-l nodes=..:ppn=..` (Torque). A directive with the other scheduler's prefix is rejected (finding
F-26).

Every job script records the command's exit code atomically in `.gmd_exit_code`. `reconcile` uses
this file when the scheduler no longer knows the job.

**Local scheduler.** The command runs as a detached session. The pid, process-group ID and start
time are stored in `.gmd_local_job.json`, so any controller can poll or cancel the job after a
restart. The local scheduler also enforces the walltime: it sends SIGTERM to the process group,
then SIGKILL after `cancel_grace_seconds`.

**Slurm.** Polling uses `squeue` and falls back to `sacct -P --format=State,ExitCode`. Every
documented state has an explicit mapping. `TIMEOUT` and `DEADLINE` map to `FAILED` with
`walltime_exceeded`.

**PBS.** Polling uses `qstat -f` and falls back to `qstat -x -f` (`history_query_enabled`).
Finished states `F`, `C` and `X` are checked against `Exit_status`. `C` without an exit status is
reported as `UNKNOWN`, not `COMPLETED`.

**Commands and errors.** Every scheduler command runs with `command_timeout_seconds`. How errors are
reported depends on whether the job could exist:

- A definite rejection raises `SubmissionRejected`, a subclass of `SchedulerError`. Examples are a
  non-zero exit from the submit command or a missing binary.
- A timeout while submitting raises `SchedulerError`, because the job may already exist.
- A failing status query leaves the job `UNKNOWN` and records `poll_error`.

`UNKNOWN` never triggers a resubmission.

## Registry and plugins

`create_backend(config, role)` reads `backends.<role>.engine` and builds the backend from the first
source that provides it, in this order:

1. a factory registered programmatically;
2. `backends.<role>.factory: "pkg.module:callable"`;
3. an entry point in the group `gmd_active_learning.adapters` named `"<role>:<name>"`;
4. a lazily imported built-in: `dft:cp2k`, `scheduler:local`, `scheduler:slurm` or
   `scheduler:pbs`.

Other areas register their own built-ins with `register_builtin(role, name, "module:attr")`.

Plugin failures are handled as follows (fixes finding F-31):

- When a plugin fails to import, the failure is stored under its normalized `role:name`. Requesting
  that backend raises `PluginLoadError` with the import error.
- A plugin can replace a built-in only when it is listed in `backends.override_builtins`.
  Otherwise the conflict is recorded and the built-in stays in place.

Capabilities are checked when a backend is created, through `required_capabilities`. When
`check_runtime_on_create` is true, the backend's runtime check also runs at creation.
`runtime_report(config)` lists the result of `check_runtime()` for every configured role and is
intended for a `doctor` command.

## Configuration sections

| Section | File | Model |
|---|---|---|
| `cp2k` | `configs/sections/cp2k.yaml` | `adapters.dft.cp2k.config.CP2KConfig` |
| `scheduler` | `configs/sections/scheduler.yaml` | `adapters.schedulers.config.SchedulerConfig` |
| `backends` | `configs/sections/backends.yaml` | `adapters.registry.BackendsConfig` |

These PHAL keys were removed:

| PHAL key | Replacement |
|---|---|
| `cp2k.periodic` | The region's PBC |
| `cp2k.charge` | `region.charge` |
| `cp2k.submit_mode` | `backends.scheduler.engine`; there is no dry-run mode |
| `cp2k.slurm_directives` | `cp2k.resources.extra_directives`, checked per scheduler |
| `cp2k.single_point.enabled` | Removed; the single point is the label |
| `pbs.history_query_enabled` | `scheduler.pbs.history_query_enabled` |

## Tests

| Location | What it covers | Needs |
|---|---|---|
| `tests/adapters/schedulers/` | Contract tests with fake `sbatch`/`squeue`/`sacct`/`scancel`/`qsub`/`qstat`/`qdel` on PATH, and real local subprocesses | Nothing external |
| `tests/labeling/` | The service, using a scripted fake CP2K | Nothing external |
| `tests/adapters/dft/test_cp2k_parse.py` | Parsing of real CP2K 2026.2 outputs in `tests/fixtures/cp2k/real_2026.2/` | Nothing external; fixtures regenerate with `python tests/fixtures/cp2k/generate_real_cp2k_fixtures.py` |
| `tests/adapters/dft/test_cp2k_real.py` | Real CP2K runs: H2, an H-capped CH3, a periodic and a slab H2, an unconverged run and its retry recovery, and the ML fallback | A CP2K binary; marker `cp2k`, skipped only when no binary is found |
