# Architecture

GMD_active is the active-learning controller of the GMD system. It decides **when** a molecular
dynamics trajectory has left the region a machine-learned potential can be trusted in, **what**
to label, and **whether** a retrained potential may replace the running one. It does not run
dynamics, train networks or solve electronic structure itself:

| Project | Owns | GMD_active talks to it through |
|---|---|---|
| **GMD** (C++) | molecular dynamics, TorchScript force evaluation, checkpoints, frame and status files, cooperative stop | `adapters/gmd/engine.py` (subprocess + files) |
| **GMD_SGT** (Python) | model architecture, training with per-atom force weights, fine-tuning, TorchScript export, inference | `adapters/sgt/backend.py` (Python API) |
| **CP2K** | DFT single points and cap-only geometry optimisations (labels) | `adapters/dft/cp2k/` + `adapters/schedulers/` |
| **GMD_active** | monitoring, events, hotspots, extraction, candidate selection, labeling state, datasets, model registry, promotion, the persistent loop | this package |

Rules that follow (also in `AGENTS.md`): no active-learning logic in GMD or GMD_SGT; every
external program is reached through an adapter; every threshold comes from YAML; every feature
can be switched off in configuration.

GMD_active replaces PHAL (`hotspot_al`, LAMMPS + Allegro + CP2K). PHAL is a *feature* reference
for the components; its end-to-end loop never closed (finding F-24), so the persistent controller
is new work. See `docs/migration/` for the ledger, findings and evidence.

## Layers

```
 entry points     cli.py   config.py   config_phal.py   workflow/assembly.py (composition root)
                      |
 application      workflow/ (controller, protocols, SQLite store)    labeling/ (label service)
                      |  protocols only
 domain           monitors/ events/ hotspots/ extraction/ active_learning/ datasets/ retraining/
                      |
 contracts        contracts/ (Frame, MonitorResult, EventRecord, ExtractedRegion, LabelTask,
                              LabelResult, DatasetVersion, ModelArtifact, stable_id, content_hash)
                      |
 foundation       core/ (exceptions, constants)   utils/ (periodic MIC, geometry, bonds, atomic I/O, YAML, logging)

 infrastructure   adapters/ (gmd, sgt, dft/cp2k, dft/scf, schedulers, registry)  importers/ (legacy LAMMPS)
                  reliability_model/ (optional torch classifier)
```

Dependencies point downwards. `tests/architecture/test_layering.py` enforces this on every module:

- **contracts** import only `core`/`utils`.
- **domain** packages import no adapter, no labeling service, no entry point, and no backend
  (`torch`, `gmd_sgt`, `subprocess`, CP2K or LAMMPS tooling). They may use
  `workflow.store` and `workflow.protocols`, nothing else from `workflow`.
- **workflow** (controller, protocols, store) reaches adapters only through
  `workflow/assembly.py`.
- **adapters** never import the composition root or the CLI.
- Nothing in `src/` imports `hotspot_al` or reads `ref/`.
- Every package must be classified, so a new package cannot slip past the guard.

The guard resolves relative imports and `importlib.import_module("...")` targets; PHAL's guard did
neither, and exempted `utils`. A second check imports every domain package in a fresh interpreter
and asserts that no backend module was loaded. `tests/architecture/test_import_all.py` imports all
modules and proves that only the optional reliability classifier needs torch at import time; the
SCF tooling, the GMD_SGT adapter and the CLI load torch lazily.

## Contracts

All subsystems exchange the typed records of `gmd_active_learning.contracts`. Units are Å, eV,
eV/Å, eV/Å³ and fs. Atoms are identified by stable `Frame.atom_ids`, never by array position.
`None` and "not evaluated" are distinct from zero: a missing committee is `UNAVAILABLE`, never a
perfect agreement (F-06). Regions record parent atom IDs and integer image shifts, so every
extracted position can be traced back to the parent frame (F-09).

## Protocols and the composition root

`workflow/protocols.py` defines what the controller needs, not how it is provided:

| Protocol | Production implementation |
|---|---|
| `MDEngine` (`run_segment`, `completed_segment`, `read_frames`) | `adapters.gmd.engine.GMDEngine` |
| `FrameEnricher` (committee / reference forces) | GMD_SGT committee through `adapters.sgt.backend` |
| `FrameMonitor` (`process`, `drain_events`, `close`, state) | `OnlineMonitorDriver` (monitors + events + hotspots) via `DriverFrameMonitor` |
| `RegionExtractor` | `extraction.EventRegionExtractor` (extract, cap, embedding data) |
| `Labeler` (`prepare`, `find_submission`, `submit`, `poll`, `reconcile`, `collect`, `retry`, `cancel`) | `labeling.service.LabelingService` (CP2K + scheduler) via `ServiceLabeler` |
| `DatasetStore` | `datasets.store.VersionedDatasetStore` |
| `Trainer` (`train`, `evaluate`, `export`, `validate_export`, `engine_smoke_test`) | `adapters.sgt.backend.GMDSGTBackend` |

`workflow/assembly.py:build_controller(config, work_dir=...)` is the **only** place that knows the
concrete classes. Backends are chosen per role by the `backends` section through
`adapters/registry.py` (built-ins: md `gmd`, mlip `gmd_sgt`, dft `cp2k`, scheduler
`local`/`slurm`/`pbs`; plugins register with `register_builtin`). Tests replace any backend with a
fake from `tests/fakes/`; production code has no fake or dry-run fallback, and a missing program
raises `CapabilityUnavailableError`.

## The persistent loop

`workflow/controller.py:ActiveLearningController` is a state machine whose position is stored in
SQLite (`<work-dir>/workflow.sqlite`, `workflow/store.py`):

```
SAMPLING ──> EXTRACTION ──> SELECTION ──> SUBMISSION ──> POLLING ──> INGESTION ──> TRAINING_DECISION
   ^                                                       │                              │ no
   │                                                   WAITING (labels outstanding      ├──────> SAMPLING (next cycle)
   │                                                    past max_poll_wait_s;           │ yes
   │                                                    `gmd-al run` resumes)           v
   └──────────── PROMOTION <── EVALUATION <── TRAINING <── PUBLICATION
                 (gates; reject keeps the active model)
DONE after workflow.max_cycles; STOPPED on `gmd-al stop` or a monitor STOP.
```

- **SAMPLING.** GMD runs in segments of `workflow.segment_steps`. Each flushed frame passes
  through the frame enricher and the monitor while GMD runs; the monitor's `FrameAction`
  (`continue`/`save`/`checkpoint`/`stop`) is delivered back to GMD through its stop and
  checkpoint-request files (ADR 005). Events and monitor state are persisted at segment end.
- **EXTRACTION / SELECTION.** Each event's trigger frame is cut into regions; candidates are
  ranked, de-duplicated by fingerprint (ADR 004) and limited by per-cycle, per-event and total
  budgets. The labeling queue is bounded (`max_pending_tasks`, `backpressure`).
- **SUBMISSION / POLLING.** Label tasks are prepared deterministically, submitted once
  (`.submit_intent` marker), polled, reconciled after a crash and retried with bounded,
  failure-class-specific ladders. An `UNKNOWN` job is never resubmitted automatically
  (`gmd-al resolve-unknown`).
- **INGESTION.** Only `VALID` labels (F-27 rule plus atom count, energy and termination checks)
  become samples.
- **PUBLICATION / TRAINING / EVALUATION / PROMOTION.** Samples are published as an immutable,
  content-addressed `DatasetVersion` with an event-aware split; GMD_SGT trains (fresh or
  fine-tuned) with per-atom force weights; the candidate is evaluated, its TorchScript export is
  validated against the checkpoint and smoke-tested in GMD; promotion gates
  (`workflow.promotion`) decide. Rejection keeps the active model; `gmd-al rollback` restores a
  previous one.

Every stage is idempotent. The stage is written before work starts and every external action
follows intent → action → result, so a killed controller resumes where it stopped and reconciles
intents whose outcome is unknown instead of repeating them (ADR 006). Identifiers of events,
regions, tasks, samples, dataset versions and models are derived from content (ADR 007), so a
rerun maps onto the same records.

## Adapters

- **GMD** (`adapters/gmd/engine.py`): writes `run.in` and the xyz input, starts GMD, tails the
  flushed `gmd-frame-v1` extxyz frames (`FrameTail`, `GMDFrameSource`), checks the model hash on
  every frame, stops cooperatively, and restarts segments from GMD checkpoint v3 (which carries
  atomic numbers, fixing F-02). The TorchScript force path is serial.
- **GMD_SGT** (`adapters/sgt/backend.py`): trains with `force_weight`/`energy_weight`/
  `stress_weight` (F-03 closed by GMD_SGT 0.4), fine-tunes, exports, validates exports, evaluates
  committees and single structures; species are checked before evaluation.
- **CP2K** (`adapters/dft/cp2k/`): input generation from the region's PBC (never changed to fit a
  solver), parsing, validation and failure classification; `labeling/service.py` owns the task
  state machine. Schedulers (`adapters/schedulers/`) share one protocol with typed resources and
  command timeouts.
- **SCF tooling** (`adapters/dft/scf/`, `tools/scf/`): experimental CP2K ML-SCF wrapper and
  measurement tools, optional and torch-optional; no acceleration benefit has been shown (F-21).
- **Legacy import** (`importers/lammps_dump.py`): converts PHAL/LAMMPS custom dumps into `Frame`s
  so historical trajectories can be monitored; never used by the live loop.

## Configuration

One YAML file, one strict pydantic model per section (`config.py:SECTIONS`); unknown keys fail.
The full reference is generated from the models: [config-reference.md](config-reference.md).
`configs/default.yaml` holds every key at its default; `configs/example_run.yaml` is a working
example; overlays merge with `utils.yaml.merge_yaml_files`.

## What lives where

| Concern | Module | User documentation |
|---|---|---|
| online monitor, events, hotspots | `monitors/`, `events/`, `hotspots/` | [monitoring.md](monitoring.md) |
| extraction, capping, embedding data, datasets | `extraction/`, `datasets/` | [extraction_and_datasets.md](extraction_and_datasets.md) |
| labeling, schedulers, registry | `labeling/`, `adapters/dft/cp2k/`, `adapters/schedulers/`, `adapters/registry.py` | [labeling_and_scheduling.md](labeling_and_scheduling.md) |
| candidate selection | `active_learning/candidates.py` | [ADR 004](adr/004-distance-histogram-dedup.md) |
| model registry, promotion, retrain trigger | `retraining/` | this page |
| ML-SCF tooling | `adapters/dft/scf/`, `tools/scf/` | [scf/README.md](scf/README.md) |
| containers, HPC | `containers/`, `tools/hpc/` | [containers/README.md](containers/README.md) |
| runtime data layout | `tools/runtime/` | [runtime-data-contract.md](runtime-data-contract.md) |
| operations tools | `tools/audit/`, `tools/benchmarks/`, `tools/smoke/` | [operations.md](operations.md) |

## Differences from PHAL's architecture document

PHAL described four layers with retries in the application layer, but its retry and lifecycle
code lived in the infrastructure package `cp2k/cp2k_task_submitter.py`; here the label lifecycle is
the application-level `labeling/service.py` and the CP2K adapter only classifies failures. PHAL's
`MDBackend` frame reading was unused in production (online mode used `LAMMPSController`); here the
controller consumes frames only through `MDEngine`. PHAL's `plugins.*` configuration was read by
no built-in; plugins now register backends explicitly and the registry reports import errors as
such (F-31).
