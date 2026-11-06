# GMD_active

GMD_active is the persistent active-learning controller of the GMD system. It watches a running
**GMD** molecular-dynamics simulation driven by a **GMD_SGT** machine-learned potential, detects
atoms whose environment the potential has not learned, cuts small regions around them, labels
those regions with **CP2K**, publishes versioned training data, retrains or fine-tunes the
potential, and promotes the new model into the running simulation only when explicit gates pass.

It is the successor to PHAL (`hotspot_al`, LAMMPS + Allegro + CP2K). PHAL's components were
ported and their confirmed defects fixed; PHAL's end-to-end loop never closed, so the persistent
loop is new.

> Dynamic active learning is for model development. Production MD should be re-run with a
> converged, fixed model.

## What it does

| Stage | What happens | Where |
|---|---|---|
| Sampling | GMD runs in segments; every flushed frame is monitored while GMD runs; STOP/CHECKPOINT decisions go back to GMD cooperatively | `adapters/gmd`, `monitors`, `events` |
| Monitoring | atom-wise staged OOD scores: forces, force bursts, displacements, close contacts, coordination change, local LJ residual (sign-corrected), MLIP and committee deviation | `monitors` ([docs](docs/monitoring.md)) |
| Events and hotspots | pre/post-trigger context, deterministic event IDs, truncation flags, spatial hotspot clusters | `events`, `hotspots` |
| Extraction | cluster, slab, graph or block regions with stable parent IDs and image shifts, H capping, periodic guard | `extraction` ([docs](docs/extraction_and_datasets.md)) |
| Selection | pair-distance-histogram diversity, score ranking, budgets, bounded queue | `active_learning` ([ADR 004](docs/adr/004-distance-histogram-dedup.md)) |
| Labeling | CP2K inputs from the region's PBC, cap-only GEO_OPT then single point, local/Slurm/PBS, validation, bounded retries, crash reconciliation | `labeling`, `adapters/dft/cp2k`, `adapters/schedulers` ([docs](docs/labeling_and_scheduling.md)) |
| Datasets | immutable, content-addressed versions with event-aware splits and GMD_SGT per-atom force weights | `datasets` |
| Training and promotion | GMD_SGT train/fine-tune/export, export validation, GMD smoke test, promotion gates, rollback | `adapters/sgt`, `retraining`, `workflow` |
| Persistence | one SQLite store per run; every stage resumable after a crash | `workflow` ([architecture](docs/architecture.md)) |

## Install

Python 3.10–3.12. From the repository root:

```bash
pip install -e ".[dev]"          # numpy, scipy, ase, pyyaml, pydantic, torch, tqdm (+ pytest)
```

The external programs are separate projects:

| Program | Needed for | Notes |
|---|---|---|
| GMD | MD | build with `-DGMD_ENABLE_TORCH=ON` (TorchScript force field); the ML path is serial |
| GMD_SGT >= 0.4 | training, export, inference | `pip install` it, or point `mlip.source_path` at a checkout |
| CP2K (tested with 2026.2) | labels | `cp2k.psmp`; `cp2k.data_dir` if the basis/potential files are not found |

`gmd-al doctor --config run.yaml` checks all of them and prints `[OK]`/`[MISS]` per check.

## Quick start

```bash
gmd-al validate-config --config configs/example_run.yaml
gmd-al doctor          --config configs/example_run.yaml
gmd-al init   --config configs/example_run.yaml --work-dir runs/water \
              --checkpoint ckpt_best.pt --export model.pt [--committee member2.pt]
gmd-al run    --config configs/example_run.yaml --work-dir runs/water    # resumes after interruption
gmd-al status --work-dir runs/water
gmd-al stop | retrain  --config configs/example_run.yaml --work-dir runs/water
gmd-al rollback        --config configs/example_run.yaml --work-dir runs/water --reason "..."
gmd-al resolve-unknown --config configs/example_run.yaml --work-dir runs/water --task TASK_ID --reason "..."
gmd-al convert-phal-config --help                                         # migrate a PHAL YAML
```

`configs/example_run.yaml` is a working example; `configs/default.yaml` lists every key at its
default; [docs/config-reference.md](docs/config-reference.md) documents every key (generated from
the models). Unknown keys are errors.

### A real closed cycle

`examples/real_cycle/run_real_cycle.py` runs the whole loop with the real programs on four water
molecules: GMD with a tiny GMD_SGT committee, monitoring, extraction, CP2K labels, dataset
publication, fine-tuning, export validation, a GMD smoke test, gated promotion and a resumed GMD
segment with the promoted model.

```bash
python examples/real_cycle/run_real_cycle.py --work-dir /tmp/gmd_al_cycle \
    --gmd ../GMD/build-torch/gmd --gmd-sgt ../GMD_SGT --cp2k-data /path/to/cp2k/data
```

It takes a few minutes. Recorded evidence, including an interrupted-and-resumed run, is in
[docs/migration/evidence/](docs/migration/evidence/README.md). It is a smoke-scale demonstration,
not a scientific validation.

### Smaller examples

`examples/01`–`06` are PHAL's examples ported to GMD_active and run on bundled data
(`examples/data`) without external programs:

| Example | Shows |
|---|---|
| `01_monitor_trajectory.py` | monitor GMD frames or a legacy LAMMPS dump |
| `02_extract_hotspots.py` | hotspot clustering, extraction and capping |
| `03_generate_cp2k_inputs.py` | CP2K inputs through the labeling gates |
| `04_parse_cp2k_forces.py` | parse and validate a real CP2K 2026.2 output |
| `05_write_sgt_dataset.py` | a labeled region published as a versioned GMD_SGT dataset |
| `06_online_monitor.py` | replay GMD frames, or monitor a live GMD run (`--live`) |

## Components

```
src/gmd_active_learning/
  contracts/        typed records shared by every layer (Frame, MonitorResult, EventRecord, ExtractedRegion, LabelTask, ...)
  monitors/ events/ hotspots/ extraction/ active_learning/ datasets/ retraining/    backend-neutral domain code
  workflow/         persistent controller, protocols, SQLite store, composition root (assembly.py)
  labeling/         persistent CP2K labeling service
  adapters/         gmd/, sgt/, dft/cp2k/, dft/scf/ (experimental ML-SCF), schedulers/, registry.py
  importers/        legacy LAMMPS dump import (re-monitor historical PHAL trajectories)
  config.py cli.py  central configuration and the gmd-al CLI
containers/         GMD, GMD_SGT and CP2K images, build.sh, versions.yaml, validators
tools/              audit, benchmarks, smoke checks, runtime tree, HPC (PBS) jobs, SCF tooling, docs generators, migration
docs/               architecture, topic guides, ADRs, containers, known issues, migration record
```

Layering is enforced by `tests/architecture/test_layering.py`: domain code never imports an
adapter, a backend or the composition root. More in [docs/architecture.md](docs/architecture.md);
operations tools in [docs/operations.md](docs/operations.md).

## Tests

```bash
bash tools/test/run_all_offline.sh                       # offline suite (CI: Python 3.10-3.12)
GMD_EXECUTABLE=... GMD_SGT_PATH=... CP2K_EXECUTABLE=... GMD_AL_RUN_INTEGRATION=1 bash tools/test/run_all_offline.sh
```

See [tests/README.md](tests/README.md) for suites, markers and the variables the integration tests
read.

## Limitations

- **Containers and HPC are unvalidated.** The Dockerfiles, `build.sh`, validators and PBS
  templates are tested statically and on the host; no image has been built and no PBS job
  submitted ([docs/containers/README.md](docs/containers/README.md)).
- **Real-program evidence is smoke-scale**: a tiny model on four water molecules, CPU, macOS arm64.
- **Uncapped metal/oxide fragments and QM/MM embedding are refused**, pending a scientific
  decision ([known issues](docs/known-issues.md) 1 and 2).
- **ML-SCF is experimental** and has shown no acceleration benefit (F-21).
- Monitoring statistics are pooled over species and never decay; the local LJ residual detects
  non-LJ-like force directions, not wrong pair strengths ([monitoring](docs/monitoring.md)).
- One controller per work directory.

Full list: [docs/known-issues.md](docs/known-issues.md). The migration from PHAL (feature ledger,
findings, evidence) is in [docs/migration/](docs/migration/README.md).

## Contributing

See [docs/contributing.md](docs/contributing.md) and `AGENTS.md`.

## Attribution

GMD_active contains code and data ported from PHAL (`hotspot_al`,
https://github.com/kiana-0423/lammps_monitor, commit 96263b6), MIT License; see [NOTICE](NOTICE).
Ported files carry a "Ported from PHAL" header. GMD, GMD_SGT and CP2K are separate projects with
their own licenses.
