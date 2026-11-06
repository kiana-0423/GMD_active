# GMD_active tests

## Running

```bash
bash tools/test/run_all_offline.sh                    # the complete offline suite (what CI runs)
bash tools/test/run_all_offline.sh -x tests/monitors  # extra arguments go to pytest
GMD_AL_RUN_INTEGRATION=1 bash tools/test/run_all_offline.sh   # + integration / cp2k tests
python -m pytest -p no:cacheprovider -q tests/contracts       # one directory (pyproject puts src/ on sys.path)
```

The runner byte-compiles `src tests examples tools` and runs pytest with
`-m "not integration and not cp2k and not slow"`. Tests that need a program which is not installed
are **skipped with a reason** (`-rs` shows them); they never pass vacuously and never fall back to a
fake backend.

## Markers

| Marker | Meaning |
|---|---|
| `integration` | needs real external programs (GMD binary, GMD_SGT) |
| `cp2k` | needs a real CP2K executable |
| `slow` | long-running |

Many tests that use a real program are not marked but skip themselves when it is absent (for
example the GMD_SGT compatibility tests and the CP2K fixture regeneration checks).

## Real programs

| Variable | Used for | Default when unset |
|---|---|---|
| `GMD_EXECUTABLE` | GMD built with `-DGMD_ENABLE_TORCH=ON` (adapter, closed loop, rollback, example 06 live mode, container validators) | `../GMD/build-torch/gmd` |
| `GMD_SGT_PATH` | GMD_SGT >= 0.4 checkout (training, export, inference, dataset compatibility, smoke chain) | `../GMD_SGT`, then an installed `gmd_sgt` |
| `CP2K_EXECUTABLE` | CP2K for labeling and the real closed loop | `cp2k.psmp` on `PATH` |
| `CP2K_BIN` / `GMD_SCF_CP2K` | CP2K for the SCF tooling tests (`tests/scf`) | `cp2k.psmp` on `PATH` |

Example (from the repository root, sibling checkouts of GMD and GMD_SGT):

```bash
export GMD_EXECUTABLE=$PWD/../GMD/build-torch/gmd GMD_SGT_PATH=$PWD/../GMD_SGT CP2K_EXECUTABLE=$(which cp2k.psmp)
GMD_AL_RUN_INTEGRATION=1 bash tools/test/run_all_offline.sh -rs
```

Docker, Apptainer and PBS are never available to the suite: container and PBS tests are static
checks plus host-level runs of the validators (`tests/tools/test_containers.py`).

## Layout

| Directory | What it covers |
|---|---|
| `contracts/` | typed contracts (`Frame`, regions, labels, artifacts, deterministic IDs) |
| `architecture/` | executable layering rules, recursive import smoke test, exception hierarchy |
| `config/` | central config: strict sections, packaged default, PHAL config conversion |
| `monitors/` | staged OOD monitor, metrics, online driver, PHAL fixture parity |
| `events/` | event buffer and event store |
| `hotspots/` | hotspot clustering |
| `extraction/` | cluster/slab/graph/block extraction, H capping, embedding data, PHAL parity |
| `datasets/` | samples, extxyz/NPZ I/O, versioned store, GMD_SGT compatibility |
| `active_learning/` | candidate fingerprints, deduplication and selection |
| `labeling/` | labeling service, task store, retry policy, PHAL parse parity |
| `adapters/` | GMD, GMD_SGT, CP2K, schedulers, backend registry |
| `workflow/` | persistent controller, SQLite store, assembly, recovery |
| `integration/` | real GMD / GMD_SGT / CP2K runs (closed loop, rollback, species checks); skipped without the programs |
| `e2e/` | end-to-end runs with fake backends from `fakes/` |
| `scf/` | experimental CP2K ML-SCF tooling (torch-optional) |
| `importers/` | legacy LAMMPS dump import and monitoring of imported trajectories |
| `examples/` | every `examples/0N_*.py` `main()` on bundled data |
| `tools/` | containers, PBS templates, audit, benchmark, smoke chain, runtime tree, config reference, CI files |
| `utils/` | periodic geometry, bonds, atomic I/O, electronic state |
| `migration/` | feature ledger, inventory and golden-fixture integrity |
| `reference/` | NumPy reference implementations used as oracles (masked force loss) |
| `fakes/` | test-only fake backends (never used by production code) |
| `fixtures/` | golden PHAL outputs (immutable), real CP2K 2026.2 outputs, GMD frames, tiny GMD_SGT models |

Top-level `test_*.py` files cover the CLI, logging/YAML utilities and the original GMD_active
reliability monitor and LJ projection.

## What passing means

Passing the offline suite does **not** mean the production loop works on your system: the GMD,
GMD_SGT and CP2K paths are proven only by the integration tests, and container images and PBS
jobs are not exercised at all (see `docs/known-issues.md`). Golden fixtures under
`tests/fixtures/phal_reference/` are historical PHAL evidence and must never be edited; where PHAL
was wrong, tests assert the corrected behaviour.

Generated files have freshness tests: `docs/config-reference.md`
(`tools/docs/render_config_reference.py`), `containers/config/runtime.yaml`
(`tools/runtime/init_runtime_tree.py`), `configs/default.yaml` and the migration ledger. After
changing a config model, regenerate them.
