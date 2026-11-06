# Operations tools

Command-line tools for running GMD_active on real systems. All run from the repository root with
the package installed (`pip install -e .`) or `PYTHONPATH=src`.

| Tool | Purpose | Ledger |
|---|---|---|
| `tools/audit/audit_artifacts.py` | audit PHAL data and GMD_active stores for corruption and integrity failures | OPS-001 |
| `tools/smoke/sgt_runtime_smoke.py` | torch / device / GMD_SGT runtime check with stable exit codes | OPS-002 |
| `tools/smoke/sgt_tiny_h2o/run_chain.py` | tiny GMD_SGT chain: synthetic H2O data, train, export, infer (CPU, seconds) | OPS-003 |
| `tools/hpc/pbs/*.pbs` | PBS validation jobs for the GMD, GMD_SGT and CP2K images (`tools/hpc/pbs/README.md`) | OPS-004 |
| `tools/runtime/init_runtime_tree.py` | runtime data tree and config overlay for any root ([contract](runtime-data-contract.md)) | OPS-008, CFG-004 |
| `tools/benchmarks/benchmark_monitor.py` | monitor kernel timings with a regression gate ([performance](performance.md)) | UTIL-008 |
| `tools/docs/render_config_reference.py` | regenerate [config-reference.md](config-reference.md) | DOC-002 |
| `tools/test/run_all_offline.sh` | the offline test suite (`tests/README.md`) | TST-007 |
| `containers/build.sh` | plan/build/push images and convert to SIF ([containers](containers/README.md)) | CTR-001 |

## Auditing artifacts

```bash
python tools/audit/audit_artifacts.py runs/water old_phal_data/ --json audit.json
```

Findings are grouped by cost: `structure` (DFT ran on the wrong structure), `label` (wrong
calculation), `integrity` (a GMD_active artifact no longer matches its record: registry checksum,
dataset tree hash, `task.json`/`result.json`), `selection` (the region choice was wrong but the
label is valid) and `unknown`. Exit status 1 on any `structure`, `label` or `integrity` finding.
The JSON report is always written. Run it before importing historical PHAL datasets into a GMD_SGT
training set, and on a run directory before trusting it after a disk incident.

## Runtime smoke checks

```bash
GMD_SGT_SMOKE_DEVICE=cpu python tools/smoke/sgt_runtime_smoke.py --json
python tools/smoke/sgt_tiny_h2o/run_chain.py --work-dir /tmp/sgt_chain --report chain.json
```

Exit codes: `0` ok, `1` runtime error, `2` import error (torch or GMD_SGT >= 0.4 missing), `3`
requested device unavailable, `4` wrong conda environment (`GMD_SGT_REQUIRED_CONDA_ENV`); the chain
adds `4` model path error, `5` model load failure, `6` inference failure, `7` element mismatch. The
chain trains on a toy harmonic H2O set (not DFT) and checks that the TorchScript export and the
checkpoint give finite, agreeing forces; it claims no accuracy.

## Importing legacy LAMMPS trajectories

```python
from gmd_active_learning.importers import iter_lammps_dump
from gmd_active_learning.monitors import OnlineMonitorDriver

frames = iter_lammps_dump("old_run.lammpstrj", type_map={1: "C", 2: "H"}, timestep_fs=0.5)
summary = OnlineMonitorDriver(config["monitor"], config["events"], config["hotspots"]).run(frames)
```

LAMMPS atom `id`s become stable `Frame.atom_ids`; positions follow PHAL's parser (origin shift,
triclinic bounds, scaled and unwrapped coordinates); only `p` boundaries are periodic. Use it to
re-monitor PHAL runs, e.g. those affected by finding F-34. Example 01 does this from the command
line.

## Packaging and repository hygiene (shared files)

`pyproject.toml` and `.gitignore` are maintained by the project lead. The operations port needs:

- **pyproject**: markers `gmd`, `gmd_sgt` and `pbs` if tests start using them (today `integration`,
  `cp2k` and `slow` suffice); a `docs` extra (`mkdocs`, `mkdocstrings[python]`) for
  `tools/docs/mkdocs.yml`; `ruff` and `mypy` in the `dev` extra with a `[tool.ruff]`/`[tool.mypy]`
  baseline so CI can gate on more than syntax errors; `pytest-timeout` is already listed.
- **.gitignore**: `runtime/` and `**/runtime/` trees, `runs/`, `*.sif`, `*.oci.tar`, scheduler logs
  (`*.o[0-9]*`, `*.e[0-9]*`, `slurm-*.out`), `benchmark*.json`/`benchmark*.csv`, `gmd_example_run/`,
  `cp2k_jobs/`, `dataset_store/`, `sgt_tiny_h2o_smoke/` (default output directories of the examples
  and tools). Do **not** ignore `*.extxyz`, `*.pt` or `*.lammpstrj` globally: tracked fixtures under
  `tests/fixtures/` and `examples/data/` use them (PHAL's global `*.extxyz` rule is why two of its
  data files were never committed, F-11).
