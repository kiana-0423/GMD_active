# Runtime data contract

Generated operational data (runs, models, datasets, labeling tasks, logs) never lives in the
source tree, a configuration directory or a container image. It lives under one **runtime root**:

- inside containers the root is the bind mount `/runtime` (see [containers/runtime.md](containers/runtime.md));
- on a workstation or cluster it is any directory you choose.

`tools/runtime/init_runtime_tree.py` creates the tree and renders the matching configuration
overlay from one table, so this page, the tree and the config paths cannot drift:

```bash
python tools/runtime/init_runtime_tree.py --root /scratch/$USER/gmd_rt                     # create the tree
python tools/runtime/init_runtime_tree.py --root /scratch/$USER/gmd_rt --overlay rt.yaml   # + overlay
python tools/runtime/init_runtime_tree.py --root /scratch/$USER/gmd_rt --check             # verify
```

`containers/config/runtime.yaml` is that overlay for `--root /runtime` (a test checks it is the
generated one). Merge it over a run configuration, later files winning:

```python
from gmd_active_learning.config import build_config
from gmd_active_learning.utils.yaml import merge_yaml_files

config = build_config(merge_yaml_files("my_run.yaml", "containers/config/runtime.yaml"), base_dir=Path("."))
```

and give every run its own work directory: `gmd-al run --config ... --work-dir /runtime/runs/<run_id>`.

## Layout

| Directory | Contents |
|---|---|
| `runs/` | one `gmd-al --work-dir` per run: `workflow.sqlite`, the model registry, the versioned dataset store, events, GMD segments |
| `models/latest/` | GMD_SGT checkpoints and TorchScript exports handed to `gmd-al init` |
| `models/archive/` | retired GMD_SGT models kept for provenance |
| `datasets/` | imported or seed training data (legacy PHAL/LAMMPS imports land here) |
| `checkpoints/` | GMD_SGT training checkpoints outside a run |
| `md/runs/` | stand-alone GMD runs (input xyz, `run.in`, `frames.extxyz`, `output.log`) |
| `md/restart/` | GMD checkpoint files (checkpoint v3) for restarts |
| `cp2k/tasks/` | labeling task directories (`cp2k.task_dir`): `task.json`, `attempts/`, `result.json` |
| `cp2k/outputs/` | CP2K outputs collected outside the labeling service |
| `cp2k/scf/` | SCF recordings (`scf.recording.output_dir`) |
| `cp2k/scf_dataset/train/` | curated ML-SCF dataset (`scf.dataset.root`), train split |
| `cp2k/scf_dataset/validation/` | validation split |
| `cp2k/scf_dataset/test/` | test split |
| `cp2k_ml/models/` | ML-SCF density models **only**, never interatomic potentials |
| `cp2k_ml/checkpoints/` | ML-SCF training checkpoints |
| `cp2k_ml/validation/` | ML-SCF validation evidence |
| `cp2k_ml/predictions/` | ML-SCF predicted densities (`scf.ml_scf.prediction_dir`) |
| `cp2k_ml/logs/` | ML-SCF logs |
| `cp2k_ml/cache/` | ML-SCF caches |
| `logs/` | job and probe logs; PBS probe summaries under `logs/<probe>/` |
| `cache/` | disposable caches |

## Overlay keys

| Key | Value under root `R` |
|---|---|
| `cp2k.task_dir` | `R/cp2k/tasks` |
| `scf.recording.enabled` | `true` (observational: recordings never change a label) |
| `scf.recording.output_dir` | `R/cp2k/scf` |
| `scf.dataset.root` | `R/cp2k/scf_dataset` |

Everything else a run writes is under its `--work-dir`. `containers/config/cp2k_ml_scf.yaml` is the
optional, experimental ML-SCF overlay (port of PHAL `configs/cp2k.yaml`); merge it after the
runtime overlay.

## Differences from PHAL

- PHAL tracked an empty `runtime/` skeleton in git and hard-coded `/runtime` in
  `configs/runtime.yaml`. The root is now an argument and the tree is created on demand.
- `md/traj/` (LAMMPS dumps) became `md/runs/` (GMD run directories); `models/` holds GMD_SGT
  models instead of Allegro models; `cp2k/inputs/` became `cp2k/tasks/` because the labeling
  service keeps a full task directory per region, not loose inputs.
- PHAL's `cp2k.labeled_dataset_dir` has no counterpart: labels enter the per-run versioned
  dataset store (`<work-dir>/datasets`).
