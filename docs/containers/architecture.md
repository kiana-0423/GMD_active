# Container architecture

Status: designed, not built (see [README](README.md)).

## Principles (carried over from PHAL)

- Dockerfiles in `containers/` are the only environment definitions. There is
  no Apptainer definition file; a SIF is always derived from an OCI image.
- `containers/versions.yaml` is the only version manifest. Dockerfiles take
  every pin as a build argument; no commit or digest is written into a
  Dockerfile (a test enforces this).
- Images are role-specific: MD (`gmd`), training (`sgt-train`), labeling
  (`cp2k`), plus the experimental CP2K-ML toolchain (`cp2k-ml`).
- Mutable data (datasets, models, checkpoints, trajectories, logs) is never in
  an image layer; it is mounted at `/runtime` ([runtime.md](runtime.md)).
- Floating references (`latest`, `stable`, `main`, `master`) are refused by
  `build.sh` and by the tests.

## Build, deploy and validate flow

```text
containers/Dockerfile.<role>  +  repository root (allowlisted by Dockerfile.<role>.dockerignore)
   + named contexts gmd_src / gmd_sgt_src  (git archive of the pinned GMD / GMD_SGT commits)
        |  containers/build.sh push <target>   external Docker Buildx builder, linux/arm64
        v
registry OCI image  (digest recorded in versions.yaml, buildx metadata in build/evidence)
        |  containers/build.sh sif <target>    Miyabi login node, module apptainer/1.3.5
        v
read-only SIF  (sha256 written to build/evidence/<target>-sif.sha256)
        |  tools/hpc/pbs/submit.sh <probe> <sif>   qsub -> PBS Pro, Miyabi-G GH200 node
        v
apptainer exec --cleanenv --nv --bind <runtime>:/runtime  ->  validators  ->  summary-<jobid>.json
```

The target platform (from `versions.yaml`): Miyabi-G, NVIDIA Grace CPU + GH200
GPU, `linux/arm64`, PBS Professional, Apptainer 1.3.5. Miyabi cannot reach a
local Docker daemon: the SIF is built from a registry URI (`docker://...`) or an
OCI archive (`oci-archive:...`).

## Images

| Target | Base | Built from | Gate at build time | Gate on GH200 (PBS) |
| --- | --- | --- | --- | --- |
| `gmd` | NGC `pytorch:25.09-py3` @ digest | GMD @ `54ac558` (`v2.5-9`), GMD_SGT @ `49ece72` (0.4.0) | probe-model export; static `gmd` check (no execution) | platform (CUDA), static, `gmd` run0 + run10 with the random model |
| `sgt-train` | same NGC base | GMD_SGT @ `49ece72` | imports, version, `gmd-train`, random tiny model on CPU | platform (CUDA), random tiny model on CPU and CUDA |
| `cp2k` | `ubuntu:24.04` @ digest | CP2K `v2026.2` @ `67b5da8` | `cp2k.psmp --version` | H2 single point with `mpirun -np N` |
| `cp2k-probe` | `cp2k` | - | H2 single point | - |
| `cp2k-ml(-probe)` | `ubuntu:24.04` @ digest | toolchain only | no CP2K source in any layer; toolchain probe | none defined |

Why one NGC base for `gmd` and `sgt-train`: GMD links the LibTorch of the
PyTorch wheel in the image, and GMD_SGT exports TorchScript with that same
torch, so the training-to-MD hand-off cannot hit a TorchScript version or ABI
mismatch. GMD's TorchScript adapter is hard-wired to `torch::kCPU`; on a GH200
node `gmd` uses the Grace CPU, and `--nv` is still needed because the NGC
LibTorch links `libcuda.so.1`.

## Source boundaries

- **GMD** and **GMD_SGT** are sibling repositories (`../GMD`, `../GMD_SGT`).
  `build.sh` exports the pinned commit with `git archive` (never the working
  tree or its `build*/` directories) and passes it as a BuildKit named context;
  the Dockerfile checks the exported commit marker. A git URL can be used
  instead (`GMD_AL_GMD_SOURCE=https://...`), which BuildKit fetches at the
  pinned commit. At the time of writing the pinned commits exist in the local
  checkouts; whether they are pushed to the public remotes was not checked.
- **CP2K** (production) is fetched inside the build from the upstream archive
  of the pinned commit.
- **cp2k-ml-scf** (experimental fork) never enters an image or the build
  context: the `cp2k-ml` image is a build environment, and the fork is
  bind-mounted at `/workspace/cp2k-ml-scf`. `../cp2k-ml-scf/VERSION_LOCK.md` is
  authoritative for its pins.
- **GMD_active itself** is not installed in any image: the controller runs on
  the host or login node and drives the engines (`cp2k.launcher`, the GMD
  executable setting in the GMD_active config). Wiring those to
  `apptainer exec` is not part of this change and has not been exercised.

## Mapping from PHAL

| PHAL (96263b6) | GMD_active | Notes |
| --- | --- | --- |
| Milestone-1 `base`/`md`/`train` scaffolds | dropped | empty templates (CTR-002) |
| PHAL train-probe (NequIP/Allegro) | `Dockerfile.sgt-train` | F-32 `COPY config` bug not reproduced |
| PHAL md-probe (LAMMPS + pair_allegro) | `Dockerfile.gmd` | multi-stage; static check at build, run under PBS |
| `Dockerfile.cp2k-ml` | `Dockerfile.cp2k-ml` | ported; plus a new production `Dockerfile.cp2k` |
| `validate_train_stack.py` | `validate_sgt_stack.py` | random GMD_SGT model instead of Allegro |
| `validate_md_stack.py`, `validate_md_binary_static.py` | `validate_gmd_stack.py` | `--mode runtime` / `--mode static` |
| `build_md_probe_model.py` | `build_gmd_probe_model.py` | GMD_SGT `script_model` export |
| `validate_cp2k_ml.py` (host-side driver) | `validate_cp2k_stack.py` (in-image) | same H2 input and "SCF run converged" gate |
| `scripts/pbs/*.pbs` | `tools/hpc/pbs/*.pbs` | SIF sha256 computed at job time (F-32), queue/group not hard-coded |
| `containers/.dockerignore` (never consulted) | `containers/Dockerfile.<target>.dockerignore` | the file BuildKit actually reads |
| `PHAL_*` variables | `GMD_AL_*` | e.g. `GMD_AL_REQUIRE_CUDA`, `GMD_AL_OCI_URI` |
