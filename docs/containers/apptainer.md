# Apptainer on Miyabi (PBS Pro)

Ported from PHAL `docs/container/apptainer.md` at 96263b6. PHAL's page listed
Slurm launch behaviour as a planned validation item although the target site
schedules with **PBS Professional**; that was a mislabel and is corrected here
(see [PBS launch semantics](#pbs-launch-semantics)).
(GMD_active's scheduler adapters support both Slurm and PBS; the container
validation path is PBS only.)

Status: not run. No SIF has been built for any GMD_active image.

## Source of truth

Apptainer has no recipe of its own. It converts an OCI image built from
`containers/Dockerfile.*`:

```text
Dockerfile -> OCI image (registry or oci-archive) -> apptainer build -> SIF
```

```bash
module purge
module load apptainer/1.3.5
export APPTAINER_CACHEDIR=<large-cache-dir>
export APPTAINER_TMPDIR=<large-tmp-dir>
GMD_AL_OCI_URI=docker://registry.example/gmd-al/gmd:<tag> containers/build.sh sif gmd
# or from an archive written by `containers/build.sh oci gmd`:
GMD_AL_OCI_URI=oci-archive:build/containers/gmd-al-gmd-<tag>-linux-arm64.oci.tar containers/build.sh sif gmd
```

`build.sh sif` refuses to run without `GMD_AL_OCI_URI` (Miyabi has no Docker
daemon), refuses floating tags, and writes the SIF's sha256, computed from the
file, to `build/evidence/<target>-sif.sha256`. SIF files are git-ignored
artifacts.

## Execution contract

Every validation job runs:

```bash
apptainer exec --cleanenv --nv --bind "<host-runtime-dir>:/runtime" <image.sif> <command>
```

- `--cleanenv`: the host environment (including PBS variables and the host's
  `OPAL_PREFIX`) does not leak into the container. Variables the container
  needs are passed explicitly as `APPTAINERENV_<NAME>` (e.g.
  `APPTAINERENV_GMD_AL_REQUIRE_CUDA=1`, `APPTAINERENV_OMP_NUM_THREADS=1`).
- `--nv`: injects the host NVIDIA driver (`libcuda.so.1`). Required for
  `sgt-train` (CUDA), and for `gmd` because the NGC LibTorch it links needs
  `libcuda.so.1` to load even though GMD evaluates on the CPU. Harmless for
  `cp2k`.
- `--bind ...:/runtime`: see [runtime.md](runtime.md).

### HPC-X UCX/UCC workaround (NGC-based images)

PHAL diagnosed (job 2362706.opbs, PHAL's md-probe) that under `--nv` the NGC
image's UCC resolved a host-injected UCX lacking `ucs_config_doc_nop`. The
working strategy was:

```bash
unset OPAL_PREFIX
export LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH:-}"
```

`validate_gmd_probe.pbs` and `validate_sgt_train_probe.pbs` wrap every
command in this prefix. `validate_cp2k_probe.pbs` applies only
`unset OPAL_PREFIX` (Ubuntu image, no HPC-X). `diagnose_hpcx.pbs` re-runs
PHAL's diagnosis for any NGC-based SIF and reports which strategy works; it has
not been run for the GMD_active images, so the workaround is inherited, not
re-verified.

## PBS launch semantics

- Templates live in `tools/hpc/pbs/`. Submit with
  `GMD_AL_PBS_GROUP=<group> tools/hpc/pbs/submit.sh <gmd|sgt-train|cp2k|hpcx> <abs-sif-path>`
  (queue from `GMD_AL_PBS_QUEUE`, default `debug-g`) or call `qsub` directly
  with `-q`, `-W group_list=` and `-v <IMAGE_VAR>=...`.
- No project group is hard-coded; the `#PBS -q debug-g` line is only a default
  that `qsub -q` overrides.
- One node (`select=1`), single process for `gmd` and `sgt-train`; the `cp2k`
  probe starts `mpirun -np ${GMD_AL_CP2K_RANKS:-2}` inside the container.
  Because `--cleanenv` drops the PBS environment, the container's OpenMPI does
  not attempt a PBS TM launch; multi-node MPI (host MPI ABI compatibility) is
  out of scope and untested.
- The arm64 images must run on Miyabi-G nodes (Grace CPU); Miyabi-C nodes are
  x86_64.
- Each job computes `sha256sum` of the SIF it runs and writes
  `runtime/logs/<probe>/summary-<jobid>.json` with the job id, node, queue,
  group, image path, SIF sha256, Apptainer version, per-stage exit status and an
  overall status, even when a stage fails. PHAL's md-probe job wrote a
  hard-coded SIF hash instead (finding F-32).
