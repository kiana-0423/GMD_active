# Probe procedure: build, convert, submit, accept

Ported from PHAL `docs/container/milestone2-train-probe.md` at 96263b6 and
generalised to the three GMD_active role images. **None of these steps has been
performed** for GMD_active; the first run must fill in `versions.yaml` and keep
its evidence.

## 0. Inspect the plan (any machine, no Docker)

```bash
containers/build.sh plan gmd
containers/build.sh plan sgt-train
containers/build.sh plan cp2k
containers/build.sh --dry-run build gmd     # exact commands
```

## 1. Build and push the OCI images (external arm64 Buildx builder)

Requirements: Docker with Buildx/BuildKit, a `linux/arm64` worker (native or
emulated), network access to `nvcr.io`, Docker Hub, PyPI, GitHub (CP2K and its
toolchain downloads) and the destination registry, the sibling checkouts
`../GMD` and `../GMD_SGT` containing the pinned commits (or
`GMD_AL_GMD_SOURCE` / `GMD_AL_GMD_SGT_SOURCE` pointing at git URLs), and roughly
10 GB for the NGC base. No GPU is needed: no build step uses CUDA.

```bash
docker buildx imagetools inspect nvcr.io/nvidia/pytorch:25.09-py3   # must list linux/arm64 and match the digest
GMD_AL_REGISTRY=registry.example containers/build.sh push sgt-train
GMD_AL_REGISTRY=registry.example containers/build.sh push gmd
GMD_AL_REGISTRY=registry.example containers/build.sh push cp2k
# optional build-time H2 check of CP2K:
containers/build.sh build cp2k-probe
```

Record each pushed digest (from `build/evidence/<target>-build-metadata.json`)
as `stacks.<stack>.oci_image` / `oci_digest`, and the probe model's sha256
(`/opt/gmd_al/probes/gmd/sgt-probe.json` in the image) as
`stacks.gmd.probe_model.sha256`.

A CPU-only check of a built image on the builder:

```bash
docker run --rm gmd-al/sgt-train:<tag> python /opt/gmd_al/containers/scripts/validate_sgt_stack.py
docker run --rm gmd-al/cp2k:<tag> sh -c 'OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 python3 /opt/gmd_al/containers/scripts/validate_cp2k_stack.py'
```

(`gmd` cannot run on a builder without `libcuda.so.1`; its build runs only the
static check.)

## 2. Convert to SIF (Miyabi login node)

```bash
module purge && module load apptainer/1.3.5
export APPTAINER_CACHEDIR=<cache> APPTAINER_TMPDIR=<tmp>
GMD_AL_OCI_URI=docker://registry.example/gmd-al/gmd@sha256:<digest> containers/build.sh sif gmd
```

Repeat for `sgt-train` and `cp2k`. `build.sh sif` prints and records the SIF
sha256. For a private registry use `apptainer registry login`; never commit
credentials.

## 3. Submit the validation jobs (PBS Pro, Miyabi-G)

```bash
export GMD_AL_PBS_GROUP=<your-group>            # GMD_AL_PBS_QUEUE defaults to debug-g
tools/hpc/pbs/submit.sh sgt-train "$PWD/build/containers/gmd-al-sgt-train-<tag>-linux-arm64.sif"
tools/hpc/pbs/submit.sh gmd       "$PWD/build/containers/gmd-al-gmd-<tag>-linux-arm64.sif"
tools/hpc/pbs/submit.sh cp2k      "$PWD/build/containers/gmd-al-cp2k-<tag>-linux-arm64.sif"
# if torch fails to load under --nv with a UCX/UCC symbol error:
tools/hpc/pbs/submit.sh hpcx      "$PWD/build/containers/gmd-al-gmd-<tag>-linux-arm64.sif"
```

`--dry-run` prints the `qsub` command. Results land in
`runtime/logs/<probe>/`: one log per stage plus `summary-<jobid>.json`. Do not
run the CUDA validators on a login node.

## 4. Acceptance criteria

**sgt-train** (ledger CTR-003, CTR-004), both on a Miyabi-G compute node:

1. `validate_platform.py` with `GMD_AL_REQUIRE_CUDA=1`: machine `aarch64`,
   Python >= 3.10, torch and gmd_sgt import, CUDA available, an H100/GH200
   device, finite CUDA matrix product; exit 0.
2. `validate_sgt_stack.py`: a seeded random tiny `UnifiedEquivariantMLIP`
   gives finite energy and finite, autograd forces on CPU and CUDA, and its
   TorchScript export reproduces the eager energy; exit 0.

**gmd** (CTR-005):

1. Platform as above (the NGC LibTorch must load under `--nv`).
2. `validate_gmd_stack.py --mode static`: `gmd` present, executable, aarch64,
   linked against LibTorch, no unresolved library.
3. `validate_gmd_stack.py --mode runtime`: `gmd h2.xyz run0.run` and
   `gmd h2.xyz run10.run` with the random GMD_SGT probe model exit 0, write
   `status.json` with state `completed` and the probe model's sha256, log steps
   0..1 and 0..10, and every PE/KE/E_total/T value is finite. GMD refuses
   `run 0`, so the single-point stage gates on the step-0 row of a one-step run.

**cp2k** (CTR-006): `cp2k.psmp --version` reports 2026.2, and the H2 single
point under `mpirun -np 2` exits 0 with `SCF run converged`, `PROGRAM ENDED`
and a finite total energy; also once in Docker (`cp2k-probe` build or the
`docker run` above).

Every job's `summary-<jobid>.json` must show `status: passed` and a
`sif_sha256` equal to the recorded one. Only then set the corresponding
`validation.*` entries and `status` in `versions.yaml`, and
`validated.selected_*_stack`.

The probe models are random (`training_steps = 0`,
`scientifically_valid = false`): they test software integration only.

## Common failures

- **Image is amd64**: the build lacked `--platform linux/arm64`; `build.sh`
  always passes the manifest platform.
- **`pinned commit ... is not present`**: the sibling checkout does not contain
  the commit from `versions.yaml`; fetch it or point `GMD_AL_GMD_SOURCE` /
  `GMD_AL_GMD_SGT_SOURCE` at a repository that does.
- **`refusing to push ... has no digest`**: record the base digest in
  `versions.yaml` (or use `GMD_AL_ALLOW_UNPINNED_BASE=1` for a throwaway image).
- **pip constraint conflict on the NGC image** (`/etc/pip/constraint.txt`):
  investigate; do not replace the inherited torch.
- **`--nv` missing**: `gmd` fails to load (`libcuda.so.1` not found) and torch
  reports no CUDA.
- **UCX/UCC symbol error (`ucs_config_doc_nop`)**: run `diagnose_hpcx.pbs` and
  adjust the prefix in the job templates to the strategy it reports.
- **CP2K loader error (`libelpa_openmp.so...` not found, exit 127)**: CP2K was
  started without the toolchain environment; call `/usr/local/bin/cp2k.psmp`
  (the wrapper), not `/opt/cp2k/bin/cp2k.psmp`.
- **OpenMPI refuses to run as root** (Docker only): set
  `OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` for that command.
- **`/tmp` fills during `apptainer build`**: point `APPTAINER_TMPDIR` at a
  large local directory.
