# GMD_active containers

Ported from PHAL (`containers/README.md`) at 96263b6 and re-targeted from
LAMMPS + NequIP/Allegro to **GMD + GMD_SGT + CP2K**. MIT License, see NOTICE.

> **Status: not built, not validated.** Docker was not running when these files
> were written and nothing could be built, pushed, converted or submitted.
> Every stack in `versions.yaml` is `not_built` / `not_validated`, no OCI digest
> or SIF hash is recorded, and the PBS templates have never run. What *is*
> tested (`tests/tools/test_containers.py`): the Dockerfiles parse and are
> self-consistent, the build plan resolves, floating refs are refused, the
> validators work on a host with GMD / GMD_SGT / CP2K, and the PBS job scripts
> hash the SIF at job time.

Dockerfiles are the only image recipes; SIF files are derived artifacts and are
never committed. Full documentation: [`docs/containers/`](../docs/containers/README.md).

## Images

| Target | Dockerfile (stage) | Role | Contents |
| --- | --- | --- | --- |
| `gmd` | `Dockerfile.gmd` (`runtime`) | MD | `gmd` built with `-DGMD_ENABLE_TORCH=ON` against the NGC PyTorch LibTorch, GMD_SGT, random probe model |
| `sgt-train` | `Dockerfile.sgt-train` (`runtime`) | training | GMD_SGT (`gmd-train`), e3nn, ASE on NGC PyTorch |
| `cp2k` | `Dockerfile.cp2k` (`runtime`) | labeling | upstream CP2K v2026.2 `cp2k.psmp` + OpenMPI, toolchain-built Libint/LibXC/ELPA/FFTW/OpenBLAS/ScaLAPACK |
| `cp2k-probe` | `Dockerfile.cp2k` (`probe`) | labeling check | `cp2k` + an H2 single point run at build time |
| `cp2k-ml` | `Dockerfile.cp2k-ml` (`development`) | experimental | toolchain + LibTorch for the external `../cp2k-ml-scf` fork, bind-mounted at `/workspace/cp2k-ml-scf`; contains no CP2K |
| `cp2k-ml-probe` | `Dockerfile.cp2k-ml` (`probe`) | experimental | toolchain self-check |

`gmd` and `sgt-train` share one base image (and therefore one LibTorch) so a
model exported by the training image loads in GMD. GMD evaluates TorchScript
models on the CPU (`torch::kCPU` in GMD's adapter).

Mapping from PHAL: its train-probe image -> `Dockerfile.sgt-train`, its
md-probe image -> `Dockerfile.gmd`, its `Dockerfile.cp2k-ml` -> ported as is,
plus a new production `Dockerfile.cp2k`. PHAL's empty Milestone-1
`base`/`md`/`train` scaffolds are dropped (ledger CTR-002).

## Files

- `build.sh` - `plan`, `build`, `push`, `build-all`, `tag`, `oci`, `sif`, with `--dry-run`.
- `versions.yaml` - the single version manifest (exact pins, platform, status).
- `Dockerfile.<target>.dockerignore` - the ignore file BuildKit actually reads for
  each Dockerfile (an allowlist). PHAL's `containers/.dockerignore` was dead
  because the build context is the repository root.
- `scripts/validate_platform.py` - ARM64/CUDA platform JSON (`GMD_AL_REQUIRE_CUDA=1`).
- `scripts/validate_sgt_stack.py` - random tiny GMD_SGT model on CPU/CUDA, TorchScript round trip.
- `scripts/build_gmd_probe_model.py` - exports the random GMD_SGT probe model for GMD.
- `scripts/validate_gmd_stack.py` - `--mode static` (no execution) or `--mode runtime` (0- and 10-step GMD runs).
- `scripts/validate_cp2k_stack.py` - CP2K version + H2 single point (`SCF run converged`).
- `scripts/validate_cp2k_ml_stack.py` - toolchain self-check for `cp2k-ml-probe`.
- `probes/gmd/`, `probes/cp2k/` - probe inputs.

## Quick reference

```bash
containers/build.sh plan gmd                 # resolved base image, digest, tag, build args
containers/build.sh --dry-run build gmd      # the exact commands, without docker
GMD_AL_REGISTRY=registry.example containers/build.sh push gmd   # on an arm64 Buildx builder
GMD_AL_OCI_URI=docker://registry.example/gmd-al/gmd:<tag> containers/build.sh sif gmd   # on Miyabi
GMD_AL_PBS_GROUP=<group> tools/hpc/pbs/submit.sh gmd "$PWD/build/containers/<sif>"
```

The GMD and GMD_SGT sources are exported from the sibling checkouts
(`../GMD`, `../GMD_SGT`, override with `GMD_AL_GMD_SOURCE` /
`GMD_AL_GMD_SGT_SOURCE`, which may also be git URLs) with `git archive` of the
pinned commit and passed as BuildKit named contexts, so an image is always built
from exactly the commit in `versions.yaml`.
