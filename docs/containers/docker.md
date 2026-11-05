# Docker, OCI and `build.sh`

Ported from PHAL `docs/container/docker.md` at 96263b6. Status: no image has
been built; Docker was not available when these files were written.

## Build context and ignore files

The build context is always the repository root (so Dockerfiles can copy
`containers/scripts/...` and `containers/probes/...`). PHAL kept a
`containers/.dockerignore` that was never consulted for that context. Here each
Dockerfile has its own ignore file next to it, `containers/Dockerfile.<name>.dockerignore`
(for example `Dockerfile.gmd.dockerignore`), which BuildKit uses in preference to
any root `.dockerignore`. Each one is an allowlist: `*` first, then `!` entries
for exactly the files that Dockerfile copies. `build.sh` always builds with
`docker buildx` (BuildKit); the legacy builder would ignore these files.

GMD and GMD_SGT sources are not in the context at all. They arrive as BuildKit
named contexts (`--build-context gmd_src=...`, `gmd_sgt_src=...`) created by
`build.sh` with `git archive <pinned commit>` from the sibling checkouts, or as
`<git-url>#<commit>`.

## Dockerfiles

| File | Stages | Notes |
| --- | --- | --- |
| `Dockerfile.gmd` | `builder` -> `runtime` | CMake + Ninja build of the `gmd` target with `-DGMD_ENABLE_TORCH=ON -DGMD_ENABLE_MPI=OFF -DBUILD_TESTING=OFF`, `CMAKE_PREFIX_PATH` from `torch.utils.cmake_prefix_path` of the base image. The runtime stage installs GMD_SGT with `--no-deps` (so pip cannot replace the NGC torch), exports the random probe model and runs `validate_gmd_stack.py --mode static`. `gmd` is not executed at build time: builders have no `libcuda.so.1`. |
| `Dockerfile.sgt-train` | `runtime` | GMD_SGT `--no-deps` install, pinned e3nn/ASE/PyYAML, `configs/` copied to `/opt/gmd_sgt/configs`, CPU run of `validate_sgt_stack.py` at build time. torch-cluster/torch-scatter are not installed (GMD_SGT falls back to pure-torch code). |
| `Dockerfile.cp2k` | `toolchain` -> `build` -> `runtime` (-> `probe`) | CP2K's own toolchain builds OpenBLAS/ScaLAPACK/FFTW/Libint/LibXC/ELPA (`--with-libtorch=no`, no `--install-all`); CP2K v2026.2 is configured with the same option set as `../cp2k-ml-scf/build.sh`'s native control build plus `-DCP2K_USE_LIBTORCH=OFF`. `/usr/local/bin/cp2k.psmp` is a wrapper that sources the toolchain environment first (it is required at run time, see `VERSION_LOCK.md`). |
| `Dockerfile.cp2k-ml` | `toolchain` -> `development` -> `probe` | PHAL's experimental development image, ported with `PHAL_*` -> `GMD_AL_*` renames. Contains no CP2K; the fork is bind-mounted. |

Common rules, enforced by `tests/tools/test_containers.py`: every `COPY` source
exists (PHAL finding F-32: the train-probe copied a removed `config/`), every
`${VAR}` is an ARG of its stage or an ENV in its stage chain, every ARG without
a default is supplied by `build.sh`, no commit or digest literal appears in a
Dockerfile, and no floating tag is used.

## `build.sh`

```text
containers/build.sh [--dry-run] plan|build|push|build-all|tag|oci|sif|help [TARGET] [TAG]
targets: gmd sgt-train cp2k cp2k-probe cp2k-ml cp2k-ml-probe
```

- `plan` prints the resolved plan as `key=value` lines: target, platform,
  Dockerfile and its ignore file, stage, build context, base image with digest,
  `base_image_digest` (or `unpinned`), image tag and reference, one
  `build_arg=NAME=value` line per build argument, named contexts and the
  manifest status. It needs only bash and awk.
- `--dry-run` (anywhere on the command line) prints the plan and the exact
  `git archive` / `docker buildx` / `apptainer` commands without running them.
- `build` uses `docker buildx build --load`; `push` adds `--push` and
  `--metadata-file build/evidence/<target>-build-metadata.json` and refuses an
  undigested base image unless `GMD_AL_ALLOW_UNPINNED_BASE=1`.
- `build-all` builds the three role images (`gmd`, `sgt-train`, `cp2k`).
- Floating references (`latest`, `stable`, `main`, `master`, also as `:tag` or
  `#ref`) are rejected for base images, image tags, `tag` destinations, OCI
  URIs, source URLs and manifest values; commits must be 40 hex characters;
  versions may not contain range operators; `<...>` placeholders are refused.
- Environment overrides are `GMD_AL_*` (see `containers/build.sh help`).
- Bash 3.2 compatible, so `plan` and `--dry-run` work on macOS as well.

## Pins (`versions.yaml`)

| Component | Pin | Source of the value |
| --- | --- | --- |
| NGC PyTorch base | `nvcr.io/nvidia/pytorch:25.09-py3@sha256:89172d8e...` | PHAL's NGC registry query (2026-07-10); not re-queried |
| inherited torch / CUDA / Python | `2.9.0a0+50eac811a6` / `13.0.1` / `3.12` | PHAL release baseline |
| GMD | `54ac558e30f9fb85a3474f7749016e24aa109a46` (`v2.5-9-g54ac558`, branch `al-integration`, CMake project version 2.4.0) | `../GMD` |
| GMD_SGT | `49ece72edabf46e1e5ee9a3949071acc650c7a13` (0.4.0, branch `al-weighted-training`) | `../GMD_SGT` |
| e3nn / ASE / PyYAML | `0.6.0` / `3.29.0` / `6.0.3` | versions in the development environment where GMD_SGT runs |
| Ubuntu base | `ubuntu:24.04@sha256:4fbb8e6a...` | PHAL / `../cp2k-ml-scf/VERSION_LOCK.md` |
| CP2K | `v2026.2` @ `67b5da876dd6a76b8b021d5a04d1c81ba79a4c50` | `VERSION_LOCK.md` (verified upstream 2026-08-04) |

Output digests (`oci_digest`, `sif_sha256`, probe-model `sha256`) are `null`
until a real build records them; nothing was invented. Whether e3nn 0.6.0 and
ASE 3.29.0 install cleanly against the NGC image's pip constraints is unknown
until the first build.
