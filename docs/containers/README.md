# Containers and HPC validation

Ported from PHAL's `docs/container/` and `containers/README.md` at 96263b6
(MIT License, see NOTICE), re-targeted to GMD, GMD_SGT and CP2K.

> **Nothing described here has been built or run.** Docker was unavailable when
> the container files were written; no image exists, no OCI digest or SIF hash
> is recorded, and no PBS job has been submitted. `containers/versions.yaml`
> marks every stack `not_built` / `not_validated`. The procedures below are the
> intended ones; the first real execution must record its evidence (see
> [Evidence](#evidence)).

| Page | Content |
| --- | --- |
| [architecture.md](architecture.md) | Image roles, OCI -> SIF -> PBS flow, source boundaries, PHAL mapping |
| [docker.md](docker.md) | Dockerfiles, build context and ignore files, `build.sh`, pins |
| [apptainer.md](apptainer.md) | SIF conversion and `apptainer exec` on Miyabi (PBS Pro) |
| [runtime.md](runtime.md) | The `/runtime` mount contract |
| [probe-procedure.md](probe-procedure.md) | Build, push, convert, submit, acceptance criteria, common failures |

Code: `containers/` (Dockerfiles, `build.sh`, `versions.yaml`, validators,
probe inputs) and `tools/hpc/pbs/` (PBS job templates and `submit.sh`).
Static tests: `tests/tools/test_containers.py`.

## Evidence

Current evidence is limited to what runs without Docker, Apptainer or PBS
(`tests/tools/test_containers.py`):

- the build plan for every target resolves to digest-pinned base images, exact
  versions and full commit hashes, and floating references are rejected;
- every `COPY` source exists and is admitted by the Dockerfile-specific ignore
  file (PHAL finding F-32 cannot recur);
- on the development host, with the sibling checkouts: the platform validator,
  a random tiny GMD_SGT model on CPU, GMD 0-/10-step runs with a random exported
  GMD_SGT model (built from `../GMD`, not in an image), and the H2 CP2K single
  point with a host CP2K v2026.2 all pass;
- the PBS job scripts, run with stub `module`/`apptainer`, write a summary JSON
  whose `sif_sha256` is computed from the image file at job time.

There is **no** image build log, OCI manifest, SIF hash or GH200 job output for
any GMD_active image. When they exist, record them in `versions.yaml`
(`oci_image`, `oci_digest`, `sif_sha256`, `validation.*`) and keep the evidence
files somewhere reachable: PHAL's release evidence lived under the git-ignored
`build/evidence/` and is lost (see `docs/migration/phal_release_baseline/`).
