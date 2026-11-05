# Runtime data contract (`/runtime`)

Ported from PHAL `docs/container/runtime.md` at 96263b6.

Images are immutable and, as SIFs, read-only. Everything a job produces or
consumes that is not software lives on the host and is bind-mounted at
`/runtime`:

```bash
apptainer exec --cleanenv --nv --bind "<host-runtime-dir>:/runtime" image.sif ...
docker run --rm -v "$PWD/runtime:/runtime" gmd-al/cp2k:<tag> ...
```

The PBS templates use `${GMD_AL_RUNTIME_DIR:-$PBS_O_WORKDIR/runtime}` as the
host directory. Layout used by the images and probes:

```text
/runtime/
|-- models/            interatomic-potential models (GMD_SGT checkpoints and TorchScript exports)
|-- datasets/          training data
|-- checkpoints/       training checkpoints
|-- md/                GMD run directories (inputs, output.log, frames, GMD checkpoints)
|-- cp2k/
|   |-- inputs/
|   |-- outputs/
|   `-- scf/           SCF recordings (config: scf recording output_dir)
|-- cp2k_ml/           experimental ML-SCF models/checkpoints/validation/logs/cache (cp2k-ml only)
`-- logs/
    |-- gmd-probe/          validate_gmd_probe.pbs
    |-- sgt-train-probe/    validate_sgt_train_probe.pbs
    |-- cp2k-probe/         validate_cp2k_probe.pbs
    `-- hpcx-diagnosis/     diagnose_hpcx.pbs
```

Rules:

- Nothing under `/runtime` is copied into an image; the Dockerfile-specific
  ignore files exclude `runtime/`, `build/`, `data/` and everything else that is
  not explicitly copied.
- Validators that must write (GMD writes `output.log` next to its input, CP2K
  writes its output and restart files) copy their probe inputs into a work
  directory first; under PBS that is `/runtime/logs/<probe>/work-<jobid>/`.
- Interatomic-potential models (`/runtime/models`) and ML-SCF models
  (`/runtime/cp2k_ml/models`) have independent identities and lifecycles and
  must not be mixed.
- Host paths come from configuration or job variables; the images assume only
  the `/runtime` mount point and their own `/opt/...` prefixes.
- The `cp2k-ml` image additionally expects the fork at `/workspace/cp2k-ml-scf`
  (bind mount); that is source code, not runtime data.
