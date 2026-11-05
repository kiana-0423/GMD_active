# PBS validation jobs (Miyabi-G, PBS Pro, Apptainer)

Ported from PHAL `scripts/pbs/` at 96263b6. **Not run yet**: written without
access to Miyabi; no job has been submitted for any GMD_active image.

| Script | Image variable | Stages |
| --- | --- | --- |
| `validate_sgt_train_probe.pbs` | `GMD_AL_SGT_TRAIN_IMAGE` | `platform` (CUDA required), `sgt_stack` (random GMD_SGT model, CPU + CUDA) |
| `validate_gmd_probe.pbs` | `GMD_AL_GMD_IMAGE` | `platform`, `gmd_static`, `gmd_runtime` (run0 + run10 with the random probe model) |
| `validate_cp2k_probe.pbs` | `GMD_AL_CP2K_IMAGE` | `cp2k_h2` (version + H2 single point, `mpirun -np ${GMD_AL_CP2K_RANKS:-2}`) |
| `diagnose_hpcx.pbs` | `GMD_AL_PROBE_IMAGE` | HPC-X UCX/UCC strategy search for NGC-based SIFs |
| `submit.sh` | - | `qsub` wrapper: queue `GMD_AL_PBS_QUEUE` (default `debug-g`), group `GMD_AL_PBS_GROUP` (required), `--dry-run` |

Each validation job hashes the SIF it runs (`sha256sum`) and writes
`runtime/logs/<probe>/summary-<jobid>.json` (job, node, queue, group, image,
`sif_sha256`, per-stage exit status, overall status), also when a stage fails.
Other overrides: `GMD_AL_RUNTIME_DIR` (host directory bound to `/runtime`),
`GMD_AL_APPTAINER_MODULE` (default `apptainer/1.3.5`).

See `docs/containers/apptainer.md` and `docs/containers/probe-procedure.md`.
