# Real-run evidence (2026-11-06)

Produced on macOS arm64 with:
- GMD `al-integration` branch, Torch build;
- GMD_SGT 0.4.0, `al-weighted-training` branch;
- CP2K 2026.2 (Homebrew, PBE/DZVP-MOLOPT-SR-GTH);
- the local scheduler.

Runs used `examples/real_cycle/run_real_cycle.py`. Scratch paths are replaced by `<scratch>`.
The same flow runs as `tests/integration/test_closed_loop_real.py` (about 2.5 minutes).

## 1. Promoted closed loop (`real_cycle_water_promoted.json`)

The system is 4 water molecules in a 7.5 Å periodic box. The initial model is a 2-member GMD_SGT
committee trained on 8 CP2K-labeled water dimers.

**Cycle 0**
- GMD ran a 40-step segment and the monitor captured 6 committee-deviation events.
- Regions were whole molecules, one per hotspot cluster. Of the 21 candidates, 5 were selected; 16 were rejected by budget or diversity.
- All 5 were labeled by real CP2K and validated VALID.
- The 5 labels were ingested and published as an immutable dataset version.
- The active model was fine-tuned on it, exported, and the export validated (max force deviation from the checkpoint 9.5e-7 eV/Å; limit 1e-4).
- A GMD smoke test passed, the gates passed, and the model was promoted.

**Cycle 1**
- GMD restarted from the cycle 0 checkpoint with the new model, under `model_change allow`.
- `real_cycle_segment1_gmd_status.json` records `model_changed_from` (the old export hash) and the new `model_sha256`, with `start_step` 40.
- Another labeling and training round promoted a third model: validation force RMSE 0.67 eV/Å, against 1.16 eV/Å for the active model on the same validation set.

`real_cycle_models.json` lists every registered model with checksums, parents, dataset versions
and committee members. The registry re-verified all of them.

The gate thresholds in this run are demonstration values (`max_force_rmse` 10 eV/Å) for a tiny model;
this is a smoke-scale run, not scientific validation.

## 2. Interrupted and resumed run (`real_cycle_water_radicals_*`, `recovery_tasks_*`)

This run used radial cluster extraction, which cut O–H bonds and produced radical fragments
(finding F-37).

- The controller was killed while polling cycle 1 labels.
- `recovery_tasks_before_resume.txt`: each cycle 1 task had exactly one submitted job (52471, 52473, 52475).
- After `run` was restarted, `recovery_tasks_after_resume.txt` shows the same job IDs adopted as attempt 1, followed by attempts 2 and 3 from the bounded retry ladder. No duplicate submissions occurred and the task set is unchanged.
- Every task ended as failed (SCF not converged). Nothing was ingested, no training ran, and the run finished with the initial model still active. This is the intended conservative behavior.
