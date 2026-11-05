# Integration gaps: GMD and GMD_SGT

Phase 0 survey, 2026-11-06. GMD at `9d8eede` (v2.4), GMD_SGT at `5c9bba5` (v0.3.0).
File references are relative to each repository. Nothing in either repository was modified.
Probes and builds ran in a temporary directory, and their numeric results are recorded here.

## Summary

| Gate | Status | Blocking for |
|---|---|---|
| GMD_SGT masked/weighted force loss | **Not supported.** A four-function extension is proven in a probe. | Phase 1 feasibility gate; all regional training |
| GMD_SGT mixed cluster + periodic batches | **Crashes** in `collate_fn` | Any dataset that mixes cluster and slab/block samples |
| GMD ML restart from checkpoint | **Silently wrong.** Atomic numbers are not checkpointed. | Segmented runs, model promotion between segments |
| GMD online frame access | File-based only. Frames lack forces, IDs, and species, and are not flushed. | Online monitoring |
| GMD external stop/checkpoint | Not supported | Stop-on-trigger, post-trigger continuation policy |
| GMD loads GMD_SGT exports | **Yes** (TorchScript contract matches). Verified end to end. | — |
| `GMD_se3gnn` is GMD_SGT's former name | Confirmed (GMD_SGT commit `926665b`) | Adapter renaming |

GMD_active's existing adapters must be replaced, not extended:
- `gmd_se3gnn_adapter.py` expects `model.pt` after training, but GMD_SGT writes `ckpt_best.pt`.
- It expects `exported_model.pt` after export, but GMD_SGT's default name is `model.pt`.
- `predict()` returns zeros unconditionally.
- `gmd_adapter.py` returns no frames outside dry-run.

## GMD_SGT (training, inference, export)

### Masked / weighted force training: the Phase 1 feasibility gate

No per-atom array reaches the force loss today. The data path is:

`reader._atoms_to_dict` (fixed key set, `gmd_sgt/data/reader.py:110-143`) → `validate_structure_item` → `collate_fn` (emits only fixed keys, `gmd_sgt/data/dataset.py:51-107`) → `Trainer` passes the whole batch as `target` (`gmd_sgt/training/trainer.py:359-366`) → `EnergyForceLoss.forward` (unweighted mean, `gmd_sgt/training/loss.py:97-102`).

The probe used an H-capped C2H6-like cluster with `force_weight = [1,1,1,1,0.5,0,0,0]` (atom 7 is a cap) and the stock reader, collate, and loss:

| Check | Result |
|---|---|
| `force_weight` on disk / in item / in batch | present / **dropped** / **dropped** |
| Perturb cap atom 7 target by (+0.5, −0.3, +0.2) eV/Å | loss 0.42901692 → 0.48935813 (**+6.0e-2**); max \|ΔGrad\| 1.8e-3 |
| Same, with `batch["force_weight"]` injected by hand | 0.48935813 (key ignored) |
| Prototype extension, all weights 1 | 0.42901692 (identical to stock) |
| Prototype, cap weight 0, perturb cap | Δloss **0.000**, max \|ΔGrad\| **0.000** |
| Prototype, cap weight 1, perturb cap | Δloss +8.6e-2 |
| Stock `Trainer.run()` with prototype | runs unchanged |

**Minimal GMD_SGT extension.** It is backward compatible, and the Trainer is unchanged:
1. `data/reader.py:_atoms_to_dict` and `read_npz`: copy `force_weight` (float32 `[N]`) into the item.
2. `data/validation.py:validate_structure_item`: require shape `[N]` and values ≥ 0.
3. `data/dataset.py:collate_fn`: concatenate `force_weight`, filling ones for items without it.
4. `training/loss.py:EnergyForceLoss.forward`: weighted rmse/mae/huber normalised by `3·Σw`.

The same pattern gives per-structure `energy_weight` and `stress_weight`.

### Other GMD_SGT gaps that affect active learning

| Gap | Evidence | Needed |
|---|---|---|
| Cluster (no cell) and periodic samples cannot share a batch: "`cell` must be present in all samples or none" | `reader.py:119-123`, `dataset.py:71-74` | Give clusters a dummy cell with `pbc=[F,F,F]` (model already handles per-graph pbc, `models/geometry.py:161-173`). Probe: cluster energy identical mixed vs alone (1.595752). |
| Stress is all-or-nothing; even with `w_stress=0`, mixed stress/no-stress batches crash | `validation.py:82-97`, `dataset.py:67-70` | Zero-fill missing stress with weight 0, or strip stress on write |
| Frames lacking energy or forces are **silently dropped** | `reader.py:81-104` | GMD_active must always write both, and should verify counts after loading |
| No per-structure energy weight; reference energies always refit by least squares on the train split; isolated-atom E0 cannot be supplied | `loss.py:89-118`, `api.py:181-184,217` | Per-structure `energy_weight` (see above) so cluster energies can be down-weighted or excluded |
| Split is random with a fixed seed of 42; no explicit validation file or indices | `split.py:11-82`, `api.py:171-175` | A split hook so related event frames stay in the same split |
| Resuming onto new data returns the **old** best model (old `best_val` kept) | `trainer.py:220-243`; probe: run B never beat 0.3292, so the epoch-3 weights were returned | A "fine-tune from weights" mode that resets `best_val`, optimizer, and epoch |
| Checkpoint lacks package version, units, and species list; export has no metadata beyond `local_cutoff`/`lr_cutoff` | `trainer.py:256-272`; `inference/export.py:117-123` | GMD_active's model registry must record these itself |
| Committee: inference only (`ensemble_forces` (M,N,3)); no spread computed; `api.train` never seeds torch | `api.py:346-363` | Committee training and spread calculation live in GMD_active; seed every member |
| Per-atom energies only from the eager model | `calculator.py:162-168` | Optional one-line addition if a monitor needs them |

## GMD (molecular dynamics)

### What exists

- CLI-only engine: `gmd input.xyz run.in [ff] [top] [--np N]`. Output is always `<xyz dir>/output.{xyz,log}`, and `model_path` resolves against the CWD (`app/gmd_main.cpp:68-110,262-264`; `src/io/config_loader.cpp:601-602`).
- No Python bindings: `GMD_BUILD_PYTHON` is declared but unused (`CMakeLists.txt:8`). There is no C API and no per-step observer; the main loop is a plain `for` loop at `app/gmd_main.cpp:995-1002`.
- ML: TorchScript via LibTorch, CPU, serial only, no virial (so no NPT) (`src/force/torchscript_adapter.cpp`, `app/gmd_main.cpp:409-441,410-415`). The contract matches GMD_SGT's export exactly. Verified: a fresh Torch build (13 s) ran a GMD_SGT-exported model for 20 steps with E_total conserved to 1e-6 eV.
- The `next/` directory is a separate early-stage engine with no MD. The root `gmd` is current.
- The committed `build/gmd` is **stale** (it rejects `output_interval`). Rebuild: about 7 s CPU, 13 s with Torch.

### Gaps

| Gap | Evidence | Consequence |
|---|---|---|
| **Checkpoint does not store atomic numbers.** After restart, every atom is fed to the model as species 0 (GMD_SGT's padding index). | `src/io/checkpoint.cpp:192,335-360`; `include/gmd/system/system.hpp:68`. Verified: PE at step 10 is −0.840180 eV continuous vs **−1.223712 eV** after restart with the same model; Python re-evaluation reproduces both (correct Z vs Z=0). | Segmented ML runs and model promotion via restart are silently wrong |
| Restart does not validate the force field or model | `app/gmd_main.cpp:743-750` | A different model is accepted silently. The registry must enforce compatibility. |
| Frames contain only a 0-based type index and positions (6 dp), plus scalar thermo | `src/io/trajectory_writer.cpp:140-167` | No forces, atom IDs, species, velocities, cell lengths, or PBC; ASE cannot read the file |
| Output is never flushed (4 KiB blocks, torn frames); SIGTERM loses buffered frames | `src/io/trajectory_writer.cpp:186` | Live tailing is unreliable |
| No external stop, pause, or checkpoint request | `app/gmd_main.cpp:995-1002` | Cannot apply a stop-on-trigger decision mid-segment |
| Checkpoint overwritten in place, not atomically | `src/io/checkpoint.cpp:144` | A crash during write can destroy the only restart point |
| Orthorhombic, fully 3D-periodic boxes only; unknown elements silently become Z=0 | `include/gmd/system/box.hpp`; `src/io/config_loader.cpp:422-452` | Slab systems need explicit vacuum. A Z=0 guard is needed when ML is active. |
| No ensembles, per-atom uncertainty, or in-process model reload | `app/gmd_main.cpp:421-432`; `src/force/ml_force_provider.cpp:25-40` | Committee deviation must be computed by GMD_active on frames |

### Recommended integration path

**Segmented runs plus a small GMD patch** (about 150–250 lines, serial path trivial), in priority order:

1. Store Z in the checkpoint (bump `kCheckpointVersion`, add Z to the MPI gather), and validate the model on restart.
2. Flush trajectory and log after each frame.
3. Add `stop_file` / `checkpoint_request_file` directives polled in the main loop (optionally a signal flag).
4. Write checkpoints atomically (temp file plus rename), optionally with step-suffixed names.
5. Write ASE-compatible frames: `Lattice=`, `pbc=`, and per-atom tag, element, and forces.

GMD_active then runs K-step segments, reads complete frames after each segment (or flushed frames during it), and evaluates monitors and committee spread in Python via GMD_SGT's `OnlinePredictor`. It decides whether to continue, stop, or swap the model, and restarts from the checkpoint.

A C++ step hook with a socket (about 300–600 lines) or Python bindings (refactor of the 800-line `main()`) is deferred until segment-restart cost is shown to matter.

Per AGENTS.md, these upstream changes add capabilities only: GMD gains frame/control I/O, and GMD_SGT gains per-atom weights. All active-learning decisions stay in GMD_active.
