# ML-SCF architecture and user guide

**Measured status.** No ML-SCF component has been shown to accelerate CP2K.

- **ML density guess.** PHAL's trained guess took **63 SCF steps where CP2K's atomic guess took 53**, a mean speedup of **0.85**. PHAL's verdict: "The ML guess is worse than the atomic guess on every system" (PHAL `ml_scf_architecture.md`).
- **Other ML components.** The second density model, the learned DIIS mixer, the adaptive controller and the strategy predictor all measured as neutral or worse ([negative_results.md](negative_results.md)).
- **The positive result.** It is the model-free retry ladder `default → nbuffer4` (12/12 converged against 9/12). It lives in `scf.retry`, not in the ML wrapper.

This document describes the optional wrapper that makes the experiment
repeatable. It does not describe an acceleration path.

PHAL's `cp2k_ml_architecture.md`, `ml_scf_architecture.md` and `ml_scf_user_guide.md`
are replaced by this page. They were internally stale: they called the ML guess "the production
acceleration path", listed neural mixing as an ML responsibility, documented an
unimplemented native retry, and named a config key the code never read. Their measurements are quoted
verbatim in [negative_results.md](negative_results.md).

## The one rule

CP2K is the backend. It builds the Kohn–Sham matrix, diagonalises it, computes the
density, energy, forces and stress, and decides convergence against `EPS_SCF`. An
ML component may only change **where the SCF starts**. If the model is missing,
wrong or unhelpful, the answer is the same and arrives more slowly; it is never a different answer.

## Repository boundary

| Repository | Owns |
| --- | --- |
| GMD_active (`gmd_active_learning.adapters.dft.scf`) | the prediction, its validation, the exchange file, the CP2K input rewrite, the environment, the capability probe, provenance, the native fallback and retry |
| `cp2k-ml-scf` (clean-room reconstruction of the lost fork) | reading `<input>.ml_density`, the CP2K-side validation, installing the density as the initial guess, the `ML-SCF\|` log lines |

The version pin is authoritative in `cp2k-ml-scf/VERSION_LOCK.md`: CP2K v2026.2 @ `67b5da876dd6a76b8b021d5a04d1c81ba79a4c50`. See [cp2k_version_history.md](cp2k_version_history.md).

## What a build can do, and how GMD_active finds out

Before every ML run the wrapper probes the configured binary with `<cp2k> --version`
under the versioned contract `gmd-cp2k-capabilities-v1` ([capabilities.md](capabilities.md)).
The probe uses the configured environment. A binary that does not advertise
`phal_ml_scf` makes the wrapper degrade the run to native CP2K, or raise when
`fallback_to_standard_scf: false`.

| Feature | Native CP2K 2026.2 (probed here) | cp2k-ml-scf (declared by its docs, not validated here) | Lost fork (PHAL docs) |
| --- | --- | --- | --- |
| `SCF_GUESS ML`, mode A (`<input>.ml_density`) | no | yes | yes |
| `SCF_GUESS ML`, mode B (in-process TorchScript) | no | no (hook 0009 placeholder) | yes |
| `MIXING METHOD ML` | no | no (out of scope) | advertised, never integrated |
| SCF controller / Pulay recorder | no | no | yes |

## Data flow for one labeling task (when `scf.ml_scf.enabled`)

The labeling area writes the CP2K input. The wrapper then:

1. `prepare_input(input, structure, task_id=..., overlap=None)` predicts with the configured predictor (`scf.ml_scf.predictor`). It validates the prediction before CP2K sees it:
   - the values are finite;
   - the spin channels and shape are right;
   - populations are non-negative and matrices symmetric (1e-8);
   - the declared count agrees with the integrated count (`Tr(P S)` for matrices);
   - the count is the **GTH valence count**, `sum(valence) − charge`, within `scf.ml_scf.electron_tolerance`;
   - the structure hash matches.

   Only then does it write `<input>.ml_density`, rewrite `SCF_GUESS ATOMIC` → `SCF_GUESS ML`, and write the `<input>.ml_scf.json` manifest (the input digest plus capability provenance). A density matrix needs the AO overlap: `density_gnn` receives `overlap=` and the exchange file carries `OVERLAP_VALUES`. Without an overlap, nothing is written.
2. `execution_request(input, output_file=...)` returns the run request. It carries `scf.ml_scf.environment` plus the variables the fork actually reads: `PHAL_ML_SCF_ENABLED`, `_PROTOCOL`, `_FALLBACK`, `_DENSITY`, `_STRUCTURE_DIGEST`, `_BASIS` and `_GUESS_ALPHA`. If ML-SCF cannot run, it returns a native request instead: the ML keywords are stripped, the ML variables are removed, and the loader environment is kept.
3. CP2K validates the density again (shape → finite → Hermitian 1e-8 → `Tr(P S)` 0.1%) and prints `ML-SCF|` lines.
4. `parse_outcome(output_text)` reads those lines. A run counts as accelerated **only** when CP2K reports an accepted guess. `retry_request(...)` returns one native retry after a failed accelerated run.
5. `calculation_provenance(...)` records the capability contract, CP2K version and commit, model digest, `scf_mode`, iterations, runtime and fallback. `scf_mode` is one of `native`, `ml_guess`, `ml_full` or `native_fallback`.

### Automatic fallback

| Situation | What happens |
| --- | --- |
| `scf.ml_scf.enabled: false` | nothing is predicted; the request is native |
| predictor missing or failing; prediction fails validation | no density file; input keeps `SCF_GUESS ATOMIC`; the reason is logged |
| density matrix without an overlap | refused (`Tr(P S)` cannot be checked) |
| CP2K binary lacks `phal_ml_scf`, or fails in the dynamic linker | run degrades to native CP2K |
| `mixing.method: ml` without a deployed model, or on a build without `phal_ml_mixing` | run degrades to native CP2K |
| model digest mismatch | run degrades to native CP2K |
| accelerated run fails | `retry_request` gives one native retry (`retry_native_on_failure`) |
| any of the degradations above with `fallback_to_standard_scf: false` | `CapabilityUnavailableError` |

## User guide

### Configuration (`configs/sections/scf.yaml`)

All switches default to off. The keys that matter for the experiment:

```yaml
scf:
  recording: {enabled: true, record_density: true, record_trajectory: false}
  retry: {enabled: true, order: [default, nbuffer4]}     # the model-free ladder
  ml_scf:
    enabled: true
    executable: /workspace/cp2k-ml-scf/build/cp2k-ml/bin/cp2k.psmp
    environment: {LD_LIBRARY_PATH: "<toolchain lib dirs>"} # needed by the probe AND the run
    predictor: {type: density_gnn, model_path: runtime/cp2k_ml/models/density.pt}
    prediction_dir: runtime/cp2k_ml/predictions
```

`ScfConfig.from_phal()` translates a PHAL configuration (`cp2k.scf_recording`,
`cp2k.ml_scf`, `cp2k_ml_scf`, `backend.dft.engine`). Some PHAL keys were never read: `cp2k_ml_scf.runtime_dir` and
`cp2k.ml_scf.mixing.fallback_method` are dropped with a warning. `electron_tolerance`,
`dataset_dir` and `trajectory_strict` now have readers.

### Collecting data

1. Turn on `scf.recording` (with `record_density` and, if needed, `record_trajectory`).
2. Add the `&AO_MATRICES` block with `recording.input.ao_matrices_print_block` / `insert_ao_matrices_print`.
3. Ingest with `python -m gmd_active_learning.adapters.dft.scf.cli scf dataset ingest --config <cfg>`.

Tests have run this end to end on native CP2K.

### Training and deploying (experimental)

The training tooling lives in `tools/scf/` and needs torch:

- `python -m tools.scf.scripts.train_density_guess` (`tools/scf/training/density_gnn.py`);
- the experimental MLP in `tools/scf/experimental/atomic_population_mlp/`. Its predictor type `torch` requires `scf.ml_scf.experimental: true`.

`deploy.export_density_guess` produces a mode-B TorchScript model, but no current CP2K build consumes it. The neural mixer export (`tools/scf/experimental/neural_mixer/export.py`, and `tools/scf/negative_results/pulay_mixer.export_mixer`) refuses to run without `experimental=True`.

### Benchmarking

- `python -m gmd_active_learning.adapters.dft.scf.cli scf benchmark --runs runs.json` compares recorded runs. A mode is accepted only if it converges everywhere and matches the native energy (≤ 1e-6 Ha), forces (≤ 1e-5) and stress, within the configured fallback rate.
- Live arms: `python -m tools.scf.scripts.benchmark_ml_guess` / `benchmark_guess_variants` / `benchmark_controller`. They need the modified binary; an ML arm that fell back makes the report invalid (exit 2).
- Pin `OMP_NUM_THREADS=1`, because SCF step counts are not thread-deterministic (cp2k-ml-scf RECONSTRUCTION.md §4g).

### Troubleshooting

- **Exit 127 or "error while loading shared libraries".** This is a dynamic-linker failure, not a missing binary. Put the toolchain `LD_LIBRARY_PATH` into `scf.ml_scf.environment`. The probe reports `loader_failure: true`.
- **Every run is `native_fallback`.** Run `scf capabilities --executable <bin> --json` and read `features` and `details`.
- **A prediction is never written.** Check the logged reason. The usual cause is an electron count that is not the valence count, or a density matrix without an overlap.
