# SCF and ML-SCF in GMD_active

**Measured status first.** PHAL recorded no ML-SCF component that accelerated CP2K:

- **ML density guess.** The trained guess used **63 SCF steps against 53** for CP2K's atomic guess, a **speedup of 0.85** (`ml_scf_architecture.md`). PHAL called it "worse than the atomic guess on every system".
- **Second density model.** It "did not help": 2 of 48 runs were better, 39 worse (`ml_density_v2.md`).
- **Learned DIIS mixer.** "Status: negative result. Experimental only — not a production backend." (`ml_pulay_objective.md`).
- **Adaptive controller.** "safe and neutral": 172 = 172 steps, and twelve of twelve counterfactual labels were `keep` (`adaptive_scf_controller.md`).
- **Strategy predictor.** "The predictor does not work at this size" (`scf_strategy_prediction.md`).

The **one positive result needs no model**. Retrying `default → nbuffer4`
converges **12/12** solids against **9/12** for the default alone, for 4.7% more
iterations. GMD_active ships that ladder as `scf.retry`; it is off by default.

Everything here is optional and capability-gated:

- Nothing in the default labeling path depends on it except plain SCF-history parsing and recording (`scf.recording`, also off by default).
- torch is imported lazily, only by the experimental models.

## What is where

| Piece | Location |
| --- | --- |
| SCF history parser and recorder (used by core labeling) | `gmd_active_learning.adapters.dft.scf.scf_history` |
| AO density, overlap and trajectory recording; the `&AO_MATRICES` input block | `adapters/dft/scf/{ao_matrices.py,recording/}` |
| Pulay (DIIS) dump reader and B-matrix diagnostics | `adapters/dft/scf/pulay_matrix.py` |
| SCF sample schema, dataset, validation, features, export | `adapters/dft/scf/{schema,dataset,validation,features,export}.py` |
| Static `&SCF` strategies and the **retry ladder** | `adapters/dft/scf/strategy.py` (`RetryLadder`) |
| Density exchange file (protocol `phal-ml-scf-v2`), predictor contract and registry | `adapters/dft/scf/density/` |
| Versioned CP2K capability contract | `adapters/dft/scf/capabilities.py`, see [capabilities.md](capabilities.md) |
| ML initial-guess wrapper (engine `cp2k_ml_scf`) | `adapters/dft/scf/cp2k_ml_backend.py`, see [architecture.md](architecture.md) |
| Configuration (`scf` section) | `adapters/dft/scf/config.py`, `configs/sections/scf.yaml` |
| CLI (`scf dataset`, `scf capabilities`, `scf benchmark`) | `adapters/dft/scf/cli.py` |
| Benchmarks, diagnostics, dataset generation, training, experimental and negative-result models | `tools/scf/` (source checkout only) |
| CP2K templates and the Pulay sampling plan (byte-identical to PHAL) | `configs/cp2k/` |
| Structure collections | `data/structures/`, see [structure_provenance.md](structure_provenance.md) |

### Module map

PHAL names used in the ported documents map to GMD_active as follows.

| PHAL | GMD_active |
| --- | --- |
| `hotspot_al.cp2k.scf_history` | `gmd_active_learning.adapters.dft.scf.scf_history` |
| `hotspot_al.cp2k.ao_matrix_parser` | `…scf.ao_matrices` |
| `hotspot_al.cp2k.density_recording` | `…scf.recording.density` |
| `hotspot_al.cp2k.scf_trajectory` | `…scf.recording.trajectory` |
| `hotspot_al.cp2k.pulay_matrix` | `…scf.pulay_matrix` |
| `hotspot_al.cp2k_ml_scf.{schema,dataset,validation,features,export}` | `…scf.{same}` |
| `hotspot_al.cp2k_ml_scf.cp2k_capabilities` | `…scf.capabilities` |
| `hotspot_al.cp2k_ml_scf.density.{prediction,exchange}` | `…scf.density.{prediction,exchange}` |
| `hotspot_al.cp2k_ml_scf.density.density_predictor` | `…scf.density.predictor` |
| `hotspot_al.cp2k_ml_scf.density.model_registry` | `…scf.density.registry` |
| `hotspot_al.cp2k_ml_scf.training.density_model` | `…scf.density.gnn_model`, training loop in `tools/scf/training/density_gnn.py` |
| `hotspot_al.cp2k_ml_scf.training.evaluate_density` | `…scf.density.gnn_predictor` |
| `hotspot_al.cp2k_ml_scf.deploy` | `…scf.deploy`; `export_mixer` in `tools/scf/experimental/neural_mixer/export.py` |
| `hotspot_al.cp2k_ml_scf.scf_strategy` | `…scf.strategy` (static part); `tools/scf/strategy_predictor.py` (k-NN) |
| `hotspot_al.backends.cp2k_ml_scf.CP2KMLSCFBackend` | `…scf.cp2k_ml_backend.CP2KMLSCFBackend` (a wrapper, no longer a subclass) |
| `hotspot_al.cp2k_ml_scf.{benchmark,guess_benchmark}`, `training.evaluate` | `tools/scf/benchmark/{modes,guess,evaluate}.py` |
| `hotspot_al.cp2k_ml_scf.{pulay_dataset,structures,difficulty,guess_diagnostics}` | `tools/scf/{pulay_dataset,structures,difficulty}.py`, `tools/scf/diagnostics/guess.py` |
| `hotspot_al.cp2k_ml_scf.{controller,controller_dataset}` | `tools/scf/controller/{policy,dataset}.py` |
| `hotspot_al.cp2k_ml_scf.{dataset_adapter,training.dataset_loader,training.train_density}` | `tools/scf/training/{dataset_adapter,dataset_loader,density_gnn}.py` |
| `hotspot_al.cp2k_ml_scf.training.{models,train}` (atomic-population MLP) | `tools/scf/experimental/atomic_population_mlp/` (torch gate: `…scf._torch`) |
| `hotspot_al.cp2k_ml_scf.training.mixer_model` | `tools/scf/experimental/neural_mixer/model.py` |
| `hotspot_al.cp2k_ml_scf.training.{pulay_mixer,ml_pulay_model}` | `tools/scf/negative_results/` |
| `scripts/*.py` (SCF) | `python -m tools.scf.scripts.<name>` |
| `cp2k.scf_recording.*`, `cp2k.ml_scf.*`, `cp2k_ml_scf.*` | `scf.recording.*`, `scf.ml_scf.*`; `ScfConfig.from_phal` translates |

## What can be re-run here, and what needs the modified CP2K

The machine this was migrated on has native CP2K 2026.2 (`/opt/homebrew/bin/cp2k.psmp`). The
cp2k-ml-scf binaries are Linux/aarch64, Docker is not running, and they cannot run there.

| Experiment / feature | Needs | Re-run here? |
| --- | --- | --- |
| SCF history parsing and recording; AO density, overlap and trajectory recording | native CP2K | **Yes.** `tests/scf/test_scf_cp2k_native.py`: H2, 10 steps, Tr(P S) = 2.000000 on all 11 frames. The converged density matches the PHAL fixture to 1e-5. |
| Capability contract and native validation | native CP2K | **Yes.** [capabilities_native_cp2k_2026.2.json](capabilities_native_cp2k_2026.2.json) |
| Static strategy grid and retry ladder (`scf_strategy_prediction.md`) | native CP2K | **Yes**, on regenerated structures. See [strategy_rerun.md](strategy_rerun.md); the MgO `R_COND` abort and the `nbuffer4` rescue are reproduced. |
| Paired-run benchmark, difficulty and Pulay B-matrix diagnostics on recorded data | nothing (offline) | **Yes**, on the committed fixtures |
| `SCF_GUESS ML` from a density file (mode A) | cp2k-ml-scf (`phal_ml_scf`) | **No.** Declared by cp2k-ml-scf (H2 replay converges in 1 step); not validated here. |
| In-process TorchScript guess (mode B), ML-guess benchmark, guess diagnostics, density-v2 α sweep | the lost fork | **No.** No current build implements mode B. |
| Adaptive controller live runs, counterfactual labels | the lost fork | **No.** Only the offline Python model is ported. |
| Pulay dataset generation (`PHAL_ML_SCF_PULAY_RECORD_DIR`) | the lost fork | **No.** Existing dumps and fixtures can be read. |
| Neural mixing (`MIXING_METHOD ML`) | none exists | **No.** Never integrated in any build. |

## Documents

Measurement records are ported verbatim from PHAL with a banner:

- [ml_density_failure_analysis.md](ml_density_failure_analysis.md)
- [ml_density_v2.md](ml_density_v2.md)
- [ml_pulay_objective.md](ml_pulay_objective.md)
- [adaptive_scf_controller.md](adaptive_scf_controller.md)
- [scf_strategy_prediction.md](scf_strategy_prediction.md)
- [pulay_dataset.md](pulay_dataset.md)
- [density_representation.md](density_representation.md)

Rewritten for GMD_active:

- [architecture.md](architecture.md) (with the user guide)
- [contracts.md](contracts.md): every data rule mapped to its validator and test
- [capabilities.md](capabilities.md)
- [negative_results.md](negative_results.md): the summary with the verbatim conclusions
- [cp2k_version_history.md](cp2k_version_history.md): the v2024.1 → v2026.2 record and the version pin
- [structure_provenance.md](structure_provenance.md)
- [strategy_rerun.md](strategy_rerun.md)
