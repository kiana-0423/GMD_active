# Alkane repair benchmark: repair_smoke (protocol smoke-1.0)

**Smoke profile: smoke-only CP2K settings and toy budgets. Mechanics only; no scientific claim.**

## Implementation status

- structures: True
- protocol_frozen: True
- plan: True
- qualification_run: True
- repair_runs: 6
- planned_runs: 6
- sealed_evaluation: True

Scientific status: **smoke profile: mechanics only, no scientific result**

## Protocol

- frozen: 2026-11-07T05:34:08+00:00; protocol sha256 `9bbc955e80f07975`

## Structures

| Partition | Counts |
|---|---|
| calibration | {"ordinary": 1} |
| development | {"torsion": 1} |
| final_test | {"ordinary": 1, "torsion": 1} |
| id_train | {"ordinary": 4} |
| id_val | {"ordinary": 1} |
| repair | {"torsion": 1} |
| repair_val | {"torsion": 1} |

## Initial model

- stage 0: 4 train / 1 val structures; id_val force RMSE 0.4760 eV/A, atom-error p95 1.3483; MD trajectories intact 2/2; criteria {'id_val_force_rmse': True, 'id_val_atom_error_p95': True, 'md_topology': True, 'no_hidden_stabilisation': True}; **accepted**
- prerequisite: met

## Fragment-fidelity audit (development challenges, DFT)

| Family | Fragments | RMSE | Max | Passes |
|---|---|---|---|---|
| torsion | 0/0 | - | - | False |

## Monitor calibration (in-domain trajectories of the frozen model)

| Arm | Frames | Alarm fraction | Cold required metrics | OK |
|---|---|---|---|---|
| C_committee | 41 | 1.0000 | [] | True |
| D_phal | 41 | 0.9024 | [] | True |

## Repair runs

| Run | Status | Events | Labels (in challenged region) | Promotions | DFT core-h |
|---|---|---|---|---|---|
| B_continued__C8-r003-torsion0__s1 | done | - | - | 1 | 0.000 |
| B_continued__C8-r003-torsion0__s2 | done | - | - | 0 | 0.000 |
| C_committee__C8-r003-torsion0__s1 | done | 4 | 1 (1) | 1 | 0.029 |
| C_committee__C8-r003-torsion0__s2 | done | 4 | 1 (1) | 1 | 0.024 |
| D_phal__C8-r003-torsion0__s1 | done | 4 | 1 (1) | 1 | 0.027 |
| D_phal__C8-r003-torsion0__s2 | done | 4 | 1 (1) | 1 | 0.046 |

## Sealed evaluation

- partial: False (missing runs: 0)

| Arm | Family | Runs | Bases | Success | Promoted | Mean rel. improvement (bases) |
|---|---|---|---|---|---|---|
| B_continued | torsion | 2 | 1 | 2 | 1 | 0.003 |
| C_committee | torsion | 2 | 1 | 1 | 2 | 0.098 |
| D_phal | torsion | 2 | 1 | 2 | 2 | 0.133 |

## Assessment (from recorded evidence only)

- **1_initial_model_reliable_in_domain**: "qualified at stage 0: id_val force RMSE 0.476 eV/A (validation set; sealed in-domain test: see evaluation)"
- **2_reference_confirmed_failure**: {"torsion": "1/1 sealed challenge structures"}
- **3_detected_and_labeled_relevant_region**: {"C_committee": {"runs": 2, "runs_with_events": 2, "runs_with_label_in_region": 2, "labels": 2, "labels_in_region": 2}, "D_phal": {"runs": 2, "runs_with_events": 2, "runs_with_label_in_region": 2, "labels": 2, "labels_in_region": 2}}
- **4_local_labels_improved_independent_predictions**: {"B_continued": {"torsion": {"success": "2/2", "mean_relative_improvement_over_bases": 0.0030455011723147414, "n_bases": 1}}, "C_committee": {"torsion": {"success": "1/2", "mean_relative_improvement_over_bases": 0.09786454076363017, "n_bases": 1}}, "D_phal": {"torsion": {"success": "2/2", "mean_relative_improvement_over_bases": 0.13271859925612794, "n_bases": 1}}}
- **5_retention**: {"B_continued__C8-r003-torsion0__s1": -0.023155582885091976, "B_continued__C8-r003-torsion0__s2": 0.0, "C_committee__C8-r003-torsion0__s1": -0.17246132963814642, "C_committee__C8-r003-torsion0__s2": 0.07528419452142986, "D_phal__C8-r003-torsion0__s1": -0.16533965718258153, "D_phal__C8-r003-torsion0__s2": 0.03854808632669899}
- **6_physical_signals**: [{"a": "B_continued", "b": "C_committee", "n_pairs": 2}, {"a": "B_continued", "b": "D_phal", "n_pairs": 2}, {"a": "C_committee", "b": "D_phal", "n_pairs": 2}]
- **7_unsupported**: ["equilibrium conformational statistics (no reference ensemble)", "chain lengths other than the benchmark's (C64/C128 not attempted)", "event-buffering benefit, unless the enabled no-buffering ablation was executed and evaluated", "efficiency (labels or core-hours to repair) beyond the measured, small-sample values", "any equivalence or 'no difference' between arms from this sample size"]

## Measured compute

- reference labels by stage (core-h): {"development": 0.009478992886013456, "final_test": 0.025281783474816216, "initial_model_q0": 0.08277128577232361, "repair_val": 0.011130763292312622}
- fidelity audit: 0.019 core-h; repair runs: 0.125 core-h
- measured CP2K CPU core-hours (wall x cores), failed attempts and retries included; repair-run labels are always newly computed (MD frames), so attributed and newly consumed compute coincide for them

