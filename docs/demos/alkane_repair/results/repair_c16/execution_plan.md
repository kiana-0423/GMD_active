# Execution plan: repair_c16 (protocol 1.0)

Costed from measured timings only (DFT label records, GMD_SGT training logs, GMD segments). Nothing is launched by this plan.

| Stage | Condition | Full labels | Fragments | DFT core-h | DFT wall h (serial) | Training h | MD h |
|---|---|---|---|---|---|---|---|
| initial_model_q0 | always | 30 | 0 | 21.9 | 4.4 | 0.6 | 0.2 |
| initial_model_q1 | only if qualification stage 0 fails | 24 | 0 | 17.5 | 3.5 | 1.1 | 0.2 |
| development |  | 8 | 0 | 5.8 | 1.2 | 0.0 | 0.0 |
| fidelity_audit |  | 0 | 12 | 2.1 | 0.4 | 0.0 | 0.0 |
| repair_val |  | 6 | 0 | 4.4 | 0.9 | 0.0 | 0.0 |
| final_test |  | 15 | 0 | 11.0 | 2.2 | 0.0 | 0.0 |
| monitor_calibration |  | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sealed_evaluation |  | 0 | 0 | 0.0 | 0.0 | 0.0 | 21.7 |

Repair runs (upper bounds):

| Arm | Kind | Runs | Labels/run | DFT core-h | Training h | MD h |
|---|---|---|---|---|---|---|
| B_continued | continued_training | 45 | 0 | 0.0 | 8.9 | 0.0 |
| C_committee | closed_loop | 45 | 6 | 47.9 | 10.0 | 0.9 |
| D_phal | closed_loop | 45 | 6 | 47.9 | 10.0 | 0.9 |
| D_noLJ | closed_loop | 45 | 6 | 47.9 | 10.0 | 0.9 |

| Budget | Projected core-h | Budget core-h | Within |
|---|---|---|---|
| initial_model | 39.5 | 45.0 | yes |
| development | 5.8 | 7.0 | yes |
| fidelity_audit | 2.1 | 3.0 | yes |
| repair_val | 4.4 | 6.0 | yes |
| final_test | 11.0 | 14.0 | yes |
| repair_runs | 143.7 | 165.0 | yes |

Total DFT upper bound: 206.4 core-h (41.3 h serial wall); if the first qualification stage passes: 188.9 core-h.

Measured inputs: full label 2630 core-s mean (n=16, p90 3892), fragment 638 core-s mean (n=50), training 0.028 s per structure-epoch (n=64), GMD 83 steps/s (n=28).
