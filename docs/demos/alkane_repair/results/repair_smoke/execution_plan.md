# Execution plan: repair_smoke (protocol smoke-1.0)

Costed from measured timings only (DFT label records, GMD_SGT training logs, GMD segments). Nothing is launched by this plan.

| Stage | Condition | Full labels | Fragments | DFT core-h | DFT wall h (serial) | Training h | MD h |
|---|---|---|---|---|---|---|---|
| initial_model_q0 | always | 5 | 0 | 0.1 | 0.0 | 0.0 | 0.0 |
| development |  | 1 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| fidelity_audit |  | 0 | 2 | 0.4 | 0.2 | 0.0 | 0.0 |
| repair_val |  | 1 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| final_test |  | 2 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| monitor_calibration |  | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sealed_evaluation |  | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |

Repair runs (upper bounds):

| Arm | Kind | Runs | Labels/run | DFT core-h | Training h | MD h |
|---|---|---|---|---|---|---|
| B_continued | continued_training | 2 | 0 | 0.0 | 0.0 | 0.0 |
| C_committee | closed_loop | 2 | 1 | 0.4 | 0.0 | 0.0 |
| D_phal | closed_loop | 2 | 1 | 0.4 | 0.0 | 0.0 |

| Budget | Projected core-h | Budget core-h | Within |
|---|---|---|---|
| initial_model | 0.1 | 1.0 | yes |
| development | 0.0 | 0.5 | yes |
| fidelity_audit | 0.4 | 0.5 | yes |
| repair_val | 0.0 | 0.5 | yes |
| final_test | 0.0 | 0.5 | yes |
| repair_runs | 0.7 | 2.0 | yes |

Total DFT upper bound: 1.2 core-h (0.6 h serial wall); if the first qualification stage passes: 1.2 core-h.

Measured inputs: full label 66 core-s mean (n=2, p90 74), fragment 644 core-s mean (n=47), training 0.028 s per structure-epoch (n=64), GMD 92 steps/s (n=32).
