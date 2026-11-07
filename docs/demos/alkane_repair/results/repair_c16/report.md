# Alkane repair benchmark: repair_c16 (protocol 1.0)

## Implementation status

- structures: True
- protocol_frozen: True
- plan: True
- qualification_run: True
- repair_runs: 0
- planned_runs: 180
- sealed_evaluation: False
- closed: True

Scientific status: **terminated before the scientific question (unaffordable: qualification_q1 is projected not to fit under the current authorization (initial_model: upper 28.3 vs remaining 13.1 core-h)); closed by user: close out after q0 failed qualification and q1 was unaffordable; no further DFT, no amendment, no protocol change**

## Execution

- started: 2026-11-07T07:25:46+00:00; terminal: {"status": "unaffordable", "message": "qualification_q1 is projected not to fit under the current authorization (initial_model: upper 28.3 vs remaining 13.1 core-h); no allowance was moved; proposal in <artifacts_root>/execution/projections/qualification_q1.json", "at": "2026-11-07T16:25:43+00:00"}
- next action: none: closed (report only)
- run statuses: {}

| Stage | Status | Detail |
|---|---|---|
| freeze | done | {"protocol_sha256": "bf54d50284226e75e99de654d35e4e0734ec0ccf6bf3ad3bb3ffc347debb96dd", "structures": {"public": "db7bc1d86d0f6aafbad52f815ec89cf4c055445092488dc5ece314016d119d73", "sealed": "d2239ce5042f278ae030eb1087b8344cfc1608e0eaa365df0fa70a6bf9b220e8"}} |
| qualification_q0 | done | {"accepted": false, "criteria": {"id_val_force_rmse": false, "id_val_atom_error_p95": false, "md_topology": false, "no_hidden_stabilisation": true}} |
| qualification_q1 | unaffordable | {"scopes": [{"scope": "initial_model", "expected_core_h": 25.496415428850387, "upper90_core_h": 28.338419783162024, "remaining_core_h": 13.129480713937017, "fits": false}], "proposal": {"feasible": false, "deficits": {"initial_model": 15.3}, "uncovered_core_h": 12.0, "transfers": [{"from": "repair_runs", "to": "initial_model", "core_h": 2.9}, {"from": "fidelity_audit", "to": "initial_model", "core |

## Closure

- closed at 2026-11-07T16:43:25+00:00 by user after 9.29 h of the window: close out after q0 failed qualification and q1 was unaffordable; no further DFT, no amendment, no protocol change
- statement: proceed with option 1 (close out now: write the deliverables for this outcome; no more compute)
- terminal stop: unaffordable: qualification_q1 is projected not to fit under the current authorization (initial_model: upper 28.3 vs remaining 13.1 core-h); no allowance was moved; proposal in <artifacts_root>/execution/projections/qualification_q1.json

## Pre-stage projection: qualification_q1

- rule: start only if the one-sided upper bound (enforcement.estimate_quantile) of the stage's DFT plus the enforcement margin fits every remaining allowance; fits: **False**; need: {"full": 24.0, "fragment": 0.0}
- full label cost: n 30, mean 1.062 core-h, sd 0.453, max 2.254 (this execution (all attempts per task))
- fragment label cost: n 50, mean 0.177 core-h, sd 0.065, max 0.568 (plan timing sources (only 0 measured in this execution))
- initial_model: expected 25.50, upper 28.34 (+ margin 0.08) vs remaining 13.13 core-h -> fits False
- redistribution (total kept): feasible False; deficits {"initial_model": 15.3}; transfers [{"from": "repair_runs", "to": "initial_model", "core_h": 2.9}, {"from": "fidelity_audit", "to": "initial_model", "core_h": 0.4}]; uncovered 12.0 core-h; the deficit of initial_model cannot be covered from projected slack without exceeding the total of 240.0 core-h; the stage is unaffordable under the current authorization

## Compute ledger (measured from every attempt record)

- consumed: {"total": 31.871, "stages": {"initial_model": 31.870519286062983}, "initial_model_q0": 31.871}
- remaining: total 208.13 core-h; stages {"initial_model": 13.13, "development": 7.0, "fidelity_audit": 3.0, "repair_val": 6.0, "final_test": 14.0, "repair_runs": 165.0}; q0 2.13
- budget cancellations: 0

## Protocol

- frozen: 2026-11-07T07:25:47+00:00; protocol sha256 `bf54d50284226e75`

## Structures

| Partition | Counts |
|---|---|
| calibration | {"ordinary": 3} |
| development | {"bend": 2, "contact": 2, "ordinary": 2, "torsion": 2} |
| final_test | {"bend": 3, "contact": 3, "ordinary": 6, "torsion": 3} |
| id_train | {"ordinary": 48} |
| id_val | {"ordinary": 6} |
| repair | {"bend": 3, "contact": 3, "torsion": 3} |
| repair_val | {"bend": 2, "contact": 2, "torsion": 2} |

## Geometry preflight (no DFT)

- passed: True; MD boxes: 33 (max 44.0 A); fragments: 60 (0 failed, 0 over the atom limit, atoms [29, 41], caps [1, 2])

## Initial model

- stage 0: 24 train / 6 val structures; id_val force RMSE 0.3031 eV/A, atom-error p95 1.0092; MD trajectories intact 2/18; criteria {'id_val_force_rmse': False, 'id_val_atom_error_p95': False, 'md_topology': False, 'no_hidden_stabilisation': True}; **not accepted**
- prerequisite: unmet: the execution was closed before a later qualification stage (initial_model/status.json still says pending)

## Repair runs

| Run | Status | Events | Labels (in challenged region) | Promotions | DFT core-h |
|---|---|---|---|---|---|
| B_continued__C16-r010-bend0__s1 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-bend0__s2 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-bend0__s3 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-bend0__s4 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-bend0__s5 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-contact0__s1 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-contact0__s2 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-contact0__s3 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-contact0__s4 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-contact0__s5 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-torsion0__s1 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-torsion0__s2 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-torsion0__s3 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-torsion0__s4 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r010-torsion0__s5 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-bend0__s1 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-bend0__s2 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-bend0__s3 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-bend0__s4 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-bend0__s5 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-contact0__s1 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-contact0__s2 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-contact0__s3 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-contact0__s4 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-contact0__s5 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-torsion0__s1 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-torsion0__s2 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-torsion0__s3 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-torsion0__s4 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r016-torsion0__s5 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-bend0__s1 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-bend0__s2 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-bend0__s3 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-bend0__s4 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-bend0__s5 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-contact0__s1 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-contact0__s2 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-contact0__s3 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-contact0__s4 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-contact0__s5 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-torsion0__s1 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-torsion0__s2 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-torsion0__s3 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-torsion0__s4 | not started (execution closed) | - | - | - | - |
| B_continued__C16-r020-torsion0__s5 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-bend0__s1 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-bend0__s2 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-bend0__s3 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-bend0__s4 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-bend0__s5 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-contact0__s1 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-contact0__s2 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-contact0__s3 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-contact0__s4 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-contact0__s5 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-torsion0__s1 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-torsion0__s2 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-torsion0__s3 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-torsion0__s4 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r010-torsion0__s5 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-bend0__s1 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-bend0__s2 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-bend0__s3 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-bend0__s4 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-bend0__s5 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-contact0__s1 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-contact0__s2 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-contact0__s3 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-contact0__s4 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-contact0__s5 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-torsion0__s1 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-torsion0__s2 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-torsion0__s3 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-torsion0__s4 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r016-torsion0__s5 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-bend0__s1 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-bend0__s2 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-bend0__s3 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-bend0__s4 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-bend0__s5 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-contact0__s1 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-contact0__s2 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-contact0__s3 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-contact0__s4 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-contact0__s5 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-torsion0__s1 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-torsion0__s2 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-torsion0__s3 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-torsion0__s4 | not started (execution closed) | - | - | - | - |
| C_committee__C16-r020-torsion0__s5 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-bend0__s1 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-bend0__s2 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-bend0__s3 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-bend0__s4 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-bend0__s5 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-contact0__s1 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-contact0__s2 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-contact0__s3 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-contact0__s4 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-contact0__s5 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-torsion0__s1 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-torsion0__s2 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-torsion0__s3 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-torsion0__s4 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r010-torsion0__s5 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-bend0__s1 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-bend0__s2 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-bend0__s3 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-bend0__s4 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-bend0__s5 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-contact0__s1 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-contact0__s2 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-contact0__s3 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-contact0__s4 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-contact0__s5 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-torsion0__s1 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-torsion0__s2 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-torsion0__s3 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-torsion0__s4 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r016-torsion0__s5 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-bend0__s1 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-bend0__s2 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-bend0__s3 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-bend0__s4 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-bend0__s5 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-contact0__s1 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-contact0__s2 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-contact0__s3 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-contact0__s4 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-contact0__s5 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-torsion0__s1 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-torsion0__s2 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-torsion0__s3 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-torsion0__s4 | not started (execution closed) | - | - | - | - |
| D_phal__C16-r020-torsion0__s5 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-bend0__s1 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-bend0__s2 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-bend0__s3 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-bend0__s4 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-bend0__s5 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-contact0__s1 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-contact0__s2 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-contact0__s3 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-contact0__s4 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-contact0__s5 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-torsion0__s1 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-torsion0__s2 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-torsion0__s3 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-torsion0__s4 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r010-torsion0__s5 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-bend0__s1 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-bend0__s2 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-bend0__s3 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-bend0__s4 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-bend0__s5 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-contact0__s1 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-contact0__s2 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-contact0__s3 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-contact0__s4 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-contact0__s5 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-torsion0__s1 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-torsion0__s2 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-torsion0__s3 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-torsion0__s4 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r016-torsion0__s5 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-bend0__s1 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-bend0__s2 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-bend0__s3 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-bend0__s4 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-bend0__s5 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-contact0__s1 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-contact0__s2 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-contact0__s3 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-contact0__s4 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-contact0__s5 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-torsion0__s1 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-torsion0__s2 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-torsion0__s3 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-torsion0__s4 | not started (execution closed) | - | - | - | - |
| D_noLJ__C16-r020-torsion0__s5 | not started (execution closed) | - | - | - | - |

## Sealed evaluation

Not executed.

## Assessment (from recorded evidence only)

- **1_initial_model_reliable_in_domain**: "no: stage 0 (24 train structures): not accepted; id_val force RMSE 0.303 eV/A, atom-error p95 1.009 eV/A, 2/18 MD trajectories intact; failed criteria ['id_val_force_rmse', 'id_val_atom_error_p95', 'md_topology']; qualification_q1 unaffordable under the authorization"
- **2_reference_confirmed_failure**: "not tested (execution closed: no qualified initial model) (no reference labels of challenge structures)"
- **3_detected_and_labeled_relevant_region**: "not tested (execution closed: no qualified initial model) (no completed closed-loop repair run)"
- **4_local_labels_improved_independent_predictions**: "not tested (execution closed: no qualified initial model) (sealed evaluation not executed)"
- **5_retention**: "not tested (execution closed: no qualified initial model) (sealed evaluation not executed)"
- **6_physical_signals**: "not tested (execution closed: no qualified initial model) (sealed evaluation not executed)"
- **7_unsupported**: ["any claim about local repair, the arms or PHAL: no repair run was executed", "equilibrium conformational statistics (no reference ensemble)", "chain lengths other than the benchmark's (C64/C128 not attempted)", "event-buffering benefit, unless the enabled no-buffering ablation was executed and evaluated", "efficiency (labels or core-hours to repair) beyond the measured, small-sample values", "any equivalence or 'no difference' between arms from this sample size"]

## Label records against the ledger

- label provenance records 43.00 core-h; ledger 31.87 core-h (the ledger (attempt files, charged to each attempt's last file write) is the enforcement basis)
- label-91ef50ccbc255dd1 (C16-r008-ord2, 2 attempts): label record 12.35, ledger 1.27 core-h

## Measured compute

- reference labels by stage (core-h): {"initial_model_q0": 43.00411586728361}
- fidelity audit: 0.000 core-h; repair runs: 0.000 core-h
- measured CP2K CPU core-hours (wall x cores), failed attempts and retries included; repair-run labels are always newly computed (MD frames), so attributed and newly consumed compute coincide for them

