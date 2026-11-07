# C16 alkane force-field repair benchmark: final report

*Execution 2026-11-07T07:25:46Z to 2026-11-07T16:43:25Z (9.3 h of the 120 h window). Closed by decision of the user after the initial model failed qualification and the next qualification stage was unaffordable.*

## 1. Outcome

| Level | Status |
|---|---|
| **Implementation** | Complete and exercised on real DFT: protocol freeze, partitions, budget ledger with amendments, pre-stage cost projections, the qualification pipeline (labels → training → accuracy and MD checks), and a recorded closure. 38 offline tests pass. |
| **Experiment** | Terminated at the first prerequisite. Qualification stage q0 ran to completion and did **not** qualify. Stage q1 was projected not to fit the initial-model allowance and was not started. No repair run, final-test label or sealed evaluation was executed. |
| **Scientific result** | **None about repair.** The benchmark's question was whether local, monitor-triggered labeling repairs a model, and how PHAL compares with a committee and with continued training. That question was not reached. The only scientific finding is narrow. With 24 full-system C16 labels from 6 base conformations, this model setup reaches an id_val force RMSE of 0.30 eV/Å, against a target of 0.10. It breaks bonds in 16 of 18 short MD runs. |

## 2. Authorization and amendments

The original authorization file is unchanged (sha256 `0984e4326a9fd84a…`). Amendments are appended in `configs/demos/alkane_repair/repair_c16_authorization_amendments.yaml`. They are verified on every load and journaled once.

| Allowance | Original | Effective | Change |
|---|---|---|---|
| Total DFT | 240 core-h | 240 | — |
| Initial model (q0 + q1) | 45 | 45 | — |
| q0 sub-limit | 30 | **34** | A1-q0-34 (2026-11-07T09:58:39Z, user): measured label cost exceeded the plan's estimate |
| Development / fidelity audit / repair_val / final test / repair runs | 7 / 3 / 6 / 14 / 165 | unchanged | — |
| Per repair run | 1.2 | 1.2 | — |
| Window | 120 h | 120 h | — |
| Final-time reserve | 0 | **25 h** | A2-final-time-reserve (2026-11-07T14:19:24Z, user). Never reached. |

The redistribution proposal for development, repair_val and final test was never granted and is now moot. It would have moved 8 core-h from initial_model to development (+3), repair_val (+1.5) and final_test (+3.5).

Structures, partitions, CP2K settings, model settings, thresholds and sealed-test restrictions were not changed. The frozen protocol hash `bf54d50284226e75` held throughout.

## 3. What was executed

1. **Protocol freeze, plan, geometry preflight** (no DFT). The preflight passed: 33 MD boxes and 60 fragments, none failed.
2. **q0 labels.**
   - 30 full-system C16 single points (24 id_train from 6 bases, plus 6 id_val from 3 bases), one CP2K job at a time on 5 cores. All 30 are valid.
   - There were 33 attempts. Three labels needed a second attempt:
     - an SCF non-convergence (physics; 1.65 core-h lost);
     - a CP2K job killed when I stopped the driver as a Claude background task (operator error; 0.63 core-h);
     - a CP2K job killed when the Claude session ended (operator error; 0.39 core-h, plus 2.2 h of idle window).
   - The driver has since been run detached from the session.
3. **q0 training and qualification.** A 3-member ensemble, trained in 526 s. The id_val accuracy check and 18 MD trajectories (9 starts × 2 seeds × 4000 steps) both failed (section 4).
4. **q1 gate.** The pre-stage projection needed 24 more full labels. Expected cost 25.5 core-h, 90 % upper bound 28.3, against 13.1 core-h remaining in initial_model. The stage was marked **unaffordable** and no allowance was moved. A total-preserving redistribution was **infeasible**: the only slack was 2.9 core-h in repair runs (165 minus the 135 × 1.2 cap bound) and 0.4 in the fidelity audit, leaving 12.0 core-h uncovered.
5. **Closure** (user decision "proceed with option 1"), recorded with `close`. `execute` now launches nothing.

## 4. q0 qualification: every acceptance criterion

| Criterion | Threshold (frozen YAML) | Measured | Pass |
|---|---|---|---|
| id_val force RMSE (6 structures, 300 atoms) | ≤ 0.10 eV/Å | **0.303** | no |
| id_val per-atom force-error p95 | ≤ 0.30 eV/Å | **1.009** (max 1.74) | no |
| MD topology: every frame intact (covalent scale 1.3) | intact fraction 1.0 in every trajectory | **2 / 18** trajectories intact; first broken bond at steps 1260–3240 (median 2140) | no |
| No hidden stabilisation (no force caps or clipping, ML force field only) | none found | none found | yes |

Reference force RMS on id_val is 1.79 eV/Å, so the model error is about 17 % of the signal. Per-structure RMSE is 0.25–0.34 eV/Å, so the failure is uniform and not driven by an outlier.

Figures (PNG, SVG and PDF in `figures/`):
- `qualification.png`: per-structure id_val RMSE against the target;
- `qualification_md.png`: step of the first broken bond per MD trajectory;
- `training_curves.png`: training and validation force RMSE per ensemble member;
- `compute.png`: consumption and the blocked q1 projection against each allowance.

**Diagnosis (observational, not tested).** The ensemble overfits.
- Training force RMSE falls to 0.075–0.086 eV/Å at the last epoch. Validation RMSE bottoms out at 0.30, 0.43 and 0.47 (best epochs 78, 26 and 12), then rises.
- The id_val set is also the early-stopping set, so the reported id_val numbers are, if anything, optimistic.
- This points to too few independent conformations rather than a training failure. Doubling the base conformations (q1) would likely help, but nothing measured here shows it would reach 0.10. My estimate was about 0.2, an extrapolation, not a result.
- Model capacity, hyperparameters and other model settings were frozen and not explored.

## 5. Compute and provenance

| | core-h |
|---|---|
| Consumed (ledger, all attempts) | **31.87**, all in initial_model / q0 |
| q0 limit after A1 | 34 (2.13 left) |
| Unspent of the 240 total | 208.13 |
| Plan's estimate per full label (timing sources from earlier pilot runs) | 0.73 |
| Measured per full label (all attempts per task, n = 30) | 1.06 (sd 0.45; 5.2–27.0 min wall at 5 cores) |

- **Ledger and label records disagree for one task.** The label provenance records total 43.00 core-h; the enforced ledger totals 31.87. 11.08 of the 11.13 core-h difference is task `label-91ef50ccbc255dd1`, the label lost in the session-end outage; the remaining 0.05 core-h is the label killed by the driver stop. The label record charges a dead attempt up to the moment a driver noticed it had vanished, so the 2.2 idle hours of the outage count as about 11 core-h. The ledger charges it up to its last file write (21:03:47 JST), when the process was killed. No CP2K process ran during the outage, so the ledger figure is the physical consumption and the enforcement basis.
- **The A1 amendment was needed regardless of the operator errors.** Without the 1.02 core-h they lost, q0 would still have cost 30.85 core-h, over the original 30.
- **The executor code changed during the execution** (commits 9a4c573, 394f991 and ed52283: amendments, restart-safe room checks, projections, time reserve). The protocol, the structures and the frozen settings did not change. The start record pins GMD_active at 6e1a321 and records GMD_SGT as dirty at the start.

## 6. Sealed final test

Never opened. No sealed label exists (`sealed/labels.jsonl` absent), and the sealed access log is empty (`sealed_access.jsonl` absent). No threshold, model or stopping decision used final-test data. The sealed structures remain usable for a future, independently authorized attempt.

## 7. Every case, seed and arm

The design has 4 repair-run arms × 9 cases (3 families × 3 bases) × 5 seeds = 180 runs. A_frozen needs no run. Every run's status is **not started (execution closed)**: B_continued 45, C_committee 45, D_phal 45, D_noLJ 45. The run order was never fixed, because the stage that fixes it comes after qualification. The per-run table is in `report.md`.

## 8. The seven questions

1. **Was the initial model reliable in-domain?** No. q0 failed three of four criteria (section 4). q1 was not run (unaffordable).
2. **Did the frozen model show reference-confirmed failures on the challenges?** Not tested. No challenge structure was labeled.
3. **Did the closed-loop arms detect and label the relevant region?** Not tested. No repair run.
4. **Did local labels improve independent (sealed) predictions?** Not tested.
5. **Was in-domain accuracy retained?** Not tested.
6. **Were there physical signals (stability, paired arm differences)?** Not tested.
7. **What remains unsupported?**
   - Any claim about local repair, the arms or PHAL.
   - Equilibrium statistics.
   - Other chain lengths (C64/C128 were not attempted).
   - Event-buffering benefit.
   - Efficiency claims.
   - Any equivalence between arms.

## 9. Implications for a future attempt (not authorized; for planning only)

- **Budget:** the measured 1.06 core-h per full C16 label means q0 + q1 (54 labels) needs about 57–60 core-h, not the planned 39.
- **Qualify the model setup first:** before re-running the benchmark, check in a separate, cheaper study whether this model setup can reach 0.10 eV/Å on C16 at any affordable data size, for example by reusing the 30 labels here. Otherwise the repair benchmark would again stop at its prerequisite.
- **Changing the protocol:** any change to the model settings or the 0.10 threshold would be a protocol revision. It would need to be declared before new labels are seen and kept separate from this sealed test set.

## 10. Reproduction

From `GMD_active`, using the `gmd-active` conda environment:

```
python -m tools.experiments.alkane_repair execute --benchmark configs/demos/alkane_repair/repair_c16.yaml   # resumable; now stops: closed
python -m tools.experiments.alkane_repair ledger  --benchmark configs/demos/alkane_repair/repair_c16.yaml
python -m tools.experiments.alkane_repair report  --benchmark configs/demos/alkane_repair/repair_c16.yaml --publish
```

- **Environment:** `environment.json` records the host, Python and package versions, the repository commits and dirty flags, CP2K 2026.2 (`cp2k.psmp`), the GMD binary sha256, and the authorization and configuration hashes.
- **Artifacts** (`../alkane_artifacts/repair_c16`):
  - `execution/journal.jsonl` and `execution/state.json` (checkpoint, closure record);
  - `execution/projections/qualification_q1.json`;
  - `reference_labels/initial_model_q0.json`;
  - `initial_model/stage0/` (dataset, checkpoints, exported model, `qualification.json`);
  - `cp2k_tasks/` (every attempt).

Files here:
- `report.md` / `report.json`: generated, complete record;
- `execution_plan.md`;
- `environment.json`;
- `figures/`;
- `STATUS_zh.md`: Chinese summary.
