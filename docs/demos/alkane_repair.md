# Alkane repair benchmark

**Question.** Can PHAL detect failures that a machine-learned force field meets outside its initial
training coverage, obtain trustworthy local reference labels, update the model, and restore accurate
and stable predictions, while preserving performance inside the original coverage?

Repair is treated as an outcome to establish, not to assume. Acquisition efficiency is a secondary
endpoint. The frozen protocol is [alkane_repair/protocol.md](alkane_repair/protocol.md).

**Status (2026-11-07).**

- *Implementation:* complete and tested. The whole chain has run on real CP2K, GMD and GMD_SGT in the
  smoke profile, including a cached re-execution that recomputed nothing.
- *Scientific result:* none yet. The confirmatory C16 profile has generated its structures and passed
  its geometry preflight; its costed plan is below. It has not been frozen or run: it needs about 190
  core-hours of DFT, and that needs an explicit compute allowance (plan in section 6).

## 1. Relation to the pilot

The alkane hotspot pilot (`docs/demos/alkane_hotspots.md`) is preserved unchanged as history: the
model studies v1, v2, v2b and v2c, the closed loops, the fidelity gate and the C32 transfer. Its results
are exploratory evidence. This benchmark is separately named, with its own configuration, protocol,
artifacts root (`../alkane_artifacts/repair_c16`) and report. It was designed after the pilot and
says so. Three pilot findings shaped it:

- **Inaccurate baseline.** The pilot's initial model was inaccurate even on ordinary configurations,
  so this benchmark first qualifies an initial model against pre-defined in-domain criteria and stops
  if none qualifies.
- **Static pools are the wrong test.** The pilot's PHAL arms were judged on static pools. Here the
  primary experiment uses real trajectories through the persistent controller, with the staged
  monitor as configured.
- **Loose gates.** The pilot's promotion gate of 1.0 eV/A was a sanity bound. Promotion here needs the
  repair-validation error of the run's family to fall and the in-domain validation error to hold.

## 2. What is reused and what is new

Reused unchanged, through composition:

- the alkane geometry, perturbations and checks;
- CP2K labeling with one settings hash and measured costs;
- the frozen fragment setting of the pilot's C16 gate (cluster, core 1.5 A, buffer 3 A), extraction and
  capping;
- GMD_SGT training and export;
- GMD_active's persistent controller with its event, hotspot, candidate, dataset, registry,
  promotion and rollback machinery, and the closed loop's extractor, budgeted labeler, held-out
  validation and image guard.

New, in `tools/experiments/alkane_repair/`:

| Module | What it does |
|---|---|
| `config.py` | the benchmark schema; every number lives in YAML |
| `generate.py` | independent base conformations; in-domain ordinary samples and out-of-coverage challenges; partitions by base; final test in a sealed root |
| `references.py` | protocol freeze; per-stage reference labels within measured budgets; fidelity audit |
| `qualify.py` | initial-model qualification and freezing |
| `failure.py` | reference-confirmed failures |
| `repair.py` | arms B, C and D on the frozen model; per-subset promotion gates; monitor calibration on in-domain trajectories, checked before any CP2K job |
| `evaluate.py` | the sealed evaluation (the only reader of final-test labels; every access logged) |
| `plan.py` | execution plan costed from measured timings |
| `preflight.py` | geometry checks without DFT |
| `report.py`, `figures.py` | report and figures |

Two changes outside the package:

- **GMD_active core.** `PromotionGateConfig.metric_gates` adds gates on named metrics (absolute,
  regression or improvement against the active model). They fail closed, and the default is empty, so
  existing configurations are unchanged. Tested.
- **Closed loop.** `initial_model.parents_manifest` lets a run's initial and validation sets be the
  frozen model's own data. The split rules are kept: train parents must be in the train splits and
  validation parents in the validation splits. Tested.

## 3. Commands

From the GMD_active root, in the `gmd-active` environment:

```bash
R="python -m tools.experiments.alkane_repair"; B=--benchmark\ configs/demos/alkane_repair/repair_c16.yaml
$R generate $B && $R preflight $B && $R plan $B          # no DFT
$R freeze-protocol $B                                     # first step of the confirmatory run
$R label --stage initial_model --qualification-stage 0 $B && $R qualify --stage 0 $B
#   (only if stage 0 is not accepted:) label --qualification-stage 1, qualify --stage 1
$R label --stage development $B && $R fidelity-audit $B && $R failures $B
$R label --stage repair_val $B && $R label --stage final_test $B
for A in C_committee D_phal D_noLJ; do $R calibrate-monitor --arm $A $B; done
$R repair-run $B                                          # or a slice: --arm/--case/--seed
$R evaluate $B && $R report --publish $B
```

Every command is resumable and refuses to run if a protocol file changed after the freeze. The smoke
profile (`repair_smoke.yaml`, C8H18, smoke-only CP2K, torsion family, permissive thresholds) runs the
same chain in about 10 minutes and carries no scientific claim.

## 4. Executed so far

**Tests.** Offline invariants of the benchmark (21 tests): split and partition isolation, generation
determinism, sealed-root separation, protocol-freeze enforcement, stage composition, paired seeds and
split rules in run configurations, family-specific promotion gates, success and stability logic,
bootstrap and paired differences, budget stop, refusal of an incomplete sealed evaluation, assessment
from evidence only, audit aggregation, measured cost fallback. Core: metric gates. Closed loop: the
parents manifest. A real-program test (`test_smoke_real.py`, marked `cp2k` and `integration`) runs the
whole smoke chain twice.

**Smoke run on real programs** (`../alkane_artifacts/repair_smoke`; report in
`results/repair_smoke/`). Mechanics only; the numbers carry no scientific meaning.

- **Labels and qualification.** 10 reference labels (9 full, 1 audit fragment; 0.14 core-h) and
  initial-model qualification (accepted at permissive smoke thresholds).
- **Audit, failures and calibration.**
  - fidelity audit: its C8 fragments cover the whole molecule, so they are flagged trivial and count
    as no evidence;
  - development failure analysis ran;
  - the monitors of C and D were calibrated on in-domain trajectories and seeded into the controller.
- **Repair runs.** Six runs (B, C and D, two paired seeds each; the four closed-loop runs used
  0.125 core-h of DFT). In the closed-loop runs:
  - every run produced events and at least one local fragment label inside the challenged region;
  - promotion went through the per-subset gates;
  - provenance, candidates (with rejection reasons), retries and gate reports were recorded.
- **Evaluation.** The sealed evaluation ran once all runs were done; its access was logged. The
  trained-model fragment-vs-parent consistency was measured.

**Cached-data re-execution.** Re-running every command of the smoke chain recomputed nothing:
- byte-identical label stores, frozen manifest, frozen export and CP2K task count;
- an identical sealed evaluation, with a second logged access.

This check found a real defect. A repeated `qualify` re-exported the model, and the TorchScript
re-export is not byte-identical. The frozen-hash check then refused every later command, and GMD
refused a cached segment of the old model, so nothing ran on changed inputs. `qualify` now returns a
finished stage from its record and never touches a frozen file. The smoke export was restored from the
registry's verified copy.

**C16 profile without DFT.**

- Structures: 28 independent bases, 95 structures. Every ordinary structure lies inside every coverage
  window (largest dihedral deviation 32 of 35 deg, angle 10.0 of 10 deg, contact 6.5 A or more). Every
  challenge lies outside its own window: torsion 97-129 deg (gauche well), bend 14-20 deg, contact
  2.1-2.5 A. Contact folds also move torsions out of the window, and this is recorded.
- Preflight: 60 challenge fragments (two shapes per challenge) cut and cap with the frozen setting, at
  29-41 atoms and 1-2 caps, within the 80-atom limit. MD boxes are at most 44 A.

## 5. Arms and endpoints (summary of the protocol)

| Arm | Description |
|---|---|
| A frozen | the initial model, unchanged |
| B continued | three fine-tuning rounds on the original data only, same settings, seeds and gates |
| C committee | committee-only local active learning |
| D PHAL | the staged online monitor as configured |
| D_noLJ | D without the LJ residual (matched ablation) |
| D_nobuffer | D without event buffering (optional; disabled in v1.0, so no buffering benefit is claimed) |

- **Units.** The independent unit is the base conformation. A repair case is one base and one family
  (3 per family). Five seeds per case are paired across arms.
- **Success** on the sealed structures of the run's family needs all of: challenged-region RMSE
  <= 0.20 eV/A; at least 30 % improvement over A on the same structures; in-domain test RMSE no more
  than 10 % above A's; and every stability trajectory intact.

## 6. Execution plan (measured costs) and the proposed staging

Costs come from measured timings on this machine:

| Measured input | Value | n |
|---|---|---|
| Full C16 label | 2630 core-s mean, p90 3892 | 16 |
| Fragment | 638 core-s | 50 |
| Training | 0.028 s per structure-epoch | 64 |
| GMD | 83 steps/s | 28 |

Repair-run DFT is an upper bound: every cycle is assumed to use both labels.

| Stage | Full labels | Fragments | DFT core-h | CP2K wall h | Other CPU |
|---|---|---|---|---|---|
| 1. initial model, qualification stage 0 | 30 | 0 | 21.9 | 4.4 | training 0.6 h, MD 0.2 h |
| 1b. stage 1 (only if stage 0 fails) | 24 | 0 | 17.5 | 3.5 | training 1.1 h |
| 2. development + fidelity audit | 8 | 12 | 7.9 | 1.6 | |
| 3. repair validation + sealed final test | 21 | 0 | 15.4 | 3.1 | |
| 4. repair runs (C, D, D_noLJ; 9 cases x 5 seeds) | 0 | <= 810 | <= 143.7 | <= 28.7 | training about 39 h (incl. B), MD 2.7 h |
| 5. sealed evaluation | 0 | 0 | 0 | 0 | stability MD <= 21.7 h |
| **Total** | | | **<= 206.4** (188.9 if stage 0 passes) | **<= 41** | |

The stage budgets in the YAML (45, 7 + 3, 6 + 14 and 165 core-h) cover the projections. A stage
that would exceed its budget does not start a job.

**Proposed staging, with a decision at each point.**

1. **Stage 1, about 22 core-h.** Freeze the protocol, then label and qualify the initial model. If it
   is not accepted, report the unmet prerequisite, or approve stage 1b (17.5 core-h).
2. **Stage 2, about 8 core-h.** Development failures and the fidelity audit. A family with no
   reference-confirmed failure, or with a failed audit, is reported as untestable for repair.
3. **Stage 3, about 15 core-h.** Repair-validation and sealed final-test labels.
4. **Stage 4.** Repair runs, first one seed of every case and arm (at most 29 core-h), then the other
   four seeds (at most 115 core-h).
5. **Stage 5, no DFT.** The sealed evaluation and the report.

Nothing beyond the smoke profile has been launched. The 550 core-hour campaign of the pilot, and
C64/C128, are out of scope.

## 7. What remains unsupported

Until stages 1-5 run: every scientific claim of the benchmark (the seven assessment questions in the
report are answered "not yet tested"). Even after a complete run, the protocol does not support:

- equilibrium conformational statistics (no reference ensemble);
- other chain lengths;
- an event-buffering benefit (the ablation is disabled);
- equivalence or "no difference" between arms from this sample size.
