# Protocol: repair of force-field failures on withheld alkane conformations (version 1.0)

Benchmark `repair_c16`. Configuration: `configs/demos/alkane_repair/repair_c16.yaml` (with
`repair_c16_alpha.yaml` and `repair_c16_model.yaml`). The YAML files are authoritative for every
number; this document gives the design and the reasons. `freeze-protocol` hashes this document, the
three YAML files and the generated structure sets together; every later command refuses to run if one
of them changed. A changed protocol is a new benchmark with a new name and artifacts root.

## 1. Status of prior evidence

All earlier alkane results (pilot model studies v1, v2, v2b, v2c, the pilot closed loops, the
fragment-fidelity gate and the C32 transfer) predate this protocol. They are exploratory. This protocol
was written after seeing them and was not registered before them. Two pilot findings shaped it:

- the pilot's initial model was inaccurate even on ordinary configurations (validation force RMSE about
  0.4-0.5 eV/A, every ground-truth region above every threshold), so an apparent "repair" could not be
  told apart from ordinary learning. This benchmark therefore requires a qualified initial model first;
- the PHAL arms of the static-pool study were evaluated on unrelated structures treated as a stream.
  Here the primary experiment uses real trajectories and the staged online monitor as configured.

## 2. Research question and endpoints

**Primary question.** Can PHAL detect failures met outside the initial training coverage, obtain
trustworthy local reference labels, update the model, and restore accurate and stable predictions,
while preserving performance inside the original coverage?

Repair is an outcome to establish, not to assume. A completed benchmark may conclude that repair was
partial, ineffective, or not attributable to PHAL's distinctive components.

**Primary endpoints** (sealed final-test structures, per repair run, against the frozen initial model
on the same structures):

1. challenged-region force error (per-component RMSE over the challenged region);
2. a pre-defined absolute accuracy criterion and a pre-defined relative improvement criterion;
3. repair success rate across independent challenge cases;
4. retention: force RMSE on the sealed in-domain test structures;
5. stability: GMD trajectories from every sealed challenge structure of the family with paired seeds
   over a fixed horizon; every trajectory counts in the denominator.

**Secondary endpoints:** labels and update rounds until repair; measured DFT core-hours (failed
attempts and retries included); training and end-to-end wall time; detection latency (first event
step), events that were rejected or never labeled, the in-domain false-alarm fraction of the calibrated
monitor; trigger frequency and repeated acquisition after a promotion.

## 3. Units of analysis

- **Independent unit: the base conformation.** Every structure derived from a base (ordinary samples,
  challenges, MD frames from a trajectory started at one of them, fragments cut from those frames)
  belongs to that base's partition.
- A **repair case** is one repair-partition challenge structure (one base, one family). The repair
  partition holds 3 bases, so each family has 3 cases.
- A **seed** is a paired repetition within a case: the GMD velocity seed and the fine-tune seed
  (offset + seed) are identical across arms. Arms C and D therefore start from the same trajectory and
  differ only from their first monitor decision on.
- Atoms, frames and the several fragments of one parent are never counted as independent repetitions.

## 4. Partitions and leakage control

| Partition | Bases | Structures per base | Closed-loop split | Role |
|---|---|---|---|---|
| id_train | 12 | 4 ordinary | initial_train | initial-model training (qualification stages use the first 6, then 12 bases) |
| id_val | 3 | 2 ordinary | validation | initial-model early stopping and acceptance; retention gate during repair |
| calibration | 3 | 1 ordinary | calibration | monitor calibration trajectories (no DFT) |
| development | 2 | 1 ordinary + 1 per family | calibration | method checks: failure confirmation, fragment-fidelity audit |
| repair | 3 | 1 per family | calibration | MD starts of the repair cases (no DFT) |
| repair_val | 2 | 1 per family | validation | promotion evidence (family improvement gate) |
| final_test | 3 | 2 ordinary + 1 per family | final_eval | sealed outcome measurement only |

Assignment is a seeded permutation of 28 independent bases. A base never straddles partitions (checked
at generation). Closed-loop splits enforce access: validation and final_eval structures can never be an
MD start or enter training. In addition, final-test structures and labels live in a separate sealed
artifacts root. Only the sealed evaluation reads it, and each access is appended to
`sealed_access.jsonl` with the protocol hash.

Final-test labels never influence training, monitor calibration, early stopping, promotion, threshold
selection or stopping decisions. Early stopping and promotion use id_val and repair_val only.

## 5. Intended distribution shift (withheld coverage)

The initial training coverage is defined operationally by three windows. A structure is in-domain
when all hold:

- every backbone C-C-C-C dihedral within 180 +- 35 deg (trans region only);
- every C-C-C angle within the ideal angle +- 10 deg;
- no atoms of chain segments at least 6 carbons apart closer than 3.4 A.

**Ordinary structures** (in-domain): a base backbone with Gaussian dihedral (8 deg) and angle (2 deg)
deviations, per-sample dihedral jitter (5 deg), and 0.04 A thermal displacements. Each must satisfy the
windows and the alkane demonstration's geometry checks (topology, bond lengths, angles, nonbonded
clashes, closed shell); otherwise it is resampled. Every rejection is recorded.

**Challenge structures** (withheld): one rigid perturbation of a base backbone, then thermal
displacement. The challenged descriptor must clear its window by a margin:

- torsion: one dihedral changed by +-120 deg, deviation from trans >= 85 deg and <= 135 deg (the gauche
  well);
- bend: one C-C-C angle changed by +-16 or 20 deg, deviation >= 14 deg;
- contact: a five-bond fold to a segment contact of 2.2 or 2.5 A, contact <= 2.8 A. A fold also moves
  torsions out of the window; this is recorded for every contact structure.

This is withheld conformational coverage, not a proof of extrapolation: a geometric perturbation alone
does not show that the model's input representation lies outside its training distribution. Whether a
challenge exposes a model failure is established separately (section 7).

## 6. Initial model and its acceptance

A GMD_SGT model (local, n_blocks 2, r_c 4 A; primary plus two committee members) is trained on the
full DFT labels of the qualification stage's id_train bases, with id_val as the early-stopping set.
Acceptance requires all of:

- id_val force RMSE <= 0.10 eV/A per component, and id_val 95th-percentile atom error <= 0.30 eV/A;
- GMD runs of 4000 steps (2 ps, 300 K) from every calibration and id_val structure with seeds 1 and 2:
  the bond topology (covalent scale 1.3) survives in every frame of every trajectory;
- the ML force field only, and no GMD directive that clips forces or constrains the dynamics. Monitor
  stop actions end a trajectory and never modify forces.

Stage 0 uses 6 id_train bases (24 structures). Stage 1 (12 bases, 48 structures) is labeled and
trained only if stage 0 is not accepted. If neither stage is accepted, the prerequisite is reported as
unmet and no repair experiment runs: a repair success is never manufactured from an inadequate
baseline.

The accepted model is frozen: checkpoint, committee and export hashes, dataset manifest, species,
model and training sections, atomic energies and seed. Repair runs use exactly these files. Its
in-domain performance is reported through the sealed evaluation; id_val is a validation set, not a
test set.

## 7. Reference-confirmed failures

For each challenge structure with a full DFT label, the challenged region is the perturbation-site
atoms plus carbons within one C-C bond, with their hydrogens. A challenge exposes a **failure** of the
initial model when its region force RMSE is at least 0.30 eV/A and at least 3 times the qualified
id_val force RMSE. The DFT force magnitude in the region is reported alongside. A physically large
force that the model reproduces is therefore not counted as a failure.

Development structures are used to check that failures exist and to configure nothing automatically.
The confirmatory failure rates come from the sealed final-test structures. Every outcome is reported,
including challenges where the initial model does not fail.

## 8. Local labels: fidelity and consistency

Local labels are capped fragments with the frozen setting of the pilot's C16 fragment-fidelity gate
(cluster, core 1.5 A, buffer 3 A). That gate passed on static C16 torsion, bend and contact
structures. Its coverage of this benchmark's geometries is **audited** on the development challenges:
for each challenge, a fragment around the challenged site and a fragment of the closed loop's hotspot
shape, with DFT core forces compared against the full parent. The targets are the pre-registered gate's:
pooled core RMSE <= 0.03 eV/A and maximum <= 0.10 eV/A. A family whose audit fails may not use local
labels.

Separately, the **trained model's** core forces on each acquired capped fragment are compared with its
forces on the full parent frame the fragment was cut from. Passing the DFT audit does not establish this
model-side consistency, nor the reverse; both are reported.

Training weights: core 1.0, inner buffer 0.3, outer buffer, boundary and caps 0. A capped fragment's
energy is never used: it is not the parent's energy.

## 9. Arms, triggers and acquisition

All arms start from the same frozen checkpoint and the same paired seeds.

| Arm | What it is | Labels |
|---|---|---|
| A frozen | the initial model, unchanged | none |
| B continued | three fine-tuning rounds on the original id_train data only, same settings, seeds and gates as C/D | none |
| C committee | closed loop; monitor scores committee force deviation only, every frame | local, CP2K |
| D PHAL | closed loop; PHAL's staged online monitor as configured in `repair_c16_model.yaml` | local, CP2K |
| D_noLJ | D without the local LJ residual (matched ablation) | local, CP2K |
| D_nobuffer | D without event buffering (optional; disabled in v1.0) | local, CP2K |

C and D share extraction, capping, labeling, training, promotion, MD and budgets. Only the monitor
configuration differs, so their comparison isolates the acquisition policy as closely as the design
allows. In C, the r_min metric is kept with weight 0 so the same MD safety stop applies.

**Closed loop** (GMD_active's persistent controller, unchanged): GMD trajectory (300 K, 0.5 fs)
started at the repair case's challenge structure in a vacuum-padded box sized for the model's 16 A
interaction range plus a 4 A margin and 3 A stretch allowance. Then:

1. online monitoring and event formation (one pre- and one post-trigger frame, 200-step cooldown);
2. hotspot grouping (merge radius 3 A);
3. local extraction and capping with the frozen setting, refused if the frame's bond topology changed;
4. CP2K with the pilot's settings;
5. dataset update and GMD_SGT fine-tuning from the active model (60 epochs);
6. export check, GMD smoke test and promotion gates;
7. GMD restart with the accepted model.

Each run has three cycles of 2000 steps (1 ps each), at most two labels per cycle and one per event,
a duplicate filter at 0.1, and a cap of 1.2 measured CP2K core-hours.

**Monitor calibration.** Before any repair run, the frozen model runs in-domain trajectories from the
calibration structures (2 seeds x 2000 steps). These frames calibrate each arm's running statistics,
which are then seeded into the controller. The challenge trajectory itself is never used for warm-up.
A run is refused before any CP2K job if one of the arm's required metrics is still in warm-up or if
the calibrated monitor triggers on more than 2 % of the in-domain frames. Required metrics: C needs
committee; D needs force, delta-force, displacement and delta-q. Each run records:

- per-stage frame counts;
- every event's scores and trigger reasons;
- every candidate with its selection outcome and rejection reason (duplicates included);
- every label's attempts, retries and validation;
- every model version and gate report;
- the potential-energy discontinuity at every model change.

**Promotion** (repair-validation and id_val, never final test):

- the candidate's repair-validation force RMSE for the run's family must fall by at least 10 % relative
  to the active model;
- its id_val force RMSE may rise by at most 10 %;
- its overall validation RMSE may rise by at most 10 % and must stay below 0.5 eV/A (a sanity bound,
  not evidence of accuracy);
- the export must reproduce the checkpoint within 1e-4 eV/A, and the model must pass a GMD smoke test.

A rejected candidate leaves the active model unchanged.

## 10. Evaluation

After every planned run has finished, the sealed evaluation computes, for the frozen model and for each
run's final model, on the same sealed structures:

- challenged-region force errors on the sealed challenges of the run's family;
- force errors on the sealed ordinary structures (retention);
- stability: GMD trajectories of 4000 steps from each sealed challenge of the family with seeds 1, 2 and
  3. A trajectory is stable when every frame keeps the bond topology; every trajectory counts.

**Repair success** for a run requires all of:

- sealed challenged-region RMSE <= 0.20 eV/A;
- a relative improvement >= 30 % over the frozen model on the same structures;
- sealed in-domain test RMSE no more than 10 % above the frozen model's;
- every stability trajectory stable.

An intact chain is structural stability, not evidence of a correct potential-energy surface.
Conformational observables, if reported, are conditioned on intact frames, and that conditioning is
stated. Without a reference ensemble, no claim about equilibrium conformational statistics is made.

**Statistics.** Raw per-run and paired results are always published. Summaries are per arm and family:
success counts, and the mean over bases of each base's mean relative improvement. A percentile bootstrap
over bases (2000 resamples) is reported with the number of bases, and is crude with three bases. No
equivalence and no "no difference" is inferred from a small, inconclusive sample.

## 11. Planned comparisons

- D vs A: does PHAL repair? (primary)
- D vs B: is the improvement more than additional optimisation on the original data?
- D vs C: what do the physical signals add over committee-only acquisition?
- D vs D_noLJ: what does the LJ residual add? (matched ablation)
- D vs D_nobuffer: event buffering, along trajectories only. Disabled in v1.0; no benefit is claimed.
- Local-vs-full labeling with a frozen candidate list is not retained in v1.0. The pilot measured the
  cost ratio; the repair question does not depend on it.

A claim that PHAL repairs failures does not require D to beat every baseline. A claim that its physical
signals improve repair requires D to beat C (and D_noLJ for the LJ residual) on the paired results.

## 12. Budgets, stages and stopping

Measured DFT CPU core-hours per stage (failed attempts and retries count):

| Stage | Budget |
|---|---|
| initial model | 45 |
| development | 7 |
| fidelity audit | 3 |
| repair validation | 6 |
| final test | 14 |
| repair runs | 165 |

A job whose estimate would exceed its stage budget is not started. Stopping conditions:

- the initial model is not qualified (stop; report the unmet prerequisite);
- a family shows no reference-confirmed failure on development structures (its repair claim is reported
  as untestable);
- a family fails the fidelity audit (no local labels for it);
- the repair-run budget is reached (no further run; the evaluation is reported as partial).

The execution plan (`plan` command) costs every stage from measured timings of earlier runs on this
machine. Expensive stages run only with an explicit compute allowance.

## 13. Deviations

Any deviation from this protocol (a failed stage, a re-run, a changed setting) is recorded in the
report. A changed setting requires a new protocol version and benchmark name. Results of a deviating
run are reported as exploratory.
