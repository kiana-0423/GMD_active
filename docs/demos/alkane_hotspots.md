# Alkane hotspot demonstration

A publication-oriented test of PHAL (hotspot-local active learning) in GMD_active on long-chain
n-alkanes. It asks whether capped fragments reproduce reference forces well enough to be used as
labels, and whether hotspot-local labeling helps a GMD_SGT potential. The hypotheses are tested, not
assumed. Negative results are kept.

Code: `tools/experiments/alkane_hotspots/` (run as `python -m tools.experiments.alkane_hotspots`).
Configurations: `configs/demos/alkane_hotspots/{smoke,pilot,campaign}.yaml`. Entry scripts:
`examples/alkane_hotspot_demo/`. Tests: `tests/experiments/alkane_hotspots/`.

Artifacts (structures, CP2K tasks, labels, fragments, reports) are written outside git to the
configuration's `artifacts_root`. The default is `../alkane_artifacts/<config name>` relative to the
GMD_active root, that is `GMD_system/alkane_artifacts/<name>/`. Only small summaries are committed,
under `docs/demos/alkane_hotspots/results/`.

**Status (2026-11-07).** *Executed:*

- the CP2K timing and cutoff-convergence probe (`results/timing_probe.json`);
- the C16 pilot: 16 parents, all 16 full-system DFT labels, and the fragment-fidelity gate on
  calibration and on final_eval (`results/fidelity_pilot/`);
- the real-CP2K smoke run (C8H18 at smoke-only settings).

The model study, the closed loop and the C32 transfer are reported in their own sections, marked
executed or planned. *Planned, not executed:* the campaign. No number appears on this page unless it
was measured.

## Commands

All commands take `--config` (and optionally `--artifacts-root`). They are deterministic and
resumable: re-running a command reuses finished work and never starts a second job for the same
calculation.

```bash
CFG=configs/demos/alkane_hotspots/pilot.yaml
python -m tools.experiments.alkane_hotspots generate         --config $CFG   # parents, checks, splits (no DFT)
python -m tools.experiments.alkane_hotspots fidelity-prepare --config $CFG --split calibration   # fragments; freezes the targets
python -m tools.experiments.alkane_hotspots status           --config $CFG   # budgets, measured timing, remaining-cost estimate
python -m tools.experiments.alkane_hotspots fidelity-run     --config $CFG --split calibration   # CP2K: calibration parents + fragments
python -m tools.experiments.alkane_hotspots fidelity-report  --config $CFG --split calibration   # metrics; selects and freezes, or records failure
python -m tools.experiments.alkane_hotspots fidelity-prepare --config $CFG --split final_eval    # frozen setting only
python -m tools.experiments.alkane_hotspots fidelity-run     --config $CFG --split final_eval
python -m tools.experiments.alkane_hotspots fidelity-report  --config $CFG --split final_eval
python -m tools.experiments.alkane_hotspots label-full       --config $CFG [--splits ...]       # remaining full-parent labels
```

`fidelity-run` and `label-full` accept `--max-wall-s N`. After N seconds the command stops waiting:
running CP2K jobs keep going detached, and the next invocation picks them up. The smoke pipeline is
also one script:
`python examples/alkane_hotspot_demo/run_fidelity_gate.py --config configs/demos/alkane_hotspots/smoke.yaml`.

## Reproducing the pilot and launching the campaign

Run everything from the GMD_active root, in the `gmd-active` environment. CP2K jobs run one at a
time; concurrency gave no throughput on the development machine. Each line is resumable.

```bash
M=python\ -m\ tools.experiments.alkane_hotspots; D=configs/demos/alkane_hotspots
# 1. C16 pilot: parents, labels, fragment-fidelity gate (~18 core-h of DFT, measured)
$M generate --config $D/pilot.yaml
$M fidelity-prepare --config $D/pilot.yaml --split calibration
$M fidelity-run     --config $D/pilot.yaml --split calibration
$M fidelity-report  --config $D/pilot.yaml --split calibration       # selects and freezes the setting
$M label-full       --config $D/pilot.yaml --splits initial_train validation
$M fidelity-prepare --config $D/pilot.yaml --split final_eval
$M fidelity-run     --config $D/pilot.yaml --split final_eval
$M fidelity-report  --config $D/pilot.yaml --split final_eval
# 2. model study + closed loop, v1 (pre-registered settings) and v2 (post-hoc fixes); same labels
for S in arms_pilot arms_pilot_v2; do
  $M ground-truth --config $D/pilot.yaml --study $D/$S.yaml
  $M arms-run     --config $D/pilot.yaml --study $D/$S.yaml
  $M evaluate     --config $D/pilot.yaml --study $D/$S.yaml
  $M figures      --config $D/pilot.yaml --study $D/$S.yaml
  $M report       --config $D/pilot.yaml --study $D/$S.yaml --publish
done
for CL in closed_loop_pilot closed_loop_pilot_v2; do
  $M closed-loop-run --config $D/pilot.yaml --closed-loop $D/$CL.yaml
  $M frozen-eval     --config $D/pilot.yaml --closed-loop $D/$CL.yaml
  $M recovery-demo   --config $D/pilot.yaml --closed-loop $D/$CL.yaml
done
# 3. diagnostics (no DFT)
python examples/alkane_hotspot_demo/initial_model_diagnostic.py --study $D/arms_pilot.yaml
python examples/alkane_hotspot_demo/md_stability_check.py --closed-loop $D/closed_loop_pilot.yaml --model NAME=PATH ...
# 4. C32H66 size transfer of the frozen C16 setting (~2.7 h wall per full label)
$M generate --config $D/pilot_c32.yaml
mkdir -p ../alkane_artifacts/pilot_c32/fidelity
cp ../alkane_artifacts/pilot/fidelity/{preregistration,selection}.json ../alkane_artifacts/pilot_c32/fidelity/
$M label-full       --config $D/pilot_c32.yaml --splits final_eval
$M fidelity-prepare --config $D/pilot_c32.yaml --split final_eval
$M fidelity-run     --config $D/pilot_c32.yaml --split final_eval
$M fidelity-report  --config $D/pilot_c32.yaml --split final_eval
```

**Main campaign, stage 1** (C16 + C32, 64 parents, 550 core-h budget, about 110 h wall serially on
the development machine; planned, not executed):

```bash
M=python\ -m\ tools.experiments.alkane_hotspots; D=configs/demos/alkane_hotspots; C=$D/campaign.yaml; S=$D/arms_campaign.yaml
$M generate --config $C && $M status --config $C
$M fidelity-prepare --config $C --split calibration && $M fidelity-run --config $C --split calibration
$M fidelity-report  --config $C --split calibration      # stop here if the gate fails (failure.json)
$M label-full --config $C --splits initial_train validation
$M fidelity-prepare --config $C --split final_eval && $M fidelity-run --config $C --split final_eval
$M fidelity-report  --config $C --split final_eval
$M ground-truth --config $C --study $S && $M arms-run --config $C --study $S
$M evaluate --config $C --study $S && $M figures --config $C --study $S && $M report --config $C --study $S --publish
```

`campaign.yaml` sets the fidelity targets, budgets and splits; its pre-registration is written at
the first `fidelity-prepare`. Before launching, check `status`: it estimates the remaining cost from
measured timings. A closed-loop configuration for the campaign is not provided. Write one from
`closed_loop_pilot_v2.yaml` once the stage-1 gate has passed.

## Systems

**Structures.** Each parent is an n-alkane C(n)H(2n+2) with a fixed atom order: carbons `0..n-1`
along the chain, then the hydrogens grouped by carbon. The `atom_id` in the extxyz file equals the
index, so a perturbation `location` (carbon index) is also an atom index.

- **Base structure.** The all-trans chain is built exactly: C-C 1.53 A, C-H 1.09 A, and every
  C-C-C, H-C-C and H-C-H angle at the tetrahedral value arccos(-1/3). The CH3 ends are staggered.
  Tests check this to 1e-9 deg.
- **Base conformation (the split group).** One chain length plus one seeded Gaussian displacement
  field, with standard deviation `families.thermal.amplitude` per Cartesian component (0.03 A in the
  pilot). Every parent of a base gets the *same* field, added after its perturbation.

**Families.** The grids come from `families.*` in the configuration. Locations are fractions along the
chain.

| Family | Construction | Changed internal coordinates |
|---|---|---|
| `normal` | base conformation only | none |
| `torsion` | rigid rotation of the side containing C(k+1) about the C(k)-C(k+1) axis by `deltas_deg` | every dihedral X-C(k)-C(k+1)-Y changes by delta; all bond lengths and angles are exact |
| `bend` | rigid rotation of the C(k+1) side about the axis through C(k) perpendicular to the C(k-1)-C(k)-C(k+1) plane | the C-C-C angle at k changes by delta. The two H-C(k)-C(k+1) angles necessarily change with it, because a rigid rotation cannot change one angle at a tetrahedral centre alone. Nothing else changes. |
| `contact` | `fold_bonds` consecutive backbone torsions from C(k), optimized deterministically (best trans/gauche starts, then Powell) | the minimum distance between atoms of two segments at least `min_segment_separation` carbons apart reaches `target_distances` within `target_tolerance`; every other nonbonded pair keeps the clash minima plus `clash_margin` |

**Perturbation site atoms** (`perturbation.atoms`):

- torsion: carbons k-1..k+2 and their H;
- bend: carbons k-1..k+1 and their H;
- contact: the two carbons of the closest inter-segment pair and their H.

**Geometry checks** (`checks.*`). A candidate is rejected, and recorded in `rejected.jsonl`, unless all
of the following hold:

- **Topology:** the bond topology from `utils.bonds.infer_bonds` equals the parent's, so nothing
  dissociated and no new bond formed.
- **Bond lengths:** every bond is within `bond_tolerance` of the sum of covalent radii.
- **Angles:** untouched angles stay within `max_angle_deviation_deg` of tetrahedral after noise.
- **Exactness of the rigid operation** (before noise): bond lengths change by at most
  `exact_tolerance` and untouched angles by at most `exact_angle_tolerance_deg`. The requested
  torsion, bend or contact target is reached.
- **Clashes:** H..H, C..H and C..C distances between atoms at least
  `nonbonded_min_bond_separation` bonds apart stay above their minima.
- **Contact window** (contact family only): the contact distance lies in `contact_window`.
- **Electronic state:** the molecule is neutral, closed shell and saturated.

The slots of a base are filled from the grid in a seeded order. If a slot cannot be filled,
generation stops with an error; no threshold is relaxed.

**Splits.** Groups are base conformations. They are allocated to `initial_train`, `calibration`,
`validation` and `final_eval` by largest remainder of `splits.fractions`. Within each chain length
the groups are ordered by a seeded hash. The strata are then interleaved and dealt in
`splits.assignment_order`, so calibration and final_eval each get one base of every chain length in
the pilot. A test checks that no group spans two splits.

**Parent record** (`parents.jsonl`): `parent_id`, `chain_length`, `formula`, `family`, `seed`,
`perturbation {kind, atoms, location, magnitude, units, ...}`, `split`, `split_group`,
`geometry_checks`, `structure_path`, and `config_hash` (the hash of the generation-relevant config
sections). The same configuration gives byte-identical `parents.jsonl` and extxyz files (tested).

**Pilot (generated, no DFT).** 24 parents: C16H34 and C32H66, 3 bases per length, one parent per
family per base. Splits: calibration and final_eval have 2 groups each (one C16 and one C32, 8
parents each); initial_train and validation have 1 group each. 3 contact candidates were rejected
for H..H clashes. Summary: [results/pilot_parents_summary.json](alkane_hotspots/results/pilot_parents_summary.json).

**Campaign (planned).** `campaign.yaml` adds C64 and C128, behind `systems.allow_long_chains`.
An all-trans C128 chain is about 160 A long, so its isolated cubic WAVELET box would be about 170 A
wide. Full-parent labels of C64/C128 are therefore probably infeasible on one workstation. The
campaign is to be re-planned from the pilot's measured costs.

## Electronic-structure settings

Parents and fragments are labeled by one `LabelingService` (CP2K adapter, local scheduler), so they
use one CP2K configuration and one settings hash. Every label record carries the hash and the
physics settings; the runner refuses a task whose hash differs. A test asserts that a parent and a
fragment share the same hash and settings.

| Setting | Pilot | Key |
|---|---|---|
| Functional | PBE | `cp2k.functional` |
| Basis / pseudopotential | DZVP-MOLOPT-SR-GTH / GTH-PBE | `cp2k.basis`, `cp2k.potential` |
| Dispersion | DFT-D3(BJ) | `cp2k.dispersion` |
| Charge / multiplicity | 0 / 1 (closed-shell RKS; `spin_policy: strict`) | `cp2k.charge`, `cp2k.multiplicity` |
| Plane-wave cutoff | 400 Ry, REL_CUTOFF 60 Ry | `cp2k.cutoff`, `cp2k.rel_cutoff` |
| SCF | OT/DIIS, EPS_SCF 1e-6, MAX_SCF 100, bounded retry ladder | `cp2k.eps_scf`, `cp2k.max_scf` |
| Boundary | isolated (`PERIODIC NONE`), WAVELET Poisson solver, cubic box = extent + 2 x 6 A | `cp2k.poisson_solver`, `cp2k.box_padding` |
| Cap relaxation | off in the pilot; when on, H-only GEO_OPT with every original atom fixed | `cp2k.h_only_opt` |
| Parallelism | 5 OpenMP threads, 1 MPI rank (`cp2k.launcher` for MPI) | `cp2k.omp_threads`, `cp2k.mpi_ranks` |

Each CP2K job runs with the per-job walltime `budgets.per_job_walltime_s`, enforced by the local
scheduler. CP2K's version (`cp2k.psmp --version`, then the version in each output) and the settings
are written to every manifest (`<artifacts_root>/manifests/*.json`) and label record.

`cp2k.data_dir: null` resolves to `$CP2K_DATA_DIR`, or else to `<prefix>/share/cp2k/data` next to
the executable. No path is hard-coded.

**Label record** (`labels.jsonl`, one per calculation, rewritten atomically):

- `label_id`: `stable_id` over the system content and the settings hash, so the same system with the
  same settings is one calculation; identical fragments shared by several parents run once;
- `kind`, `parent_id`, `fragment {policy, hops | core_radius + buffer_radius, buffer, core_parent_ids,
  n_atoms, n_core, n_caps, ...}` or null;
- `region_path` (the labeled region, with relaxed caps if any) and `forces_path` (npy, eV/A, in the
  region's atom order);
- `status`: valid, invalid, failed or budget_skipped;
- `energy_ev`: the system's own energy, never the parent's;
- `scf_steps`, `cp2k {version, settings_hash, settings, smoke_only}`, `task_id`, `created_at`;
- `cost {wall_s, cpu_core_s, attempts, retries_wall_s, stage_wall_s, timing_sources}`.

**Measured cost.** The wall time of each attempt is the time from the local scheduler's recorded
start to the modification time of the job's exit-code file. `cpu_core_s` is that wall time multiplied
by the allocated cores. Nothing is estimated from atom or call counts.

**Budgets** (`budgets.*`) are checked before each new job is launched:

- the atom limits for full parents and for fragments;
- the number of full and of fragment labels;
- the total CPU core-hours: spent so far, plus the estimates for running jobs, plus the estimate for
  the new job.

The estimate is the measured mean of finished labels of the same kind (full: same chain length).
Before any measurement exists it is the walltime x cores upper bound. Refused jobs get a
`budget_skipped` record with the reason and the estimate, which is also printed. `status` reports
spent and remaining budget, measured timing, and the estimated cost of the remaining work.

**Measured so far.**

- *Smoke, smoke-only settings* (C8H18; 200 Ry, REL_CUTOFF 40, EPS_SCF 1e-5, 4 A padding; 2 OpenMP
  threads; CP2K 2026.2):
  - full parents: 38.2 s and 28.2 s wall (20 and 21 SCF steps);
  - one graph-hop-2 fragment of 23 atoms with one cap: 122.9 s wall, of which 80.2 s was the
    cap-only GEO_OPT and 42.7 s the single point.

  These numbers check the pipeline only; they are not pilot costs.
- *Pilot settings, measured by the lead outside this pipeline:* one full C16H34 single point took
  801 s wall (44 SCF steps) at 5 OpenMP threads, which is 1.1 core-hours. 5 MPI ranks were not faster.

## Fragment-fidelity gate

**Question.** Does a capped fragment reproduce the parent's DFT forces on the same original core
atoms? This is the first scientific gate. If it fails, fragment labels are not used as training
targets, and the failure is reported.

**Sites.** Each calibration parent provides one or two sites (`fragments.sites`):

- the perturbation site;
- a normal-site control: a torsion-like site (carbons k-1..k+2 and their H) placed as far as
  possible from the perturbation in bonds. For a normal parent it sits at `normal_location`.

**Fragments.** Fragments are cut with the existing extraction and capping
(`extract_regions` + `prepare_region`) and `extraction.terminal_atoms: follow_partner`, so every
carbon keeps all its H and no H is left without its carbon.

| Policy | Fragment | Core (atoms whose error is scored) |
|---|---|---|
| graph, hops h (`fragments.graph_hops`) | carbons within h C-C bonds of the site carbons, plus their H | the site atoms |
| cluster (`core_radius`, `buffer_radius`) | whole CH_n units within core + buffer of the site atoms | atoms within `core_radius` of the site atoms (region role CORE) |

Every cut C-C bond is capped with H at 1.09 A along the bond. Partial capping is an error. Parent atom
IDs, roles and cap records are kept. A cap closer than `fragments.cap_clash_distance` to another atom
rejects the fragment. That rejection counts as a failure of the setting.

**Identical settings.** Fragments use the same CP2K settings as their parents. With
`cp2k.h_only_opt.enabled`, only the caps move, and every original atom is checked to stay fixed. No
alignment or relaxation of the original atoms takes place.

**Pre-registration.** The first `fidelity-prepare --split calibration` writes
`fidelity/preregistration.json`. It holds the targets, the sweep, the CP2K physics settings and the
calibration parents. Any later change to those is refused, and thresholds are never lowered. The
pilot targets are:

- pooled `core_force_rmse` at most 0.03 eV/A;
- maximum core force-vector error at most 0.10 eV/A;
- `max_failures: 0`.

**Metrics** (`fidelity-report`). For each fragment:

- `core_force_rmse = sqrt(mean((F_frag_core - F_parent_core)^2))` over the core atoms and their
  Cartesian components;
- the maximum `|F_frag - F_parent|` over the core atoms;
- per core atom: the error against the bond distance to the nearest cut bond, the distance to the
  nearest cap and to the nearest excluded parent atom, and the number of excluded parent atoms
  within `fidelity.environment_radius` (missing environment);
- fragment size, number of caps, failures, and the measured cost next to the parent's.

Outputs are `fidelity/metrics_<split>.jsonl`, `per_atom_<split>.csv` and `report_<split>.json`.

**Selection** (calibration only). A setting passes when it is complete, has at most `max_failures`
failures, and its pooled RMSE and maximum error meet the targets. The passing setting with the lowest
measured mean fragment cost is frozen in `fidelity/selection.json`; later reports do not change it.
If settings are still incomplete (unlabeled or budget-skipped), the decision stays `undecided`. If
every setting is complete and none passes, `fidelity/failure.json` records the diagnosis:

- the error trend with distance to the boundary;
- the correlation between missing environment and error, with the worst fragments;
- label and SCF failures;
- the closest setting.

**Final evaluation.** `fidelity-prepare --split final_eval` refuses to run without a frozen selection
and builds fragments for the selected setting only. Its report never feeds back into the selection.

**Pilot sizing (from the measured timing probe).** The probe (`results/timing_probe.json`) changed
the plan before any pilot label was run:

- a C32H66 full label takes about 2.7 h wall (13.5 core-h), extrapolated from SCF steps measured
  alone;
- concurrent CP2K jobs and MPI give no throughput on this machine, so jobs run one at a time;
- 300/50 Ry differs from 400/60 Ry by 0.006 eV/A RMS and 0.021 eV/A at most on C16H34. That is below
  the 0.03 eV/A target, so parents and fragments both use 300/50 Ry.

The pilot (`pilot.yaml`) is therefore C16H34 only: 16 parents (4 per split, one per family) and a
40 core-h budget. C32H66 is a separate transfer check (`pilot_c32.yaml`, below). C64 and C128 were
not attempted: their isolated cubic boxes are not affordable on this machine.

### Executed: C16 pilot fidelity gate (2026-11-06/07)

Copies: `results/fidelity_pilot/`. These are the pre-registration, the selection, both reports, and
the sha256 of the labels and per-atom tables they were computed from. Pre-registration hash
`640c118d…`. CP2K 2026.2, PBE-D3, DZVP-MOLOPT-SR-GTH, 300/50 Ry, isolated wavelet box, 5 OpenMP
threads.

**Calibration** (4 parents, 7 sites; the normal parent has only its control site):

- six settings, 42 fragment specs, sharing 22 distinct fragment calculations;
- every label is valid and every setting is complete, with 0 failures and 0 retries;
- mean measured cost: full C16 parent 2370 core-s; fragments 508-1095 core-s, depending on the
  setting.

| Setting | Pooled core RMSE (eV/A) | Max core error (eV/A) | Mean atoms | Mean caps | Mean cost (core-s) | Passes |
|---|---|---|---|---|---|---|
| graph h2 | 0.0278 | **0.2125** | 23.7 | 1.71 | 508 | no (max error) |
| graph h3 | 0.0108 | 0.0521 | 28.6 | 1.43 | 692 | yes |
| graph h4 | 0.0113 | 0.0552 | 32.9 | 1.29 | 904 | yes |
| cluster c1.5 b3 | 0.0118 | 0.0747 | 28.6 | 1.43 | 692 | yes (**selected**) |
| cluster c1.5 b4 | 0.0113 | 0.0552 | 33.6 | 1.43 | 879 | yes |
| cluster c1.5 b5 | 0.0095 | 0.0515 | 37.9 | 1.14 | 1095 | yes |

- **Graph h2 fails.** It passes the pooled-RMSE target narrowly but has a maximum error of
  0.21 eV/A. Its errors sit on core atoms two to three bonds from a cut:
  - [2,3) bonds: mean 0.067 eV/A, max 0.145;
  - [3,4) bonds: mean 0.036, max 0.212;
  - [4,6) bonds: mean 0.008;
  - [6,∞) bonds: mean 0.001.

  So the failure comes from the boundary, not from the perturbation.
- **The selected setting**, by the pre-registered rule (lowest measured cost among passing settings),
  is cluster c1.5 b3. For C16 it builds the same fragments as graph h3 (same atom counts, same shared
  calculations, same measured cost). The two differ only in which atoms count as core: graph uses the
  site atoms, cluster uses atoms within 1.5 A of the site. The tie was broken by the fixed setting
  order. With cluster c1.5 b3, the error falls with the distance to the cut:

  | Bond distance to cut | n | Mean (eV/A) | Max (eV/A) |
  |---|---|---|---|
  | [2,3) | 2 | 0.055 | 0.075 |
  | [3,4) | 10 | 0.034 | 0.052 |
  | [4,6) | 47 | 0.009 | 0.048 |
  | [6,∞) | 22 | 0.005 | 0.046 |

- **Perturbation sites vs normal controls:** 0.0139 vs 0.0098 eV/A RMSE. Distorted regions are not
  reproduced notably worse.

**Final evaluation** (4 untouched parents; frozen setting only; 7 specs, 6 CP2K calculations):

- pooled core RMSE **0.0100 eV/A**; max core error **0.049 eV/A**; 0 failures. **The frozen setting
  passes on held-out parents.**
- perturbation sites 0.0093, normal controls 0.0101 eV/A;
- mean fragment 28.6 atoms and 561 core-s, against 2535 core-s for a full C16 parent;
- by bond distance to the cut: [3,4) mean 0.032; [4,6) mean 0.008; [6,∞) mean 0.006 eV/A.

**What this supports.** At these DFT settings, for C16H34 torsion, bend and contact distortions, a
capped fragment of about 29 atoms (C16H34 has 50) reproduces the parent's forces on the scored core
to about 0.01 eV/A RMS. It costs about a quarter of the full single point: 0.22-0.29 measured.

**What it does not support.**

- Larger savings for longer chains. They are expected, because fragment cost does not grow with
  chain length, but they are not measured here; see the C32 transfer below.
- Fidelity for atoms near the boundary. Fragments are scored on their core only. Buffer atoms near
  the cut carry errors of 0.03-0.2 eV/A and must get zero or small training weight
  (`training.force_weights`).
- Any claim beyond 4+4 parents. These are small samples, and no confidence interval is
  claimed.

Total measured DFT for the pilot labels (16 full + 28 calibration and final_eval fragments): 17.9
core-h.

**Smoke result (smoke-only settings, not a scientific result).** C8H18 torsion parent, perturbation
site, graph hops 2, one cap at two bonds from the nearest core atom:

- core RMSE 0.061 eV/A, maximum error 0.179 eV/A;
- the gate failed against the 0.03/0.10 targets and wrote `failure.json`.

This shows that the gate can fail and does record the failure. Whether the pilot settings pass is
open.

## Model study (part beta): commands

The model-study commands are registered on the same CLI. They take the demo configuration (`--config`,
which supplies `artifacts_root`) and the study configuration (`--study`,
`configs/demos/alkane_hotspots/arms_{smoke,pilot,campaign}.yaml`); without `--study` they read the
demo configuration's `model_side` section. Outputs go to `<artifacts_root>/model_study/`.

```bash
CFG=configs/demos/alkane_hotspots/pilot.yaml; STUDY=configs/demos/alkane_hotspots/arms_pilot.yaml
python -m tools.experiments.alkane_hotspots receptive-field --config $CFG --study $STUDY   # no DFT
python -m tools.experiments.alkane_hotspots ground-truth    --config $CFG --study $STUDY   # needs full labels (initial_train, validation, calibration normals)
python -m tools.experiments.alkane_hotspots arms-run        --config $CFG --study $STUDY [--only phal_local] [--repetitions 0]
python -m tools.experiments.alkane_hotspots evaluate        --config $CFG --study $STUDY
python -m tools.experiments.alkane_hotspots figures         --config $CFG --study $STUDY
python -m tools.experiments.alkane_hotspots report          --config $CFG --study $STUDY [--publish]
```

`arms-run` is resumable: every (arm, repetition) keeps `model_study/arms/<arm>/rep<k>/progress.json`,
rewritten atomically after each round, and a finished round is never redone. A progress file written
under a different configuration is refused. Trainings with identical data, seed and training section
share one directory under `model_study/models/`, so the round-0 model is trained once per seed and
shared by every arm (and by `ground-truth`). `report --publish` copies `report.json`/`report.md` to
`docs/demos/alkane_hotspots/results/model_study_<name>/` with machine paths replaced by
`<artifacts_root>` / `<GMD_active>`.

## Model receptive field

`receptive_field.py` measures how the force on a core atom of the demo's GMD_SGT model depends on
atoms that extraction removes. The model is either the configured architecture with a seeded random
initialisation, or a trained checkpoint. GMD_SGT zero-initialises its last readout layer, so every
force of a fresh model is exactly 0. The random-init mode therefore redraws every all-zero weight
matrix from N(0, `reinit_zero_weights_std`) with a fixed seed and records which ones. This measures
the architecture's dependency structure, not trained magnitudes. The model runs in float64 and eval mode.

Nominal values. Message passing builds edges within `local_cutoff` r_c. After `n_blocks` T blocks a
per-atom energy depends on atoms at most T hops, so at most T r_c, away. The force
F_i = -dE/dr_i = -sum_j dE_j/dr_i collects every E_j that depends on r_i, so **the force depends on atoms
up to 2T hops (2 T r_c) away**, not T r_c. Any `long_range_type` other than `none` is global within a
structure: every atom can influence every force.

Measurements on the core atom's force (eV/A):

- `perturb`: displace one atom by `displacement` along a seeded random unit vector.
- `delete`: remove one atom.
- `truncate`: keep only the atoms within R of the core, with no caps.

Each measurement gives three radii: absolute `tolerance`, `relative_tolerance` times the largest
single-atom influence, and `structural_tolerance` (above float64 round-off, i.e. any dependence).

The tests use tiny real GMD_SGT models. For a local model (T = 1, r_c = 2 A), every atom beyond
2 T r_c changes the core force by less than 1e-12 eV/A. For an `invariant_attention` model, atoms far
outside every cutoff still change it by more than the tolerance.

## Hotspot ground truth

`ground_truth.py` defines hotspots from reference DFT errors, never from PHAL scores. The definitions
are also in the module docstring.

- Atom error: e_i = |F_model,i - F_DFT,i|. Only configurations with a valid full-system label are used.
  F_model is the primary model's force.
- Regions: one per heavy atom. A region holds the heavy atoms within `radius_bonds` bond hops of its
  centre (covalent graph, `bond_scale`) plus their hydrogens. Regions overlap, so they are correlated
  samples; uncertainties are therefore bootstrapped over `split_group`, not over regions.
- Region error E_R is the max, rms or mean of e_i. A region is positive at threshold tau when E_R >= tau
  (`reference_thresholds`, eV/A). The region score is the max of the signal over the region.
- Metrics:
  - average precision (the PR-curve area) and ROC-AUC (ties count 1/2);
  - precision, recall, FPR and F1 at the signal's YAML operating threshold;
  - a threshold sweep;
  - percentile-bootstrap intervals (`bootstrap.n`, `seed`).
  
  An undefined value (no positives or no negatives) is `null` with a reason, never 0.
- Normal configurations:
  - `fpr_normal` = FP/(FP+TN) over regions of `normal` configurations;
  - `alarm_rate_normal` = the fraction of `normal` configurations with at least one flagged region,
    whatever their error.
- Localisation: is the max-error region at the perturbation site, and does each signal's top region
  hit the max-error region?

Signals come from GMD_active's real monitor code on the same frames:

- raw values from `StagedOODMonitor` with every stage forced and lazy LJ off: `committee`,
  `lj_residual`, `force`, `rmin` (scored as -r_min), and `delta_force` (see below);
- `coordination` = |q_i - <q>_Z|, with the mean taken over the calibration `normal` frames;
- monitor variants, scored as the final-stage per-atom score:
  - `phal_staged`: the configured staged monitor;
  - `phal_staged_no_lj`: the same with LJ disabled;
  - `committee_monitor`: the committee term alone;
  - `committee_plus_lj`: the committee and LJ terms.

Running statistics are calibrated once on the calibration `normal` frames, and every evaluated frame
starts from that state.

The LJ monitor is evaluated separately:

- its normal-configuration FPR and alarm rate;
- AP(`committee_plus_lj`) - AP(`committee_monitor`), its incremental value beyond the committee;
- AP(`phal_staged`) - AP(`phal_staged_no_lj`), LJ enabled vs disabled.

Each difference comes with a bootstrap interval. The report states an interval containing 0 as "no
detectable difference".

Temporal metrics (`delta_force`, `displacement`, `delta_q`) need an MD trajectory. Using the
unperturbed base conformation as the previous frame would make `displacement` equal to the applied
perturbation, which hands the monitor the answer. So `temporal_reference: none` is the default and
those signals are reported as unavailable. In the synthetic known-answer test a committee that tracks
the error gets AP = 1, while the LJ residual of non-LJ forces is uninformative and fires on normal
configurations.

## Acquisition arms

`arms.py` runs four arms on one fixed, recorded candidate pool.

The pool has one candidate per `stride`-th carbon of every configuration in `pool.splits`. It never
includes final_eval; the pilot uses `calibration`, and `validation` stays for early stopping. With
`pool.source: gate_selection` each candidate is built by alpha's own `build_fragment` with the
setting frozen by the fidelity gate (`fidelity/selection.json`), so pool fragments are byte-identical
to alpha's label requests and share their label cache. A fragment label belongs to a candidate when its
region has the same parent, the same core atoms and the same non-cap atoms. The pool is written to
`model_study/pool_<hash>.json`.

| Arm | Ranking | Labels |
|---|---|---|
| A `random_local` | seeded random permutation (seed per repetition and round) | fragment |
| B `committee_local` | max committee force deviation over the candidate's core | fragment |
| C `phal_local` | each split group's configurations streamed through `OnlineMonitorDriver` (staged physical + committee signals, event buffering, hotspot clustering); per hotspot group the best-scoring candidate whose core holds a hotspot atom; `CandidateSelector` diversity filter (also against labeled fingerprints) and per-event budget | fragment, only from a gate-approved setting with status `valid` |
| D `phal_full_system` | C's acquisition | the selected candidate's whole parent (one charge per parent) |

Shared by every arm:

- the model architecture and training section (`training`);
- the initial data: full-system labels of `initial_splits`;
- the validation data;
- the label records;
- the evaluation protocol;
- the per-round budget and the per-repetition seed.

Supervision (`training.py`):

- force weights per role, from `training.force_weights`;
- energy weight 0 for every capped fragment, with the reason written into the dataset manifest;
- full-system labels carry the molecule's own energy, weighted by `energy.full_system_weight`.

A test checks, through GMD_SGT's own reader, collate, model and loss, that perturbing zero-weight
targets (cap and boundary forces, the capped fragment's energy) changes neither the loss nor any
parameter gradient, while perturbing a core target does.

Ablations are variants of C:

- `physical_signals: false`: committee only, every frame escalated;
- `lj: false`;
- `event_buffering: false`: every triggered frame is its own event;
- `force_weights`: other core, buffer or boundary weights;
- `extraction`: another fragment setting, for example another buffer size. A setting that did not
  pass the gate is refused unless `allow_unvalidated_overrides` is set.

The following raise a configuration error:

- PHAL-only options on A or B;
- weight or extraction overrides on D;
- `physical_signals: false` combined with `lj: false`;
- temporal metrics on static pools;
- a local arm without a passed gate;
- a final_eval pool;
- repeated seeds.

Repetitions use distinct seeds (`arms.seeds`), and their counts are set per profile: smoke 1x1,
pilot 2 repetitions x 2 rounds, campaign 5 x 5.

## Cost accounting

Every acquisition is charged the label record's measured `cost.cpu_core_s`: wall time times allocated
cores, from the labeling service's attempt records. Nothing is estimated from atom or call counts.

- A calculation acquired by several arms is one physical calculation. It is charged to each arm at its
  measured value. Within one arm it is charged once.
- `invalid` and `failed` labels that consumed CPU time are charged but never trained on.
- A launch failure without measured time is skipped and recorded.
- `budget_skipped` labels are skipped.

The per-round budget is `budget.cpu_core_s` and/or `budget.labels`, and both are reported.
`overshoot: allow` keeps the calculation that crosses the budget and records the overshoot.
`forbid` skips a recorded label whose measured cost would exceed the budget.

Evaluation (`evaluation.py`) is computed on final_eval full-DFT configurations:

- force RMSE;
- the 95th and 99th percentile and the maximum of the per-atom force error;
- the RMSE over the perturbation-site region;
- a per-family breakdown.

Accuracy at a fixed budget and cost to reach a target are interpolated linearly between the two
measured points that bracket the value. Outside the measured range the result is `null` with a
reason; nothing is extrapolated. Variation is reported across repetitions (seeds), never across atoms
or frames.

## Figures

`figures.py` (matplotlib) writes PDF, SVG and PNG to `model_study/figures/`, and `index.json` records
each figure's sources (with sha256) and whether every panel had data:

- receptive field: core-force change vs distance, and vs truncation radius, with the nominal radii;
- system and hotspot view: the chain projected on its principal axes, coloured by reference error and
  by a signal;
- region risk vs region error;
- core-force error vs fragment size, from alpha's `fidelity/report_calibration.json`;
- accuracy vs cumulative measured DFT cost per arm, and vs label count;
- ablations: the final metric per variant, with every repetition shown;
- closed-loop timeline: segments coloured by model, event triggers, and promotions or rollbacks.
  The input is the controller's `summarize` JSON, optionally with `event_steps`.

A panel without data says "NOT EXECUTED / no data" and why. When any label record is smoke-only, every
data figure carries a red "SMOKE-ONLY" stamp.

## Pilot model study: results

`report` writes `report.json` and `report.md`. Each executed section carries its artifact path,
sha256, config hash, label counts, measured costs and the numbers read from it. Planned or
unexecuted parts are listed separately. Published copies are in
`results/model_study_{smoke,pilot,pilot_v2}/`. All numbers below were measured. The pilot is small:
4 parents per split, 2 repetitions, and a trainer that is not bitwise reproducible (see below). No
difference between arms is claimed from it.

### Executed: pilot v1 (`arms_pilot.yaml`, 2026-11-07), pre-registered settings

Artifacts are in `<artifacts_root>/model_study/`. The study config hash is `6d363dce…`;
ground_truth.json sha256 `42f8c6c5…`; evaluation.json sha256 `912f06e0…`.

**Receptive field (random init, no DFT).** The model is a local GMD_SGT with n_blocks = 2 and
r_c = 4 A.

- Nominal receptive field: 16 A for forces, 8 A for the per-atom energy.
- Measured on a 98-atom C32H66 normal structure from the pre-sizing parent set: a dependence above
  round-off reaches at most 8.95 A (delete) and 7.64 A (perturb). Truncating at R >= 3 A changes the
  core force by less than 1e-4 eV/A.

These are the magnitudes of a random model. The measurement on a trained checkpoint is in pilot v2.

**Ground truth (validation split: 4 configurations, 64 regions).**

- The initial model was trained on the 4 initial_train structures with the pre-registered training
  settings (batch_size 4, 150 epochs, patience 30). Its region errors (max |F_model - F_DFT| per
  region) have a median of 2.39 eV/A and a 99th percentile of 3.36 eV/A.
- At every reference threshold (0.1, 0.2, 0.4 eV/A), all 64 regions are positive. AP is therefore
  undefined: with no negatives, any ranking scores 1. The report says so instead of printing 1.0;
  this was fixed in `ground_truth.py` after the first run printed AP = 1.
- The staged-monitor signals are unavailable: 1 normal calibration frame is fewer than
  `warmup_frames` = 3.
- Localisation: in 3 of 3 perturbed configurations the max-error region contains the perturbation
  site. No raw signal (committee, LJ, force, r_min, coordination) put its top region on it (0/3
  each). This is a 3-configuration observation, not a rate.

**Acquisition arms (2 repetitions x 2 rounds x 4 labels; evaluated on final_eval).**

| Arm | Valid labels | Measured DFT (core-s) | Force RMSE, round 0 -> 2 (eV/A) |
|---|---|---|---|
| A random_local | 8, 8 | 3393, 3073 | 0.827 -> 0.664; 0.831 -> 0.634 |
| B committee_local | 8, 8 | 2922, 3738 | 0.827 -> 0.637; 0.831 -> 0.756 |
| C phal_local | 0, 0 | 0, 0 | not exercised |
| D phal_full_system | 0, 0 | 0, 0 | not exercised |
| ablations (4) | 0 | 0 | not exercised |

- **Arms C, D and the ablations were not exercised.** The staged monitor never left warm-up (one
  calibration configuration), so it never triggered, and the arms recorded zero acquisitions as
  "complete". `arms-run` now refuses such a variant before any training
  (`_check_monitor_calibration`, tested).
- A and B both lower the final_eval RMSE from about 0.83 eV/A, but not monotonically: one A
  repetition rose to 0.962 after round 1. With 2 repetitions they cannot be told apart.
- The acquired fragments had 14-31 atoms and cost 1.75 core-h for 15 labels, a mean of about
  420 core-s each. A full C16 label costs about 2630 core-s.

**Initial-model sizing diagnostic (no DFT)**, `examples/alkane_hotspot_demo/initial_model_diagnostic.py`,
result in `results/model_study_pilot/initial_model_diagnostic.json`:

- single models, seed 11;
- evaluated on the validation full labels, which are also the early-stopping set;
- final_eval is never read.

| Training set | Training settings | Median atom error (C / H), eV/A | Atoms < 0.2 / < 0.4 eV/A |
|---|---|---|---|
| initial_train (4) | pilot (batch 4, 150 epochs, patience 30) | 1.11 (2.12 / 0.99) | 4 % / 12 % |
| initial_train (4) | 1500 epochs, batch 1, patience 300 | 0.85 (0.67 / 0.91) | 2 % / 7 % |
| + calibration full (8) | same | 0.45 (0.62 / 0.40) | 12 % / 40 % |
| + calibration fragments of the frozen setting (13) | same | 0.38 (0.50 / 0.34) | 8 % / 54 % |

An earlier run of the same procedure, with the same data, seed and thread count, gave 1.11, 0.68,
0.64 and 0.54 eV/A. **GMD_SGT CPU training was not bitwise reproducible here.** Single-seed
differences of about 0.2 eV/A are noise. Both runs agree that the pilot model is step-limited and
data-limited, and remains far above the 0.1-0.4 eV/A thresholds.

**MD stability (no DFT)**, `examples/alkane_hotspot_demo/md_stability_check.py`:

- the closed loop's GMD setup, 300 K, 0.5 fs, 2000 steps, 2 velocity seeds;
- the parent bond topology is checked at every 10th step.

| Model | First broken step (seeds 7, 8) | Intact frames |
|---|---|---|
| v1 initial model (pilot training) | 20, 20 | 1 % |
| initial_train, 1500 epochs, batch 1 | none, none | 100 % |
| + calibration full | none, none | 100 % |
| + calibration fragments | none, none | 100 % |

The v1 settings stopped training after too few optimiser steps (one step per epoch on 4 structures),
and the resulting model dissociates C16H34 almost at once. With more steps on the same 4 structures
the chain stays intact for 1 ps.

### Pilot v2 (`arms_pilot_v2.yaml`), post-hoc and not pre-registered

v2 changes only the two defects above:

- training uses batch_size 1, 1500 epochs, patience 300 and warmup_steps 50;
- the arms' monitor is calibrated on the normal configurations of initial_train, calibration and
  validation (3; final_eval is never used).

Ground truth keeps validation out of its calibration, so its staged-monitor signals stay unavailable
(2 < 3 frames). Labels, parents, the fidelity setting, thresholds, budgets, arms, ablations and seeds
are unchanged.

Results: see "Executed: pilot v2" below, or "planned" if it has not run yet.

## Closed-loop execution and recovery

Part gamma runs the real closed loop on GMD_active's persistent controller and demonstrates its
recovery. Code: `closed_loop.py`, `recovery.py`, `md_box.py`. Configurations:
`configs/demos/alkane_hotspots/closed_loop_{smoke,pilot}.yaml`. Entry script:
`examples/alkane_hotspot_demo/closed_loop.py`. Outputs go to
`<artifacts_root>/closed_loop/<name>/` (outside git).

```bash
CFG=configs/demos/alkane_hotspots/pilot.yaml; CL=configs/demos/alkane_hotspots/closed_loop_pilot.yaml
python -m tools.experiments.alkane_hotspots closed-loop-run --config $CFG --closed-loop $CL [--max-wall-s N]   # resumable
python -m tools.experiments.alkane_hotspots frozen-eval     --config $CFG --closed-loop $CL
python -m tools.experiments.alkane_hotspots recovery-demo   --config $CFG --closed-loop $CL
```

`--config` must be the closed-loop YAML's `base_config`; otherwise the command refuses to run.

### What runs, and what is reused

Nothing of the loop is re-implemented. `workflow.assembly.build_controller` builds the controller
with GMD_active's components:

- the GMD adapter;
- the GMD_SGT adapter (training, fine-tuning, export, export validation, committee enrichment);
- `LabelingService` (CP2K and the local scheduler);
- `VersionedDatasetStore`;
- `ModelRegistry`, the promotion gates and rollback.

The controller runs every stage: GMD sampling, monitoring, events, extraction, selection, CP2K
submission, polling and validation, ingestion, dataset publication, fine-tuning, export validation,
GMD smoke test, gated promotion, and a GMD checkpoint restart with the new model
(`model_change allow`).

The alkane-specific parts are attached by composition, each implementing a `workflow.protocols`
interface around the existing component:

| Component | Protocol | Role |
|---|---|---|
| `AlkaneRegionExtractor` | `RegionExtractor` | Unwraps the event's trigger frame along its bonds into an isolated molecule. Full mode labels the whole chain. Local mode cuts alpha's fragment (`fidelity.build_fragment`) with the frozen gate setting around each hotspot group. |
| `BudgetedLabeler` | `Labeler` | Refuses a new CP2K job beyond the CPU-hour budget or the atom limits, and records every decision in `budget_ledger.json`. |
| `HeldOutValidationStore` | `DatasetStore` | Training data is the cumulative versioned store. Validation data is always the fixed, checksum-verified `validation` split. |
| `RecordingTrainer` | `Trainer` | Records every evaluation, export check and smoke test, so each gate decision can be re-derived. |
| `md_box.ImageGuard` | `FrameEnricher` | Checks every monitored frame for interaction between periodic images. |

The CP2K settings hash of the closed loop must equal alpha's, or the command refuses to run.

### MD box

GMD needs a fully periodic orthorhombic box. The reference calculations stay isolated: CP2K uses
`PERIODIC NONE` with the WAVELET solver, and this is unchanged. Only the MD runs in a periodic box.

The box is cubic, with edge `L >= D + R_force + margin + stretch_allowance`:

- `D` is the chain diameter;
- `R_force = 2 x n_blocks x local_cutoff` is the force receptive field (16 A for the demo model; see
  "Model receptive field").

For any rigid rotation or translation of the chain, every atom of another image is therefore at
least `L - D >= R_force + margin` away. Within the model, images cannot interact.

This is checked twice:

- when the box is built, from the rotation-invariant bound and the exact minimum image distance;
- on every monitored MD frame, and on every frame of the frozen-model runs, by `ImageGuard`. The
  molecule is unwrapped along its bonds, and the exact minimum distance to any atom of another image
  must exceed `R_force`.

A violation raises `ImageInteractionError`. A model with a global long-range term is refused.

The box, the receptive field, the margin and the measured minimum over the trajectory are written
to the run manifest, `md_box/box.json`, `run/image_check.json` and the run report.

### Labeling mode

`labeling.mode: auto` labels local fragments only when alpha's fidelity gate has passed and frozen
a setting (`fidelity/selection.json`). The fragment is alpha's `build_fragment` with that setting, on
a site shaped like the gate's sites: carbons k-1..k+2 and their H, where k is the hotspot group's
top-scoring carbon.

If the gate failed (`failure.json`) or is undecided, full-system labels are used. `mode: local`
without a passed gate is refused. Unvalidated local labels are never produced.

Before any CP2K job, the event geometry must still be the intact alkane (`label_geometry`): the bond
graph from `infer_bonds` must equal the parent's, and no interatomic distance may be below 0.7 A.
Otherwise the event is recorded as `extraction_failed`, with its reasons in
`rejected_events.jsonl`.

### Initial model, validation and gates

- **Initial model.** GMD_SGT is trained from scratch on the full DFT labels of `initial_train`. It
  uses beta's training section (architecture, force weights, energy policy), the committee from
  `training.sgt.committee_size`, and the validation split for early stopping.
- **Fine-tuning.** Each cycle starts from the active model (`finetune.train`). GMD_SGT's adapter
  initialises every committee member from the active primary checkpoint, with seeds `seed+k`.
- **Promotion gates.** The YAML `promotion` gates are evaluated on the validation split: the force
  RMSE limit, no regression beyond `max_relative_regression` against the active model on the same
  validation data, export versus checkpoint, and the GMD smoke test.
- **The initial model's gates** are recorded but not enforced, because there is no parent. Its export
  check and smoke test must pass.
- **final_eval** is never read by the loop.

### Bounds

| Bound | Key |
|---|---|
| Cycles | `loop.cycles` |
| MD steps per segment | `loop.segment_steps` |
| Labels per cycle | `loop.labels_per_cycle` (`workflow.candidates.max_per_cycle`) |
| Labels per event | `loop.max_per_event` |
| Measured CP2K CPU core-hours | `budget.max_cpu_core_hours` |
| Atoms per job | alpha's `budgets.max_atoms_*`, or `budget.max_atoms_*` |

The CPU check before each job adds three terms: the measured cost of the finished closed-loop tasks
(`cost.measured_cost`), the estimates for running ones, and the estimate for the new job. The
estimate is the measured mean of alpha's labels of the same kind. Training CPU is not part of this
budget.

With `exclusive_cp2k: true` (the pilot), a command that may launch CP2K refuses to start while a CP2K
process outside its own output directory is running.

### Bookkeeping

- **Model changes.** At every model change, the report records the potential-energy jump
  `E_new - E_old` (`model_changes`). `E_old` comes from the old segment's last frame and `E_new` from
  the restarted segment's first frame: the same MD step and coordinates, recomputed by GMD with the
  new model. Both exports are also re-evaluated on that configuration (`reevaluated`). The record is
  labelled `model_change`, not integration drift. When the coordinates differ, no jump is claimed.
- **Provenance.** Every promoted model has a chain in `provenance`: model -> dataset version ->
  samples -> CP2K tasks and job ids -> regions -> events (trigger step, frame steps, frames path) ->
  parent MD frames -> the generated parent structure. Initial samples resolve to their alpha label
  and parent. Unresolved samples are listed, never dropped.
- **Report.** `closed_loop_report.json` holds:
  - the controller's `summarize`, the transitions, events and rejected events;
  - the measured cost of every label task, and the budget ledger;
  - the trainings and the re-derived gate reports;
  - the model changes and the provenance;
  - the initial model's validation metrics.

  `timeline.json` is the input of beta's closed-loop timeline figure (`figures.timeline_report`).

### Frozen-model evaluation

`frozen-eval` runs fresh GMD trajectories with the frozen initial and final models. Both use the same
seeds, the same start structure in the same box, and the same steps.

It reports, after `discard_steps`:

- the end-to-end C1-Cn distance;
- the radius of gyration of the carbons;
- the C-C-C-C dihedral histogram and gauche fraction (|phi| < 120 deg).

Results are given per seed and pooled. This is a pilot-scale comparison of two trajectories per
seed; it supports no claim beyond them, and in particular none about the accuracy of either model.

### Recovery demonstration

`recovery-demo` uses its own run (`recovery.run_id`, directory `recovery/`) and is bounded by
`recovery.*`:

1. **Interrupt.** The recovery run starts in a subprocess. It is SIGKILLed as soon as the
   controller has recorded a submitted CP2K job (`recovery/interruption.json`), and the report
   records whether that job's process group is still alive.
2. **Resume.** The run resumes in-process. The verification passes only if:
   - each task submitted before the kill keeps one initial attempt with the same job id (it was
     adopted, not resubmitted);
   - no task directory disappeared;
   - there is at most one sample per task, and no duplicate label files, dataset versions or job ids.
3. **Strict gate audit.** The run's promoted model is audited with `recovery.strict_promotion`,
   using `evaluate_gates` on its recorded validation metrics, export check and smoke test. If it
   fails, `ModelRegistry.rollback` restores the previous model (the failed one becomes `rejected`).
   GMD then restarts from the run's last checkpoint with the restored model, and the energy jump at
   that boundary is recorded.

   The recovery run's own gates (`recovery.promotion`) are deliberately loose in both YAMLs, so a
   model is promoted and can be audited. The strict gate is the quality gate in this demonstration.
4. **Cancellation.** A real CP2K job is submitted. Once `cp2k.psmp` is running, a new
   `LabelingService` instance cancels it. This is the situation after a controller restart: the new
   instance has no in-memory process handle.

### Cancellation: findings and operational guidance

Killing the controller leaves locally scheduled CP2K jobs running, by design. The local scheduler
starts each job in its own session and persists the pid, process group and start time
(`.gmd_local_job.json`), so a restarted controller can adopt the job.

`LabelingService.cancel` and `LocalScheduler.cancel` terminate a running job, including from a fresh
instance. They send SIGTERM to the job's process group, then SIGKILL after the grace period. They
write `.gmd_cancelled` and record the attempt as `cancelled` (failure class `cancelled`, "cancelled
jobs are never retried automatically"). The task is finalised, and a later poll does not resubmit.

One defect was found and fixed in GMD_active (`workflow.assembly.ServiceLabeler`, commit 65c90e5).
The workflow bridge reported a cancelled task as COMPLETED. The controller would then have collected
it and recorded it as an *invalid label* instead of a cancelled job. CANCELLED is now kept, and the
controller records the task as failed with "execution cancelled".

Operational guidance:

- **To stop the workflow without stopping CP2K**, kill or stop the controller (`gmd-al stop` or a
  signal). Running jobs finish and are adopted at the next start.
- **To stop CP2K work too**, cancel the tasks explicitly with `LabelingService.cancel(task_id)`,
  using the run's CP2K configuration. Any process may do this.
- **Never kill `cp2k.psmp` by hand** while a controller may adopt it. The job then has no exit code,
  the scheduler reports it VANISHED, and the service's retry ladder may resubmit it.
- **MPI launchers.** Cancellation was verified with the configured single-rank launch (`launcher:
  []`). With an MPI launcher, ranks that leave the job's process group are not covered by this check.

### Executed: smoke, 2026-11-07 (mechanics only, not a scientific result)

Setup:

- `closed_loop_smoke.yaml` on C8H18, with CP2K at alpha's smoke-only settings (200 Ry).
- The initial model has one training structure and one validation structure from the same base.
- The gates are deliberately loose, and the monitor thresholds are lowered so that 100-step
  segments produce events.
- Artifacts are under `<artifacts_root>/closed_loop/smoke/`, with `closed_loop_report.json`,
  `frozen_eval/frozen_eval.json` and `recovery/recovery_report.json`.

None of the numbers below says anything about model quality.

**Closed loop.** Two cycles reached `done`.

- **Labeling mode:** full system, because the smoke fidelity gate failed.
- **Box:** 32 A cubic, chain diameter 10.6 A, force receptive field 16 A plus a 4 A margin. The
  minimum image distance over the 41 monitored frames was 21.1 A.
- **Events:** 4 events. One was refused before CP2K because a C-H bond exceeded 1.2 x the covalent
  sum.
- **Labels:** 2 CP2K labels, 32.5 s and 57.8 s wall at 2 threads, 0.050 core-h in total. Both were
  valid on the first attempt and ingested.
- **Models:** 2 fine-tuned models, both promoted. Validation force RMSE went 0.519 -> 0.494 -> 0.487
  eV/A, each on one validation structure.
- **Model change:** at step 100 the energy jump was +1.169 eV, recorded with GMD's restart frame
  and the re-evaluation at the same coordinates.
- **Provenance:** every promoted model's chain resolves to the initial full-DFT label or to
  event, frame and parent structure.

**Frozen evaluation.** Initial versus final model, 2 seeds x 200 steps, 62 frames each, after
discarding 50 steps:

| Model | End-to-end distance (A) | Radius of gyration (A) | Gauche fraction |
|---|---|---|---|
| Initial | 9.28 ± 0.49 | 3.08 | 0.013 |
| Final | 9.13 ± 0.31 | 3.05 | 0.016 |

These are two short trajectories per model; no conclusion is drawn.

**Recovery.**

- **Interrupt and resume:** the controller was SIGKILLed 4 s after its first CP2K submission, and
  the CP2K job was still alive. On resume, the job was adopted: the same job id, one initial attempt,
  no new task directory, one sample, one label file. The verification passed.
- **Strict gate audit:** the promoted model failed `force_rmse` (0.494 > 0.05 eV/A) and was rolled
  back.
- **GMD restarts:** both restarts had `model_changed_from` set by GMD. The energy jump was +1.168 eV
  at the promotion boundary (step 100) and +0.831 eV at the rollback boundary (step 150).
- **Cancellation:** `cp2k.psmp` was running (41 kB of output) when a new service instance cancelled
  it. The cancellation took 0.11 s, no live process remained in the job's process group, and the
  task records `cancelled` without a retry.

The cancellation was also checked by `tests/experiments/alkane_hotspots/test_recovery.py`
(real CP2K, marked `cp2k`).

### Executed: pilot v1 (`closed_loop_pilot.yaml`, 2026-11-07), failed to exercise the loop

This was a real run: C16H34 `C16-b01-normal0`, GMD at 300 K and 0.5 fs, labeling `local` with the
frozen fidelity setting, artifacts under `<artifacts_root>/closed_loop/pilot/`.

- **Initial model:** the study's pilot training settings, trained on the 4 initial_train
  structures. Validation force RMSE 0.715 eV/A.
- **What happened:** one event at step 70. Its triggers were `force_large`, `committee_deviation`
  and `lj_residual`, with a maximum score of 8.6. The extractor refused it before CP2K: by then the
  bond topology had changed (5 parent bonds missing, 4 spurious ones).
- **Segments:** the controller stopped segment 0 at step 204 and segment 1 at step 224, with no
  further event.
- **Outcome:** no label, no training, no model change, 0 core-h of DFT; the loop ended at `done`.
  The refusal itself is correct behaviour: a dissociated chain is not an alkane configuration, and
  labeling it would put non-alkane data into the set.
- **Frozen evaluation:** initial and final are the same model. In the first version of frozen-eval
  the reported chain statistics (end-to-end minimum 0.46 A, radius of gyration 2.5 A) were averages
  over a broken molecule. frozen-eval now checks the topology of every frame and reports chain
  observables over intact frames only.
- **Recovery demo:** failed explicitly (`the recovery run exited before any CP2K submission; nothing
  to interrupt`), because the same model produced no labelable event.
- **Cause:** the MD stability check above. The v1 initial model breaks the chain at step 20 with
  both seeds; a model trained with enough optimiser steps on the same data keeps it intact for 1 ps.

The smoke run's mechanics (adoption after SIGKILL, gate rejection and rollback, cancellation,
energy jumps at model changes) remain the only executed demonstration of those paths on real
programs.

### Pilot v2 (`closed_loop_pilot_v2.yaml`), post-hoc

Identical to v1 except for three things:

- the study is `arms_pilot_v2.yaml`, which gives the initial model enough optimiser steps;
- the output directory is `closed_loop_v2`;
- the run id is new.

It runs after the C32 transfer has released CP2K (`exclusive_cp2k`). Results: "Executed: pilot v2"
below, or "not executed" if absent.
