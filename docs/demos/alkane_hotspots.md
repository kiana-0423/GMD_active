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

**Status (2026-11-07).** *Executed:* parent generation for the pilot (no DFT), and the real-CP2K
smoke run (C8H18 at smoke-only settings; see below). *Planned, not executed:* the pilot DFT labels,
the pilot fragment-fidelity gate, and the campaign. No pilot or campaign number appears on this page
until it has been measured.

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

**Pilot sweep and expected cost (planned).** The calibration split has 8 parents and 14 sites. Graph
hops {2, 3, 4} and cluster (core 1.5 A, buffer {3, 4, 5} A) give 84 fragment specs. Because
identical fragments are shared, these need 52 CP2K calculations of 20-56 atoms. The calibration
split also needs 8 full parents (4 x C16, 4 x C32).

Using the lead's C16 measurement of 1.1 core-hours, a rough estimate is:

- 4 C16 parents: about 4.5 core-hours;
- C32 parents: not measured yet. The cubic box is 52.5 A wide instead of 32.5 A for C16, about 4.2
  times the grid volume with twice the atoms, so expect roughly 4-8 times the C16 cost each;
- fragments: probably below a C16 parent each.

`status` replaces this with measured means as soon as the first labels finish. The pilot budget is
120 core-hours (`budgets.max_total_cpu_core_hours`). Labeling all 12 C32 parents of the pilot could
exhaust it on its own, so run the calibration split first and check `status`.

**Smoke result (smoke-only settings, not a scientific result).** C8H18 torsion parent, perturbation
site, graph hops 2, one cap at two bonds from the nearest core atom:

- core RMSE 0.061 eV/A, maximum error 0.179 eV/A;
- the gate failed against the 0.03/0.10 targets and wrote `failure.json`.

This shows that the gate can fail and does record the failure. Whether the pilot settings pass is
open.
