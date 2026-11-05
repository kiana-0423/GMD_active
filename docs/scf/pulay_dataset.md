> **Ported from PHAL** (`docs/pulay_dataset.md` at 96263b6, MIT License, see NOTICE). The body below is
> kept **verbatim**, including every measured number. Those measurements were taken by
> PHAL on the lost `cp2k-ml-scf` fork, with models and datasets that no longer exist;
> they cannot be re-run here as written. Generation needs the Pulay recorder (`PHAL_ML_SCF_PULAY_RECORD_DIR`), which no current CP2K build has. The structure collections it names were never committed and have been regenerated; see [structure_provenance.md](structure_provenance.md).
> Code paths named below (`hotspot_al.*`, `scripts/*`) map to GMD_active as listed in
> [README.md](README.md#module-map). Status of each feature: [README.md](README.md).

# The Pulay training dataset

One sample is one converged SCF run, kept whole:

```
runtime/cp2k/pulay_dataset/
├── dataset.json              manifest: counts, per-sample verdicts
├── dataset_report.json       statistics
└── <sample_id>/
    ├── sample.json           structure, settings, convergence, validation
    ├── cp2k.out              the run it came from
    ├── pulay_records/        raw dumps from ml_pulay_recorder.F
    └── trajectory/
        ├── overlap.npy
        ├── metadata.json
        ├── iter_0000/{density.npy, pulay_matrix.npy, metadata.json}
        └── final/
```

Samples are **not** aggregated into one array at generation time. That is where
the AO-reconstruction mistake became invisible: once matrices are stacked, a
wrong metric looks exactly like a right one. Every sample stays independently
checkable, and `representation` / `metric` travel with each matrix.

## Generating

Three ways to say what to run.

**A structure collection and a template** — xyz, extxyz or cif:

```bash
scripts/generate_pulay_dataset.py \
    --structures data/structures/pulay_sampling/small_molecules.extxyz \
    --template configs/cp2k/pulay_dataset.inp.template \
    --cp2k-data ../cp2k-ml-scf/cp2k/data \
    --binary build/cp2k-ml/bin/cp2k.psmp \
    --scheduler local          # or slurm, pbs
```

**A sampling plan** — several families, each with the CP2K settings that suit
it (`configs/cp2k/pulay_sampling_plan.json`):

```bash
scripts/generate_pulay_dataset.py --plan configs/cp2k/pulay_sampling_plan.json \
    --cp2k-data ../cp2k-ml-scf/cp2k/data --binary build/cp2k-ml/bin/cp2k.psmp
```

**An explicit task list**, when the inputs already exist:

```bash
scripts/generate_pulay_dataset.py --tasks tasks.json --binary build/cp2k-ml/bin/cp2k.psmp
```

```json
[{"task_id": "h2o-r0.96", "input": "h2o-r0.96.inp",
  "structure": {"system": "H2O", "OH_A": 0.96}}]
```

Every task is submitted before any is polled, so a cluster runs them
concurrently. A failed task becomes a rejected sample with a reason rather than
an exception — one bad geometry must not discard the rest.

Each run needs `&AO_MATRICES DENSITY .TRUE. OVERLAP .TRUE.` in its input; the
generator sets `PHAL_ML_SCF_PULAY_RECORD_DIR` itself.

### Basis sets are looked up, not remembered

`resolve_kinds` reads CP2K's own `BASIS_MOLOPT` and `POTENTIAL`. The MOLOPT
alias states the valence it was optimised for (`DZVP-MOLOPT-SR-GTH-q12`), so
pairing it with `GTH-PBE-q12` is a lookup. An element with no usable entry
raises at task-construction time rather than aborting an hour into a queued
job.

### Sampling

The shipped plan is three families, because a dataset drawn from one teaches a
mixer that family and nothing else:

| group | systems | why |
| --- | --- | --- |
| `small_molecules` | 17 closed-shell molecules, equilibrium and strained | cheap, small `n_ao`, short SCF |
| `periodic_main_group` | Si, C, AlN, and ionic crystals | periodicity changes the g-space residual, and therefore `B` |
| `transition_metal_oxides` | TiO₂ (rutile, anatase), NiO, MnO, ZnO, Cu₂O | open d shells: the slow, hard SCFs a mixer is actually wanted for |

Strained geometries are in there deliberately. A dataset of only relaxed
structures never shows the mixer a hard SCF.

## Validation

A sample is accepted only if all of these hold. Each rejection names one thing,
so the report says *why* runs were dropped, not just how many.

| Rejected when | Because |
| --- | --- |
| SCF did not converge | no reliable final density, so the hindsight label is meaningless |
| no Pulay matrices recorded | the input the model is actually fed is absent |
| `representation` is not `cp2k_pulay_metric` | an AO reconstruction is a different matrix in a different space |
| `metric` is not `gspace_kerker` | same |
| overlap missing or non-finite | `Ne = Tr(P S)` is uncheckable without `S` |
| any density non-finite or non-Hermitian | corrupt frame |
| a matrix fails to reproduce CP2K's own coefficients | the export is not the matrix CP2K solved |
| fewer than 4 frames | no mixing history to learn from |

The last one is the strongest. Every dump carries the coefficients CP2K derived
from that matrix, and `B^-1 1 / (1^T B^-1 1)` must reproduce them. A
reconstruction reproduces nothing, because it never produced anything.

## Statistics

`dataset_report.json` covers systems, chemistry, iterations, matrix dimensions,
element distribution, conditioning and convergence. From the shipped sampling
plan, run through the container on 2026-08-01:

| | |
| --- | --- |
| systems | 27 accepted / 30 (90%) |
| elements / formulas | 13 / 21 |
| families | 17 molecular, 10 periodic |
| trajectory frames | 460 |
| Pulay matrices | 379 |
| history depth | 2–8 (median 8) |
| `n_ao` | 10–208 |
| SCF steps | 7–52 |
| `log10 \|B_ij\|` | −11.56 to +4.85 |
| `log10 cond(B)` | 0.40 to 14.36 |
| matrices needing regularisation | 86 / 379 (23%) |
| `\|\|c\|\|₁` | 1.00 to 102.5 |
| max verification deviation | 1.25×10⁻⁷ (MnO) |

### What to read before training on this

**`B` spans sixteen decades.** A model that normalises by `max|B|` removes the
overall scale, but the spread *within* each matrix survives, and the smallest
elements are near float64 noise relative to the largest.

**Condition numbers reach 10¹⁴·⁴, and 23% of matrices exceed 10⁸.** Successive
SCF densities are nearly collinear by construction, so `B` approaches singular
late in a run — exactly where a learned model's coefficients become most
sensitive to its input. A model consuming `B` will need its own regularisation;
`regularization_shift` in each frame's metadata says how much would be needed
to reach a given condition limit. Nothing shifts the matrix CP2K solved.

**`||c||₁` reaches 102.** The native solve is sometimes an extrapolation
relying on cancellation between large opposite-signed terms, not an
interpolation. A small relative error in a predicted `c` is not a small error
in the mixed density there.

**The verification deviation grew from 10⁻¹⁰ to 1.25×10⁻⁷**, and its worst case
is MnO — the system with the most matrices and the worst conditioning. This is
the closed-form check losing digits on a near-singular `B`, not a faithfulness
problem: the tolerance is 10⁻⁶ and every matrix passes.

## Per-frame conditioning

Each `iter_xxxx/metadata.json` carries `pulay_diagnostics` for that frame:
eigenvalues, rank (and `rank_deficient`), condition number, `positive_definite`,
`asymmetry`, `coefficient_l1`, and the `regularization_shift` that would bring
the condition number to `condition_limit` (10⁸ by default).

They are recorded at generation time rather than derived later so that the
difficulty of each solve is a property of the dataset, not something a training
script has to reconstruct.

## Systems CP2K itself refuses

Three ionic crystals — MgO, NaCl and LiH — abort inside CP2K's own
`pulay_mixing`:

```
[ABORT] Bad condition number R_COND = 7.103E-17 (smaller than the machine
        working precision)                              common/mathlib.F:739
```

This is upstream behaviour, not something the ML hooks cause: the unmodified
`cp2k-native` binary aborts identically on the same input. With `NBUFFER 8` on
a two-atom cell the DIIS history becomes linearly dependent and CP2K's own
inversion gives up. They are recorded as rejected samples with the reason, and
would need a smaller `NBUFFER` to contribute.

It is worth noticing what this says about the problem: the systems where a
better mixer would help most are the ones where the native solve is already at
the edge of what float64 supports.
