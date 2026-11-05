> **Ported from PHAL** (`docs/adaptive_scf_controller.md` at 96263b6, MIT License, see NOTICE). The body below is
> kept **verbatim**, including every measured number. Those measurements were taken by
> PHAL on the lost `cp2k-ml-scf` fork, with models and datasets that no longer exist;
> they cannot be re-run here as written. The live controller was Fortran in the lost fork and is not in the reconstruction; only the offline Python model is ported (`tools/scf/controller/`).
> Code paths named below (`hotspot_al.*`, `scripts/*`) map to GMD_active as listed in
> [README.md](README.md#module-map). Status of each feature: [README.md](README.md).

# The adaptive SCF controller

CP2K keeps the Kohn-Sham solve, the density update, the mixer, and the
`EPS_SCF` test. The controller answers one question CP2K currently answers with
a fixed setting — *given how this SCF is going, should the strategy change?* —
and may answer it in only four ways: `KEEP`, `RESET_DIIS`, `SWITCH_MIXING`,
`INCREASE_STABILITY`.

It is wired into `scf_env_do_scf_inner_loop`, after the convergence check so the
residual and energy are final, and before the exit branch so a finishing run is
never restrategised. Off unless `PHAL_ML_SCF_CONTROLLER=1`.

## Why strategy and not density

Density prediction was measured to exhaustion: four model variants, four mixing
weights, 54 runs, and no configuration beat CP2K's atomic guess
([ml_density_v2.md](ml_density_v2.md)). The measurement that closed that line
was the oracle sweep — an *exact* density saves 66% of iterations, but 75% of
the way there saves 17%, so there is no partial credit.

Strategy looked different because it has no such threshold. A stale DIIS
history reset at the right iteration either helps or is rolled back.

## Safety

No action touches the density. They change how the *next* mix is computed,
which is why a rollback restores settings and never an electronic state.

| Guard | Value | Why |
| --- | --- | --- |
| first action no earlier than | iteration 4 | nothing to judge before that |
| spacing between actions | 3 iterations | a reset needs iterations to rebuild what it discarded |
| actions per run | 2 | see below |
| no action past | 50% of `MAX_SCF` | an intervention cannot repay its cost near the budget |
| residual growth after an action | 10× | divergence, not the transient a reset causes |
| rejections before stopping | 1 if irreversible, else 2 | a discarded history cannot be restored |

The last two rows were written by measurement, not by taste:

* Forcing `reset_diis` on CO₂ and letting the controller carry on cost **53
  steps against 26** for stopping after the first irreversible mistake. An
  action that cannot be undone and turns out wrong ends the controller.
* The first policy also fired on runs that were merely *slow*. On the
  transition-metal oxides that cost two converged runs — MnO intervened on
  eight times, turning a 52-step success into a failure. Slowness is not
  trouble; the rule now requires a rising or genuinely stalled residual.

## Benchmark: native CP2K vs the controller

Twelve systems including the four that fail natively. One environment variable
apart.

| | native | controller |
| --- | --- | --- |
| converged | 8/12 | 8/12 |
| total SCF steps | **172** | **172** |
| rescued / lost | — | 0 / 0 |
| actions taken | 0 | 5 |

| system | native | controller | actions |
| --- | --- | --- | --- |
| `tio2-rutile` | 19 | 19 | 14:increase_stability |
| `nio-rocksalt` | 31 | 31 | 10:increase_stability, 13:increase_stability |
| `zno-wurtzite` | 25 | 25 | — |
| `cu2o-cuprite` | 19 | 19 | — |
| `mno-rocksalt` | 52 | 52 | 11:increase_stability, 14:increase_stability |
| `si-diamond` | 8 | 8 | — |
| `c-diamond` | failed | failed | — |
| `mgo-rocksalt` | failed | failed | — |
| `nacl-rocksalt` | failed | failed | — |
| `lih-rocksalt` | failed | failed | — |
| `aln-wurtzite` | 10 | 10 | — |
| `si-diamond-strained` | 8 | 8 | — |

The controller is **safe and neutral**: it acted five times and changed
nothing, and it rescued none of the four systems that abort inside CP2K's own
Pulay solve.

## Counterfactual labels: there was nothing to learn

The benchmark shows the *policy* did not help. The labels ask whether any
policy could: the same SCF replayed with each action forced at the same
iteration, labelled by the measured remaining cost.

| system | iteration | keep | reset_diis | switch_mixing | increase_stability | best |
| --- | --- | --- | --- | --- | --- | --- |
| `mno-rocksalt` | 5 | 47 | 88 | failed | 47 | **keep** |
| `nio-rocksalt` | 5 | 26 | 88 | failed | 26 | **keep** |
| `tio2-rutile` | 5 | 14 | 17 | 16 | 14 | **keep** |
| `zno-wurtzite` | 5 | 20 | 32 | 48 | 20 | **keep** |
| `mno-rocksalt` | 8 | 44 | 89 | 77 | 44 | **keep** |
| `nio-rocksalt` | 8 | 23 | failed | 28 | 23 | **keep** |
| `tio2-rutile` | 8 | 11 | 30 | 42 | 11 | **keep** |
| `zno-wurtzite` | 8 | 17 | 39 | 50 | 17 | **keep** |
| `mno-rocksalt` | 12 | 40 | 40 | 40 | 40 | **keep** |
| `nio-rocksalt` | 12 | 19 | 19 | 19 | 19 | **keep** |
| `tio2-rutile` | 12 | 7 | 30 | 15 | 7 | **keep** |
| `zno-wurtzite` | 12 | 13 | 36 | 22 | 13 | **keep** |

**Twelve of twelve labels are `keep`.** Not one intervention, at any tested
iteration, on any tested system, left less work to do than doing nothing.
Several were catastrophic: resetting the DIIS history at iteration 5 of MnO
turned 47 remaining steps into 88.

A supervised controller trained on these labels would learn to output `keep`.
That is the correct answer, and it does not need a model.

## Where the leverage actually is

`increase_stability` produced *identical* step counts to `keep` in every
counterfactual — halving `ALPHA` mid-run does nothing, because under Pulay
mixing the DIIS coefficients dominate the update. But the same parameter
chosen **before** the run matters a great deal:

| system | `ALPHA` 0.4 | `ALPHA` 0.2 |
| --- | --- | --- |
| TiO₂ rutile | 19 | **14** |
| ZnO wurtzite | 25 | **23** |
| NiO rocksalt | **31** | 65 |
| MnO rocksalt | **52** | 86 |

TiO₂ converges 26% faster at the smaller `ALPHA`; NiO takes 110% longer. The
right value is system-dependent and worth a factor of two — and it is chosen
once, from the structure, before any SCF state exists.

That is a **static prediction problem, not a control problem**, and unlike
density prediction it has no accuracy threshold: predicting `ALPHA` 0.2 instead
of 0.4 for TiO₂ is a discrete choice with a measurable payoff.

## What this line delivered, and what it did not

Delivered: a controller inside CP2K's SCF loop with four safe actions, a
rollback that cannot corrupt an electronic state, a measured safety envelope,
per-run reporting, and a counterfactual labelling harness.

Not delivered: any acceleration. Both the live benchmark (172 = 172) and the
counterfactual labels (12/12 `keep`) say mid-run strategy switching on these
systems has nothing to offer.

The evidence points at SCF *parameter selection* — `ALPHA`, `NBUFFER`, the
mixer itself — predicted per system before the run starts. The controller
machinery is kept: it is the mechanism that would apply such a choice, and the
harness that measured this one.

## Reproducing

```bash
# native vs controller
scripts/benchmark_controller.py --structures ... --template ... --binary ...

# counterfactual labels at chosen iterations
scripts/benchmark_controller.py ... --label-mode 5 8 12
```

`PHAL_ML_SCF_CONTROLLER=1` enables it; `PHAL_ML_SCF_CONTROLLER_FORCE=5:reset_diis`
forces one action, which is how the labels above were produced.
