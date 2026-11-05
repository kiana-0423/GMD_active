> **Ported from PHAL** (`docs/scf_strategy_prediction.md` at 96263b6, MIT License, see NOTICE). The body below is
> kept **verbatim**, including every measured number. Those measurements were taken by
> PHAL on the lost `cp2k-ml-scf` fork, with models and datasets that no longer exist;
> they cannot be re-run here as written. **Exception:** the strategy grid uses only native CP2K settings, so it *can* be re-run with an unmodified CP2K; the GMD_active re-run on regenerated structures is recorded in [strategy_rerun.md](strategy_rerun.md).
> Code paths named below (`hotspot_al.*`, `scripts/*`) map to GMD_active as listed in
> [README.md](README.md#module-map). Status of each feature: [README.md](README.md).

# Predicting SCF settings from the structure

Three attempts to put ML *inside* the SCF were measured and none paid: the
density guess ([v2](ml_density_v2.md)), the DIIS coefficients
([why](ml_pulay_objective.md)), and mid-run control
([12/12 `keep`](adaptive_scf_controller.md)).

All three pointed at the same untested place: the settings chosen *before* the
run. `ALPHA` alone was measured to take TiO₂ from 19 SCF steps to 14 and NiO
from 31 to 65. This is that experiment.

## The dataset

Six configurations on twelve solids, 72 CP2K runs. Each configuration changes
one thing from the default, so a result attributes to that thing.

| system | default | α=0.2 | α=0.1 | nbuffer 4 | broyden | broyden α=0.2 |
| --- | --- | --- | --- | --- | --- | --- |
| `aln-wurtzite` | **10** | 10 | 11 | 11 | 11 | 14 |
| `c-diamond` | **7** | 7 | 7 | 7 | 9 | 9 |
| `cu2o-cuprite` | **19** | 26 | 22 | 86 | 19 | 19 |
| `lih-rocksalt` | FAIL | FAIL | FAIL | **7** | 10 | 13 |
| `mgo-rocksalt` | FAIL | FAIL | FAIL | **10** | 13 | 14 |
| `mno-rocksalt` | 71 | **30** | 118 | 59 | 65 | 39 |
| `nacl-rocksalt` | FAIL | FAIL | FAIL | **9** | 12 | 14 |
| `nio-rocksalt` | **33** | 86 | 66 | FAIL | 39 | 51 |
| `si-diamond` | **8** | 8 | 8 | 8 | 8 | 8 |
| `si-diamond-strained` | **8** | 8 | 8 | 8 | 8 | 8 |
| `tio2-rutile` | 19 | 14 | **14** | 18 | 16 | 14 |
| `zno-wurtzite` | 21 | 22 | 21 | FAIL | **19** | 21 |

Bold is the cheapest converged configuration for that system.

**Configuration matters, and it matters more than anything measured so far.**

* Three systems that the default **cannot converge at all** — MgO, NaCl and
  LiH, which abort inside CP2K's Pulay solve on a linearly dependent history —
  converge in 7–10 steps with `NBUFFER 4`. Halving the buffer removes the
  dependency that causes the abort.
* MnO goes from 71 steps to 30 with `ALPHA 0.2`: **−58%**.
* The same change costs NiO 86 steps against 33: **+161%**. The right answer is
  system-specific in both directions.

Oracle total for the nine systems the default converges: 174 steps against 196.

## The predictor does not work at this size

k-nearest-neighbours over structure features, leave-one-out, so every system is
predicted by a model that never saw it.

| min confidence | label accuracy | total steps (default 196) | improved / degraded | failures rescued |
| --- | --- | --- | --- | --- |
| 0.5 | 25% | 250 | 0 / 2 | 0 |
| 0.6 | 33% | 197 | 0 / 1 | 0 |
| 0.7 | 50% | 196 | 0 / 0 | 0 |
| 0.8 | 50% | 196 | 0 / 0 | 0 |
| 0.9 | 50% | 196 | 0 / 0 | 0 |

Read the top row first: at a permissive threshold the predictor is **worse than
doing nothing** — 250 steps against 196. It recommended `alpha0.2` for NiO on a
52% vote, which cost 86 steps instead of 33.

Now read the bottom rows. At a safe threshold it declines to guess so often
that it becomes the default: identical steps, zero improvements, zero
degradations, and **zero rescues**. Its 50% "accuracy" is the accuracy of
always answering `default`, which is right for six of the twelve systems.

The label is simply not predictable from these features at n=12. Neighbouring
oxides disagree with each other: NiO wants `default`, MnO wants `alpha0.2`, ZnO
wants `broyden`, TiO₂ wants `alpha0.1`. There is no neighbourhood to average
over.

The safety design did its job — nothing was degraded at the shipped threshold —
but a predictor whose safe operating point is "always defer" is not a
predictor.

## What does work, and needs no model

Try configurations in a fixed order until one converges. A failed attempt is
charged its full `MAX_SCF` of 120 iterations, which is what it actually costs.

| ladder | converged | total steps incl. failed attempts |
| --- | --- | --- |
| `default` | 9/12 | 556 |
| `default -> nbuffer4` | 12/12 | 582 |
| `default -> nbuffer4 -> broyden` | 12/12 | 582 |

**`default → nbuffer4` converges 12 of 12 against the default's 9, for 4.7%
more iterations.** It captures the entire robustness win — all three rescues —
deterministically, with no features, no model and no training set.

That is the recommendation. The three failing systems are 25% of this test set,
and turning a failed calculation into a successful one is worth far more than
the 11% of iterations an oracle configuration would save on the ones that
already work.

## When ML would be worth revisiting

The measurement says where the value is and is not:

* **Robustness** — large (3/12 systems), and already fully captured by a
  two-step ladder. A model would have to beat "retry once", which it cannot,
  because retrying is not a prediction at all.
* **Speed** — the oracle saves 11% over the default on convergent systems.
  Real, but it needs a *correct* prediction, and at n=12 the label is not
  learnable. This becomes interesting at a dataset size where chemically
  similar systems agree — which is a data-collection question, and the
  generator for it is `scripts/benchmark_scf_strategies.py`.

## What is shipped

* `SCFStrategy` — a configuration and the `&SCF` block it renders. The section
  is replaced wholesale, so two inputs differ only in what was chosen.
* `scripts/benchmark_scf_strategies.py` — the grid runner and dataset
  generator, with `--evaluate` for leave-one-out.
* `StrategyPredictor` — k-NN with distance-weighted votes, a confidence
  threshold defaulting to 0.7, and a documented measurement of what it does
  below that.
* `scripts/predict_scf_input.py` — predict, render, stop. CP2K runs the input
  and remains the authority; three safety rules fall back to defaults, and a
  sidecar records what was recommended and why.
* `retry_ladder()` — the evaluation that found the thing that actually works.

The predictor is kept because the harness around it is what measured its
failure, and because the same harness answers the question again at a larger
dataset. It is not recommended for use as it stands, and its own default
threshold makes it a no-op rather than a risk.
