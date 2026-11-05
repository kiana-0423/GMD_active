> **Ported from PHAL** (`docs/ml_density_v2.md` at 96263b6, MIT License, see NOTICE). The body below is
> kept **verbatim**, including every measured number. Those measurements were taken by
> PHAL on the lost `cp2k-ml-scf` fork, with models and datasets that no longer exist;
> they cannot be re-run here as written. Needs the modified CP2K (GUESS_ALPHA, in-process models) and the lost checkpoints.
> Code paths named below (`hotspot_al.*`, `scripts/*`) map to GMD_active as listed in
> [README.md](README.md#module-map). Status of each feature: [README.md](README.md).

# SCF-cost-aware density initialisation (v2)

The first density model was physically valid and did not accelerate anything.
The diagnosis ([ml_density_failure_analysis.md](ml_density_failure_analysis.md))
showed why the obvious fix would not work: density distance predicts SCF cost at
Spearman **+0.29**, so a better-fitting model is not, by itself, a faster one.

v2 changes what is optimised and what is measured:

* **trajectory targets** — a model can be trained to predict the density after
  *k* SCF iterations instead of the converged one;
* **SCF-cost labels** — every target frame carries the iterations that remained
  from it, and the residual and energy reduction rates over them;
* **hybrid initialisation** — `P = (1 - α) P_atomic + α P_ML`, swept;
* **chemistry holdout** — whole formula families held out, never geometries;
* **the v1 final-density model is kept** as the baseline it now is.

## Why an early-SCF target should have helped

A converged density is a *fixed point*: the SCF never passes through it, so
nothing about it tells a model what a good *starting* density looks like. The
density after one iteration is a state CP2K actually produced — reachable by
construction, with its error already in the modes the mixer clears well. That
was the hypothesis.

## Result: it did not help

Six systems, 9 arms, 54 CP2K runs. Every run converged.

| arm | total SCF steps | converged | mean speedup | improved | worsened |
| --- | --- | --- | --- | --- | --- |
| `atomic` | 86 | 6/6 | 1.000 | 0 | 0 |
| `final-a0.25` | 96 | 6/6 | 0.901 | 0 | 5 |
| `final-a0.5` | 98 | 6/6 | 0.884 | 0 | 5 |
| `final-a0.75` | 101 | 6/6 | 0.863 | 1 | 5 |
| `final-a1` | 109 | 6/6 | 0.800 | 0 | 6 |
| `early-a0.25` | 96 | 6/6 | 0.905 | 0 | 4 |
| `early-a0.5` | 98 | 6/6 | 0.899 | 1 | 4 |
| `early-a0.75` | 102 | 6/6 | 0.852 | 0 | 5 |
| `early-a1` | 109 | 6/6 | 0.811 | 0 | 5 |

Per system:

| system | atomic | final α=0.25 | early α=0.25 | final α=1 | early α=1 |
| --- | --- | --- | --- | --- | --- |
| `h2o` | 14 | 14 | 14 | 16 | 14 |
| `nh3` | 14 | 15 | 14 | 16 | 16 |
| `co2` | 17 | 20 | 20 | 25 | 26 |
| `si` | 8 | 9 | 9 | 10 | 10 |
| `hcooh` | 18 | 20 | 21 | 23 | 24 |
| `n2h4` | 15 | 18 | 18 | 19 | 19 |

Three things follow.

**The early-SCF target is not better than the final-density target.** 96 steps
against 96 at α=0.25; 109 against 109 at α=1. The hypothesis was reasonable and
is not supported.

**α is monotonic in the wrong direction.** More ML is uniformly worse: 86
(none) → 96 (α=0.25) → 109 (α=1). Hybrid mixing does not find a sweet spot; it
finds that the best available weight is the smallest one, and the limit α → 0
is the atomic guess.

**Nothing improved.** Across 48 ML runs, two beat the baseline on a single
system and 39 were worse.

## The measurement that reframes this

The obvious question is whether initial-density prediction can help *at all*.
It can, and by a lot. Feeding back the recorded **exact converged density** as
the guess, and mixing it with the atomic one at the same α:

| system | α=0 (atomic) | 0.25 | 0.5 | 0.75 | α=1 (exact) |
| --- | --- | --- | --- | --- | --- |
| `h2o` | 14 | 14 | 14 | 12 | **6** |
| `nh3` | 14 | 14 | 14 | 12 | **5** |
| `co2` | 17 | 15 | 14 | 14 | **6** |
| `si` | 8 | 8 | 7 | 6 | **1** |
| **total** | **53** | 51 | 49 | 44 | **18** |

**A perfect predictor saves 66% of all SCF steps.** The concept is sound and
the headroom is large — Si converges in a single iteration.

But look at the shape of the curve. Going **75% of the way** to the exact
density saves **17%**. The last quarter of the distance
delivers the remaining 49%. The payoff is
strongly non-linear and almost all of it sits at the very end.

That explains everything else on this page:

* why hybrid mixing cannot rescue a mediocre model — the curve is flat until you
  are nearly exact, so diluting a bad prediction just walks back toward atomic;
* why a model that is "closer than the atomic guess" still costs iterations —
  closer is not close;
* why changing the target from `P_final` to `P_iter1` changed nothing — the
  problem is not *which* nearby density is predicted, it is that the prediction
  is not near enough.

## What this says about the next step

There is **no partial credit** in initial-density prediction. A model must be
close to exact before it returns anything, and the current one — 27 training
systems, a 2.5–3.9× train/holdout MAE gap on a chemistry holdout — is nowhere
near that bar. Incremental improvement is not an attractive path: a model twice
as accurate as this one would still sit in the flat part of the curve.

Options that follow from the measurement, in the order the evidence supports:

1. **Restrict the claim.** The guess pays off inside a trajectory, where the
   previous geometry's converged density is available and genuinely close —
   which is what CP2K's own `SCF_GUESS RESTART` already exploits. That is a
   real use case and the measured curve says it is where the benefit lives.
2. **Change what is predicted.** Not a density, but a correction to a density
   the SCF is already near — the same reason `RESTART` works.
3. **Spend the effort on the controller instead.** Robustness on hard systems
   is a separate axis and does not need this accuracy bar.

## What was kept

The v1 final-density model is retained as the baseline
(`runtime/cp2k_ml/models/density_final.pt`), alongside the v2 early-SCF model
(`density_iter1.pt`). Both deploy, both run in process, and both are measured
above. Neither is a production accelerator.

## Reproducing

```bash
scripts/train_density_guess.py --target-frame 1 --train-on split   # honest holdout
scripts/train_density_guess.py --target-frame 1 --train-on all --deploy ...

scripts/benchmark_guess_variants.py \
    --model final=.../guess_final.ts.pt --model early=.../guess_iter1.ts.pt \
    --alpha 0.25 0.5 0.75 1.0 --max-parallel 3 ...
```

`PHAL_ML_SCF_GUESS_ALPHA` is read by CP2K itself; the mixing happens where the
atomic density already exists, in the install hook, so α = 1 reproduces the
pure prediction exactly.
