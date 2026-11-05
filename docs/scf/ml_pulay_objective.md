> **Ported from PHAL** (`docs/ml_pulay_objective.md` at 96263b6, MIT License, see NOTICE). The body below is
> kept **verbatim**, including every measured number. Those measurements were taken by
> PHAL on the lost `cp2k-ml-scf` fork, with models and datasets that no longer exist;
> they cannot be re-run here as written. Status: negative result. The offline analysis is ported to `tools/scf/negative_results/`; MIXING_METHOD ML has no CP2K consumer.
> Code paths named below (`hotspot_al.*`, `scripts/*`) map to GMD_active as listed in
> [README.md](README.md#module-map). Status of each feature: [README.md](README.md).

# What an ML Pulay mixer can and cannot learn

> **Status: negative result. Experimental only — not a production backend.**
>
> Direct learning of DIIS/Pulay coefficients was tried, measured, and stopped.
> The code, dataset and this analysis are kept because the reason it fails is
> reusable, not because the line is being continued. Do not optimise coefficient
> imitation further. The active direction is
> [ML initial density + adaptive control](ml_scf_architecture.md).
>
> Section 1 is the whole argument in one line: **native DIIS already solves the
> minimisation exactly, so a model trained to reproduce it has nothing left to
> learn.**

`MIXING_METHOD ML_PULAY` predicts the DIIS coefficient vector. This document
records what that model should be trained on, because the obvious objective is
a tautology and the obvious safety gate forbids the answer.

## 1. `min c^T B c` is native DIIS, not a learning target

The constrained problem

```
minimise  sqrt(c^T B c)      subject to   sum(c) = 1
```

has the closed form

```
c = B^-1 1 / (1^T B^-1 1)
```

which is exactly what `qs_gspace_mixing.F` computes: invert `B`, sum the
columns, normalise. Measured agreement with CP2K's own solve is 1e-14 to
1e-10, and of 10000 random feasible vectors, **0** beat it.

So a model trained toward that target can, at best, reproduce native Pulay.
There is no signal in it.

## 2. The signal is in *which* `B`

| `B` from | meaning of `c^T B c` | native DIIS is | available online |
| --- | --- | --- | --- |
| online `r_i = P_{i+1} - P_i` | modelled residual | optimal | yes |
| hindsight `r_i = P_i - P_final` | `\|\|sum c_i P_i - P_final\|\|^2` | optimal | **no** |

With `sum(c) = 1` the hindsight quadratic form *is* the squared distance to the
converged density, so minimising it is minimising the real error — but it needs
the answer, so a running SCF cannot form it.

Native DIIS minimises the first and lands 1.9x-3.1x from optimal on the second,
measured on recorded trajectories. **That gap is the learnable content**: map
the online `B` (which the SCF has) onto the hindsight-optimal `c` (which it
does not).

Building `B` from hindsight residuals collapses the two objectives into one and
makes the targets identical to native DIIS. The first version of the sample
generator did exactly that.

## 3. The residual gate forbids the improvement

The run-time gate accepts a proposal only if its `c^T B c` is no worse than
native's, on CP2K's online `B`. Native minimises that quantity, so **any**
proposal that is better in truth is necessarily worse by this measure and is
rejected — by 1.3x to 36000x on recorded samples.

The gate is therefore not a divergence guard at the default setting; it is an
optimality requirement, and it makes learned improvement impossible.
`PHAL_ML_SCF_PULAY_TOLERANCE` raises the factor deliberately. The default stays
strict, so a bad model costs nothing and a good one needs an explicit decision.

## 4. Train on CP2K's `B`, not a reconstruction of it

CP2K's `mixing_store%pulay_matrix` is built from **g-space** residuals,
Kerker-weighted (`kerker_eff`), weighted again by `special_metric`, and scaled
by `vol^2`. A `B` reconstructed from recorded AO densities is a different
matrix in a different space under a different metric.

Normalising by `max|B|` removes the scale but not the shape. A model trained on
the reconstruction is fed something else at run time, and the measured result is
that it degrades convergence monotonically with acceptance rate (11 -> 15 -> 37
SCF steps as the tolerance is widened).

**Recording `pulay_matrix` per iteration is a prerequisite for training a
deployable mixer.** Reconstructing it is not sufficient.

## 5. Off-policy training

Targets come from trajectories some *other* mixer produced. Once the ML mixer
is live it changes the history it will next be asked about, so errors compound
in a way offline training cannot observe. Even with a matched `B`, expect to
need either on-policy data collection or a conservative acceptance factor.

## 6. Why this line was stopped

Every route out of section 1 was checked and each is blocked:

| Route | Why it fails |
| --- | --- |
| Train `B -> c_native` | That map is `B^-1 1 / (1^T B^-1 1)`, computed exactly by CP2K. A perfect model matches native; an imperfect one is worse. Best case: zero gain. |
| Train `B -> c_hindsight` | The improvement is real (native lands 1.9x–3.1x from optimal) but the target needs the converged density, and the run-time safety gate — which accepts only if `c^T B c` does not worsen — rejects every such proposal by construction, because native already minimises that form. |
| Widen the gate | Measured: convergence degrades monotonically with acceptance rate. 11 -> 15 -> 37 SCF steps. |
| Better features on the same target | Does not change that the target is the solver's own output. |

The deeper point: **DIIS is not the weak link.** Given a history, it already
extrapolates optimally under its own metric. What determines whether an SCF is
fast or fails is *where it starts* and *what strategy it runs* — which history
to keep, when to reset it, when to change method. Those are the two things ML
can decide and CP2K cannot decide for itself, and they are what the current
architecture targets.

## 7. What is kept, and why

* `ml_pulay_mixer.F`, `ml_pulay_recorder.F`, `MIXING_METHOD ML_PULAY` — kept and
  still tested. The recorder is now a **feature source** for the adaptive
  controller: `B`'s condition number, rank and coefficient norm are among the
  strongest signals that an SCF is in trouble.
* `runtime/cp2k/pulay_dataset/` — kept. It is the training set for the
  controller, and the only place the conditioning of real `B` matrices is
  measured.
* `export_mixer()` — kept, but now requires `experimental=True`. It cannot be
  wired into a production configuration by accident.

The dataset outlived the model it was collected for. That is a normal outcome,
not a wasted effort.
