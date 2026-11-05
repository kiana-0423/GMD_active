> **Ported from PHAL** (`docs/ml_density_failure_analysis.md` at 96263b6, MIT License, see NOTICE). The body below is
> kept **verbatim**, including every measured number. Those measurements were taken by
> PHAL on the lost `cp2k-ml-scf` fork, with models and datasets that no longer exist;
> they cannot be re-run here as written. Needs the in-process guess model (mode B), which the cp2k-ml-scf reconstruction does not implement.
> Code paths named below (`hotspot_al.*`, `scripts/*`) map to GMD_active as listed in
> [README.md](README.md#module-map). Status of each feature: [README.md](README.md).

# Why the ML density guess is slower

The ML initial guess costs 63 SCF steps where CP2K's atomic superposition costs
53. The pipeline is correct — the model runs in process, the density is
installed, `Tr(P S) = Ne` holds — so the deficit is algorithmic. This is the
measurement that says which of the four candidate causes it is.

Nothing here was retrained. Every number comes from running each system twice,
identical except for `SCF_GUESS`, with the density printed at every iteration.

## What was measured

Per system and per guess: `||P_guess − P_final||` in both the plain Frobenius
norm and the overlap metric `sqrt(Tr(dP S dP S))`; the residual after the first
iteration; the per-iteration reduction rate; the energy path; and the SCF step
count. `P_final` is the **atomic** run's converged density for both arms — one
target, so the two distances are comparable.

`||dP||` is reported in the overlap metric because `P` lives in a
non-orthogonal AO basis, where a plain Frobenius norm weights each orbital by
however diffuse it happens to be.

## Result

| system | fitted on | `d(atomic)` | `d(ML)` | excess | SCF steps | cause |
| --- | --- | --- | --- | --- | --- | --- |
| `co2` | yes | 2.53 | 3.89 | +54% | 17 → 22 | A |
| `h2o` | yes | 1.42 | 1.04 | -27% | 14 → 15 | B |
| `nh3` | yes | 1.75 | 1.86 | +6% | 14 → 16 | A |
| `si` | yes | 4.23 | 1.21 | -71% | 8 → 10 | B |
| `h2cchf` | no | 2.73 | 4.08 | +49% | 19 → 21 | A |
| `hcf3` | no | 2.56 | 5.44 | +112% | 17 → 18 | A |
| `hcooh` | no | 2.69 | 4.83 | +80% | 18 → 20 | A |
| `n2h4` | no | 2.55 | 5.35 | +110% | 15 → 18 | A |

MgO is excluded: Mg is absent from the model's basis table, so its ML arm fell
back and both rows are the same run. A fallback is not evidence about a model.

## Verdict: C, generalisation failure — with a decisive caveat

**On the chemistry it was fitted on, the model is not worse than the atomic
guess: mean excess distance −9.6%. On chemistry it has never seen, it is +87.6%
worse.** A gap of ~1.0 in relative distance between fitted and held-out systems
is the signature of a model that memorised 19 systems rather than learning a
transferable map. Note that every held-out system is built from elements the
model *does* know (H, C, N, O, F) — this is not a missing-element problem, it
is a missing-generalisation problem.

That is the primary cause. It is not the whole answer.

### B is present independently, and it is what decides whether fixing C helps

Three measurements say so:

**Closer densities still converge slower.** Si starts **71% closer** to the
converged density than the atomic guess and with a first residual 6.8× smaller
— and takes 10 SCF steps against 8. H₂O starts 27% closer and takes 15 against
14. Both are systems the model was fitted on.

**Distance barely predicts cost.** Spearman correlation between excess distance
and extra SCF steps: **+0.29** across 8 systems. The quantity the model is
trained to minimise is not the quantity that governs convergence.

**The ML guess starts at a much higher energy — on 7 of 8 systems.** Mean
`|E_first − E_converged|`: **2.89 Ha for ML against 0.42 Ha for atomic.** A
density can be closer to the answer in norm while being a far less physical
state: an atomic superposition is a sum of real atomic densities, correctly
idempotent per atom, so its error lies in the few slow chemical modes DIIS
extrapolates well. A regressed density matrix has no such structure, and its
error is spread over many modes — including high-frequency ones the Kerker
damping deliberately mixes slowly. That is visible in the per-iteration
reduction rate, which is worse for ML on every system (Si: 0.198 against
0.098).

### D is visible but secondary

The deficit tracks difficulty: `medium` systems are +74% worse than atomic,
`easy` ones +4%. The training set is 27 systems of which most are small
molecules at equilibrium, so this is a property of the data, not of the
architecture.

### A is a symptom, not a cause

Six of eight systems classify as "poor prediction", but five of those six are
held-out chemistry. Read alone it would suggest "fit a better model"; read with
the generalisation split it means "the model was never asked to generalise".

## What this implies for the next step

**Do not simply retrain on more data.** More data addresses C, and C alone
would leave B untouched — and B is why Si, where the prediction is already
excellent, still costs two extra iterations.

The three things the measurements actually point at:

1. **Train against SCF cost, not density distance.** The correlation is +0.29.
   A model that halves `||dP||` has not been shown to save a single iteration.
2. **Predict something the SCF can use.** The energy gap suggests the target
   should be constrained to be a physically admissible density — idempotency
   under `S`, correct occupation structure — not merely a matrix at the right
   distance with the right trace.
3. **Then broaden the dataset**, which is necessary for C but not sufficient
   for anything.

The dataset for (1) already exists: `runtime/cp2k/pulay_dataset` records every
iteration of every SCF, so "how many steps remain from this density" is a
measured label, not a guess.

## Reproducing

```bash
scripts/diagnose_ml_guess.py \
    --structures data/structures/benchmark/{molecules,periodic,unseen}.extxyz \
    --template configs/cp2k/diagnostic_guess.inp.template \
    --cp2k-data ../cp2k-ml-scf/cp2k/data \
    --binary build/cp2k-ml/bin/cp2k.psmp \
    --guess-model runtime/cp2k_ml/models/active/guess.ts.pt \
    --max-parallel 3
```

Writes `ml_density_failure_report.json`. `--max-parallel` matters: every run
prints its density each iteration, and the first attempt lost three runs to
memory pressure, which looks exactly like a system that failed to converge.
