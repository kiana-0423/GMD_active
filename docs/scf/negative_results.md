# ML-SCF: the negative-result record

> All measurements below were taken by PHAL on the **lost** `cp2k-ml-scf` fork, with
> models (`density_final.pt`, `density_iter1.pt`) and datasets that **no longer exist**.
> They are quoted verbatim. They cannot be re-run in GMD_active; see the
> [re-run matrix](README.md#what-can-be-re-run-here-and-what-needs-the-modified-cp2k).
> The full reports are ported verbatim alongside this page.

**Bottom line: no acceleration benefit was ever demonstrated for any ML component.**
The positive results that exist are of two kinds:

- **Mechanism checks.** Replaying an exact converged density converges H2/H2O in 1 step. This proves installation, not prediction.
- **The model-free retry ladder.** `default → nbuffer4` converges 12/12 against 9/12.

## ML initial density guess — [ml_density_failure_analysis.md](ml_density_failure_analysis.md)

> "The ML initial guess costs 63 SCF steps where CP2K's atomic superposition costs
> 53."

| | atomic | ML (torchscript) |
| --- | --- | --- |
| total SCF steps | **53** | 63 |
| mean speedup | 1.00 | **0.85** |
| max energy error | — | 9.3×10⁻¹⁰ Ha |
| max force error | — | 1.1×10⁻⁴ Ha/bohr |

> "**The ML guess is worse than the atomic guess on every system.**" (PHAL
> `ml_scf_architecture.md`)

> "**On the chemistry it was fitted on, the model is not worse than the atomic
> guess: mean excess distance −9.6%. On chemistry it has never seen, it is +87.6%
> worse.**"

> "Spearman correlation between excess distance and extra SCF steps: **+0.29**
> across 8 systems."

## Density model v2 (cost-aware, early-SCF target, hybrid α) — [ml_density_v2.md](ml_density_v2.md)

> "The first density model was physically valid and did not accelerate anything."
> "Result: it did not help."
> "**Nothing improved.** Across 48 ML runs, two beat the baseline on a single
> system and 39 were worse."
> "**A perfect predictor saves 66% of all SCF steps.**" "Going **75% of the way** to the exact
> density saves **17%**."
> "There is **no partial credit** in initial-density prediction."
> "Neither is a production accelerator."

## Learned DIIS / Pulay mixer — [ml_pulay_objective.md](ml_pulay_objective.md)

> "**Status: negative result. Experimental only — not a production backend.**"
> "**native DIIS already solves the minimisation exactly, so a model trained to
> reproduce it has nothing left to learn.**"
> "Measured: convergence degrades monotonically with acceptance rate. 11 -> 15 -> 37 SCF steps."
> "**DIIS is not the weak link.**"

No CP2K build consumes a neural mixer: `MIXING_METHOD ML` was never integrated, and
cp2k-ml-scf declares it out of scope. The model is kept as gated experimental tooling
(`tools/scf/experimental/neural_mixer/`, ledger SCF-030). The coefficient analysis
lives in `tools/scf/negative_results/`.

## Adaptive SCF controller — [adaptive_scf_controller.md](adaptive_scf_controller.md)

| | native | controller |
| --- | --- | --- |
| converged | 8/12 | 8/12 |
| total SCF steps | **172** | **172** |
| rescued / lost | — | 0 / 0 |
| actions taken | 0 | 5 |

> "The controller is **safe and neutral**."
> "**Twelve of twelve labels are `keep`.**"
> "Not delivered: any acceleration."

The ported offline model (`tools/scf/controller/policy.py`) is inconsistent with that
document in three ways; ledger SCF-016 records them:

- it lacks the documented "actions per run 2", "no action past 50% of MAX_SCF" and "1 rejection if irreversible" guards;
- its `observe()` docstring speaks of restoring "the previous electronic state";
- the doc, by contrast, says rollback "restores settings and never an electronic state".

The live controller was Fortran in the lost fork and does not exist in the reconstruction.

## Static strategy prediction — [scf_strategy_prediction.md](scf_strategy_prediction.md)

> "The predictor does not work at this size" … "It is not recommended for use as it
> stands, and its own default threshold makes it a no-op rather than a risk."

| min confidence | label accuracy | total steps (default 196) |
| --- | --- | --- |
| 0.5 | 25% | 250 |
| ≥ 0.7 | 50% | 196 |

## The one result that needs no model

> "**`default → nbuffer4` converges 12 of 12 against the default's 9, for 4.7%
> more iterations.**"

| ladder | converged | total steps incl. failed attempts |
| --- | --- | --- |
| `default` | 9/12 | 556 |
| `default -> nbuffer4` | 12/12 | 582 |

This ladder is re-implemented as `gmd_active_learning.adapters.dft.scf.strategy.RetryLadder`
(`scf.retry`, off by default). It reproduces these numbers from the documented table
(`tests/scf/test_scf_retry_ladder.py`). It was re-run on native CP2K 2026.2 with regenerated
structures: the three rescues reproduce ([strategy_rerun.md](strategy_rerun.md)).

Caveat: PHAL's "default" uses `NBUFFER 8`, while CP2K's own default is 4. The rescue is
therefore partly a return to upstream's default.
