# Strategy grid and retry ladder, re-run on native CP2K 2026.2

PHAL's static-strategy experiment ([scf_strategy_prediction.md](scf_strategy_prediction.md)) uses
only native CP2K settings: `ALPHA`, `NBUFFER` and Broyden vs Pulay mixing. Unlike every
ML-SCF experiment, it can therefore be repeated without the modified fork. It was re-run on
2026-11-06 with these settings:

- **Binary.** `/opt/homebrew/bin/cp2k.psmp`, CP2K 2026.2, `git:c92cc08` (macOS arm64, Homebrew).
- **Inputs.** The ported `configs/cp2k/strategy.inp.template` (byte-identical to PHAL), with basis and potential resolved from that build's data directory.
- **Structures.** The **regenerated** `periodic.extxyz` and `transition_metal_oxides.extxyz` ([structure_provenance.md](structure_provenance.md)).
- **Threads.** `OMP_NUM_THREADS=1`.
- **Grid.** 6 configurations × 12 solids = 72 runs.

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -m tools.scf.scripts.benchmark_scf_strategies \
  --structures data/structures/pulay_sampling/periodic.extxyz \
               data/structures/pulay_sampling/transition_metal_oxides.extxyz \
  --template configs/cp2k/strategy.inp.template \
  --cp2k-data /opt/homebrew/Cellar/cp2k/2026.2/share/cp2k/data \
  --binary /opt/homebrew/bin/cp2k.psmp --max-parallel 6 --evaluate
```

Raw result: [strategy_rerun_dataset.json](strategy_rerun_dataset.json).

## SCF steps: re-run (PHAL's recorded value)

| system | default | α=0.2 | α=0.1 | nbuffer 4 | broyden | broyden α=0.2 |
| --- | --- | --- | --- | --- | --- | --- |
| `aln-wurtzite` | 11 (10) | 10 (10) | 11 (11) | 11 (11) | 11 (11) | 12 (14) |
| `c-diamond` | 7 (7) | 7 (7) | 7 (7) | 7 (7) | 9 (9) | 9 (9) |
| `cu2o-cuprite` | 14 (19) | 14 (26) | 15 (22) | 40 (86) | 14 (19) | 16 (19) |
| `lih-rocksalt` | FAIL (FAIL) | FAIL (FAIL) | FAIL (FAIL) | **7 (7)** | 10 (10) | 13 (13) |
| `mgo-rocksalt` | FAIL (FAIL) | FAIL (FAIL) | FAIL (FAIL) | **10 (10)** | 13 (13) | 14 (14) |
| `mno-rocksalt` | 78 (71) | 31 (30) | FAIL (118) | FAIL (59) | 68 (65) | 39 (39) |
| `nacl-rocksalt` | FAIL (FAIL) | FAIL (FAIL) | FAIL (FAIL) | **9 (9)** | 12 (12) | 14 (14) |
| `nio-rocksalt` | 37 (33) | 100 (86) | FAIL (66) | FAIL (FAIL) | 39 (39) | 51 (51) |
| `si-diamond` | 8 (8) | 8 (8) | 8 (8) | 8 (8) | 8 (8) | 8 (8) |
| `si-diamond-strained` | 8 (8) | 8 (8) | 8 (8) | 8 (8) | 8 (8) | 9 (8) |
| `tio2-rutile` | 19 (19) | 14 (14) | 14 (14) | 18 (18) | 16 (16) | 14 (14) |
| `zno-wurtzite` | 19 (21) | 21 (22) | 21 (21) | FAIL (FAIL) | 20 (19) | 21 (21) |

**How the failures happen.**

- **MgO, NaCl and LiH with `NBUFFER 8`** abort inside CP2K's own Pulay solve. MgO reports `Bad condition number R_COND = 7.103E-17`, the very number PHAL's `pulay_dataset.md` records.
- **The other failures** are `SCF run NOT converged` within `MAX_SCF 120`.

**Agreement with PHAL.**

- Exact agreement for 8 of the 12 systems on most configurations, and all six for `tio2-rutile`. This is strong evidence that those regenerated geometries match PHAL's.
- Cu2O, NiO and MnO differ. Their geometries and spin set-up are uncertain (see the provenance page).
- Some differences sit within thread and build noise.

## The retry ladder

| ladder | converged | total steps incl. failed attempts (failure = 120) | PHAL |
| --- | --- | --- | --- |
| `default` | 9/12 | 561 | 9/12, 556 |
| `default -> nbuffer4` | **12/12** | **587** | 12/12, 582 |
| `default -> nbuffer4 -> broyden` | 12/12 | 587 | 12/12, 582 |

**PHAL's positive result is reproduced on native CP2K.**

- `default → nbuffer4` converges 12/12 against 9/12.
- It costs 4.6% more iterations; PHAL measured 4.7%.
- The rescued systems are the same three: LiH, MgO and NaCl.

This is the ladder shipped as `scf.retry` (`RetryLadder`). `tests/scf/test_scf_cp2k_native.py::test_the_retry_ladder_rescues_mgo_on_native_cp2k`
repeats the MgO rescue on every CP2K-enabled test run.

## The k-NN predictor (leave-one-out)

| min confidence | label accuracy | recommended steps (default 201) | improved / degraded | rescues |
| --- | --- | --- | --- | --- |
| 0.5 | 33% | 264 | 0 / 1 | 0 |
| 0.6 | 42% | 201 | 0 / 0 | 0 |
| 0.7 | 50% | 201 | 0 / 0 | 0 |
| 0.8 | 50% | 201 | 0 / 0 | 0 |
| 0.9 | 50% | 201 | 0 / 0 | 0 |

The re-run gives the same verdict as PHAL:

- at a permissive threshold the predictor is worse than doing nothing;
- at the shipped threshold (0.7) it is a no-op;
- it rescues nothing.

The oracle total over the default-convergent systems is 174 against 201.
