# Provenance of the SCF structure data

PHAL's SCF tooling names seven structure files. Only one of them was ever
committed (`tio2_anatase.cif`). The others were excluded by PHAL's
`.gitignore` rule `*.extxyz` (finding F-11). They could not be recovered. Searches covered:

- the GMD_system workspace;
- the cp2k-ml-scf repository;
- PHAL's history at 96263b6.

The missing files were therefore **regenerated**. The table states for each file
whether it is original or regenerated.

| File | Status | Frames | SHA-256 |
| --- | --- | --- | --- |
| `data/structures/pulay_sampling/tio2_anatase.cif` | **original** (byte-identical to PHAL 96263b6, hash in `docs/migration/phal_inventory.json`) | Ti4O8 | `8a4f6ccd…560d` |
| `data/structures/pulay_sampling/small_molecules.extxyz` | regenerated | 17 | `196d704b…f29e` |
| `data/structures/pulay_sampling/periodic.extxyz` | regenerated | 7 | `e0abe7d8…9374` |
| `data/structures/pulay_sampling/transition_metal_oxides.extxyz` | regenerated | 5 | `a49ae73d…e89e` |
| `data/structures/benchmark/molecules.extxyz` | regenerated | 3 | `0ec8fd91…1827` |
| `data/structures/benchmark/periodic.extxyz` | regenerated | 1 | `b2a1af41…0305` |
| `data/structures/benchmark/unseen.extxyz` | regenerated | 4 | `4b4d1661…09d3` |

## How the replacements were made

The generator is `tools/scf/generate_structures.py` (version
`gmd-scf-structures-v1`). Run it as follows:

```bash
python -m tools.scf.generate_structures           # rewrite data/structures/...
python -m tools.scf.generate_structures --check   # verify the committed files
```

It builds every frame deterministically:

- Molecules come from `ase.build.molecule` (the G2 set). Fluoroform is the exception: it is not in G2 and is built from a tetrahedral geometry (C–F 1.33 Å, C–H 1.09 Å).
- Crystals come from `ase.build.bulk` and `ase.spacegroup.crystal`, with literature lattice constants.
- Strained molecules scale every interatomic distance about the centroid by the stated factor. A uniform jitter of ±0.02 Å is then added from `numpy.random.default_rng(20260801 + k)`.
- Every frame carries `label`, `family` and `generator` in its extxyz comment line. PHAL's reader uses those keys.

The **membership** of each file comes from PHAL's own records:

- `small_molecules`: "17 closed-shell molecules, equilibrium and strained" (`pulay_dataset.md`).
- 13 elements and 21 formulas over the 27 accepted systems, of which 17 are molecular and 10 periodic.
- The fitted-on and held-out molecules of `ml_density_failure_analysis.md`.
- The system names in the strategy and controller tables.

PHAL lists no individual molecules, so the choice is ours: 12 equilibrium molecules (H2, H2O, NH3, CH4, CO2, CO, N2, HF, LiH, C2H4, HCN and H2CO) plus 5 strained ones (H2O, NH3, CH4, CO2, N2).

The periodic file holds the 7 main-group solids of the 12-solid table, `si-diamond`
first, because the committed fixture records `source_index: 0`. The transition-metal
oxide file holds the 5 oxides of the table. Anatase comes from the original CIF.

## What matches the originals, and what cannot be compared

**Verified identical.** `si-diamond` (a = 5.431 Å) matches the original. It reproduces the
descriptors stored in the committed fixture
`tests/fixtures/scf/pulay_sample_periodic/sample.json`:

- cell volume 40.04786949774999 Å³;
- minimum distance 2.351691983976643 Å;
- extent 0.678875 Å.

The match is asserted by `tests/scf/test_scf_structures_data.py`.

**Behaviourally consistent with the originals.** The regenerated solids were run through PHAL's strategy grid on native
CP2K 2026.2 ([strategy_rerun.md](strategy_rerun.md)). Several systems match PHAL's table exactly:

- MgO aborts with `R_COND = 7.103E-17`, the same number PHAL records.
- NaCl and LiH also abort with the default settings, then converge with `NBUFFER 4` in 9 and 7 steps.
- Si, strained Si and C reproduce PHAL's step counts.
- TiO2 rutile reproduces all six configurations: 19, 14, 14, 18, 16 and 14 steps.

This is strong evidence that those geometries match. It is not proof.

**Cannot be compared.** These PHAL numbers came from the lost structures, the lost fork, or the lost models. They are
quoted verbatim in the ported docs and are **not** reproducible with these files:

- **Pulay dataset statistics** in `pulay_dataset.md`: 27 of 30 accepted, 460 frames, 379 matrices, cond(B) up to 10^14.4, and so on. They also need the Pulay recorder, which no current build has.
- **ML guess results**: 53 vs 63 steps, failure-analysis distances, the v2 α sweep. They need the lost checkpoints and mode B.
- **Controller replay and counterfactuals**: 433 iterations, precision 0.35, 12/12 keep. They need the lost dataset and the lost controller.
- **Strategy-predictor leave-one-out**: 25–50% accuracy. Its features depend on the exact geometries; the re-run reports its own numbers.
- **The fitted valence/AO tables of the 27-system dataset.**

**Uncertain geometry.**

- The strained molecules and `si-diamond-strained`: PHAL's strain factors are unknown.
- NiO and MnO: built as non-magnetic primitive rocksalt cells, because PHAL's spin set-up is unknown.
- AlN and ZnO: wurtzite with u = 0.382.
- Cu2O: cuprite, a = 4.27 Å.
- All benchmark molecules. Any of these may differ from PHAL's originals.

## Tests that use the replacements

These tests consume the regenerated `periodic.extxyz`:

- `tests/scf/test_scf_dataset_adapter.py`
- `tests/scf/test_scf_cost_aware_targets.py`

At PHAL 96263b6 they error, with 12 errors in all. `tests/scf/test_structure_tasks.py::test_the_shipped_plan_and_templates_agree`
needs every file named by `configs/cp2k/pulay_sampling_plan.json` and failed at 96263b6. Both now pass.
`tests/scf/test_scf_cp2k_native.py::test_the_retry_ladder_rescues_mgo_on_native_cp2k` runs
the regenerated MgO frame through real CP2K.
