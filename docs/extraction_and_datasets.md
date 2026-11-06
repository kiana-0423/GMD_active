# Extraction, capping, embedding data and datasets

This page describes `gmd_active_learning.extraction` and `gmd_active_learning.datasets`. Both packages
replace PHAL's `hotspot_al.extraction`, `training/{mask_generator,region_codes,allegro_adapter,loss_mask}`,
`datasets/writer.py` and the merge step of `RetrainTrigger`, as of PHAL commit 96263b6.
Defect IDs (F-NN) refer to `docs/migration/findings.md`.

## Pipeline

```
Frame + hotspot atom IDs
  -> extract_regions (cluster | slab | graph | block)   ExtractedRegion: roles, weights, parent IDs, image shifts
  -> add_h_caps (separate step, any mode)               region.with_caps(): all per-atom arrays grow together
  -> build_embedding (point-charge data; disabled)      metadata["embedding"] plus capability flags
  -> [labeling: check_labelable, require_embedding_backend, resolve_region_electronic_state]
  -> build_sample(region, LabelResult)                  energy/stress policy, split group
  -> VersionedDatasetStore.add_sample / publish         immutable DatasetVersion (GMD_SGT-ready files)
```

`EventRegionExtractor` (workflow `RegionExtractor`) runs the first three steps for an `EventRecord`.
`VersionedDatasetStore` implements the workflow `DatasetStore` protocol (`add_sample`, `publish`, `get`).

## Configuration sections

The default YAML files are in `configs/sections/`. Each section is a pydantic model with
`extra="forbid"`:

| Section | Model | Switch |
|---|---|---|
| `extraction` | `extraction.config.ExtractionConfig` | `enabled`, `mode` |
| `capping` | `extraction.config.CappingConfig` | `enabled` |
| `embedding` | `extraction.config.EmbeddingConfig` | `enabled` (default false) |
| `datasets` | `datasets.config.DatasetsConfig` | `enabled`, `export.*`, `split.enabled` |

## Regions

Every region records the following:

- `parent_atom_ids`: stable `Frame.atom_ids`, with -1 for caps.
- `image_shifts`: the integer lattice vector of each atom, so that
  `position = parent_position + image_shift @ parent_cell + metadata["origin_translation"]`.
  `origin_translation` is one rigid vector per region that places the region inside its box. The
  contract docstring omits this term; see "Contract notes" below.
- `roles`, `force_weights`, `frozen`.

`frozen` marks boundary atoms when `extraction.freeze_boundary` is true. In block mode the boundary
role is the frozen shell.

`region_id` is `stable_id` over the run, step, mode, event, hotspots, parent IDs, shifts and the
mode's configuration. Re-extracting the same input therefore gives the same ID.

### Roles and force weights (`extraction.force_weights`)

| Role | Weight | Code |
|---|---|---|
| core (block: label_core) | 1.0 | 0 |
| inner_buffer | **0.3** | 1 |
| outer_buffer | 0.0 | 2 |
| boundary (block: frozen_boundary) | 0.0 | 3 |
| h_cap | 0.0 | 4 |
| unassigned (min_atoms padding, slab corners) | 0.0 | -1 |

The integer codes and the extxyz `region_label_map` JSON are byte-identical to PHAL.

F-18: PHAL's block metadata used 0.2 for the inner buffer, but that value was informational only.
The mask actually written to datasets used 0.3, so 0.3 is now the single default for every mode.

### Modes

- **cluster**: every atom within `extract_radius` of any hotspot (default `mlip_cutoff + buffer_radius`).
  Atoms are placed by minimum image from the hotspot centroid inside a non-periodic box. Radial roles
  use `core_radius` and `boundary_thickness`. PHAL padded to `min_atoms` and trimmed to `max_atoms`
  silently. Here `min_atoms_policy` is pad, error or allow, and `max_atoms_policy` is trim or error.
  Padded and trimmed atom IDs are recorded in `metadata["size_policy"]`. Hotspot atoms are never
  trimmed: if they alone exceed `max_atoms`, extraction raises.
- **slab**: a lateral window `(lx, ly)` by `2 * z_margin`, with the surface normal along z. A lateral
  axis stays periodic only when the source axis is periodic and the window covers at least one period.
  - F-08: PHAL built a multi-period cell that held a single image. `slab.multi_period` now chooses
    between `replicate` (default: fill the supercell with images, which repeats parent IDs with
    distinct shifts), `single_period` (recorded as `window_reduced`) and `error`.
  - A skewed source cell with both axes at full period keeps the source lateral vectors. If only one
    axis is at full period, `slab.skewed_policy` decides: `error` (default) or `nonperiodic`.
  - z vacuum is `slab.z_vacuum`, 6 A as in PHAL.
  - Hotspots outside the window are listed in `hotspots_outside_window`.
- **graph**: breadth-first search over the covalent graph (`graph.covalent_scale`, `graph.hops`).
  With `radial_shell` (default), an `extract_radius` shell is added around every graph atom.
  Without it, the region is exactly the subgraph and every atom is core.
- **block**: a spatial grid of `block.size` cells. Adjacent blocks merge (26-neighbourhood, aware of
  periodic boundaries) up to `max_merged_blocks`. The roles are label core, an inner buffer
  (`buffer.inner`), an outer buffer, and a frozen shell of `frozen.thickness` out to
  `max(buffer.outer, halo, inner)`.
  - Trimming to `max_atoms` (frozen, then outer, then inner, farthest first) is recorded. A core
    larger than `max_atoms` raises.
  - `min_atoms_policy` is warn, error or allow.
  - `BlockCooldownTracker` state can be serialised (`to_dict`, `from_dict`, `save`, `load`). It is
    persisted through `block.cooldown_state_path`, and `release()` frees blocks after a failed label.

### Terminal atoms (`extraction.terminal_atoms`)

`as_selected` (default) keeps the cluster/graph selection unchanged. `follow_partner` makes every
terminal atom (exactly one covalent bond, e.g. H; bonds from `graph.covalent_scale`) part of the region
if and only if its bonded partner is: a lone H whose carbon was cut away is dropped, and the H atoms of
an included carbon are added. This keeps molecular fragments free of isolated H atoms, so only
heavy-atom bonds are cut and capped. Added and removed atom IDs are recorded in
`metadata["size_policy"]["terminal_atoms"]`; a hotspot terminal atom whose partner is outside the
selection raises. The default value is left out of the region-ID payload, so existing region IDs do
not change.

### Periodic guard (F-09)

A region is *torn* when two of its atoms lie within `periodic_guard.pair_cutoff` (3 A) of each other
in the periodic parent but sit at different relative images in the region. The check is exact: it
compares integer shifts against the shifts of an ASE neighbour list.

- Torn pairs that involve an atom with non-zero force weight are a violation. Under the default
  `policy: error` such regions are refused. Under `flag` they are kept with a warning, and `allow`
  only records the result.
- Torn pairs between zero-weight atoms are ordinary surface cuts and are only recorded. A torn
  covalent bond is capped.
- `radius_exceeds_half_cell` (extraction radius larger than half the smallest periodic width) is
  always recorded and flagged.

In the PHAL fixtures, the ethane and polyethylene block regions are almost as wide as their 13.8 A
and 15.2 A cells, and the Pt(111) regions are wider than the 9.6 A slab. These regions are torn
through weighted atoms. PHAL emitted them silently; they are now refused by default.

## H capping (`capping`)

Capping is a separate step that works on any mode:
`add_h_caps(region, frame, CappingConfig)`. It uses `ExtractedRegion.with_caps`, so roles, weights,
frozen flags, parent IDs and shifts always have `n_atoms` entries. This is the fix for F-04: capped
block regions now write datasets.

- A bond counts as cut when its partner image is not in the region, decided with integer shifts.
  Each cut bond gets one cap at `parent + L_XH * unit(bond)`. The bond-length table is
  `capping.bond_lengths` (C 1.09, Si 1.48, O 0.96, N 1.01, S 1.34, P 1.42, B 1.19).
- Capping is skipped, and recorded as `h_capping_skipped`/`h_capping_skipped_reason`, when it is
  disabled (`disabled_by_config`), when the region contains a metal (`metal_system`), or when the
  region is an oxide or any cut bond is O-(metal or Si) (`oxide_system`).
- Capping is applied to boundary and outer-buffer atoms. **`allow_core_capping` does what PHAL's
  code did, not what its name says: it adds inner-buffer atoms.** PHAL never capped core atoms.
  `cap_core_atoms` is the switch that really caps core atoms.
- `metadata["h_capping"]` lists every cut bond left dangling (`uncapped_cut_bonds`, with the reason)
  and sets `partial`. PHAL did not tag partial capping.
- `check_labelable(region, config, periodic_labeling=...)` is the gate for uncapped fragments, kept
  from PHAL known issue 1:
  - An isolated calculation of a skipped (metal or oxide) fragment with dangling bonds raises under
    `uncapped_policy: error` (default).
  - Partial capping follows `partial_policy` (default `warn`).
- Charge and multiplicity: `resolve_region_electronic_state(region, settings)` calls
  `utils.electronic_state` on the capped atoms. Precedence is region metadata, then settings, then
  derivation from parity. `spin_policy` is auto or strict.
- Not done, as in PHAL: no clash check, no cell enlargement, and no H-only relaxation (F-17;
  `optimize_h_only` is recorded for the labeling area).

## Embedding data (`embedding`)

`build_embedding` computes point charges for environment atoms within `cutoff` and places them in
the region frame using the recorded shifts and translation.

- The status is disabled (default), unsupported, unavailable or ok.
- Atoms without a charge are skipped and their IDs recorded.
- `EMBEDDING_CAPABILITIES` states that `qmmm_execution` is `{"cp2k": False}`, with no periodic images
  and no charge neutralisation.
- `require_embedding_backend(region, "cp2k")` raises `CapabilityUnavailableError` for any enabled
  embedding. This is known issue 2 and stays fail-closed; the labeling area must call it.

## Samples and the energy/stress policy (`datasets`)

`build_sample(region, LabelResult, DatasetsConfig)` accepts only VALID labels for the same region,
with forces of shape `(n_atoms, 3)`.

- `sample_id = stable_id("sample", region_id, task_id)`. Re-adding the same label is a no-op; a
  different label under the same ID is refused.
- **Energy:** the stored energy is the region's own DFT energy (`energy_scope = "region"`,
  `energy_is_parent_energy = False`). It is never attached to the parent frame. Whether it is
  trained is set by `energy_weight`:
  - `energy.nonperiodic_weight` (default 0) or `energy.periodic_weight` (default 0).
  - Capped regions get 0 unless `energy.allow_capped` is set.
  - The energy is still written as provenance when `write_energy` is true.
  - PHAL wrote no energy at all (F-12).
- **Stress:** kept only when labeled, the region is periodic and `stress.enabled` is true. Otherwise
  the stress is dropped and `stress_weight` is 0. Stress must be symmetric, because extxyz stores
  Voigt components.
- `split_group` is `event:<id>`, or `run:<id>` when the sample has no event. It can also be
  `region:<id>` (`split.group_by`).

## Files

- **extxyz** (`write_samples_extxyz`): GMD_SGT keys plus provenance.
  - Per-atom columns: `species pos forces force_weight region_code frozen parent_atom_id image_shift`.
  - Info keys: `energy` (when present), `energy_weight`, `stress` (when present), `stress_weight`,
    `split_group`, `region_label_map`, and `gmd_provenance` (JSON with the sample, region, event and
    run IDs, parent step, task, model version, label engine and version, convergence, validation,
    output sha256).
  - Floats are written with their shortest exact repr, so the round trip is bit-exact. ASE's own
    writer rounds to 8 decimals.
- **Canonical NPZ** (`write_samples_npz`): pickle-free concatenated arrays plus a JSON sidecar that
  includes the region metadata. This is what the store keeps.
- **GMD_SGT NPZ** (`write_sgt_npz`): `R Z E F cell pbc stress force_weight energy_weight
  stress_weight split_group`. Missing labels are NaN. Ragged sets are stored as object arrays, which
  GMD_SGT loads with `allow_pickle`.
- **Per-sample triple** (`write_dataset_entry`): PHAL's `<prefix>.extxyz`, `_labels.npz` and
  `_metadata.json` keys, plus the new fields.

All writes are atomic.

## Versioned store

`VersionedDatasetStore(root)` uses this layout:

- `samples/<sample_id>.npz|json`
- `versions/<version>/{samples.npz, samples.json, all.extxyz, train.extxyz, val.extxyz, sgt.npz, split.json}`.
  These directories are staged, renamed into place and made read-only.
- `registry/<version>.json`
- `index.json`

How publishing works:

- `publish(sample_paths, parent=..., split_groups=...)` creates a cumulative version: the parent's
  samples plus new ones, with duplicates removed.
- The version ID is derived from the parent, the ordered sample IDs, the groups and the export
  settings. Publishing the same samples returns the same version, and publishing only samples the
  parent already holds returns the parent.
- `DatasetVersion.sha256` is `tree_sha256` over the version directory, and `get()` re-verifies it.
- `metadata` holds absolute `train_file`, `val_file`, `all_file`, `sgt_npz`, `samples_file` and
  `split_file`.

How the split works:

- The split is event-aware: a group's side is a function of `(seed, group)` (hash compared with
  `val_fraction`).
- Groups inherited from the parent keep their side, so validation data never moves into training.
- When `val_fraction > 0` and there are at least two groups, at least one new group goes to
  validation.

`retraining.dataset_merger.merge_datasets(existing_dataset, labeled_dir, output_dir)` keeps its name
and argument order but now publishes through the store. `existing_dataset` must be a version ID in
that store, or None. PHAL's merger had no versioning, deduplication or split; GMD_active's old
merger wrote a two-line manifest with the literal parent string `existing_dataset` (F-10).

## Verification against GMD_SGT

`tests/datasets/test_gmd_sgt_compat.py` reads published `train_file`, `val_file` and `sgt.npz` with
`gmd_sgt.data.reader`. It checks `force_weight`, `energy_weight`, `stress_weight` and `split_group`.
It also compares `EnergyForceLoss._weighted` on a collated batch with the NumPy oracle in
`tests/reference/masked_force_mse.py` (TRN-003), including zero gradient on zero-weight atoms.

The test runs when GMD_SGT 0.4 or later is importable: through `GMD_SGT_PATH`, the sibling
checkout, or an installed package. It is skipped, with the reason, for older GMD_SGT versions.

## PHAL parity

`tests/extraction/test_extraction_phal_parity.py` replays all 42 golden regions (4 modes in
3 scenarios) and checks the following:

- **Matches PHAL:** atom sets, roles, positions and cells, hotspots and graph atoms. The tolerance is
  5e-8 A because the input extxyz stores 8 decimals.
- **Capped regions:** caps, masks and codes are checked. Where the region is torn, PHAL's caps must
  be a subset of the new caps.
- **Corrected behaviour:**
  - Image shifts reproduce every position.
  - Capped block regions write datasets. PHAL raised "could not broadcast" for all of them.
  - Torn regions are refused by the default guard.
- **PHAL's dataset files:** `mask_weights` and `region_code` in PHAL's written datasets are compared
  for the non-block modes.

`tests/extraction/test_phal_live_comparison.py` compares radial partitioning, block assignment and
merging, and cooldown on randomized inputs against PHAL itself. It is skipped once `ref/` is removed.

## Contract notes

- `ExtractedRegion`'s docstring says `position = parent_position + image_shift @ cell`. The extracted
  box adds a rigid `origin_translation`, recorded in metadata. The contract docstring (or an explicit
  field) should mention it.
- Replicated slab images repeat parent atom IDs with different `image_shifts`. The contract allows
  this, but consumers must not assume parent IDs are unique within a region.
