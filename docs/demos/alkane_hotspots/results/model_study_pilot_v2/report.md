# Alkane hotspot model study: pilot_v2

Config `<GMD_active>/configs/demos/alkane_hotspots/arms_pilot_v2.yaml` (hash `e0ec2c12091ff26c`), generated 2026-11-07T02:55:12+00:00.

## Label records

- file: `<artifacts_root>/labels.jsonl` (62 records)
  - fragment/valid: 46
  - full/valid: 16
  - measured full cost: 42081 CPU core-s over 16 calculations
  - measured fragment cost: 30028 CPU core-s over 46 calculations

## Executed

### Receptive field

- artifact: `<artifacts_root>/model_study_v2/receptive_field.json`
- model source: random_init
- nominal: {"energy_receptive_field_A": 8.0, "force_receptive_field_A": 16.0, "force_receptive_field_hops": 4, "force_receptive_field_local_A": 16.0, "global": false, "local_cutoff": 4.0, "long_range_type": "none", "n_blocks": 2, "note": "force depends on atoms within 2 x n_blocks x local_cutoff (per-atom energy: n_blocks x local_cutoff); a long-range term is global"}
- measured: {"delete_influence_radius_A": 2.756180622197681, "delete_relative_influence_radius_A": 2.756180622197681, "delete_structural_radius_A": 9.652040789782424, "perturb_influence_radius_A": 2.728535710578001, "perturb_relative_influence_radius_A": 2.756180622197681, "perturb_structural_radius_A": 7.586083941475614, "truncation_converged_radius_A": 3.0}
- local model: structural influence radius 9.65 A (any dependence above round-off) vs nominal force receptive field 16.00 A (2 x n_blocks x local_cutoff)

### Receptive field, trained checkpoint

- artifact: `<artifacts_root>/model_study_v2/receptive_field_checkpoint.json`
- model source: checkpoint
- nominal: {"energy_receptive_field_A": 8.0, "force_receptive_field_A": 16.0, "force_receptive_field_hops": 4, "force_receptive_field_local_A": 16.0, "global": false, "local_cutoff": 4.0, "long_range_type": "none", "n_blocks": 2, "note": "force depends on atoms within 2 x n_blocks x local_cutoff (per-atom energy: n_blocks x local_cutoff); a long-range term is global"}
- measured: {"delete_influence_radius_A": 7.586083941475614, "delete_relative_influence_radius_A": 7.402143471859699, "delete_structural_radius_A": 10.90267957540656, "perturb_influence_radius_A": 7.554198644054336, "perturb_relative_influence_radius_A": 7.402143471859699, "perturb_structural_radius_A": 10.90267957540656, "truncation_converged_radius_A": 8.0}
- local model: structural influence radius 10.90 A (any dependence above round-off) vs nominal force receptive field 16.00 A (2 x n_blocks x local_cutoff)

### Hotspot ground truth

- artifact: `<artifacts_root>/model_study_v2/ground_truth.json`
- 4 configurations, 64 regions

Reference threshold 0.1 eV/A (64 positive regions), average precision:

| signal | AP | why undefined |
|---|---|---|
| committee | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| committee_monitor | undefined | signal unavailable on every region |
| committee_plus_lj | undefined | signal unavailable on every region |
| coordination | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| delta_force | undefined | signal unavailable on every region |
| force | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| lj_residual | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| phal_staged | undefined | signal unavailable on every region |
| phal_staged_no_lj | undefined | signal unavailable on every region |
| rmin | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |

Reference threshold 0.2 eV/A (64 positive regions), average precision:

| signal | AP | why undefined |
|---|---|---|
| committee | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| committee_monitor | undefined | signal unavailable on every region |
| committee_plus_lj | undefined | signal unavailable on every region |
| coordination | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| delta_force | undefined | signal unavailable on every region |
| force | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| lj_residual | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| phal_staged | undefined | signal unavailable on every region |
| phal_staged_no_lj | undefined | signal unavailable on every region |
| rmin | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |

Reference threshold 0.4 eV/A (64 positive regions), average precision:

| signal | AP | why undefined |
|---|---|---|
| committee | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| committee_monitor | undefined | signal unavailable on every region |
| committee_plus_lj | undefined | signal unavailable on every region |
| coordination | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| delta_force | undefined | signal unavailable on every region |
| force | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| lj_residual | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |
| phal_staged | undefined | signal unavailable on every region |
| phal_staged_no_lj | undefined | signal unavailable on every region |
| rmin | undefined | no negative regions (every region error >= tau): AP is trivially 1 and uninformative |

- tau=0.1: AP(committee_plus_lj) - AP(committee_monitor) undefined (committee_plus_lj: signal unavailable on every region; committee_monitor: signal unavailable on every region)
- tau=0.1: AP(phal_staged) - AP(phal_staged_no_lj) undefined (phal_staged: signal unavailable on every region; phal_staged_no_lj: signal unavailable on every region)
- tau=0.1: AP(phal_staged) - AP(committee) undefined (phal_staged: signal unavailable on every region; committee: no negative regions (every region error >= tau): AP is trivially 1 and uninformative)
- tau=0.2: AP(committee_plus_lj) - AP(committee_monitor) undefined (committee_plus_lj: signal unavailable on every region; committee_monitor: signal unavailable on every region)
- tau=0.2: AP(phal_staged) - AP(phal_staged_no_lj) undefined (phal_staged: signal unavailable on every region; phal_staged_no_lj: signal unavailable on every region)
- tau=0.2: AP(phal_staged) - AP(committee) undefined (phal_staged: signal unavailable on every region; committee: no negative regions (every region error >= tau): AP is trivially 1 and uninformative)
- tau=0.4: AP(committee_plus_lj) - AP(committee_monitor) undefined (committee_plus_lj: signal unavailable on every region; committee_monitor: signal unavailable on every region)
- tau=0.4: AP(phal_staged) - AP(phal_staged_no_lj) undefined (phal_staged: signal unavailable on every region; phal_staged_no_lj: signal unavailable on every region)
- tau=0.4: AP(phal_staged) - AP(committee) undefined (phal_staged: signal unavailable on every region; committee: no negative regions (every region error >= tau): AP is trivially 1 and uninformative)

### Acquisition arms

| variant | reps complete/planned | labels | cumulative CPU core-s | final force RMSE (eV/A) | monitor-triggered frames |
|---|---|---|---|---|---|
| random_local | 2/2 | [8, 8] | 3393, 3073 | 0.3903, 0.3279 | n/a |
| committee_local | 2/2 | [8, 8] | 3606, 3468 | 0.4552, 0.4108 | n/a |
| phal_local | 2/2 | [0, 0] | 0, 0 | 0.6454, 0.7490 | 0/8, 0/8 |
| phal_full_system | 2/2 | [0, 0] | 0, 0 | 0.6454, 0.7490 | 0/8, 0/8 |
| no_lj | 2/2 | [0, 0] | 0, 0 | 0.6454, 0.7490 | 0/8, 0/8 |
| no_event_buffering | 2/2 | [0, 0] | 0, 0 | 0.6454, 0.7490 | 0/8, 0/8 |
| committee_signals_only | 2/2 | [2, 3] | 1255, 1893 | 0.4839, 0.4806 | 2/8, 4/8 |
| core_only_weights | 2/2 | [0, 0] | 0, 0 | 0.6454, 0.7490 | 0/8, 0/8 |

- **phal_local: not exercised: the phal_local monitor triggered on 0 of 8 streamed pool frames in every repetition, so nothing was acquired and the final model is the round-0 model**
- **phal_full_system: not exercised: the phal_full_system monitor triggered on 0 of 8 streamed pool frames in every repetition, so nothing was acquired and the final model is the round-0 model**
- **no_lj: not exercised: the phal_local monitor triggered on 0 of 8 streamed pool frames in every repetition, so nothing was acquired and the final model is the round-0 model**
- **no_event_buffering: not exercised: the phal_local monitor triggered on 0 of 8 streamed pool frames in every repetition, so nothing was acquired and the final model is the round-0 model**
- **core_only_weights: not exercised: the phal_local monitor triggered on 0 of 8 streamed pool frames in every repetition, so nothing was acquired and the final model is the round-0 model**

### Budget analysis

- artifact: `<artifacts_root>/model_study_v2/evaluation.json`

### Figures

- ablations: ablations.pdf, ablations.svg, ablations.png
- fidelity: fidelity.pdf, fidelity.svg, fidelity.png
- fidelity_distance: fidelity_distance.pdf, fidelity_distance.svg, fidelity_distance.png
- fidelity_max: fidelity_max.pdf, fidelity_max.svg, fidelity_max.png
- learning_curves: learning_curves.pdf, learning_curves.svg, learning_curves.png
- learning_curves_labels: learning_curves_labels.pdf, learning_curves_labels.svg, learning_curves_labels.png
- receptive_field: receptive_field.pdf, receptive_field.svg, receptive_field.png
- receptive_field_checkpoint: receptive_field_checkpoint.pdf, receptive_field_checkpoint.svg, receptive_field_checkpoint.png
- timeline: timeline.pdf, timeline.svg, timeline.png

## Planned / not executed

- figures: {"rendered_without_data (panels say NOT EXECUTED)": {"hotspots": ["signal phal_staged unavailable"], "risk_vs_error": ["phal_staged unavailable on every region"]}}
