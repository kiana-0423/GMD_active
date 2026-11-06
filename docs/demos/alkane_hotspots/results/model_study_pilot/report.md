# Alkane hotspot model study: pilot

Config `<GMD_active>/configs/demos/alkane_hotspots/arms_pilot.yaml` (hash `6d363dceb7ded3a2`), generated 2026-11-06T20:52:11+00:00.

## Label records

- file: `<artifacts_root>/labels.jsonl` (59 records)
  - fragment/valid: 43
  - full/valid: 16
  - measured full cost: 42081 CPU core-s over 16 calculations
  - measured fragment cost: 28816 CPU core-s over 43 calculations

## Executed

### Receptive field

- artifact: `<artifacts_root>/model_study/receptive_field.json`
- model source: random_init
- nominal: {"energy_receptive_field_A": 8.0, "force_receptive_field_A": 16.0, "force_receptive_field_hops": 4, "force_receptive_field_local_A": 16.0, "global": false, "local_cutoff": 4.0, "long_range_type": "none", "n_blocks": 2, "note": "force depends on atoms within 2 x n_blocks x local_cutoff (per-atom energy: n_blocks x local_cutoff); a long-range term is global"}
- measured: {"delete_influence_radius_A": 2.751629342241791, "delete_relative_influence_radius_A": 2.751629342241791, "delete_structural_radius_A": 8.94929770129875, "perturb_influence_radius_A": 2.751629342241791, "perturb_relative_influence_radius_A": 2.751629342241791, "perturb_structural_radius_A": 7.641195081606981, "truncation_converged_radius_A": 3.0}
- local model: structural influence radius 8.95 A (any dependence above round-off) vs nominal force receptive field 16.00 A (2 x n_blocks x local_cutoff)

### Hotspot ground truth

- artifact: `<artifacts_root>/model_study/ground_truth.json`
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

| variant | reps complete/planned | labels | cumulative CPU core-s | final force RMSE (eV/A) |
|---|---|---|---|---|
| random_local | 2/2 | [8, 8] | 3393, 3073 | 0.6639, 0.6344 |
| committee_local | 2/2 | [8, 8] | 2922, 3738 | 0.6375, 0.7565 |
| phal_local | 2/2 | [0, 0] | 0, 0 | 0.8272, 0.8312 |
| phal_full_system | 2/2 | [0, 0] | 0, 0 | 0.8272, 0.8312 |
| no_lj | 2/2 | [0, 0] | 0, 0 | 0.8272, 0.8312 |
| no_event_buffering | 2/2 | [0, 0] | 0, 0 | 0.8272, 0.8312 |
| committee_signals_only | 2/2 | [0, 0] | 0, 0 | 0.8272, 0.8312 |
| core_only_weights | 2/2 | [0, 0] | 0, 0 | 0.8272, 0.8312 |

### Budget analysis

- artifact: `<artifacts_root>/model_study/evaluation.json`

### Figures

- ablations: ablations.pdf, ablations.svg, ablations.png
- fidelity: fidelity.pdf, fidelity.svg, fidelity.png
- fidelity_max: fidelity_max.pdf, fidelity_max.svg, fidelity_max.png
- learning_curves: learning_curves.pdf, learning_curves.svg, learning_curves.png
- learning_curves_labels: learning_curves_labels.pdf, learning_curves_labels.svg, learning_curves_labels.png
- receptive_field: receptive_field.pdf, receptive_field.svg, receptive_field.png

## Planned / not executed

- receptive_field_checkpoint: {"reason": "no receptive_field_checkpoint.json (receptive-field --checkpoint <trained model>)"}
- figures: {"rendered_without_data (panels say NOT EXECUTED)": {"hotspots": ["signal phal_staged unavailable"], "risk_vs_error": ["phal_staged unavailable on every region"], "timeline": ["no closed-loop controller report"]}}
