# Alkane hotspot model study: pilot_v2c

Config `<GMD_active>/configs/demos/alkane_hotspots/arms_pilot_v2c.yaml` (hash `9c1cd8448963cdab`), generated 2026-11-07T04:13:31+00:00.

## Label records

- file: `<artifacts_root>/labels.jsonl` (62 records)
  - fragment/valid: 46
  - full/valid: 16
  - measured full cost: 42081 CPU core-s over 16 calculations
  - measured fragment cost: 30028 CPU core-s over 46 calculations

## Executed

### Acquisition arms

| variant | reps complete/planned | labels | cumulative CPU core-s | final force RMSE (eV/A) | monitor-triggered frames |
|---|---|---|---|---|---|
| random_local | 2/2 | [8, 8] | 3393, 3073 | 0.3903, 0.3279 | n/a |
| committee_local | 2/2 | [8, 8] | 3606, 3468 | 0.4552, 0.4108 | n/a |
| phal_local | 2/2 | [5, 6] | 2823, 2467 | 0.5462, 0.4377 | 8/8, 8/8 |
| phal_full_system | 2/2 | [4, 4] | 10037, 10037 | 0.5110, 0.4937 | 4/4, 4/4 |
| no_lj | 2/2 | [3, 4] | 1887, 2234 | 0.5248, 0.3885 | 3/8, 4/8 |


### Budget analysis

- artifact: `<artifacts_root>/model_study_v2c/evaluation.json`

### Figures

- ablations: ablations.pdf, ablations.svg, ablations.png
- fidelity: fidelity.pdf, fidelity.svg, fidelity.png
- fidelity_distance: fidelity_distance.pdf, fidelity_distance.svg, fidelity_distance.png
- fidelity_max: fidelity_max.pdf, fidelity_max.svg, fidelity_max.png
- learning_curves: learning_curves.pdf, learning_curves.svg, learning_curves.png
- learning_curves_labels: learning_curves_labels.pdf, learning_curves_labels.svg, learning_curves_labels.png
- timeline: timeline.pdf, timeline.svg, timeline.png

## Planned / not executed

- figures: {"rendered_without_data (panels say NOT EXECUTED)": {"hotspots": ["no ground-truth per-atom results", "no ground-truth per-atom results"], "receptive_field": ["no receptive-field result", "no receptive-field result"], "risk_vs_error": ["no region table for committee", "no region table for lj_residual", "no region table for phal_staged"]}}
