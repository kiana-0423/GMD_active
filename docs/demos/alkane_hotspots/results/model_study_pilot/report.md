# Alkane hotspot model study: pilot

Config `<GMD_active>/configs/demos/alkane_hotspots/arms_pilot.yaml` (hash `3f2fc68d1e7c45da`), generated 2026-11-06T16:03:42+00:00.

## Label records

- file: `<artifacts_root>/labels.jsonl` (0 records)
  - measured full cost: 0 CPU core-s over 0 calculations
  - measured fragment cost: 0 CPU core-s over 0 calculations

## Executed

### Receptive field

- artifact: `<artifacts_root>/model_study/receptive_field.json`
- model source: random_init
- nominal: {"energy_receptive_field_A": 8.0, "force_receptive_field_A": 16.0, "force_receptive_field_hops": 4, "force_receptive_field_local_A": 16.0, "global": false, "local_cutoff": 4.0, "long_range_type": "none", "n_blocks": 2, "note": "force depends on atoms within 2 x n_blocks x local_cutoff (per-atom energy: n_blocks x local_cutoff); a long-range term is global"}
- measured: {"delete_influence_radius_A": 2.751629342241791, "delete_relative_influence_radius_A": 2.751629342241791, "delete_structural_radius_A": 8.94929770129875, "perturb_influence_radius_A": 2.751629342241791, "perturb_relative_influence_radius_A": 2.751629342241791, "perturb_structural_radius_A": 7.641195081606981, "truncation_converged_radius_A": 3.0}
- local model: structural influence radius 8.95 A (any dependence above round-off) vs nominal force receptive field 16.00 A (2 x n_blocks x local_cutoff)

### Figures

- receptive_field: receptive_field.pdf, receptive_field.svg, receptive_field.png

## Planned / not executed

- ground_truth: {"reason": "no ground_truth.json"}
- arms: {"random_local": {"reason": "not run", "repetitions_planned": 2, "rounds_planned": 2}, "committee_local": {"reason": "not run", "repetitions_planned": 2, "rounds_planned": 2}, "phal_local": {"reason": "not run", "repetitions_planned": 2, "rounds_planned": 2}, "phal_full_system": {"reason": "not run", "repetitions_planned": 2, "rounds_planned": 2}, "no_lj": {"reason": "not run", "repetitions_planned": 2, "rounds_planned": 2}, "no_event_buffering": {"reason": "not run", "repetitions_planned": 2, "rounds_planned": 2}, "committee_signals_only": {"reason": "not run", "repetitions_planned": 2, "rounds_planned": 2}, "core_only_weights": {"reason": "not run", "repetitions_planned": 2, "rounds_planned": 2}}
- evaluation: {"reason": "no evaluation.json (run `evaluate` after `arms-run`)"}
- figures: {"rendered_without_data (panels say NOT EXECUTED)": {"ablations": ["no executed ablation runs"], "fidelity": ["no fidelity results with mean_n_atoms and pooled_core_force_rmse"], "fidelity_max": ["no fidelity results with mean_n_atoms and max_core_force_error"], "hotspots": ["no ground-truth per-atom results", "no ground-truth per-atom results"], "learning_curves": ["no executed arm runs"], "learning_curves_labels": ["no executed arm runs"], "risk_vs_error": ["no region table for committee", "no region table for lj_residual", "no region table for phal_staged"], "timeline": ["no closed-loop controller report"]}}
