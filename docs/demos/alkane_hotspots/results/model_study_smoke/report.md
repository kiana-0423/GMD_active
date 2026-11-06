# Alkane hotspot model study: smoke

Config `<GMD_active>/configs/demos/alkane_hotspots/arms_smoke.yaml` (hash `8486bdbbcfd6b983`), generated 2026-11-06T16:03:43+00:00.

## Label records

- file: `<artifacts_root>/labels.jsonl` (3 records)
  - fragment/valid: 1
  - full/valid: 2
  - measured full cost: 133 CPU core-s over 2 calculations
  - measured fragment cost: 246 CPU core-s over 1 calculations
  - **3 smoke-only records**: smoke-only labels (reduced CP2K settings) are pipeline checks; their numbers are not results

## Executed

### Figures

- fidelity: fidelity.pdf, fidelity.svg, fidelity.png
- fidelity_max: fidelity_max.pdf, fidelity_max.svg, fidelity_max.png

## Planned / not executed

- receptive_field: {"reason": "no receptive_field.json"}
- ground_truth: {"reason": "no ground_truth.json"}
- arms: {"random_local": {"reason": "not run", "repetitions_planned": 1, "rounds_planned": 1}, "committee_local": {"reason": "not run", "repetitions_planned": 1, "rounds_planned": 1}, "phal_local": {"reason": "not run", "repetitions_planned": 1, "rounds_planned": 1}, "phal_full_system": {"reason": "not run", "repetitions_planned": 1, "rounds_planned": 1}, "no_lj": {"reason": "not run", "repetitions_planned": 1, "rounds_planned": 1}, "no_event_buffering": {"reason": "not run", "repetitions_planned": 1, "rounds_planned": 1}, "committee_signals_only": {"reason": "not run", "repetitions_planned": 1, "rounds_planned": 1}, "core_only_weights": {"reason": "not run", "repetitions_planned": 1, "rounds_planned": 1}}
- evaluation: {"reason": "no evaluation.json (run `evaluate` after `arms-run`)"}
- figures: {"rendered_without_data (panels say NOT EXECUTED)": {"ablations": ["no executed ablation runs"], "hotspots": ["no ground-truth per-atom results", "no ground-truth per-atom results"], "learning_curves": ["no executed arm runs"], "learning_curves_labels": ["no executed arm runs"], "receptive_field": ["no receptive-field result", "no receptive-field result"], "risk_vs_error": ["no region table for committee", "no region table for lj_residual", "no region table for phal_staged"], "timeline": ["no closed-loop controller report"]}}
