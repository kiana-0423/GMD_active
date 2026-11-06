# Contributing

Ported from PHAL `CONTRIBUTING.md` (commit 96263b6) and rewritten for GMD_active. `AGENTS.md` lists
the project rules; this page explains how to follow them.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # or the gmd-active conda environment
python -m pip install -U pip
python -m pip install -e ".[dev]"
pip install pre-commit && pre-commit install           # optional; same gates as CI
```

GMD, GMD_SGT and CP2K are not Python dependencies. Tests that need them skip with a reason; see
`tests/README.md` for how to run them.

## Before a pull request

```bash
bash tools/test/run_all_offline.sh
PYTHONPATH=src python tools/docs/render_config_reference.py --check   # after config model changes
python tools/migration/apply_status.py --check                        # after ledger status changes
```

## Rules

1. **Layering.** Domain packages (`monitors`, `events`, `hotspots`, `extraction`,
   `active_learning`, `datasets`, `retraining`) and `contracts` never import adapters, the
   labeling service, the composition root or a backend (`torch`, `gmd_sgt`, `subprocess`, ...).
   `tests/architecture/test_layering.py` enforces it; a new package must be classified there.
2. **No active-learning logic in GMD or GMD_SGT.** If GMD_active needs a capability from them,
   add a minimal, general extension upstream (see `docs/migration/integration_gaps.md`) and call
   it through the adapter.
3. **External programs only through adapters**, chosen by the `backends` section. No fake or
   dry-run fallback in production code: a missing program raises `CapabilityUnavailableError`.
   Fakes live in `tests/fakes/`.
4. **Configuration.** Every threshold, tolerance, path and resource limit comes from a pydantic
   section model with `extra="forbid"` and a default; every feature has an `enabled` switch with a
   test of the disabled behaviour. Register new sections in `config.SECTIONS`, add
   `configs/sections/<name>.yaml`, then regenerate `configs/default.yaml` and
   `docs/config-reference.md`.
5. **Contracts.** Exchange `contracts` types; keep stable atom IDs; distinguish "not evaluated" or
   "unavailable" from zero; use Å, eV, eV/Å, eV/Å³ and fs.
6. **Identity and persistence.** Derive IDs with `contracts.stable_id`/`content_hash`, never from
   random numbers or clocks; write files with `utils.atomic_io`; record workflow state in the
   store before acting.
7. **Code.** Type hints, `get_logger(__name__)`, exceptions from `core.exceptions`, NumPy for
   geometry (`utils.periodic` for minimum image), no unnecessary large dependencies, no hard-coded
   absolute paths.
8. **Tests.** Every new behaviour and every fixed defect gets a test. Golden PHAL fixtures under
   `tests/fixtures/phal_reference/` are immutable; where PHAL was wrong, assert the corrected
   behaviour and document the deviation.
9. **Ported code** starts with `# Ported from PHAL (<module>) at 96263b6; MIT License, see NOTICE.`

## Adding an extraction mode

Implement `extract_<mode>(frame, hotspot_atom_ids, config, *, event_id=None) -> ExtractedRegion`
in `extraction/`, using `extraction/_placement.py` for image shifts, the periodic guard and region
assembly, so parent IDs, image shifts, roles and force weights are always complete. Register it
in `extraction/dispatch.py`, add its settings to `ExtractionConfig`, and test roles, shifts, the
periodic guard and capping.

## Adding a monitor metric

Compute per-atom values in `monitors/`, return `MetricValues` with an honest `MetricStatus`
(`MEASURED` with an `evaluated` mask, `NOT_EVALUATED` with a reason, or `UNAVAILABLE`), add
`metrics.<name>.{enabled, weight, ...}` to `MonitorConfig`, assign it a stage (light, physics,
full) and add its trigger reason. Keep expensive metrics lazy or staged, and extend the
benchmark in `tools/benchmarks/benchmark_monitor.py` if it has a fast path.

## Adding a backend

Implement the role's protocol (`workflow/protocols.py` or the scheduler/DFT adapter protocol),
declare its capabilities, and register it with `adapters.registry.register_builtin` (or a plugin
entry point). Unsupported combinations raise `UnsupportedCombinationError`; never change physics
(e.g. boundary conditions) to fit a backend.

## Documentation

Topic pages live in `docs/`; architecture decisions in `docs/adr/`. `docs/config-reference.md`
is generated. Build the site with `mkdocs build --strict -f tools/docs/mkdocs.yml` (needs the
`mkdocs` and `mkdocstrings[python]` packages).
