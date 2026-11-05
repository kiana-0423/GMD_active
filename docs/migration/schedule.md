# Revised implementation schedule

The Phase 0 proposal estimated 11–17 engineer-weeks for one developer and treated GMD and GMD_SGT
as stable providers. Phase 0 and the implementation since then showed that estimate was not a
sound commitment. It did not include:

- **Upstream repairs.** GMD needed checkpoint v3 with atomic numbers (F-02), flushed ASE-readable
  frames, cooperative stop/checkpoint control, atomic checkpoints and restart model validation. It also
  had a silent ML edge-shift bug (F-36). GMD_SGT needed per-atom force weights (F-03), mixed periodic
  batches, per-structure energy/stress weights, event-aware splits and a fine-tune mode distinct from
  resume.
- **A persistent state machine.** PHAL never closed its own loop (F-24), so submission,
  reconciliation, ingestion, training, export validation, gated promotion, rollback and recovery are
  new functionality, not a port.
- **Correctness defects in both codebases** (F-01, F-33, F-34, F-04, F-35, ...), each needing
  independent tests, not parity tests.

## Re-estimate (one experienced developer, excluding cluster queue time)

| Work package | Estimate | Status |
|---|---:|---|
| Phase 0 ledger, inventory, golden fixtures | 1 wk | done |
| GMD upstream (checkpoint v3, frames, control, edge-shift fix, tests) | 2–3 wk | done (local branch `al-integration`) |
| GMD_SGT upstream (weights, mixed batches, splits, fine-tune, metadata) | 2 wk | done (local branch `al-weighted-training`) |
| Contracts, utilities, config, registry | 1–2 wk | done |
| Monitoring, events, hotspots (parity + corrections) | 2–3 wk | done |
| Extraction, capping, embedding prep, datasets | 3 wk | done |
| CP2K labeling, schedulers, retry ladder | 2–3 wk | done (Slurm/PBS only against fake commands) |
| Persistent controller, registry, gates, rollback, recovery | 3–4 wk | done |
| GMD / GMD_SGT adapters and real integration | 2 wk | done |
| SCF / ML-SCF tooling (experimental) | 3 wk | done except experiments that need the lost fork or data |
| Operations, containers, docs, examples, CLI | 2 wk | see the ledger |
| **Total** | **23–30 wk** | |

The remaining risk is validation on production hardware, not code:

- real Slurm/PBS runs (Miyabi PBS jobs are templates here);
- container builds (Docker unavailable here);
- MPI GMD runs (ML in GMD is serial-only);
- scientific acceptance on representative reactive systems. The real cycle in this repository is a
  smoke-scale water demonstration, not a scientific validation.

Budget 2–4 weeks for those once cluster access is available.
