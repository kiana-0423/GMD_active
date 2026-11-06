# GMD_active documentation

GMD_active is the persistent active-learning controller connecting GMD (molecular dynamics),
GMD_SGT (machine-learned potentials) and CP2K (labels). The repository README has the
installation steps and the `gmd-al` quick start. Then:

- [Architecture](architecture.md): layers, protocols, the persistent state machine, adapters and
  ownership boundaries.
- Topic guides: [monitoring, events and hotspots](monitoring.md),
  [extraction, capping and datasets](extraction_and_datasets.md),
  [labeling, schedulers and backends](labeling_and_scheduling.md).
- [Configuration reference](config-reference.md) (generated from the models).
- [Operations tools](operations.md), [runtime data contract](runtime-data-contract.md),
  [performance benchmark](performance.md).
- [Containers and HPC](containers/README.md).
- [Known issues](known-issues.md).
- [Architecture decisions](adr/README.md).
- [Experimental CP2K ML-SCF tooling](scf/README.md).
- [API reference](api.md).
- [Contributing](contributing.md).
- [Migration from PHAL](migration/README.md): feature ledger, findings, evidence and the archived
  PHAL release baseline.

## Core concepts

| Concept | Type | Produced by |
|---|---|---|
| MD frame with stable atom IDs | `contracts.Frame` | GMD adapter, legacy importer |
| Per-atom scores, stage, action | `contracts.MonitorResult` | `monitors` |
| Trigger context | `contracts.EventRecord` | `events` |
| Region to label | `contracts.ExtractedRegion` | `extraction` |
| Labeling job and result | `contracts.LabelTask`, `contracts.LabelResult` | `labeling` |
| Training data version | `contracts.DatasetVersion` | `datasets` |
| Registered model | `contracts.ModelArtifact` | `retraining.registry` |
