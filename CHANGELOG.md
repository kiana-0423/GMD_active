# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]

---

## [v0.4.0] - 2026-11-06

GMD_active becomes the complete successor to PHAL (`hotspot_al`, commit 96263b6): GMD replaces
LAMMPS, GMD_SGT replaces Allegro/NequIP, CP2K remains the labeling engine. Every PHAL capability
is tracked in `docs/migration/feature_ledger.yaml`; defects are listed in
`docs/migration/findings.md` with the evidence of their fixes.

### Added

- **Persistent closed loop** (`workflow/`): SQLite-backed state machine (sampling, extraction,
  selection, submission, polling, ingestion, training decision, publication, training,
  evaluation, promotion) with write-ahead intents, crash reconciliation, bounded labeling queue,
  promotion gates and rollback; composition root `workflow/assembly.py`; `gmd-al` CLI
  (`validate-config`, `doctor`, `convert-phal-config`, `init`, `run`, `status`, `stop`, `retrain`,
  `rollback`, `resolve-unknown`).
- **Central configuration** (`config.py`): one strict pydantic model per section, generated
  `configs/default.yaml` and `docs/config-reference.md`, PHAL config converter.
- **Adapters**: GMD (segmented runs, flushed frames, cooperative stop/checkpoint, model change at
  restart), GMD_SGT (weighted training, fine-tuning, committee, export validation), CP2K (inputs
  from region PBC, cap-only GEO_OPT, validation, bounded retry ladders), local/Slurm/PBS
  schedulers, backend registry with plugins.
- **Domain ports from PHAL**: staged atom-wise OOD monitor, events, hotspots, cluster/slab/graph/
  block extraction with image shifts and H capping, embedding data, candidate selection,
  versioned datasets with GMD_SGT per-atom force weights, model registry.
- **Experimental CP2K ML-SCF tooling** (`adapters/dft/scf/`, `tools/scf/`), torch-optional.
- **Legacy import**: `gmd_active_learning.importers` converts LAMMPS custom dumps into `Frame`s.
- **Operations**: artifact audit (`tools/audit/`, now also registry checksums, dataset tree
  hashes and task integrity), GMD_SGT runtime smoke and tiny H2O chain (`tools/smoke/`), monitor
  benchmark with regression gate (`tools/benchmarks/`), runtime tree and config overlays
  (`tools/runtime/`, `containers/config/`), offline test runner (`tools/test/`), CI on Python
  3.10-3.12 and pre-commit hooks.
- **Containers and HPC** (not built or run yet): `containers/Dockerfile.{gmd,sgt-train,cp2k,cp2k-ml}`,
  `build.sh` with `--dry-run` and floating-ref rejection, exact pins in `versions.yaml`,
  validators, PBS validation jobs that hash the SIF at job time (`tools/hpc/pbs/`).
- **Examples** `examples/01`-`06` (PHAL's examples on GMD frames, CP2K labels and GMD_SGT datasets)
  and `examples/real_cycle/` (real GMD + GMD_SGT + CP2K cycle with recorded evidence).
- **Documentation**: architecture, topic guides, ADRs 004-007, known issues, operations, runtime
  data contract, container docs, contributing guide, mkdocs configuration, archived PHAL
  milestone-3 release baseline (`docs/migration/phal_release_baseline/`).
- **Architecture tests**: executable layering guard and a recursive import smoke test.
- `utils.yaml.merge_dicts`/`merge_yaml_files`/`dump_yaml`; `utils.logging.configure_logging` with
  text/JSON output, file handler and `GMD_AL_LOG_{LEVEL,FILE,FORMAT}` overrides; PHAL exception
  classes (`BackendError`, `InferenceError`, `MDRuntimeError`, `DFTRuntimeError`,
  `CP2KRuntimeError`, `DataError`, `CooldownSkip`).

### Fixed (PHAL and former GMD_active defects)

- LJ force-basis sign (F-01) and unscaled ridge penalty (F-33); minimum image for skewed,
  partially periodic cells (F-34); WAVELET on non-cubic cells (F-35).
- Missing committee no longer scored as perfect agreement (F-06); truncated events flagged
  (F-07); deterministic event, region, task, sample, dataset and model IDs (F-15, F-24).
- Capped block regions write datasets (F-04); labeling uses the configured extraction mode
  (F-05); image shifts recorded and torn regions refused (F-09); H-only cap relaxation runs
  (F-17); NaN fingerprints (F-30); retry compounding and missing native fallback (F-25).
- Placeholder workflow, adapters, merger and CLI of GMD_active 0.2 replaced (F-10).

### Changed

- `configure_logging` now attaches handlers to the `gmd_active_learning` logger instead of calling
  `logging.basicConfig`; existing calls keep working.
- README rewritten in English for the current system.

### Removed

- PHAL's empty Milestone-1 container scaffolds and config templates are not carried over
  (ledger CTR-002, CFG-005).

---


## [v0.2] - 2026-04-21

### 项目简介

`GMD_active_learning` 是连接 `GMD`（分子动力学核心）与 `GMD_se3gnn`（机器学习势训练）的工程化主动学习控制层。项目不侵入 MD 核心或训练核心，通过 adapter 模式实现模块解耦，使主动学习策略可以独立演化与替换。

**核心特色：**

- **在线可靠性监控**：集成 LJ 投影监控器、集成偏差监控、物理性检查等多种 OOD 预警手段
- **LJ 投影预警**：将 MLIP 原子力投影到元素对 Lennard-Jones 参数空间，通过拟合残差、参数物理性与历史跳变量触发主动学习，仅作为 OOD 检测触发器，不替代 MLIP
- **风险聚合**：多监控器输出加权聚合为统一风险分，通过 YAML 配置权重与阈值
- **候选管理**：候选构型队列化管理，支持 pair distance histogram 去重（可扩展至 SOAP/ACSF）
- **DFT 标注**：生成 VASP / CP2K 输入文件与作业脚本
- **重训练编排**：通过 adapter 调用 `GMD_se3gnn` 完成训练、导出与模型注册
- **全配置驱动**：所有阈值、目录和模块开关均通过 YAML 配置文件控制，支持 dry-run 模式

**安装：**

```bash
pip install -e ".[dev]"
```

**快速开始：**

```bash
gmd-al monitor-example
python examples/minimal_lj_monitor_example.py
python examples/minimal_active_learning_loop.py
```

**配置文件：**

| 文件 | 用途 |
|------|------|
| `configs/active_learning.yaml` | 总流程、目录、dry-run 开关 |
| `configs/monitor.yaml` | 监控器开关、阈值、LJ 参数、风险权重 |
| `configs/dft_labeling.yaml` | DFT 标注任务模板与作业脚本 |
| `configs/retraining.yaml` | `GMD_se3gnn` 调用方式与导出命令 |

**运行测试：**

```bash
pytest
```

---
