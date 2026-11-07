"""Repair runs: arms x cases x seeds, all starting from the frozen, qualified initial model.

A *case* is one repair-partition challenge structure (one base, one family). A *seed* is paired across
arms: it sets the GMD velocity seed and the fine-tune seed, so arms C and D start from identical
trajectories until their monitors first act.

* ``continued_training`` (arm B): ``repair.cycles`` rounds of fine-tuning on the original (id_train)
  data only, with the fine-tune settings, seeds and promotion gates of the closed-loop arms; no new
  labels. Controls for improvement from additional optimisation alone.
* ``closed_loop`` (arms C, D and ablations): GMD_active's persistent controller, unchanged:
  GMD trajectory -> staged online monitor -> events -> hotspot grouping -> gate-validated local
  extraction and capping -> CP2K -> dataset update -> GMD_SGT fine-tuning -> export check, GMD smoke
  test and promotion gates -> GMD restart with the accepted model. Arms differ only in the monitor
  configuration (``arms[].monitor_overrides``) and, for an ablation, the event buffering.

What the benchmark adds around the controller:

* the frozen initial model is registered as the run's initial model (never retrained per run);
* the initial dataset is the frozen model's training set; the held-out validation set is its id_val
  structures plus the repair-validation challenge structures;
* promotion gates on per-subset metrics (``promotion.metric_gates``): the candidate must improve the
  repair-validation force RMSE of the run's challenge family by a minimum fraction and may not regress
  on id_val beyond a tolerance;
* the monitor's running statistics are calibrated on in-domain trajectories of the frozen model
  (``monitor_calibration``) and seeded into the controller before the first segment, after checking
  warm-up of the metrics an arm depends on and the in-domain alarm fraction; a run whose calibration
  fails is refused before any CP2K job.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gmd_active_learning.contracts.ids import content_hash, file_sha256
from gmd_active_learning.core.exceptions import ConfigurationError, WorkflowError

from tools.experiments.alkane_hotspots.families import load_positions
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology
from tools.experiments.alkane_hotspots.manifest import utc_now

from .common import model_artifact, read_json, run_md, write_json
from .config import LoadedBenchmark
from .generate import load_records
from .qualify import require_qualified, training_config
from .references import audit_status, full_label_for, require_frozen

LOGGER = logging.getLogger(__name__)

RUNS_SUBDIR = "repair_runs"


# ---------------------------------------------------------------------------------------------- run matrix


@dataclass(frozen=True)
class RunSpec:
    arm: str
    kind: str
    case: str
    family: str
    base_id: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.arm}__{self.case}__s{self.seed}"


def repair_cases(loaded: LoadedBenchmark) -> list[dict[str, Any]]:
    return sorted((r for r in load_records(loaded.root) if r["partition"] == "repair" and r["kind"] == "challenge"), key=lambda r: r["parent_id"])


def run_matrix(loaded: LoadedBenchmark, *, arms: list[str] | None = None, seeds: list[int] | None = None,
               cases: list[str] | None = None) -> list[RunSpec]:
    config = loaded.config
    specs = []
    for arm in config.arms:
        if not arm.enabled or arm.kind == "frozen" or (arms and arm.name not in arms):
            continue
        for case in repair_cases(loaded):
            if cases and case["parent_id"] not in cases:
                continue
            for seed in config.repair.seeds:
                if seeds and seed not in seeds:
                    continue
                specs.append(RunSpec(arm.name, arm.kind, case["parent_id"], case["family"], case["base_id"], int(seed)))
    return specs


def run_dir(loaded: LoadedBenchmark, spec: RunSpec) -> Path:
    return loaded.root / RUNS_SUBDIR / spec.name


# ---------------------------------------------------------------------------------------------- frozen model


def frozen_trained_model(frozen: dict[str, Any]) -> Any:
    from tools.experiments.alkane_hotspots.training import TrainedModel

    return TrainedModel([item["path"] for item in frozen["checkpoints"]], list(frozen["species"]), frozen.get("cutoff"), {})


def frozen_predictor(loaded: LoadedBenchmark, frozen: dict[str, Any]) -> Callable[[np.ndarray, np.ndarray], Any]:
    from tools.experiments.alkane_hotspots.training import make_backend, predictor_for

    return predictor_for(make_backend(training_config(loaded), root=loaded.repo_path(".")), frozen_trained_model(frozen))


def parents_manifest(loaded: LoadedBenchmark, frozen: dict[str, Any]) -> Path:
    """Initial dataset = the frozen model's training set; validation = its id_val + all repair_val structures."""

    repair_val = sorted(r["parent_id"] for r in load_records(loaded.root) if r["partition"] == "repair_val")
    payload = {"train": [item["parent_id"] for item in frozen["samples"]["train"]],
               "validation": [item["parent_id"] for item in frozen["samples"]["validation"]] + repair_val}
    path = loaded.root / RUNS_SUBDIR / "initial_parents.json"
    write_json(path, payload)
    return path


def promotion_section(loaded: LoadedBenchmark, family: str) -> dict[str, Any]:
    p = loaded.config.repair.promotion
    return {"max_force_rmse": p.max_force_rmse, "max_relative_regression": p.overall_max_relative_regression, "require_validation_samples": 1,
            "require_export_validation": p.require_export_validation, "export_max_force_deviation": p.export_max_force_deviation,
            "require_engine_smoke_test": p.require_engine_smoke_test,
            "metric_gates": [{"metric": f"force_rmse.repair_val.{family}", "min_relative_improvement": p.family_min_relative_improvement},
                             {"metric": "force_rmse.id_val", "max_relative_regression": p.id_val_max_relative_regression}]}


# ---------------------------------------------------------------------------------------------- closed-loop configuration


def closed_loop_raw(loaded: LoadedBenchmark, spec: RunSpec, manifest: Path) -> dict[str, Any]:
    config = loaded.config
    arm = config.arm(spec.arm)
    r = config.repair
    return {
        "name": spec.name,
        "description": f"repair benchmark {config.name} (protocol {config.protocol_version}): arm {arm.name}, case {spec.case}, seed {spec.seed}",
        "base_config": str(loaded.resolve(config.alpha_config)),
        "study": str(loaded.resolve(config.model_config_file)),
        "output_subdir": RUNS_SUBDIR,
        "exclusive_cp2k": not config.smoke_only,
        "md": {"executable": r.executable, "parent": {"parent_id": spec.case}, "allowed_splits": ["calibration"],
               "directives": {**r.md_directives, "velocity_seed": spec.seed, "velocity_init": "random"}, "frame_interval": r.frame_interval,
               "env": dict(r.env)},
        "md_box": dict(r.md_box),
        "loop": {"run_id": spec.name, "cycles": r.cycles, "segment_steps": r.segment_steps, "segments_per_cycle": 1,
                 "labels_per_cycle": r.labels_per_cycle, "max_per_event": r.max_per_event, "diversity_threshold": r.diversity_threshold,
                 "min_score": None, "min_new_samples": r.min_new_samples, "max_pending_tasks": 4, "poll_interval_s": r.poll_interval_s,
                 "max_poll_wait_s": r.max_poll_wait_s},
        "budget": {"max_cpu_core_hours": r.budget_core_h_per_run},
        "labeling": {"mode": "local", "fidelity_dir": "fidelity"},
        "label_geometry": dict(r.label_geometry),
        "initial_model": {"train_splits": ["initial_train"], "validation_splits": ["validation"], "parents_manifest": str(manifest),
                          "seed": config.initial_model.seed},
        "finetune": {"train": dict(r.finetune_train), "seed": r.finetune_seed_offset + spec.seed},
        "promotion": promotion_section(loaded, spec.family),
        "monitor_overrides": dict(arm.monitor_overrides),
        "events": dict(arm.events if arm.events is not None else r.events),
        "hotspots": dict(r.hotspots),
    }


def write_run_config(loaded: LoadedBenchmark, spec: RunSpec, manifest: Path) -> Path:
    path = loaded.root / RUNS_SUBDIR / "configs" / f"{spec.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(closed_loop_raw(loaded, spec, manifest), sort_keys=True)
    if path.is_file() and path.read_text(encoding="utf-8") != text:
        raise WorkflowError(f"{path} exists with a different configuration; a run's configuration never changes")
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------- subset metrics


def subset_version(samples: list[Any], directory: Path, name: str) -> Any:
    """A validation-only dataset version (``val_file`` = all of ``samples``) for per-subset metrics."""

    from gmd_active_learning.datasets.config import DatasetsConfig
    from gmd_active_learning.datasets.store import VersionedDatasetStore

    if not samples:
        raise ConfigurationError(f"validation subset {name} is empty")
    store = VersionedDatasetStore(directory / name, DatasetsConfig.model_validate({"split": {"enabled": False}}))
    version = store.publish_samples(list(samples))
    return replace(version, metadata={**version.metadata, "val_file": version.metadata["all_file"], "subset": name})


def subset_samples(loaded: LoadedBenchmark, val_samples: list[Any], val_record: list[dict[str, Any]]) -> dict[str, list[Any]]:
    by_id = {r["parent_id"]: r for r in load_records(loaded.root)}
    groups: dict[str, list[Any]] = {"id_val": []}
    for sample, item in zip(val_samples, val_record):
        record = by_id[item["parent_id"]]
        if record["partition"] == "id_val":
            groups["id_val"].append(sample)
        elif record["partition"] == "repair_val":
            groups.setdefault(f"repair_val.{record['family']}", []).append(sample)
        else:
            raise WorkflowError(f"validation sample of {record['parent_id']} ({record['partition']}) is neither id_val nor repair_val")
    return groups


class SubsetTrainer:
    """``Trainer`` around the closed loop's ``RecordingTrainer``: ``evaluate`` adds ``force_rmse.<subset>``."""

    def __init__(self, recording: Any, subsets: dict[str, Any]) -> None:
        self.recording = recording
        self.inner = recording.inner  # the GMD_SGT backend (build_report re-evaluates energies through it)
        self.subsets = dict(subsets)

    def train(self, dataset: Any, *, parent: Any, work_dir: Path) -> Any:
        return self.recording.train(dataset, parent=parent, work_dir=work_dir)

    def evaluate(self, checkpoint: Path, dataset: Any) -> dict[str, float]:
        metrics = dict(self.recording.evaluate(checkpoint, dataset))
        for name, version in self.subsets.items():
            metrics[f"force_rmse.{name}"] = float(self.inner.evaluate(checkpoint, version)["force_rmse"])
        self.recording._record("subset_metrics", checkpoint=str(checkpoint), metrics={k: v for k, v in metrics.items() if k.startswith("force_rmse.")})
        return metrics

    def export(self, checkpoint: Path, work_dir: Path) -> Path:
        return self.recording.export(checkpoint, work_dir)

    def validate_export(self, checkpoint: Path, export: Path, dataset: Any) -> dict[str, Any]:
        return self.recording.validate_export(checkpoint, export, dataset)

    def engine_smoke_test(self, export: Path, work_dir: Path) -> dict[str, Any]:
        return self.recording.engine_smoke_test(export, work_dir)

    def history(self) -> list[dict[str, Any]]:
        return self.recording.history()


# ---------------------------------------------------------------------------------------------- monitor calibration


def calibration_trajectories(loaded: LoadedBenchmark, frozen: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    """In-domain GMD trajectories of the frozen model from the calibration partition (resumable, shared by arms)."""

    config = loaded.config
    cal = config.monitor_calibration
    primary, export = Path(frozen["checkpoints"][0]["path"]), Path(frozen["export"]["path"])
    artifact = model_artifact(export, primary, frozen["species"], "frozen-initial")
    starts = sorted((r for r in load_records(loaded.root) if r["partition"] in cal.partitions and r["kind"] == "ordinary"), key=lambda r: r["parent_id"])
    if not starts:
        raise ConfigurationError("monitor calibration needs ordinary structures in the calibration partition")
    out = []
    for start in starts:
        topology = AlkaneTopology(int(start["chain_length"]))
        for seed in cal.seeds:
            frames: list[Any] = []
            name = f"{start['parent_id']}-s{seed}"
            result = run_md(artifact=artifact, positions=load_positions(loaded.root, start), topology=topology, directives=config.repair.md_directives,
                            seed=seed, steps=cal.steps, frame_interval=cal.frame_interval, out_dir=loaded.root / "monitor_calibration" / "trajectories" / name,
                            run_id=f"calibration-{name}", executable=config.repair.executable, env=config.repair.env, md_box=config.repair.md_box,
                            model_section=frozen["model_section"], covalent_scale=config.initial_model.acceptance.md.covalent_scale, on_frame=frames.append)
            if result["status"] != "completed" or result["n_broken"]:
                raise WorkflowError(f"calibration trajectory {name} is not an intact in-domain trajectory ({result['status']}, "
                                    f"{result['n_broken']} broken frames); the monitor cannot be calibrated on it")
            out.append((name, frames))
    return out


def calibrate_monitor(loaded: LoadedBenchmark, frozen: dict[str, Any], arm_name: str, sections: dict[str, Any]) -> dict[str, Any]:
    """Calibrated driver state for one arm's monitor configuration (cached; identical for every run of the arm)."""

    from gmd_active_learning.monitors.config import load_monitor_config
    from gmd_active_learning.monitors.online import OnlineMonitorDriver, build_frame_monitor

    from tools.experiments.alkane_hotspots.training import make_backend

    arm = loaded.config.arm(arm_name)
    cal = loaded.config.monitor_calibration
    key = content_hash({"export": frozen["export"]["sha256"], "monitor": sections["monitor"], "events": sections["events"],
                        "hotspots": sections["hotspots"], "calibration": cal.model_dump(mode="json"), "required": arm.required_warm_metrics}, length=16)
    path = loaded.root / "monitor_calibration" / f"{arm_name}.json"
    if path.is_file():
        cached = read_json(path)
        if cached.get("key") == key:
            return cached
    monitor_cfg = load_monitor_config(sections["monitor"])
    backend = make_backend(training_config(loaded), root=loaded.repo_path("."))
    members = [item["path"] for item in frozen["checkpoints"]]
    monitor = build_frame_monitor(monitor_cfg)
    driver = OnlineMonitorDriver(monitor_cfg, sections["events"], sections["hotspots"], monitor=monitor)
    n_frames = n_triggered = 0
    per_trajectory = []
    for name, frames in calibration_trajectories(loaded, frozen):
        if hasattr(monitor, "reset_temporal_state"):
            monitor.reset_temporal_state()
        triggered = 0
        for frame in frames:
            if len(members) >= 2:
                cell = frame.cell if any(frame.pbc) else None
                frame.committee_forces = backend.predict_committee(members, frame.positions, frame.numbers, cell, frame.pbc, species=frozen["species"])
            result, _ = driver.process(frame)
            triggered += int(result.triggered)
        driver.finish("calibration_end")
        n_frames += len(frames)
        n_triggered += triggered
        per_trajectory.append({"trajectory": name, "n_frames": len(frames), "n_triggered": triggered})
    monitor_state = monitor.to_state()
    if "previous" in monitor_state:
        monitor_state["previous"] = None  # no temporal state crosses into the repair trajectory
    fresh = OnlineMonitorDriver(monitor_cfg, sections["events"], sections["hotspots"], monitor=build_frame_monitor(monitor_cfg))
    state = fresh.to_state()
    state["monitor"] = monitor_state
    warmup = monitor_cfg.running_stats.warmup_frames
    counts = {name: int(stats.get("frame_count", 0)) for name, stats in (monitor_state.get("stats") or {}).items()}
    cold = [m for m in arm.required_warm_metrics if counts.get(m, 0) < warmup]
    fraction = n_triggered / n_frames if n_frames else 1.0
    record = {"key": key, "arm": arm_name, "created_at": utc_now(), "state": state, "n_frames": n_frames, "n_triggered": n_triggered,
              "alarm_fraction": fraction, "max_alarm_fraction": cal.max_alarm_fraction, "warmup_frames": warmup, "stats_frame_counts": counts,
              "required_warm_metrics": arm.required_warm_metrics, "cold_required_metrics": cold, "driver_stats": dict(driver.stats),
              "per_trajectory": per_trajectory, "ok": not cold and fraction <= cal.max_alarm_fraction,
              "interpretation": "alarm_fraction = triggered frames / in-domain calibration frames (frozen initial model): an estimate of the "
                                "monitor's in-domain false-alarm rate per frame"}
    write_json(path, record)
    return record


def monitor_sections(loaded: LoadedBenchmark, spec: RunSpec, frozen: dict[str, Any]) -> dict[str, Any]:
    """The merged monitor/events/hotspots sections a run of ``spec``'s arm uses (no controller is built)."""

    from tools.experiments.alkane_hotspots.closed_loop import gmd_active_raw, load_closed_loop

    cl = load_closed_loop(write_run_config(loaded, spec, parents_manifest(loaded, frozen)))
    raw = gmd_active_raw(cl, kind="main", initial_structure=Path("unused.xyz"), task_dir=Path("unused"))
    return {"monitor": raw["monitor"], "events": raw["events"], "hotspots": raw["hotspots"]}


def calibrate_arm(loaded: LoadedBenchmark, arm_name: str) -> dict[str, Any]:
    """Calibrate (or verify) one closed-loop arm's monitor before any expensive work."""

    require_frozen(loaded)
    frozen = require_qualified(loaded)
    specs = [s for s in run_matrix(loaded, arms=[arm_name]) if s.kind == "closed_loop"]
    if not specs:
        raise ConfigurationError(f"arm {arm_name} has no closed-loop runs")
    return calibrate_monitor(loaded, frozen, arm_name, monitor_sections(loaded, specs[0], frozen))


def require_calibrated(record: dict[str, Any]) -> None:
    if not record["ok"]:
        raise WorkflowError(f"monitor calibration of arm {record['arm']} failed: cold required metrics {record['cold_required_metrics']} "
                            f"(warmup {record['warmup_frames']} frames), in-domain alarm fraction {record['alarm_fraction']:.3f} "
                            f"(max {record['max_alarm_fraction']}); the run is refused before any CP2K job")


# ---------------------------------------------------------------------------------------------- guards


def require_audit(loaded: LoadedBenchmark, family: str) -> None:
    if not loaded.config.fidelity.require_audit:
        return
    audit = audit_status(loaded)
    if audit is None:
        raise WorkflowError("no fragment-fidelity audit (fidelity/audit.json): run `fidelity-audit` on the development structures first")
    verdict = (audit.get("families") or {}).get(family)
    if not verdict or not verdict.get("passes"):
        raise WorkflowError(f"the frozen fragment setting failed (or has not completed) the fidelity audit for {family}: {verdict}; "
                            "local labels of this family are not validated")


def repair_spent_core_h(loaded: LoadedBenchmark) -> float:
    total = 0.0
    for report in sorted((loaded.root / RUNS_SUBDIR).glob("*/repair_run.json")):
        total += float(read_json(report).get("measured_dft_cpu_core_hours") or 0.0)
    return total


def require_budget(loaded: LoadedBenchmark, spec: RunSpec) -> dict[str, Any]:
    if (run_dir(loaded, spec) / "repair_run.json").is_file() or spec.kind != "closed_loop":
        return {"checked": False}
    spent = repair_spent_core_h(loaded)
    cap = loaded.config.repair.budget_core_h_per_run
    limit = loaded.config.budgets.repair_runs
    if spent + cap > limit:
        raise WorkflowError(f"repair-run budget: measured {spent:.3f} core-h + this run's cap {cap:.3f} > budgets.repair_runs {limit:.3f}; stop")
    return {"checked": True, "spent_core_h": spent, "run_cap_core_h": cap, "limit_core_h": limit}


# ---------------------------------------------------------------------------------------------- closed-loop arms


def register_frozen_initial(cl: Any, asm: Any, frozen: dict[str, Any]) -> Any:
    """Register the frozen initial model as the run's initial (active) model; idempotent."""

    from tools.experiments.alkane_hotspots.closed_loop import initial_samples

    controller = asm.controller
    run = controller.store.ensure_run(asm.run_id, config_hash=controller.config_hash, initial_model_id=None, stage="sampling")
    if run.active_model_id is not None:
        return controller.registry.get(run.active_model_id)
    train, _val, record = initial_samples(cl, asm.service_settings_hash)
    expected = [item["parent_id"] for item in frozen["samples"]["train"]]
    if [item["parent_id"] for item in record["train"]] != expected:
        raise WorkflowError("the run's initial dataset is not the frozen initial model's training set")
    datasets = controller.datasets
    for sample in train:
        datasets.inner.write_sample(sample)
    dataset = datasets.publish_samples(train, parent=None)
    primary, export = Path(frozen["checkpoints"][0]["path"]), Path(frozen["export"]["path"])
    trainer = controller.trainer
    export_check = trainer.validate_export(primary, export, dataset)
    smoke = trainer.engine_smoke_test(export, cl.out_dir / "initial_model" / "engine_smoke")
    metrics = trainer.evaluate(primary, dataset)
    limit = float(controller.config.promotion.export_max_force_deviation)
    if export_check.get("max_force_deviation") is None or float(export_check["max_force_deviation"]) > limit:
        raise WorkflowError(f"frozen export does not reproduce the frozen checkpoint: {export_check}")
    if not smoke.get("passed"):
        raise WorkflowError(f"frozen initial model failed the GMD engine smoke test: {smoke}")
    write_json(cl.out_dir / "initial_model" / "manifest.json", {
        "dataset_version": dataset.version, "validation_version": asm.validation.version, "training_hash": frozen["training_hash"],
        "samples": record, "checkpoints": [item["path"] for item in frozen["checkpoints"]], "export": str(export), "metrics_validation": metrics,
        "export_check": export_check, "engine_smoke": smoke, "gates_recorded_not_enforced": None, "best_val": None,
        "frozen_initial_model": {"qualification_stage": frozen["qualification_stage"], "export_sha256": frozen["export"]["sha256"]}})
    artifact = controller.registry.register(
        checkpoint=primary, export=export, dataset_version=dataset.version, parent_model_id=None, species=list(frozen["species"]),
        cutoff=frozen.get("cutoff"), metrics=metrics, train_mode="fresh", backend="gmd_sgt",
        metadata={"initial_model": True, "frozen_initial_model": True, "initial_samples": record, "export_check": export_check, "engine_smoke": smoke},
        extra_files={f"committee_{k}": item["path"] for k, item in enumerate(frozen["checkpoints"][1:], start=1)})
    controller.initialise(artifact)
    controller.store.update_run(asm.run_id, active_dataset_version=dataset.version)
    return artifact


def run_closed_loop_arm(loaded: LoadedBenchmark, spec: RunSpec, frozen: dict[str, Any], manifest: Path, *, max_wall_s: float | None,
                        sleep: Callable[[float], None]) -> dict[str, Any]:
    from tools.experiments.alkane_hotspots.closed_loop import assemble, build_report, initial_samples, load_closed_loop, require_exclusive_cp2k

    cl = load_closed_loop(write_run_config(loaded, spec, manifest))
    require_exclusive_cp2k(cl, [cl.out_dir, loaded.root])
    asm = assemble(cl, "main")
    sections = {"monitor": asm.gmd_config.raw["monitor"], "events": asm.gmd_config.raw["events"], "hotspots": asm.gmd_config.raw["hotspots"]}
    calibration = calibrate_monitor(loaded, frozen, spec.arm, sections)
    require_calibrated(calibration)
    _train, val, record = initial_samples(cl, asm.service_settings_hash)
    groups = subset_samples(loaded, val, record["validation"])
    if f"repair_val.{spec.family}" not in groups:
        raise WorkflowError(f"no repair-validation structures of family {spec.family}: the promotion gate would have no evidence")
    subsets = {name: subset_version(samples, asm.work_dir / "subset_store", name) for name, samples in groups.items()}
    asm.controller.trainer = SubsetTrainer(asm.trainer, subsets)
    register_frozen_initial(cl, asm, frozen)
    blob = f"{asm.run_id}:monitor"
    seeded = False
    if asm.controller.store.get_blob(blob) is None:
        asm.controller.store.put_blob(blob, calibration["state"])
        seeded = True
    started = time.monotonic()
    while True:
        stage = asm.controller.run()
        if stage != "waiting":
            break
        if max_wall_s is not None and time.monotonic() - started > max_wall_s:
            LOGGER.warning("run %s: stopping after %.0f s with labels outstanding; re-run to resume", spec.name, max_wall_s)
            break
        sleep(cl.config.loop.poll_interval_s)
    report = build_report(cl, asm)
    report["final_stage"] = stage
    store = asm.controller.store
    final_state = store.get_blob(blob) or {}
    report["repair"] = {
        "spec": spec.__dict__, "run_name": spec.name, "monitor_calibration": {k: v for k, v in calibration.items() if k != "state"},
        "monitor_seeded_from_calibration": seeded, "monitor_driver_stats_at_end": final_state.get("stats"),
        "candidates": [{k: row[k] for k in ("candidate_id", "cycle", "event_id", "score", "state", "reason")} for row in store.candidates(asm.run_id)],
        "validation_subsets": {name: {"n_structures": len(samples), "version": subsets[name].version} for name, samples in groups.items()},
        "final_model": final_model_record(asm.controller.registry, report["final_model_id"]),
        "ancestry": {"md_start": spec.case, "base_id": spec.base_id, "family": spec.family},
    }
    write_json(cl.out_dir / "repair_run.json", {"spec": spec.__dict__, "kind": spec.kind, "final_stage": stage,
                                                "measured_dft_cpu_core_hours": report["measured_dft_cpu_core_hours"],
                                                "final_model": report["repair"]["final_model"], "report": "closed_loop_report.json"})
    write_json(cl.out_dir / "closed_loop_report.json", report)
    return report


def final_model_record(registry: Any, model_id: str | None) -> dict[str, Any] | None:
    if model_id is None:
        return None
    artifact = registry.get(model_id)
    extras = (artifact.metadata or {}).get("extra_files") or {}
    committee = [extras[name]["path"] for name in sorted(extras) if name.startswith("committee_")]
    return {"model_id": model_id, "checkpoint": artifact.checkpoint_path, "export": artifact.export_path,
            "checkpoints": [artifact.checkpoint_path, *committee], "initial": bool((artifact.metadata or {}).get("initial_model"))}


# ---------------------------------------------------------------------------------------------- continued training (B)


def run_continued_training(loaded: LoadedBenchmark, spec: RunSpec, frozen: dict[str, Any]) -> dict[str, Any]:
    """Arm B: fine-tune rounds on the original data only, gated exactly like the closed-loop arms."""

    from gmd_active_learning.retraining.registry import PromotionGateConfig, evaluate_gates

    from tools.experiments.alkane_hotspots.labels import LabelStore
    from tools.experiments.alkane_hotspots.training import full_sample, make_backend, train_model, write_dataset

    config = loaded.config
    out = run_dir(loaded, spec)
    if (out / "repair_run.json").is_file():
        return read_json(out / "repair_run.json")
    base = training_config(loaded)
    training = base.model_copy(update={"mode": "finetune", "train": {**base.train, **config.repair.finetune_train}})
    backend = make_backend(training, root=loaded.repo_path("."))
    store = LabelStore(loaded.root)
    records = {r["parent_id"]: r for r in load_records(loaded.root)}
    manifest = read_json(parents_manifest(loaded, frozen))

    def samples(ids: list[str]) -> list[Any]:
        return [full_sample(full_label_for(store, records[pid]), records[pid], training, loaded.root) for pid in ids]

    train = samples(manifest["train"])
    val = samples(manifest["validation"])
    dataset = write_dataset(train, val, out / "dataset", version=f"{spec.name}-original-only")
    groups = subset_samples(loaded, val, [{"parent_id": pid} for pid in manifest["validation"]])
    subsets = {name: subset_version(items, out / "subset_store", name) for name, items in groups.items()}

    def evaluate(checkpoint: str) -> dict[str, float]:
        metrics = dict(backend.evaluate(Path(checkpoint), dataset))
        for name, version in subsets.items():
            metrics[f"force_rmse.{name}"] = float(backend.evaluate(Path(checkpoint), version)["force_rmse"])
        return metrics

    gates = PromotionGateConfig.model_validate(promotion_section(loaded, spec.family))
    active = frozen_trained_model(frozen)
    active_metrics = evaluate(active.primary)
    rounds = []
    start = records[spec.case]
    topology = AlkaneTopology(int(start["chain_length"]))
    for index in range(config.repair.cycles):
        started = time.monotonic()
        candidate = train_model(backend, dataset, out / f"round{index}" / "train", seed=config.repair.finetune_seed_offset + spec.seed, parent=active,
                                config=training)
        wall = time.monotonic() - started
        metrics = evaluate(candidate.primary)
        export = backend.export(Path(candidate.primary), out / f"round{index}" / "export")
        export_check = backend.validate_export(Path(candidate.primary), export, dataset)
        smoke_md = run_md(artifact=model_artifact(export, Path(candidate.primary), candidate.species, f"{spec.name}-r{index}"),
                          positions=load_positions(loaded.root, start), topology=topology, directives=config.repair.md_directives, seed=spec.seed, steps=5,
                          frame_interval=1, out_dir=out / f"round{index}" / "engine_smoke", run_id=f"{spec.name}-r{index}-smoke",
                          executable=config.repair.executable, env=config.repair.env, md_box=config.repair.md_box, model_section=frozen["model_section"],
                          covalent_scale=config.initial_model.acceptance.md.covalent_scale)
        smoke = {"passed": smoke_md["status"] == "completed", "status": smoke_md["status"], "frames": smoke_md["n_frames"]}
        report = evaluate_gates(gates, metrics=metrics, parent_metrics=active_metrics, export_check=export_check, engine_smoke=smoke)
        rounds.append({"round": index, "checkpoints": candidate.checkpoints, "export": str(export), "train_wall_s": wall, "metrics": metrics,
                       "parent_metrics": active_metrics, "gate_report": report.to_dict(), "promoted": report.passed})
        if report.passed:
            active, active_metrics = candidate, metrics
            active_export = export
    promoted = any(r["promoted"] for r in rounds)
    final = {"model_id": None, "checkpoint": active.primary, "checkpoints": active.checkpoints,
             "export": str(active_export) if promoted else frozen["export"]["path"], "initial": not promoted}
    result = {"spec": spec.__dict__, "kind": spec.kind, "final_stage": "done", "measured_dft_cpu_core_hours": 0.0, "rounds": rounds,
              "final_model": final, "ungated_last": {"checkpoints": rounds[-1]["checkpoints"], "export": rounds[-1]["export"]} if rounds else None,
              "data": {"train": manifest["train"], "validation": manifest["validation"],
                       "note": "original id_train data only; no new labels"}}
    write_json(out / "repair_run.json", result)
    return result


# ---------------------------------------------------------------------------------------------- entry point


def run_repair(loaded: LoadedBenchmark, spec: RunSpec, *, max_wall_s: float | None = None, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    require_frozen(loaded)
    frozen = require_qualified(loaded)
    if spec.kind == "closed_loop":
        require_audit(loaded, spec.family)
    budget = require_budget(loaded, spec)
    manifest = parents_manifest(loaded, frozen)
    LOGGER.info("repair run %s (budget check %s)", spec.name, budget)
    if spec.kind == "continued_training":
        return run_continued_training(loaded, spec, frozen)
    if spec.kind == "closed_loop":
        return run_closed_loop_arm(loaded, spec, frozen, manifest, max_wall_s=max_wall_s, sleep=sleep)
    raise ConfigurationError(f"arm kind {spec.kind!r} has no run")


def run_all(loaded: LoadedBenchmark, specs: list[RunSpec], *, max_wall_s: float | None = None, echo: Callable[[str], None] = print) -> dict[str, Any]:
    results = {}
    for spec in specs:
        report = run_repair(loaded, spec, max_wall_s=max_wall_s)
        results[spec.name] = {"final_stage": report.get("final_stage"), "dft_core_h": report.get("measured_dft_cpu_core_hours")}
        echo(f"{spec.name}: {results[spec.name]}")
    return results


__all__ = ["RUNS_SUBDIR", "RunSpec", "SubsetTrainer", "calibrate_monitor", "closed_loop_raw", "frozen_predictor", "frozen_trained_model", "parents_manifest",
           "promotion_section", "register_frozen_initial", "repair_cases", "run_all", "run_matrix", "run_repair", "subset_samples", "subset_version"]
