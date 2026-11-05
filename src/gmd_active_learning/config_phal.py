"""Convert a PHAL (hotspot_al 96263b6) configuration into a GMD_active configuration (CFG-002).

``PHAL_KEY_MAP`` covers every leaf of PHAL's ``CONFIG_SCHEMA``: the value is either the GMD key the
leaf is carried to (``section.key[.key]``, several targets separated by ``" + "``) or
``"dropped: <reason>"``. :func:`convert_phal_config` applies the table and returns the raw GMD
configuration (validate it with :func:`gmd_active_learning.config.build_config`) together with a
report of what was mapped, dropped, and which dropped keys carried non-default content.

PHAL concepts with no GMD counterpart (LAMMPS input generation and dump decoding, Allegro command
templates and model paths, dry-run modes, ad-hoc state and output directories) are dropped with a
reason; the run directory (``gmd-al run --work-dir``) owns every state, dataset and model path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from gmd_active_learning.core.exceptions import ConfigurationError
from gmd_active_learning.utils.logging import get_logger

LOGGER = get_logger(__name__)

_LAMMPS = "dropped: LAMMPS is replaced by GMD (adapters/gmd); "
_ALLEGRO = "dropped: Allegro is replaced by GMD_SGT (adapters/sgt); "
_WORKDIR = "dropped: the run directory (gmd-al --work-dir) owns this path; "
_NEVER_READ = "dropped: PHAL never read this key; "

PHAL_KEY_MAP: dict[str, str] = {
    # project / backend / plugins
    "project.name": "workflow.run_id",
    "project.method_name": _NEVER_READ + "no GMD equivalent",
    "project.description": "dropped: descriptive text only; use a YAML comment",
    "backend.md.engine": "backends.md.engine",
    "backend.mlip.engine": "backends.mlip.engine",
    "backend.dft.engine": "backends.dft.engine + scf.ml_scf.enabled",
    "backend.scheduler.engine": "backends.scheduler.engine",
    "plugins": "dropped: external backends are configured through backends.<role>.factory or entry points",
    # lammps
    "lammps.executable": _LAMMPS + "md.executable (converter argument gmd_executable)",
    "lammps.input_template": _NEVER_READ + "GMD run.in is written by the adapter",
    "lammps.dump_style": _NEVER_READ + "GMD writes extxyz frames (gmd-frame-v1)",
    "lammps.dump_fields": _LAMMPS + "the GMD frame format is fixed (ids, species, positions, forces, cell, pbc)",
    "lammps.type_map": _LAMMPS + "GMD frames carry element symbols; species come from md.initial_structure",
    "lammps.timestep_fs": "md.directives.time_step",
    "lammps.units": _LAMMPS + "units are fixed (Angstrom, eV, fs; contracts.UNITS)",
    "lammps.atom_style": _LAMMPS + "no atom styles",
    "lammps.boundary": _LAMMPS + "periodicity comes from the GMD structure and frames (Frame.pbc)",
    "lammps.thermo_freq": _LAMMPS + "GMD writes its own log; only frames are read by the controller",
    "lammps.run_steps": "workflow.segment_steps",
    "lammps.full_input_template": _LAMMPS + "extra GMD directives go to md.directives",
    # allegro
    "allegro.enabled": _NEVER_READ + "the MLIP backend is backends.mlip.engine",
    "allegro.device": "mlip.device",
    "allegro.model_paths": _ALLEGRO + "the initial model is registered with `gmd-al init --checkpoint --export`",
    "allegro.deployed_model_paths": _ALLEGRO + "the deployed model is the registry's promoted model",
    "allegro.dataset_dir": _WORKDIR + "versioned datasets live under <work_dir>/datasets",
    "allegro.train_output_dir": _WORKDIR + "training runs live under <work_dir>/training",
    "allegro.checkpoint_path": _ALLEGRO + "checkpoints are the paths GMD_SGT reports, recorded by the model registry",
    "allegro.train_config_path": _ALLEGRO + "GMD_SGT training settings are inline in mlip.train",
    "allegro.train_command_template": _ALLEGRO + "GMD_SGT is called through its Python API, not command templates (BE-006)",
    "allegro.export_command_template": _ALLEGRO + "GMD_SGT export is called through its Python API (BE-006)",
    "allegro.lammps_pair_style": _ALLEGRO + "GMD loads the TorchScript export directly (model_path)",
    "allegro.committee_mode": _NEVER_READ + "committees are mlip.committee_size",
    "allegro.type_names": _ALLEGRO + "GMD_SGT exports carry their species list; frames are checked against it",
    "allegro_model_input": _ALLEGRO + "the GMD_SGT calculator builds its own graph inputs",
    # monitor
    "monitor.light_interval": "monitor.stages.light_interval",
    "monitor.physics_interval": "monitor.stages.physics_interval",
    "monitor.committee_interval": "monitor.stages.full_interval",
    "monitor.force_z_threshold": "monitor.metrics.force.z_threshold",
    "monitor.delta_force_z_threshold": "monitor.metrics.delta_force.z_threshold",
    "monitor.delta_q_threshold": "monitor.metrics.delta_q.threshold",
    "monitor.rmin_threshold": "monitor.metrics.rmin.threshold",
    "monitor.displacement_z_threshold": "monitor.metrics.displacement.z_threshold",
    "monitor.coordination_smoothing_power": "monitor.metrics.delta_q.power",
    "monitor.coordination_scale": "monitor.metrics.delta_q.scale",
    "monitor.lj_cutoff": "monitor.neighbors.lj_cutoff",
    "monitor.coordination_cutoff": "monitor.neighbors.coordination_cutoff",
    # online
    "online.enabled": "dropped: the GMD workflow always monitors online (segmented runs with flushed frames); "
                      "offline replay feeds frames to monitors.online.OnlineMonitorDriver",
    "online.work_dir": _WORKDIR + "MD segments run in <work_dir>/md/segment_NNNNN",
    "online.dump_file": "dropped: the GMD frame file name is owned by the adapter (frames.extxyz)",
    "online.dump_freq": "md.frame_interval",
    "online.monitor_freq": "monitor.stages.scheduled_full_interval",
    "online.summary_interval": "monitor.driver.summary_interval",
    "online.progress_file": "monitor.driver.progress_file",
    "online.continue_on_error": "monitor.driver.continue_on_error",
    "online.max_errors": "monitor.driver.max_errors",
    "online.max_walltime": "dropped: only a dead fallback for the CP2K walltime in PHAL; use cp2k.resources.walltime",
    "online.event_dir": _WORKDIR + "events are written to <work_dir>/events",
    "online.run_steps": "workflow.segment_steps",
    # logging
    "logging.level": "dropped: logging is configured on the command line (gmd-al --log-level)",
    "logging.file": "dropped: logging is configured by the caller (gmd-al writes to stderr)",
    "logging.format": "dropped: logging is configured by the caller",
    # ood_score
    "ood_score.weights": "monitor.metrics.<metric>.weight",
    "ood_score.screen_threshold": "monitor.stages.screen_threshold",
    "ood_score.physics_threshold": "monitor.stages.physics_threshold",
    "ood_score.label_threshold": "monitor.stages.label_threshold",
    "ood_score.lj_lazy_threshold": "monitor.metrics.lj_residual.lazy_threshold",
    "ood_score.min_trigger_atoms": "monitor.stages.min_trigger_atoms",
    "ood_score.running_stats.enabled": "monitor.running_stats.enabled",
    "ood_score.running_stats.warmup_frames": "monitor.running_stats.warmup_frames",
    "ood_score.running_stats.min_std": "monitor.running_stats.min_std",
    # buffer / hotspot
    "buffer.pre_trigger_frames": "events.pre_trigger_frames",
    "buffer.post_trigger_frames": "events.post_trigger_frames",
    "buffer.maxlen": "events.maxlen",
    "hotspot.threshold": "dropped: second hotspot threshold removed (F-16); hotspot atoms come from monitor.stages.label_threshold",
    "hotspot.merge_radius": "hotspots.merge_radius",
    # extraction
    "extraction.mode": "extraction.mode",
    "extraction.mlip_cutoff": "extraction.mlip_cutoff",
    "extraction.core_radius": "extraction.core_radius",
    "extraction.buffer_radius": "extraction.buffer_radius",
    "extraction.extract_radius": "extraction.extract_radius",
    "extraction.boundary_thickness": "extraction.boundary_thickness",
    "extraction.max_atoms": "extraction.max_atoms",
    "extraction.min_atoms": "extraction.min_atoms",
    "extraction.vacuum_padding": "extraction.vacuum_padding",
    "extraction.slab.lx": "extraction.slab.lx",
    "extraction.slab.ly": "extraction.slab.ly",
    "extraction.slab.z_margin": "extraction.slab.z_margin",
    "extraction.graph.hops": "extraction.graph.hops",
    "extraction.graph.radial_shell": "extraction.graph.radial_shell",
    "extraction.block.scheme": "extraction.block.scheme",
    "extraction.block.size": "extraction.block.size",
    "extraction.block.halo": "extraction.block.halo",
    "extraction.block.merge_adjacent": "extraction.block.merge_adjacent",
    "extraction.block.max_merged_blocks": "extraction.block.max_merged_blocks",
    "extraction.block.cooldown_steps": "extraction.block.cooldown_steps",
    "extraction.block.label_region.type": _NEVER_READ + "block labels always cover the block core",
    "extraction.block.label_region.shrink": _NEVER_READ + "block labels always cover the block core",
    "extraction.block.buffer.inner": "extraction.block.buffer.inner",
    "extraction.block.buffer.outer": "extraction.block.buffer.outer",
    "extraction.block.frozen.enabled": "extraction.block.frozen.enabled",
    "extraction.block.frozen.thickness": "extraction.block.frozen.thickness",
    "extraction.block.max_atoms": "extraction.block.max_atoms",
    "extraction.block.min_atoms": "extraction.block.min_atoms",
    "extraction.block.min_atoms_policy": "extraction.block.min_atoms_policy",
    # embedding / capping
    "embedding.enabled": "embedding.enabled",
    "embedding.method": "embedding.method",
    "embedding.cutoff": "embedding.cutoff",
    "embedding.charges": "embedding.charges",
    "embedding.default_charge": "embedding.default_charge",
    "h_capping.enabled": "capping.enabled",
    "h_capping.covalent_scale": "capping.covalent_scale",
    "h_capping.allow_core_capping": "capping.allow_core_capping",
    "h_capping.optimize_h_only": "capping.optimize_h_only",
    "h_capping.disabled_for_metals": "capping.disabled_for_metals",
    "h_capping.disabled_for_oxides_by_default": "capping.disabled_for_oxides_by_default",
    "h_capping.bond_lengths": "capping.bond_lengths",
    # cp2k
    "cp2k.executable": "cp2k.executable",
    "cp2k.submit_mode": "dropped: there is no dry-run submission; the scheduler is backends.scheduler.engine",
    "cp2k.task_dir": "cp2k.task_dir",
    "cp2k.labeled_dataset_dir": _WORKDIR + "labeled samples go to the versioned dataset store",
    "cp2k.max_retries": "cp2k.retry.max_retries",
    "cp2k.max_unknown_polls": "cp2k.max_unknown_polls_warning",
    "cp2k.max_walltime_seconds": "cp2k.resources.walltime",
    "cp2k.slurm_directives": "cp2k.resources.extra_directives",
    "cp2k.functional": "cp2k.functional",
    "cp2k.basis": "cp2k.basis + scf.ml_scf.basis_set",
    "cp2k.potential": "cp2k.potential",
    "cp2k.cutoff": "cp2k.cutoff",
    "cp2k.rel_cutoff": "cp2k.rel_cutoff",
    "cp2k.charge": "dropped: the region's charge (set by extraction) is used",
    "cp2k.multiplicity": "cp2k.multiplicity",
    "cp2k.spin_policy": "cp2k.spin_policy",
    "cp2k.poisson_solver": "cp2k.poisson.non_periodic_solver",
    "cp2k.periodic": "dropped: CP2K periodicity follows the region's pbc",
    "cp2k.uncapped_fragment_policy": "cp2k.uncapped_fragment_policy",
    "cp2k.scf_eps": "cp2k.scf_eps",
    "cp2k.max_scf": "cp2k.max_scf",
    "cp2k.use_ot": "cp2k.use_ot",
    "cp2k.dispersion": "cp2k.dispersion",
    "cp2k.h_only_opt.enabled": "cp2k.h_only_opt.enabled",
    "cp2k.h_only_opt.max_iter": "cp2k.h_only_opt.max_iter",
    "cp2k.single_point.enabled": "dropped: the single point is the label; it cannot be disabled",
    "cp2k.single_point.print_forces": _NEVER_READ + "forces are always printed",
    "cp2k.scf_recording.enabled": "scf.recording.enabled",
    "cp2k.scf_recording.output_dir": "scf.recording.output_dir",
    "cp2k.scf_recording.record_density": "scf.recording.record_density",
    "cp2k.scf_recording.density_strict": "scf.recording.density_strict",
    "cp2k.scf_recording.record_trajectory": "scf.recording.record_trajectory",
    "cp2k.scf_recording.trajectory_strict": "scf.recording.trajectory_strict",
    "cp2k.ml_scf.enabled": "scf.ml_scf.enabled",
    "cp2k.ml_scf.predictor.type": "scf.ml_scf.predictor.type",
    "cp2k.ml_scf.predictor.model_path": "scf.ml_scf.predictor.model_path",
    "cp2k.ml_scf.predictor.options": "scf.ml_scf.predictor.options",
    "cp2k.ml_scf.prediction_dir": "scf.ml_scf.prediction_dir",
    "cp2k.ml_scf.dataset_dir": "scf.dataset.root",
    "cp2k.ml_scf.electron_tolerance": "scf.ml_scf.electron_tolerance",
    "cp2k.ml_scf.mixing.method": "scf.ml_scf.mixing.method",
    "cp2k.ml_scf.mixing.fallback_method": _NEVER_READ + "the native fallback is scf.ml_scf.fallback_to_standard_scf",
    "cp2k.ml_scf.mixing.history_length": "scf.ml_scf.mixing.history_length",
    "cp2k.ml_scf.mixing.alpha": "scf.ml_scf.mixing.alpha",
    "cp2k_ml_scf.executable": "scf.ml_scf.executable",
    "cp2k_ml_scf.model_path": "scf.ml_scf.model_path",
    "cp2k_ml_scf.model_digest": "scf.ml_scf.model_digest",
    "cp2k_ml_scf.runtime_dir": _NEVER_READ + "no GMD equivalent",
    "cp2k_ml_scf.fallback_to_standard_scf": "scf.ml_scf.fallback_to_standard_scf",
    "cp2k_ml_scf.require_final_scf_convergence": "scf.ml_scf.require_final_scf_convergence",
    "cp2k_ml_scf.controller_protocol": "scf.ml_scf.protocol",
    "cp2k_ml_scf.environment": "scf.ml_scf.environment",
    # retraining / registry / datasets
    "retraining.min_new_samples": "workflow.retraining.min_new_samples",
    "retraining.interval_hours": "workflow.retraining.interval_hours",
    "retraining.dry_run": "dropped: no dry-run training; workflow.retraining.enabled switches retraining off",
    "retraining.state_path": "dropped: the trigger state lives in the workflow store (<work_dir>/workflow.sqlite)",
    "retraining.export_dir": _WORKDIR + "exports live next to their training run and are copied into the registry",
    "retraining.train_output_dir": _WORKDIR + "training runs live under <work_dir>/training",
    "retraining.checkpoint_path": "dropped: checkpoints are the paths GMD_SGT reports, recorded by the model registry",
    "retraining.exported_model_path": "dropped: exports are the paths GMD_SGT reports, recorded by the model registry",
    "model_registry.root_dir": _WORKDIR + "the model registry is <work_dir>/models (read by nothing in PHAL)",
    "datasets.labeled_dir": _WORKDIR + "the versioned dataset store holds labeled samples (datasets.store_root)",
    "datasets.training_dir": _WORKDIR + "the versioned dataset store holds published training sets (datasets.store_root)",
    # training mask
    "training_mask.core": "extraction.force_weights.core",
    "training_mask.label_core": "dropped: block label-core atoms share extraction.force_weights.core",
    "training_mask.inner_buffer": "extraction.force_weights.inner_buffer",
    "training_mask.outer_buffer": "extraction.force_weights.outer_buffer",
    "training_mask.boundary": "extraction.force_weights.boundary",
    "training_mask.frozen_boundary": "dropped: frozen boundary atoms share extraction.force_weights.boundary",
    "training_mask.h_cap": "extraction.force_weights.h_cap",
    "training_mask.energy_weight": "datasets.energy.nonperiodic_weight + datasets.energy.periodic_weight",
    "training_mask.stress_weight": "datasets.stress.weight",
    # candidate pool
    "candidate_pool.deduplicate": "workflow.candidates.deduplicate",
    "candidate_pool.max_candidates_per_round": "workflow.candidates.max_per_cycle",
    "candidate_pool.fingerprint": "workflow.candidates.fingerprint",
    "candidate_pool.diversity_threshold": "workflow.candidates.diversity_threshold",
}

ENGINE_MAP: dict[str, dict[str, str]] = {
    "md": {"lammps": "gmd", "gmd": "gmd"},
    "mlip": {"allegro": "gmd_sgt", "gmd_sgt": "gmd_sgt"},
    "dft": {"cp2k": "cp2k", "cp2k_ml_scf": "cp2k"},  # ML-SCF is scf.ml_scf.enabled on the CP2K backend
    "scheduler": {"local": "local", "slurm": "slurm", "pbs": "pbs"},
}
FINGERPRINT_ALIASES: dict[str, str] = {
    "pair_distance_histogram": "pair_distance_histogram",
    "pair_distance": "pair_distance_histogram",
    "distance": "pair_distance_histogram",
    "type_weighted_pair_distance_histogram": "type_weighted_pair_distance_histogram",
    "type_weighted_pair_distance": "type_weighted_pair_distance_histogram",
    "weighted_pair": "type_weighted_pair_distance_histogram",
}


@dataclass(slots=True)
class ConversionReport:
    mapped: dict[str, str] = field(default_factory=dict)  # PHAL leaf -> GMD target
    dropped: dict[str, str] = field(default_factory=dict)  # PHAL leaf -> reason
    dropped_with_content: dict[str, Any] = field(default_factory=dict)  # dropped leaves that carried non-empty values
    warnings: list[str] = field(default_factory=list)


_MISSING = object()


def _get(data: Mapping[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _set(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        data = data.setdefault(part, {})
    data[parts[-1]] = value


def _leaves(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Leaf paths of a PHAL mapping, using the schema's notion of a leaf (free-form dicts are leaves)."""

    out: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if path in PHAL_KEY_MAP or not isinstance(value, Mapping):
            out[path] = value
        else:
            out.update(_leaves(value, path))
    return out


def convert_phal_config(
    phal: Mapping[str, Any],
    *,
    gmd_executable: str = "gmd",
    initial_structure: str = "initial_structure.xyz",
    run_id: str | None = None,
    legacy_scoring: bool = False,
) -> tuple[dict[str, Any], ConversionReport]:
    """Translate a full PHAL configuration (as loaded from its YAML) into a raw GMD configuration.

    ``gmd_executable`` / ``initial_structure`` fill the GMD fields PHAL has no value for (PHAL ran
    LAMMPS on a data file it generated). ``legacy_scoring`` keeps PHAL's warmup/first-frame
    scoring conventions (parity replays); the default uses GMD's corrected conventions. Unknown
    PHAL keys raise ``ConfigurationError`` (PHAL's schema rejected them too).
    """

    from gmd_active_learning.active_learning.candidates import CandidateConfig
    from gmd_active_learning.adapters.dft.scf.config import ScfConfig
    from gmd_active_learning.events.config import events_config_from_phal
    from gmd_active_learning.monitors.config import monitor_config_from_phal

    report = ConversionReport()
    leaves = _leaves(phal)
    unknown = sorted(path for path in leaves if path not in PHAL_KEY_MAP)
    if unknown:
        raise ConfigurationError(f"unknown PHAL config key(s): {unknown}")
    for path, value in leaves.items():
        target = PHAL_KEY_MAP[path]
        if target.startswith("dropped:"):
            report.dropped[path] = target[len("dropped:"):].strip()
            if value not in (None, "", [], {}, False) and path not in _DROPPED_DEFAULTS_OK:
                report.dropped_with_content[path] = value
        else:
            report.mapped[path] = target

    def get(dotted: str, default: Any = None) -> Any:
        value = _get(phal, dotted, _MISSING)
        return default if value is _MISSING else value

    raw: dict[str, Any] = {}
    # ---------------------------------------------------------------- workflow
    workflow: dict[str, Any] = {"run_id": run_id or str(get("project.name") or "phal-run")}
    run_steps = int(get("online.run_steps", 0) or 0) or int(get("lammps.run_steps", 0) or 0)
    if run_steps > 0:
        workflow["segment_steps"] = run_steps
    else:
        report.warnings.append("PHAL online.run_steps/lammps.run_steps is 0 (no MD run); workflow.segment_steps keeps its default")
    candidates: dict[str, Any] = {}
    for key, gmd_key in (("deduplicate", "deduplicate"), ("max_candidates_per_round", "max_per_cycle"),
                         ("diversity_threshold", "diversity_threshold")):
        if get(f"candidate_pool.{key}") is not None:
            candidates[gmd_key] = get(f"candidate_pool.{key}")
    fingerprint = get("candidate_pool.fingerprint")
    if fingerprint is not None:
        try:
            candidates["fingerprint"] = FINGERPRINT_ALIASES[str(fingerprint).lower()]
        except KeyError as exc:
            raise ConfigurationError(f"candidate_pool.fingerprint {fingerprint!r} is not a PHAL fingerprint mode") from exc
    # PHAL's diversity_threshold is an L2 distance between *density* histograms of plain (non-PBC)
    # pair distances; keep that meaning when the candidate model offers the switches.
    fields = CandidateConfig.model_fields
    if "histogram_normalisation" in fields:
        candidates["histogram_normalisation"] = "density"
    if "pair_distances" in fields:
        candidates["pair_distances"] = "plain"
    if "max_per_event" in fields and not fields["max_per_event"].is_required():
        if fields["max_per_event"].annotation is not int:  # Optional: PHAL had no per-event budget
            candidates["max_per_event"] = None
    if candidates:
        workflow["candidates"] = candidates
    retraining = {key: get(f"retraining.{key}") for key in ("min_new_samples", "interval_hours") if get(f"retraining.{key}") is not None}
    if retraining:
        retraining["interval_hours"] = float(retraining.get("interval_hours", 24.0))
        workflow["retraining"] = retraining
    raw["workflow"] = workflow

    # ---------------------------------------------------------------- md / mlip / backends
    md: dict[str, Any] = {"executable": gmd_executable, "initial_structure": initial_structure}
    directives = {"time_step": 0.5, "velocity": 300.0, "velocity_init": "random", "velocity_seed": 1}
    if get("lammps.timestep_fs") is not None:
        directives["time_step"] = float(get("lammps.timestep_fs"))
    md["directives"] = directives
    if get("online.dump_freq") is not None:
        md["frame_interval"] = int(get("online.dump_freq"))
    raw["md"] = md
    raw["mlip"] = {"device": str(get("allegro.device", "cpu"))} if get("allegro.device") is not None else {}
    backends: dict[str, Any] = {}
    for role in ("md", "mlip", "dft", "scheduler"):
        engine = get(f"backend.{role}.engine")
        if engine is None:
            continue
        try:
            backends[role] = {"engine": ENGINE_MAP[role][str(engine).lower()]}
        except KeyError as exc:
            raise ConfigurationError(f"backend.{role}.engine {engine!r} has no GMD_active counterpart; "
                                     f"known: {sorted(ENGINE_MAP[role])}") from exc
    if backends:
        raw["backends"] = backends

    # ---------------------------------------------------------------- monitoring / events / hotspots
    raw["monitor"] = monitor_config_from_phal(dict(phal), legacy_scoring=legacy_scoring).model_dump(mode="json")
    progress = get("online.progress_file")
    if progress is not None:
        raw["monitor"]["driver"]["progress_file"] = progress
    if get("buffer") is not None:
        raw["events"] = events_config_from_phal(dict(phal)).model_dump(mode="json")
    if get("hotspot.merge_radius") is not None:
        raw["hotspots"] = {"merge_radius": float(get("hotspot.merge_radius"))}

    # ---------------------------------------------------------------- extraction / capping / embedding / datasets
    extraction = _copy_mapped(phal, "extraction", "extraction")
    mask = {key: get(f"training_mask.{key}") for key in ("core", "inner_buffer", "outer_buffer", "boundary", "h_cap")
            if get(f"training_mask.{key}") is not None}
    if mask:
        extraction["force_weights"] = mask
    for key in ("label_core", "frozen_boundary"):
        shared = "core" if key == "label_core" else "boundary"
        if get(f"training_mask.{key}") is not None and get(f"training_mask.{shared}") is not None \
                and float(get(f"training_mask.{key}")) != float(get(f"training_mask.{shared}")):
            report.warnings.append(f"training_mask.{key}={get(f'training_mask.{key}')} differs from training_mask.{shared}; "
                                   f"GMD uses extraction.force_weights.{shared} for both")
    if extraction:
        raw["extraction"] = extraction
    capping = _copy_mapped(phal, "h_capping", "capping")
    if capping:
        raw["capping"] = capping
    embedding = _copy_mapped(phal, "embedding", "embedding")
    if embedding:
        raw["embedding"] = embedding
    datasets: dict[str, Any] = {}
    if get("training_mask.energy_weight") is not None:
        weight = float(get("training_mask.energy_weight"))
        datasets["energy"] = {"nonperiodic_weight": weight, "periodic_weight": weight}
    if get("training_mask.stress_weight") is not None:
        datasets["stress"] = {"weight": float(get("training_mask.stress_weight"))}
    if datasets:
        raw["datasets"] = datasets

    # ---------------------------------------------------------------- cp2k / scf
    cp2k = _copy_mapped(phal, "cp2k", "cp2k")
    if get("cp2k.max_retries") is not None:
        cp2k.setdefault("retry", {})["max_retries"] = int(get("cp2k.max_retries"))
    if get("cp2k.max_unknown_polls") is not None:
        cp2k["max_unknown_polls_warning"] = max(1, int(get("cp2k.max_unknown_polls")))
    resources: dict[str, Any] = {}
    if get("cp2k.max_walltime_seconds") is not None:
        resources["walltime"] = int(get("cp2k.max_walltime_seconds"))
    directives_text = str(get("cp2k.slurm_directives") or "")
    if directives_text.strip():
        resources["extra_directives"] = [line.strip() for line in directives_text.splitlines() if line.strip()]
    if resources:
        cp2k["resources"] = resources
    if get("cp2k.poisson_solver") is not None:
        cp2k["poisson"] = {"non_periodic_solver": str(get("cp2k.poisson_solver")).upper()}
    raw["cp2k"] = cp2k
    if get("cp2k.scf_recording") is not None or get("cp2k.ml_scf") is not None or get("cp2k_ml_scf") is not None:
        raw["scf"] = ScfConfig.from_phal(dict(phal)).model_dump(mode="json")

    for path in report.dropped_with_content:
        LOGGER.info("PHAL key %s dropped: %s", path, report.dropped[path])
    return raw, report


# Dropped keys whose PHAL *default* values are not worth a report line.
_DROPPED_DEFAULTS_OK = frozenset({"project.method_name", "project.description", "allegro.enabled", "allegro.committee_mode",
                                  "allegro.lammps_pair_style", "lammps.units", "lammps.atom_style", "lammps.dump_style",
                                  "lammps.dump_fields", "lammps.boundary", "lammps.thermo_freq", "logging.level",
                                  "logging.format", "cp2k.single_point.enabled", "cp2k.single_point.print_forces",
                                  "online.dump_file", "online.work_dir", "online.event_dir", "online.max_walltime",
                                  "cp2k.submit_mode", "cp2k.labeled_dataset_dir", "cp2k.periodic", "retraining.dry_run",
                                  "retraining.state_path", "retraining.export_dir", "model_registry.root_dir",
                                  "datasets.labeled_dir", "datasets.training_dir", "cp2k_ml_scf.runtime_dir",
                                  "extraction.block.label_region.type", "cp2k_ml_scf.controller_protocol"})


def _copy_mapped(phal: Mapping[str, Any], phal_section: str, gmd_section: str) -> dict[str, Any]:
    """Copy the leaves of ``phal_section`` that map 1:1 into ``gmd_section`` (same relative key)."""

    out: dict[str, Any] = {}
    section = phal.get(phal_section)
    if not isinstance(section, Mapping):
        return out
    for path, value in _leaves(section, phal_section).items():
        target = PHAL_KEY_MAP.get(path, "")
        relative = path[len(phal_section) + 1:]
        if target == f"{gmd_section}.{relative}" or target.startswith(f"{gmd_section}.{relative} + "):
            _set(out, relative, value)
    return out


__all__ = ["ENGINE_MAP", "PHAL_KEY_MAP", "ConversionReport", "convert_phal_config"]
