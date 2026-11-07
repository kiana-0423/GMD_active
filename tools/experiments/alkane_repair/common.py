"""Shared helpers of the repair benchmark: records, challenged regions, force errors, frozen-model MD."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.core.exceptions import ConfigurationError, WorkflowError
from gmd_active_learning.utils.atomic_io import write_json_atomic

from tools.experiments.alkane_hotspots.geometry import AlkaneTopology

LOGGER = logging.getLogger(__name__)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)
    return path


# ---------------------------------------------------------------------------------------------- regions and errors


def challenged_region(record: Mapping[str, Any], topology: AlkaneTopology, bond_hops: int) -> list[int]:
    """Perturbation-site atoms plus every carbon within ``bond_hops`` C-C bonds of a site carbon (with H)."""

    site = [int(a) for a in (record.get("perturbation") or {}).get("atoms") or []]
    if not site:
        raise ValueError(f"{record.get('parent_id')}: no perturbation site (ordinary structures have no challenged region)")
    carbons = {a if a < topology.n_carbons else topology.carbon_of[a] for a in site}
    distances = topology.graph_distances
    near = {c for c in range(topology.n_carbons) for s in carbons if distances[c, s] <= bond_hops}
    return sorted(set(site) | set(topology.site_atoms(sorted(near))))


def force_errors(model_forces: np.ndarray, reference_forces: np.ndarray, atoms: Sequence[int] | None = None) -> dict[str, Any]:
    """Per-component RMSE, per-atom |dF| statistics and force magnitudes over ``atoms`` (all if None)."""

    model = np.asarray(model_forces, dtype=float)
    reference = np.asarray(reference_forces, dtype=float)
    if model.shape != reference.shape:
        raise ValueError(f"force arrays differ in shape: {model.shape} vs {reference.shape}")
    index = np.arange(len(model)) if atoms is None else np.asarray(list(atoms), dtype=int)
    delta = model[index] - reference[index]
    per_atom = np.linalg.norm(delta, axis=1)
    return {"n_atoms": int(len(index)), "force_rmse": float(np.sqrt(np.mean(delta * delta))),
            "atom_error_mean": float(per_atom.mean()), "atom_error_max": float(per_atom.max()),
            "atom_error_p95": float(np.percentile(per_atom, 95)),
            "reference_force_rms": float(np.sqrt(np.mean(reference[index] ** 2))),
            "model_force_rms": float(np.sqrt(np.mean(model[index] ** 2)))}


def pooled_errors(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """Force errors pooled over several structures (all atoms)."""

    if not pairs:
        return {"n_structures": 0}
    model = np.concatenate([np.asarray(m, dtype=float) for m, _ in pairs])
    reference = np.concatenate([np.asarray(r, dtype=float) for _, r in pairs])
    return {"n_structures": len(pairs), **force_errors(model, reference)}


# ---------------------------------------------------------------------------------------------- frozen-model MD


def model_artifact(export: Path, checkpoint: Path, species: Sequence[int], model_id: str) -> Any:
    from gmd_active_learning.contracts import ModelArtifact
    from gmd_active_learning.contracts.ids import file_sha256

    return ModelArtifact(model_id=model_id, checkpoint_path=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                         export_path=str(export), export_sha256=file_sha256(export), dataset_version=None, parent_model_id=None,
                         species=tuple(sorted(int(z) for z in species)), cutoff=None, units={}, metrics={}, train_mode="fresh",
                         backend="gmd_sgt", created_at="")


def run_md(*, artifact: Any, positions: np.ndarray, topology: AlkaneTopology, directives: Mapping[str, Any], seed: int, steps: int,
           frame_interval: int, out_dir: Path, run_id: str, executable: str, env: Mapping[str, str], md_box: Mapping[str, Any],
           model_section: Mapping[str, Any], covalent_scale: float, on_frame: Callable[[Any], None] | None = None,
           segment_timeout_s: float = 3600.0) -> dict[str, Any]:
    """One frozen-model GMD trajectory in a vacuum-padded box; every frame's bond topology is checked.

    Every trajectory reports its full frame count; ``intact_fraction`` counts broken frames against
    all frames (none is dropped). Resumable: a completed segment is re-read, not re-run.
    """

    from gmd_active_learning.adapters.gmd.engine import GMDConfig, GMDEngine
    from gmd_active_learning.contracts import FrameAction
    from gmd_active_learning.workflow.protocols import SegmentRequest

    from tools.experiments.alkane_hotspots.closed_loop import REPO_ROOT, chain_observables
    from tools.experiments.alkane_hotspots.md_box import ImageGuard, MDBoxConfig, build_box, force_receptive_field, write_gmd_xyz

    box_config = MDBoxConfig.model_validate(dict(md_box))
    receptive = force_receptive_field(dict(model_section), override=box_config.receptive_field_override)
    box = build_box(positions, receptive, box_config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = write_gmd_xyz(out_dir / "start.xyz", topology.numbers, box.positions, box.lengths)
    exe = Path(executable)
    md = {"executable": str(exe if exe.is_absolute() else (REPO_ROOT / exe).resolve()), "initial_structure": str(xyz),
          "directives": {**dict(directives), "velocity_seed": int(seed), "velocity_init": "random"}, "frame_interval": int(frame_interval),
          "control_poll_interval": 1, "segment_timeout_s": float(segment_timeout_s), "env": dict(env), "smoke_steps": 5}
    engine = GMDEngine(GMDConfig.model_validate(md))
    guard = ImageGuard(None, bonds=topology.bonds, receptive_field=receptive, margin=box_config.margin, stats_path=out_dir / "image_check.json")
    rows: list[dict[str, Any]] = []

    def handle(frame: Any) -> Any:
        guard.check(frame)
        rows.append(chain_observables(frame, topology, covalent_scale=covalent_scale))
        if on_frame is not None:
            on_frame(frame)
        return FrameAction.CONTINUE

    request = SegmentRequest(run_id, 0, out_dir / "segment", artifact, int(steps), None, None)
    result = engine.completed_segment(request)
    if result is not None and result.status == "completed":
        for frame in engine.read_frames(result, request):
            handle(frame)
    else:
        result = engine.run_segment(request, handle)
    broken = [r["step"] for r in rows if r["topology_intact"] is False]
    return {"run_id": run_id, "seed": int(seed), "status": result.status, "detail": getattr(result, "detail", None), "steps": int(steps),
            "n_frames": len(rows), "n_broken": len(broken), "first_broken_step": min(broken) if broken else None,
            "intact_fraction": (len(rows) - len(broken)) / len(rows) if rows else 0.0, "box": box.to_dict(), "image_check": guard.flush()}


def require(condition: bool, message: str, error: type[Exception] = WorkflowError) -> None:
    if not condition:
        raise error(message)


def scan_directives(directives: Mapping[str, Any], forbidden: Sequence[str]) -> list[str]:
    """Directive keys that would clip forces or constrain the dynamics (case-insensitive substring match)."""

    keys = [str(k).lower() for k in directives]
    return sorted({key for key in keys for word in forbidden if word.lower() in key.split("_") or key == word.lower()})


__all__ = ["ConfigurationError", "challenged_region", "force_errors", "model_artifact", "pooled_errors", "read_json", "read_jsonl", "require",
           "run_md", "scan_directives", "write_json"]
