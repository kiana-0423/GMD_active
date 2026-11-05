"""GMD_SGT adapter: training, evaluation, export, export validation and committee inference.

Replaces PHAL's Allegro backends (ledger BE-005/006/007) and GMD_active's former
``GMD_se3gnn`` subprocess stub, which returned zero forces and assumed artifact names. All
artifact paths here are the ones GMD_SGT actually reports (``api.train`` return value,
``training_summary.json``, ``api.export_model`` return value, embedded export metadata).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from pydantic import BaseModel, ConfigDict, Field

from gmd_active_learning.contracts import DatasetVersion, Frame, ModelArtifact
from gmd_active_learning.core.exceptions import AdapterError, CapabilityUnavailableError, ConfigurationError
from gmd_active_learning.utils.logging import get_logger
from gmd_active_learning.workflow.protocols import TrainOutput

LOGGER = get_logger(__name__)
MIN_GMD_SGT_VERSION = (0, 4, 0)


class SGTConfig(BaseModel):
    """``mlip`` config section for the GMD_SGT backend."""

    model_config = ConfigDict(extra="forbid")

    source_path: str | None = Field(None, description="GMD_SGT checkout to import from when it is not installed")
    device: str = "cpu"
    train: dict[str, Any] = Field(default_factory=dict, description="merged into GMD_SGT's train config (model/train sections)")
    finetune_from_active: bool = True
    seed: int = 1234
    committee_size: int = Field(0, ge=0, description="extra committee members trained with seeds seed+1..; 0 disables")
    export_filename: str = "model.pt"
    validation_max_structures: int = Field(50, ge=1)


def import_gmd_sgt(config: SGTConfig) -> Any:
    """Import ``gmd_sgt`` (optionally from a configured checkout) and check its version."""

    if config.source_path is not None:
        source = str(Path(config.source_path).expanduser().resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
    try:
        module = importlib.import_module("gmd_sgt")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CapabilityUnavailableError("gmd_sgt is not importable; install GMD_SGT or set mlip.source_path") from exc
    version = tuple(int(part) for part in str(module.__version__).split(".")[:3])
    if version < MIN_GMD_SGT_VERSION:
        raise CapabilityUnavailableError(
            f"GMD_SGT {module.__version__} lacks per-atom force weights / fine-tuning; >= {'.'.join(map(str, MIN_GMD_SGT_VERSION))} required"
        )
    importlib.import_module("gmd_sgt.api")
    importlib.import_module("gmd_sgt.inference.calculator")
    return module


def _graph(atoms: Any, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Full directed neighbour graph with Cartesian image shifts (the GMD/TorchScript convention)."""

    if any(atoms.pbc):
        i, j, shifts = neighbor_list("ijS", atoms, cutoff)
        edge_shift = shifts.astype(float) @ atoms.cell.array
    else:
        i, j = neighbor_list("ij", atoms, cutoff)
        edge_shift = np.zeros((len(i), 3))
    return np.vstack([i, j]).astype(np.int64), edge_shift


class GMDSGTBackend:
    """Implements ``workflow.protocols.Trainer`` and ``FrameEnricher`` on top of GMD_SGT >= 0.4."""

    name = "gmd_sgt"
    role = "mlip"
    capabilities = frozenset({"train", "finetune", "per_atom_force_weights", "mixed_pbc_batches", "event_aware_splits",
                              "torchscript_export", "export_metadata", "committee"})

    @classmethod
    def from_config(cls, config: Any) -> GMDSGTBackend:
        """Registry factory: ``config`` is the full configuration mapping (section ``mlip``)."""

        section = config.get("mlip") if hasattr(config, "get") else None
        return cls(SGTConfig.model_validate(section or {}))

    def check_runtime(self) -> Any:
        from gmd_active_learning.adapters.registry import BackendRole, RuntimeStatus

        return RuntimeStatus(self.name, BackendRole.MLIP, True, f"gmd_sgt {self.sgt.__version__}")

    def set_engine_smoke(self, engine_smoke: Any) -> None:
        self._engine_smoke = engine_smoke

    def __init__(self, config: SGTConfig, *, engine_smoke: Any | None = None) -> None:
        self.config = config
        self.sgt = import_gmd_sgt(config)
        from gmd_sgt import api
        from gmd_sgt.inference.calculator import MLIPCalculator

        self.api = api
        self.calculator_cls = MLIPCalculator
        self._engine_smoke = engine_smoke
        self._calculators: dict[str, Any] = {}

    # ------------------------------------------------------------------ training
    def _train_config(self, dataset: DatasetVersion, seed: int) -> dict[str, Any]:
        config = json.loads(json.dumps(self.config.train))
        config.setdefault("train", {})
        config["train"]["seed"] = seed
        config["train"].setdefault("device", self.config.device)
        config.setdefault("data", {})["val_file"] = dataset.metadata["val_file"]
        return config

    def _train_one(self, dataset: DatasetVersion, parent: ModelArtifact | None, out: Path, seed: int) -> Path:
        summary_path = out / "training_summary.json"
        if summary_path.is_file():  # idempotent: a finished run in this directory is reused
            summary = json.loads(summary_path.read_text())
            checkpoint = Path(summary["best_checkpoint"])
            if checkpoint.is_file():
                LOGGER.info("reusing finished training run %s", out)
                return checkpoint
        init = None
        if parent is not None and self.config.finetune_from_active and parent.backend == "gmd_sgt" and parent.train_mode != "external-incompatible":
            init = parent.checkpoint_path
        for key in ("train_file", "val_file"):
            if not Path(dataset.metadata.get(key, "")).is_file():
                raise AdapterError(f"dataset {dataset.version} has no readable {key}")
        result = self.api.train(dataset.metadata["train_file"], self._train_config(dataset, seed), str(out), init_checkpoint=init)
        checkpoint = Path(result)
        if not checkpoint.is_file():
            raise AdapterError(f"GMD_SGT reported {checkpoint}, which does not exist")
        if not summary_path.is_file():
            raise AdapterError(f"GMD_SGT did not write {summary_path}")
        return checkpoint

    def train(self, dataset: DatasetVersion, *, parent: ModelArtifact | None, work_dir: Path) -> TrainOutput:
        work_dir = Path(work_dir)
        primary = self._train_one(dataset, parent, work_dir / "member_0", self.config.seed)
        summary = json.loads((work_dir / "member_0" / "training_summary.json").read_text())
        committee = [
            str(self._train_one(dataset, parent, work_dir / f"member_{k}", self.config.seed + k))
            for k in range(1, self.config.committee_size + 1)
        ]
        species = [int(z) for z in summary.get("species") or []]
        if not species:
            raise AdapterError("training summary does not list the species the model was trained on")
        if summary.get("dry_run"):
            raise AdapterError("GMD_SGT reports a dry run; refusing to treat it as a trained model")
        cutoff = summary.get("cutoff")
        if isinstance(cutoff, dict):
            cutoff = cutoff.get("local_cutoff")
        return TrainOutput(primary, species, None if cutoff is None else float(cutoff), str(summary.get("mode", "fresh")),
                           {**summary, "committee_checkpoints": committee})

    # ------------------------------------------------------------------ evaluation
    def calculator(self, checkpoint: str | Path) -> Any:
        key = str(Path(checkpoint).resolve())
        if key not in self._calculators:
            self._calculators[key] = self.calculator_cls.from_checkpoint(key, device=self.config.device)
        return self._calculators[key]

    def _validation_structures(self, dataset: DatasetVersion) -> list[Any]:
        frames = read(dataset.metadata["val_file"], index=":")
        return frames[: self.config.validation_max_structures]

    def evaluate(self, checkpoint: Path, dataset: DatasetVersion) -> dict[str, float]:
        """Force RMSE weighted by ``force_weight`` (core atoms count fully, caps/buffers per weight)."""

        calc = self.calculator(checkpoint)
        sq_force, weight_sum, sq_energy, n_energy = 0.0, 0.0, 0.0, 0
        structures = self._validation_structures(dataset)
        for atoms in structures:
            weights = np.asarray(atoms.arrays.get("force_weight", np.ones(len(atoms))), dtype=float)
            reference = atoms.get_forces() if atoms.calc is not None else atoms.arrays["forces"]
            out = calc.compute(atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array if any(atoms.pbc) else None, atoms.pbc)
            sq_force += float(np.sum(weights[:, None] * (np.asarray(out["forces"]) - reference) ** 2))
            weight_sum += 3.0 * float(np.sum(weights))
            energy_weight = float(atoms.info.get("energy_weight", 1.0))
            if energy_weight > 0 and atoms.calc is not None and "energy" in atoms.calc.results:
                sq_energy += ((float(out["energy"]) - atoms.get_potential_energy()) / len(atoms)) ** 2
                n_energy += 1
        metrics = {"n_validation": float(len(structures)), "force_rmse": float(np.sqrt(sq_force / weight_sum)) if weight_sum > 0 else float("nan")}
        if n_energy:
            metrics["energy_rmse_per_atom"] = float(np.sqrt(sq_energy / n_energy))
        return metrics

    # ------------------------------------------------------------------ export
    def export(self, checkpoint: Path, work_dir: Path) -> Path:
        path = Path(self.api.export_model(str(checkpoint), str(work_dir), {"device": "cpu", "filename": self.config.export_filename}))
        metadata = self.api.load_export_metadata(path)
        if not metadata or not metadata.get("species"):
            raise AdapterError(f"export {path} carries no species metadata")
        return path

    def validate_export(self, checkpoint: Path, export: Path, dataset: DatasetVersion) -> dict[str, Any]:
        """Independent check: TorchScript forward vs eager checkpoint on the same external graph."""

        import torch

        scripted = torch.jit.load(str(export), map_location="cpu")
        cutoff = float(scripted.local_cutoff)
        calc = self.calculator(checkpoint)
        deviation, n = 0.0, 0
        for atoms in self._validation_structures(dataset):
            edge_index, edge_shift = _graph(atoms, cutoff)
            out = scripted(
                torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.int64),
                torch.as_tensor(atoms.get_positions(), dtype=torch.float32),
                torch.as_tensor(edge_index, dtype=torch.int64),
                torch.as_tensor(edge_shift, dtype=torch.float32),
            )
            reference = calc.compute(atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array if any(atoms.pbc) else None,
                                     atoms.pbc, edge_index=edge_index, edge_shift=edge_shift)
            deviation = max(deviation, float(np.max(np.abs(out["forces"].detach().numpy() - np.asarray(reference["forces"])))))
            n += 1
        metadata = self.api.load_export_metadata(export) or {}
        return {"max_force_deviation": deviation if n else None, "n_structures": n, "cutoff": cutoff, "species": metadata.get("species")}

    def engine_smoke_test(self, export: Path, work_dir: Path) -> dict[str, Any]:
        if self._engine_smoke is None:
            return {"passed": False, "reason": "no MD engine configured for the smoke test"}
        return dict(self._engine_smoke(export, work_dir))

    # ------------------------------------------------------------------ committee enrichment
    def committee_checkpoints(self, model: ModelArtifact) -> list[str]:
        extras = model.metadata.get("extra_files") or {}
        members = [extras[name]["path"] for name in sorted(extras) if name.startswith("committee_")]
        return [model.checkpoint_path, *members] if members else []

    def enrich(self, frame: Frame, model: ModelArtifact) -> Frame:
        """Attach committee forces (members re-evaluated on the frame). Without committee members
        the frame is returned unchanged, so the committee metric is reported as unavailable."""

        members = self.committee_checkpoints(model)
        if not members:
            return frame
        missing = set(frame.numbers.tolist()) - set(model.species)
        if missing:
            raise ConfigurationError(f"frame contains species {sorted(missing)} unknown to model {model.model_id}")
        cell = frame.cell if any(frame.pbc) else None
        forces = [np.asarray(self.calculator(path).compute(frame.positions, frame.numbers, cell, frame.pbc)["forces"]) for path in members]
        frame.committee_forces = np.stack(forces)
        return frame
