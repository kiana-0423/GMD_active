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
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
from ase.data import chemical_symbols
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


@dataclass(frozen=True, slots=True)
class Prediction:
    """One model evaluation in GMD units (eV, eV/Angstrom)."""

    energy: float
    forces: np.ndarray  # (n_atoms, 3)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """What a GMD_SGT artifact declares about itself (read from its own metadata)."""

    path: str
    kind: str  # "checkpoint" | "export"
    species: tuple[int, ...]
    cutoff: float
    lr_cutoff: float | None
    gmd_sgt_version: str | None


def check_species(numbers: Sequence[int] | np.ndarray, model_species: Sequence[int], *, model: str = "model") -> None:
    """Refuse a structure containing species the model was not trained on (before any evaluation)."""

    if not model_species:
        raise AdapterError(f"{model} declares no species; refusing to evaluate (GMD_SGT >= 0.4 records them)")
    unknown = sorted(set(int(z) for z in np.asarray(numbers).tolist()) - set(int(z) for z in model_species))
    if unknown:
        names = [chemical_symbols[z] if 0 < z < len(chemical_symbols) else str(z) for z in unknown]
        raise ConfigurationError(f"structure contains species {names} (Z={unknown}) unknown to {model} (species {sorted(model_species)})")


def validate_species_map(species_map: Mapping[int, str] | Sequence[str], model_species: Sequence[int], *, model: str = "model") -> dict[int, int]:
    """Check an engine type map against the model's species; returns ``{type_id: atomic_number}``.

    Replaces PHAL's Allegro ``type_names``/``lammps.type_map`` check (ledger MD-005): every name must
    be a chemical symbol the model was trained on, and type ids must be contiguous ``1..n``
    (a sequence is taken as types ``1..n`` in order). Names may not repeat.
    """

    mapping = {index + 1: name for index, name in enumerate(species_map)} if not isinstance(species_map, Mapping) else dict(species_map)
    if not mapping:
        raise ConfigurationError("species map is empty")
    try:
        ids = sorted(int(key) for key in mapping)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"species map keys must be integer type ids, got {list(mapping)}") from exc
    if ids != list(range(1, len(ids) + 1)):
        raise ConfigurationError(f"species map type ids must be contiguous 1..{len(ids)}, got {ids}")
    names = [str(mapping[key] if key in mapping else mapping[str(key)]) for key in ids]  # type: ignore[index]
    if len(set(names)) != len(names):
        raise ConfigurationError(f"species map repeats element names: {names}")
    unknown_symbols = [name for name in names if name not in ASE_ATOMIC_NUMBERS]
    if unknown_symbols:
        raise ConfigurationError(f"species map contains unknown element symbols {unknown_symbols}")
    numbers = {type_id: int(ASE_ATOMIC_NUMBERS[name]) for type_id, name in zip(ids, names)}
    absent = [name for name in names if ASE_ATOMIC_NUMBERS[name] not in set(int(z) for z in model_species)]
    if absent:
        raise ConfigurationError(f"species map names {absent} are absent from {model}'s species {sorted(model_species)}")
    return numbers


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
        # A private deep copy: nothing done through this backend (seeds, train overrides) leaks
        # into the caller's configuration object (PHAL's RealAllegroBackend contract).
        self.config = config.model_copy(deep=True)
        self.sgt = import_gmd_sgt(config)
        from gmd_sgt import api
        from gmd_sgt.inference.calculator import MLIPCalculator

        self.api = api
        self.calculator_cls = MLIPCalculator
        self._engine_smoke = engine_smoke
        self._calculators: dict[str, Any] = {}
        self._scripted: dict[str, Any] = {}
        self._info: dict[str, ModelInfo] = {}

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

    def train(self, dataset: DatasetVersion, *, parent: ModelArtifact | None, work_dir: Path, seed: int | None = None) -> TrainOutput:
        """Train the primary model (``seed``, default ``mlip.seed``) and ``committee_size`` extra
        members with seeds ``seed+1..``; the configuration object is never modified."""

        work_dir = Path(work_dir)
        seed = self.config.seed if seed is None else int(seed)
        primary = self._train_one(dataset, parent, work_dir / "member_0", seed)
        summary = json.loads((work_dir / "member_0" / "training_summary.json").read_text())
        committee = [
            str(self._train_one(dataset, parent, work_dir / f"member_{k}", seed + k))
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

    # ------------------------------------------------------------------ inference
    @staticmethod
    def _is_export(path: Path) -> bool:
        try:
            with zipfile.ZipFile(path) as archive:
                return any("/code/" in name for name in archive.namelist())
        except zipfile.BadZipFile as exc:
            raise AdapterError(f"{path} is not a GMD_SGT checkpoint or TorchScript export") from exc

    def model_info(self, model: str | Path) -> ModelInfo:
        """Species, cutoff and kind of a checkpoint or TorchScript export, from its own metadata."""

        path = Path(model).expanduser().resolve()
        key = str(path)
        if key in self._info:
            return self._info[key]
        if not path.is_file():
            raise AdapterError(f"model file {path} does not exist")
        if self._is_export(path):
            metadata = self.api.load_export_metadata(path) or {}
            if not metadata.get("species"):
                raise AdapterError(f"export {path} carries no species metadata (export with GMD_SGT >= 0.4)")
            info = ModelInfo(key, "export", tuple(int(z) for z in metadata["species"]), float(metadata["local_cutoff"]),
                             None if metadata.get("lr_cutoff") is None else float(metadata["lr_cutoff"]), metadata.get("version"))
        else:
            import torch
            from gmd_sgt.metadata import checkpoint_species

            try:
                checkpoint = torch.load(key, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001 - any load failure is reported explicitly
                raise AdapterError(f"cannot load GMD_SGT checkpoint {path}: {exc}") from exc
            species = checkpoint_species(checkpoint) or []
            if not species:
                raise AdapterError(f"checkpoint {path} declares no species")
            calc = self.calculator(path)
            info = ModelInfo(key, "checkpoint", tuple(int(z) for z in species), float(calc.cutoff), float(calc.lr_cutoff),
                             (checkpoint.get("metadata") or {}).get("gmd_sgt_version"))
        self._info[key] = info
        return info

    def _scripted_model(self, path: Path) -> Any:
        import torch

        key = str(path)
        if key not in self._scripted:
            self._scripted[key] = torch.jit.load(key, map_location="cpu")
        return self._scripted[key]

    def predict(
        self,
        model: str | Path,
        positions: np.ndarray,
        numbers: np.ndarray,
        cell: np.ndarray | None = None,
        pbc: Sequence[bool] = (False, False, False),
        *,
        species: Sequence[int] | None = None,
    ) -> Prediction:
        """Energy and forces of one structure with one checkpoint or TorchScript export.

        ``species`` (e.g. ``ModelArtifact.species``) overrides the model's own species list for the
        pre-evaluation species check. Wrong shapes, unknown species, model exceptions and non-finite
        outputs raise (``ConfigurationError`` / ``AdapterError``); nothing is turned into NaN.
        """

        path = Path(model).expanduser().resolve()
        positions = np.asarray(positions, dtype=float)
        numbers = np.asarray(numbers, dtype=np.int64)
        n_atoms = len(numbers)
        if positions.shape != (n_atoms, 3):
            raise AdapterError(f"positions have shape {positions.shape}, expected ({n_atoms}, 3)")
        pbc = tuple(bool(flag) for flag in pbc)
        info = self.model_info(path)
        check_species(numbers, info.species if species is None else species, model=path.name)
        try:
            if info.kind == "export":
                import torch
                from ase import Atoms

                atoms = Atoms(numbers=numbers, positions=positions, cell=cell if cell is not None else None, pbc=pbc)
                edge_index, edge_shift = _graph(atoms, info.cutoff)
                out = self._scripted_model(path)(
                    torch.as_tensor(numbers, dtype=torch.int64), torch.as_tensor(positions, dtype=torch.float32),
                    torch.as_tensor(edge_index, dtype=torch.int64), torch.as_tensor(edge_shift, dtype=torch.float32),
                )
                energy = float(out["energy"].detach().sum().item())
                forces = out["forces"].detach().cpu().numpy().astype(np.float64)
            else:
                out = self.calculator(path).compute(positions, numbers, cell if any(pbc) else None, pbc)
                energy, forces = float(out["energy"]), np.asarray(out["forces"], dtype=np.float64)
        except (AdapterError, ConfigurationError):
            raise
        except Exception as exc:  # noqa: BLE001 - a model failure is reported, never converted to NaN
            raise AdapterError(f"GMD_SGT evaluation with {path.name} failed: {type(exc).__name__}: {exc}") from exc
        if forces.shape != (n_atoms, 3):
            raise AdapterError(f"{path.name} returned forces of shape {forces.shape}, expected ({n_atoms}, 3)")
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            raise AdapterError(f"{path.name} returned non-finite energy/forces")
        return Prediction(energy, forces)

    def predict_frame(self, frame: Frame, model: str | Path, *, species: Sequence[int] | None = None) -> Prediction:
        return self.predict(model, frame.positions, frame.numbers, frame.cell, frame.pbc, species=species)

    def predict_committee(
        self,
        models: Sequence[str | Path],
        positions: np.ndarray,
        numbers: np.ndarray,
        cell: np.ndarray | None = None,
        pbc: Sequence[bool] = (False, False, False),
        *,
        species: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Committee forces with shape ``(n_models, n_atoms, 3)``; every member must succeed."""

        if len(models) < 2:
            raise AdapterError(f"a committee needs at least two members, got {len(models)}")
        stacked = np.stack([self.predict(path, positions, numbers, cell, pbc, species=species).forces for path in models])
        expected = (len(models), len(np.asarray(numbers)), 3)
        if stacked.shape != expected:  # pragma: no cover - guarded per member already
            raise AdapterError(f"committee forces have shape {stacked.shape}, expected {expected}")
        return stacked

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
        check_species(frame.numbers, model.species, model=f"model {model.model_id}")
        cell = frame.cell if any(frame.pbc) else None
        frame.committee_forces = self.predict_committee(members, frame.positions, frame.numbers, cell, frame.pbc, species=model.species)
        return frame
