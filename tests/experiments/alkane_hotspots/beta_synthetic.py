"""Synthetic artifacts with known answers for the model-study tests (part beta). Test-only: nothing here
is used by production code. Parents are real alkane geometries from alpha's geometry module; "DFT"
forces are seeded random fields; the stub model's error is a Gaussian bump at the perturbation site whose
amplitude shrinks with the number of training samples that cover a perturbation site of that family."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ase import Atoms  # noqa: E402
from ase.io import write  # noqa: E402

from tools.experiments.alkane_hotspots.geometry import AlkaneTopology, apply_bend, apply_torsion, build_all_trans  # noqa: E402

SGT_PATH = REPO_ROOT.parent / "GMD_SGT"
FAMILIES = ("normal", "torsion", "bend")
SPLITS = ("initial_train", "calibration", "validation", "final_eval")
BASE_AMPLITUDE = 1.0
WIDTH = 1.6


def import_sgt() -> Any:
    """GMD_SGT from the sibling checkout, or pytest.skip."""

    import pytest

    if str(SGT_PATH) not in sys.path and SGT_PATH.is_dir():
        sys.path.append(str(SGT_PATH))
    try:
        import gmd_sgt  # noqa: F401
        from gmd_sgt.data import validation
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"GMD_SGT not importable: {exc}")
    if not hasattr(validation, "FORCE_WEIGHT_KEY"):
        pytest.skip("GMD_SGT without force_weight support")
    return gmd_sgt


def alkane(n_carbons: int, family: str, seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    topology = AlkaneTopology(n_carbons)
    positions = build_all_trans(n_carbons)
    rng = np.random.default_rng(seed)
    positions = positions + rng.normal(0.0, 0.01, positions.shape)
    vertex = n_carbons // 2
    site: list[int] = []
    if family == "torsion":
        positions, _ = apply_torsion(positions, topology, (vertex - 1, vertex), 120.0)
        site = [vertex - 1, vertex]
    elif family == "bend":
        positions, _ = apply_bend(positions, topology, vertex, 20.0)
        site = [vertex]
    return topology.numbers, positions, site


def make_parents(root: Path, *, n_carbons: int = 8, bases_per_split: int = 1, splits: tuple[str, ...] = SPLITS) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "structures").mkdir(exist_ok=True)
    parents = []
    seed = 0
    for split in splits:
        for base in range(bases_per_split):
            group = f"{split}-b{base}"
            for family in FAMILIES:
                seed += 1
                numbers, positions, site = alkane(n_carbons, family, seed)
                parent_id = f"{group}-{family}"
                atoms = Atoms(numbers=numbers, positions=positions)
                atoms.arrays["atom_id"] = np.arange(len(numbers))
                write(root / "structures" / f"{parent_id}.extxyz", atoms, format="extxyz")
                parents.append({"parent_id": parent_id, "chain_length": n_carbons, "formula": f"C{n_carbons}H{2 * n_carbons + 2}",
                                "family": family, "seed": seed,
                                "perturbation": {"kind": family, "atoms": site, "location": n_carbons // 2, "magnitude": 1.0, "units": "deg"}
                                if site else {"kind": "none", "atoms": [], "location": None, "magnitude": 0.0, "units": None},
                                "split": split, "split_group": group, "geometry_checks": {}, "structure_path": f"structures/{parent_id}.extxyz",
                                "config_hash": "synthetic"})
    (root / "parents.jsonl").write_text("".join(json.dumps(p) + "\n" for p in parents))
    return parents


def dft_forces(parent: dict[str, Any], n_atoms: int) -> np.ndarray:
    return np.random.default_rng(1000 + int(parent["seed"])).normal(0.0, 0.1, (n_atoms, 3))


def _label(label_id: str, kind: str, parent_id: str, cost: float, forces_path: str, *, status: str = "valid", region_path: str | None = None,
           fragment: dict[str, Any] | None = None, energy: float | None = -100.0) -> dict[str, Any]:
    return {"label_id": label_id, "kind": kind, "parent_id": parent_id, "fragment": fragment, "region_path": region_path, "status": status,
            "forces_path": forces_path if status == "valid" else None, "energy_ev": energy if status == "valid" else None, "scf_steps": 12,
            "cp2k": {"version": "synthetic", "settings_hash": "s", "settings": {}},
            "cost": {"wall_s": cost / 5.0, "cpu_core_s": cost, "attempts": 1, "retries_wall_s": 0.0}, "task_id": label_id,
            "created_at": "2026-11-07T00:00:00+00:00"}


def make_labels(root: Path, parents: list[dict[str, Any]], pool: list[Any] | None = None, *, invalid_every: int = 0) -> list[dict[str, Any]]:
    """Full labels for every parent; a fragment label for every pool candidate (cost 10 core-s per atom)."""

    (root / "labels").mkdir(exist_ok=True)
    by_id = {p["parent_id"]: p for p in parents}
    labels = []
    for parent in parents:
        n = 3 * parent["chain_length"] + 2
        path = f"labels/full-{parent['parent_id']}.npy"
        np.save(root / path, dft_forces(parent, n))
        labels.append(_label(f"full-{parent['parent_id']}", "full", parent["parent_id"], 100.0 * n, path))
    for index, candidate in enumerate(pool or []):
        parent = by_id[candidate.parent_id]
        region = candidate.region
        full = dft_forces(parent, 3 * parent["chain_length"] + 2)
        forces = np.zeros((region.n_atoms, 3))
        real = region.parent_atom_ids >= 0
        forces[real] = full[region.parent_atom_ids[real]]
        label_id = f"frag-{candidate.candidate_id}"
        np.save(root / f"labels/{label_id}.npy", forces)
        (root / f"labels/{label_id}.region.json").write_text(json.dumps(region.to_dict()))
        status = "invalid" if invalid_every and index % invalid_every == invalid_every - 1 else "valid"
        labels.append(_label(label_id, "fragment", candidate.parent_id, 10.0 * region.n_atoms, f"labels/{label_id}.npy", status=status,
                             region_path=f"labels/{label_id}.region.json", energy=-50.0,
                             fragment={"policy": "graph", "hops": 2, "buffer": 0, "core_parent_ids": sorted(candidate.core_ids),
                                       "n_atoms": region.n_atoms, "n_core": len(candidate.core_ids), "n_caps": int((~real).sum())}))
    (root / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
    return labels


# ---------------------------------------------------------------------------------------------- stub model


def stub_errors(numbers: np.ndarray, positions: np.ndarray, site: list[int], amplitude: float) -> np.ndarray:
    if not site:
        return np.zeros(len(numbers))
    distance = np.min(np.linalg.norm(positions[:, None, :] - positions[None, site, :], axis=2), axis=1)
    return amplitude * np.exp(-(distance**2) / (2 * WIDTH**2))


class StubWorld:
    """Stub trainer + predictor sharing the label/parent tables (counts trainer calls)."""

    def __init__(self, parents: list[dict[str, Any]], labels: list[dict[str, Any]], root: Path):
        self.parents = {p["parent_id"]: p for p in parents}
        self.labels = {r["label_id"]: r for r in labels}
        self.root = root
        self.calls = 0
        self.by_position: dict[bytes, dict[str, Any]] = {}
        from ase.io import read

        for parent in parents:
            atoms = read(root / parent["structure_path"])
            self.by_position[np.round(atoms.get_positions(), 6).tobytes()] = parent

    def covers_site(self, label: dict[str, Any]) -> bool:
        parent = self.parents[label["parent_id"]]
        site = set(parent["perturbation"]["atoms"])
        if not site:
            return False
        if label["kind"] == "full":
            return True
        return bool(site & set(label["fragment"]["core_parent_ids"]))

    def train(self, dataset: Any, work: Path, seed: int) -> Any:
        self.calls += 1
        manifest = json.loads(Path(dataset.metadata["manifest"]).read_text())
        counts = {family: 0 for family in FAMILIES}
        for item in manifest["train"]:
            label = self.labels[item["label_id"]]
            if self.covers_site(label):
                counts[self.parents[label["parent_id"]]["family"]] += 1
        spec = json.dumps({"counts": counts, "seed": seed})
        return _Model([f"stub:{spec}:{k}" for k in range(3)])

    def predictor(self, model: Any):
        spec = json.loads(model.checkpoints[0].split(":", 1)[1].rsplit(":", 1)[0])

        def predict(numbers: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            parent = self.by_position[np.round(positions, 6).tobytes()]
            n = len(numbers)
            amplitude = BASE_AMPLITUDE / (1.0 + spec["counts"].get(parent["family"], 0))
            error = stub_errors(numbers, positions, parent["perturbation"]["atoms"], amplitude)
            direction = np.array([1.0, 0.0, 0.0])
            truth = dft_forces(parent, n)
            primary = truth + error[:, None] * direction
            noise = np.random.default_rng(7 + int(parent["seed"]))  # committee disagreement of a well-sampled region
            members = [primary + (k - 1) * 0.5 * error[:, None] * np.array([0.0, 1.0, 0.0]) + noise.normal(0.0, 0.01, (n, 3))
                       for k in range(3)]
            return primary, np.stack(members)

        return predict


class _Model:
    def __init__(self, checkpoints: list[str]):
        self.checkpoints = checkpoints

    @property
    def primary(self) -> str:
        return self.checkpoints[0]
