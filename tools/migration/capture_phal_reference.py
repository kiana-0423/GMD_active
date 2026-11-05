"""Capture golden PHAL reference outputs for migration parity tests.

Runs the PHAL reference implementation (``ref/lammps_monitor``) on deterministic
synthetic trajectories and stores, under ``tests/fixtures/phal_reference/<scenario>/``:

* ``input_frames.extxyz``: the exact frames fed to PHAL (positions, forces, cell, pbc, step, time);
* ``config.json``: the full PHAL configuration used;
* ``monitor_results.json`` + ``monitor_arrays.npz``: per-frame stage/trigger decisions and
  per-atom scores for every evaluated metric (absent metrics were not evaluated);
* ``events.json``: buffered events (pre/trigger/post frame steps, hotspot atoms, reasons);
* ``regions/<event>/<mode>.json``: extracted regions for cluster/slab/graph/block, both as
  extracted and after PHAL's labeling-time ``add_h_caps``, with original-atom mapping,
  region roles, masks and region codes, or the error PHAL raised;
* ``datasets/<event>/<mode>/``: PHAL's Allegro extxyz + metadata written for the capped region
  (with deterministic synthetic DFT forces);
* ``manifest.json``: PHAL commit, library versions, determinism patches, file hashes.

The migrated GMD_active implementation must reproduce these outputs from
``input_frames.extxyz`` + ``config.json`` without importing PHAL.

Only PHAL's ``uuid4`` event-ID suffix is patched (to a counter) for determinism.
MLIP and committee forces are precomputed and stored in ``input_frames.extxyz``
(arrays ``committee_forces_<k>``; member 0 is the primary model), so the parity
tests can replay them without any MLIP backend.

Usage (from the GMD_active repository root, while ref/lammps_monitor exists)::

    python tools/migration/capture_phal_reference.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

SEED = 20261106
MODES: tuple[str, ...] = ("cluster", "slab", "graph", "block")


# --------------------------------------------------------------------------- helpers


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _dump_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    else:
        text = json.dumps(_jsonable(payload), indent=1, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- scenarios

COMMITTEE = ("committee-a.pth", "committee-b.pth", "committee-c.pth", "committee-d.pth")
Injection = Callable[[np.ndarray, np.ndarray], list[int]]
Frame = tuple[Any, np.ndarray, np.ndarray, int]  # atoms, MD forces, committee forces (M, N, 3), step


def _harmonic_trajectory(
    base: Any,
    n_frames: int,
    *,
    step_stride: int,
    rng: np.random.Generator,
    walk_sigma: float,
    spring: float,
    noise: float,
    injections: dict[int, Injection],
    committee_sigma: float = 0.02,
    uncertain_sigma: float = 1.0,
) -> list[Frame]:
    """Random walk with harmonic MD forces plus injected anomalies.

    Committee forces follow the noise-free harmonic force with a small per-model
    spread; atoms returned by an injection get a large spread on that frame, so
    committee deviation is localized at the anomalies.
    """

    frames: list[Frame] = []
    reference = base.get_positions().copy()
    drift = np.zeros_like(reference)
    for index in range(n_frames):
        drift += rng.normal(0.0, walk_sigma, size=drift.shape)
        positions = reference + drift
        clean = -spring * drift
        forces = clean + rng.normal(0.0, noise, size=drift.shape)
        uncertain = injections[index](positions, forces) if index in injections else []
        committee = clean[None] + rng.normal(0.0, committee_sigma, size=(len(COMMITTEE), *drift.shape))
        for atom in uncertain:
            committee[:, atom] += rng.normal(0.0, uncertain_sigma, size=(len(COMMITTEE), 3))
        atoms = base.copy()
        atoms.set_positions(positions)
        atoms.wrap()
        frames.append((atoms, forces, committee, index * step_stride))
    return frames


def _burst(atom: int, vector: list[float]) -> Injection:
    def apply(_positions: np.ndarray, forces: np.ndarray) -> list[int]:
        forces[atom] += np.asarray(vector)
        return [atom]

    return apply


def _ethane_box() -> tuple[list[Frame], dict[str, Any]]:
    from ase import Atoms
    from ase.build import molecule

    rng = np.random.default_rng(SEED)
    spacing, repeats = 4.6, 3
    box = Atoms(cell=[spacing * repeats] * 3, pbc=True)
    for i, j, k in itertools.product(range(repeats), repeat=3):
        mol = molecule("C2H6")
        mol.rotate(float(rng.uniform(0.0, 360.0)), tuple(rng.normal(size=3)), center="COM")
        mol.translate(np.array([i + 0.5, j + 0.5, k + 0.5]) * spacing - mol.get_center_of_mass())
        box += mol
    box.wrap()
    positions = box.get_positions()
    hydrogens = [index for index, symbol in enumerate(box.get_chemical_symbols()) if symbol == "H"]
    corner_h = min(hydrogens, key=lambda index: float(positions[index].sum()))
    jumper = max(hydrogens, key=lambda index: float(positions[index, 2]))
    center_mol, neighbor_mol = 13, 14  # grid (1,1,1) and (1,1,2)
    center_hs = [index for index in range(center_mol * 8, center_mol * 8 + 8) if index in hydrogens]
    neighbor_cs = [index for index in range(neighbor_mol * 8, neighbor_mol * 8 + 8) if index not in hydrogens]
    center_h, neighbor_c = min(
        itertools.product(center_hs, neighbor_cs),
        key=lambda pair: float(np.linalg.norm(positions[pair[0]] - positions[pair[1]])),
    )
    second_burst = next(index for index in center_hs if index != center_h)

    def close_contact(frame_positions: np.ndarray, forces: np.ndarray) -> list[int]:
        direction = frame_positions[neighbor_c] - frame_positions[center_h]
        distance = float(np.linalg.norm(direction))
        frame_positions[center_h] = frame_positions[neighbor_c] - direction / distance * 0.85
        forces[center_h] += -direction / distance * 4.0
        return [center_h, neighbor_c]

    def boundary_jump(frame_positions: np.ndarray, forces: np.ndarray) -> list[int]:
        frame_positions[jumper, 2] += 0.9
        forces[jumper, 2] += 3.0
        return [jumper]

    injections = {
        7: _burst(corner_h, [6.0, 0.0, 0.0]),
        13: close_contact,
        14: _burst(second_burst, [0.0, -5.0, 2.0]),
        19: boundary_jump,
    }
    frames = _harmonic_trajectory(
        box, 24, step_stride=10, rng=rng, walk_sigma=0.01, spring=5.0, noise=0.05, injections=injections
    )
    overrides = {
        "online": {"monitor_freq": 5},
        "buffer": {"pre_trigger_frames": 3, "post_trigger_frames": 2, "maxlen": 16},
        "extraction": {"core_radius": 3.0, "buffer_radius": 3.0, "extract_radius": 6.0, "min_atoms": 10,
                       "slab": {"lx": 9.0, "ly": 9.0, "z_margin": 4.0},
                       "block": {"size": [6.9, 6.9, 6.9], "halo": 3.0, "min_atoms": 10, "max_atoms": 600}},
        "monitor": {"lj_cutoff": 5.0, "coordination_cutoff": 5.0},
    }
    description = {
        "system": "3x3x3 ethane molecules, periodic cubic cell 13.8 A, 216 atoms",
        "frames": 24,
        "step_stride": 10,
        "injections": {
            "frame 7": f"force burst +6 eV/A x on H {corner_h} (nearest the origin corner)",
            "frame 13": f"close contact: H {center_h} placed 0.85 A from C {neighbor_c}",
            "frame 14": f"force burst on H {second_burst} (inside the frame-13 event window)",
            "frame 19": f"H {jumper} (highest z) jumps +0.9 A in z across the periodic boundary",
        },
    }
    return frames, {"overrides": overrides, "description": description}


def _pt_slab_with_co() -> tuple[list[Frame], dict[str, Any]]:
    from ase.build import add_adsorbate, fcc111, molecule

    rng = np.random.default_rng(SEED + 1)
    slab = fcc111("Pt", size=(4, 4, 4), vacuum=8.0, periodic=True)
    slab.pbc = (True, True, False)
    add_adsorbate(slab, molecule("CO"), height=1.9, position="ontop", mol_index=0)
    slab.info.pop("adsorbate_info", None)
    carbon, oxygen = len(slab) - 2, len(slab) - 1
    surface_pt = 48  # first atom of the top Pt layer

    def co_burst(_positions: np.ndarray, forces: np.ndarray) -> list[int]:
        forces[carbon] += [0.0, 0.0, 7.0]
        forces[oxygen] += [0.0, 0.0, -7.0]
        return [carbon, oxygen]

    def pt_kick(positions: np.ndarray, forces: np.ndarray) -> list[int]:
        positions[surface_pt, 0] += 0.5
        forces[surface_pt, 0] += 4.0
        return [surface_pt]

    frames = _harmonic_trajectory(
        slab, 16, step_stride=5, rng=rng, walk_sigma=0.008, spring=8.0, noise=0.04,
        injections={6: co_burst, 11: pt_kick},
    )
    overrides = {
        "online": {"monitor_freq": 4},
        "buffer": {"pre_trigger_frames": 2, "post_trigger_frames": 2, "maxlen": 12},
        "extraction": {"core_radius": 3.0, "buffer_radius": 3.0, "extract_radius": 6.0, "min_atoms": 10,
                       "slab": {"lx": 9.0, "ly": 9.0, "z_margin": 6.0},
                       "block": {"size": [5.6, 4.9, 30.0], "halo": 3.0, "min_atoms": 10, "max_atoms": 600}},
        "monitor": {"lj_cutoff": 5.0, "coordination_cutoff": 5.0},
    }
    description = {
        "system": "Pt(111) 4x4x4 slab, 8 A vacuum, pbc=(T,T,F), CO adsorbed on top; 66 atoms",
        "frames": 16,
        "step_stride": 5,
        "injections": {
            "frame 6": f"opposing z force burst on C {carbon} / O {oxygen}",
            "frame 11": f"surface Pt {surface_pt} displaced +0.5 A in x",
        },
    }
    return frames, {"overrides": overrides, "description": description}


def _polyethylene_chains() -> tuple[list[Frame], dict[str, Any]]:
    """Infinite all-trans CH2 chains along x: extraction must cut C-C bonds, so H capping is exercised."""

    from ase import Atoms

    rng = np.random.default_rng(SEED + 2)
    units, chains, spacing = 6, 3, 4.5
    rise, zig = 1.27, 0.43  # C-C projection on x and zig-zag offset (C-C ~1.54 A)
    symbols: list[str] = []
    positions: list[list[float]] = []
    for cy, cz in itertools.product(range(chains), repeat=2):
        y0, z0 = (cy + 0.5) * spacing, (cz + 0.5) * spacing
        for carbon in range(2 * units):
            sign = 1.0 if carbon % 2 == 0 else -1.0
            x, y = carbon * rise, y0 + sign * zig
            symbols.append("C")
            positions.append([x, y, z0])
            for hz in (0.89, -0.89):
                symbols.append("H")
                positions.append([x, y + sign * 0.63, z0 + hz])
    chain_atoms = Atoms(symbols, positions=positions, cell=[2 * units * rise, chains * spacing, chains * spacing], pbc=True)
    chain_atoms.wrap()
    per_chain = 6 * units
    center = 4 * per_chain  # chain at grid (1,1)
    first_c, last_c = center + 3 * 3, center + 3 * 8

    frames = _harmonic_trajectory(
        chain_atoms, 12, step_stride=10, rng=rng, walk_sigma=0.01, spring=6.0, noise=0.05,
        injections={5: _burst(first_c, [0.0, 6.0, 0.0]), 11: _burst(last_c, [0.0, 0.0, -6.0])},
    )
    overrides = {
        "online": {"monitor_freq": 3},
        "buffer": {"pre_trigger_frames": 2, "post_trigger_frames": 3, "maxlen": 12},
        "extraction": {"core_radius": 2.5, "buffer_radius": 2.5, "extract_radius": 5.5, "min_atoms": 10,
                       "slab": {"lx": 9.0, "ly": 9.0, "z_margin": 4.0},
                       "block": {"size": [7.62, 6.75, 6.75], "halo": 3.0, "min_atoms": 10, "max_atoms": 600}},
        "monitor": {"lj_cutoff": 5.0, "coordination_cutoff": 5.0},
    }
    description = {
        "system": "3x3 infinite all-trans polyethylene chains along x, cell 15.24 x 13.5 x 13.5 A, 324 atoms",
        "frames": 12,
        "step_stride": 10,
        "injections": {
            "frame 5": f"force burst +6 eV/A y on backbone C {first_c}",
            "frame 11": f"force burst -6 eV/A z on backbone C {last_c} on the final frame (post-context cut off)",
        },
    }
    return frames, {"overrides": overrides, "description": description}


SCENARIOS: dict[str, Callable[[], tuple[list[Frame], dict[str, Any]]]] = {
    "ethane_box": _ethane_box,
    "polyethylene_chains": _polyethylene_chains,
    "pt111_co_slab": _pt_slab_with_co,
}


def committee_evaluator(committee_by_step: dict[int, np.ndarray]) -> Callable[..., np.ndarray]:
    """PHAL ``force_evaluator`` replaying precomputed committee forces by frame step and model name."""

    def evaluate(atoms: Any, model_path: Any, _config: dict[str, Any]) -> np.ndarray:
        member = COMMITTEE.index(Path(str(model_path)).name) if model_path is not None else 0
        return committee_by_step[int(atoms.info["step"])][member].copy()

    return evaluate


# --------------------------------------------------------------------------- capture


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _install_phal(ref_root: Path) -> None:
    for entry in (ref_root / "src",):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import hotspot_al.buffer.rolling_buffer as rolling_buffer

    counter = itertools.count()

    class _DeterministicUUID:
        def __init__(self, value: int) -> None:
            self.hex = f"{value:08x}" + "0" * 24

    rolling_buffer.uuid4 = lambda: _DeterministicUUID(next(counter))  # type: ignore[assignment]


def capture_scenario(name: str, out_root: Path, ref_root: Path) -> list[Path]:
    _install_phal(ref_root)
    from ase.io import write

    from hotspot_al.active_learning.workflow import extract_regions_for_result
    from hotspot_al.backends.allegro_runner import AllegroRunner
    from hotspot_al.config import load_config
    from hotspot_al.extraction.h_capping import add_h_caps
    from hotspot_al.models import FrameData
    from hotspot_al.monitor.online_monitor import OnlineMonitor
    from hotspot_al.training.allegro_adapter import write_allegro_dataset
    from hotspot_al.training.mask_generator import build_atom_mask, generate_region_labels
    from hotspot_al.training.region_codes import region_codes_for_labels

    frames, scenario = SCENARIOS[name]()
    out = out_root / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    config = load_config()
    config["allegro"]["model_paths"] = list(COMMITTEE)
    _deep_update(config, scenario["overrides"])
    config["online"]["event_dir"] = str(out / "_scratch_events")
    config["online"]["work_dir"] = str(out / "_scratch_work")
    config["logging"] = {**config.get("logging", {}), "level": "WARNING", "file": None}

    timestep = float(config["lammps"]["timestep_fs"])
    frame_data = []
    inputs = []
    for atoms, forces, committee, step in frames:
        atoms.info["step"] = step
        atoms.info["time"] = step * timestep
        frame_data.append(FrameData(atoms=atoms, step=step, time=step * timestep, forces=forces))
        stored = atoms.copy()
        stored.arrays["forces"] = np.asarray(forces, dtype=float)
        for member, member_forces in enumerate(committee):
            stored.arrays[f"committee_forces_{member}"] = member_forces
        inputs.append(stored)
    write(out / "input_frames.extxyz", inputs, format="extxyz")

    events: list[Any] = []
    monitor = OnlineMonitor(
        config=config,
        runner=AllegroRunner(force_evaluator=committee_evaluator({step: committee for _, _, committee, step in frames})),
        frame_source=frame_data,
        on_event=events.append,
    )
    results = monitor.run()

    arrays: dict[str, np.ndarray] = {}
    records = []
    for index, (frame, result) in enumerate(zip(frame_data, results)):
        arrays[f"frame{index:03d}/atom_scores"] = np.asarray(result.atom_scores, dtype=float)
        for metric, values in sorted(result.metric_scores.items()):
            arrays[f"frame{index:03d}/metric/{metric}"] = np.asarray(values, dtype=float)
        records.append({
            "index": index,
            "step": frame.step,
            "stage": result.stage,
            "triggered": result.triggered,
            "max_score": result.max_score,
            "hotspot_indices": list(result.hotspot_indices),
            "trigger_reason": list(result.trigger_reason),
            "evaluated_metrics": sorted(result.metric_scores),
            "metadata": result.metadata,
        })
    np.savez_compressed(out / "monitor_arrays.npz", **arrays)
    _dump_json(out / "monitor_results.json", records, compact=True)

    event_records = []
    result_by_step = {frame.step: result for frame, result in zip(frame_data, results)}
    for event in events:
        state = event.__getstate__()
        event_records.append({
            "event_id": event.event_id,
            "step": event.step,
            "time": event.time,
            "pre_frame_steps": [frame.step for frame in event.pre_frames],
            "post_frame_steps": [frame.step for frame in event.post_frames],
            "hotspot_atoms": list(event.hotspot_atoms),
            "trigger_reason": list(event.trigger_reason),
            "ood_scores": state["ood_scores"],
            "backend": event.backend,
            "model_version": event.model_version,
            "metadata": event.metadata,
        })
    _dump_json(out / "events.json", event_records)

    labels_rng = np.random.default_rng(SEED + 2)
    for event in events:
        result = result_by_step[event.step]
        result.metadata = {**result.metadata, "event_id": event.event_id, "backend": "fake_allegro"}
        for mode in MODES:
            mode_config = json.loads(json.dumps(_jsonable(config)))
            mode_config["extraction"]["mode"] = mode
            target = out / "regions" / str(event.event_id) / f"{mode}.json"
            try:
                regions = extract_regions_for_result(event.trigger_frame, result, config=mode_config)
            except Exception as exc:  # noqa: BLE001 - the failure itself is reference behaviour
                _dump_json(target, {"error": {"type": type(exc).__name__, "message": str(exc)}})
                continue
            payload = []
            for region_index, region in enumerate(regions):
                raw = copy.deepcopy(region)
                # PHAL caps regions at labeling time (CP2KTaskSubmitter), not during extraction.
                try:
                    capped = add_h_caps(event.trigger_frame.atoms, region, config=mode_config)
                    capping_error = None
                except Exception as exc:  # noqa: BLE001 - the failure itself is reference behaviour
                    capped, capping_error = None, {"type": type(exc).__name__, "message": str(exc)}
                record: dict[str, Any] = {"raw_region": raw.__getstate__(), "capping_error": capping_error}
                if capped is not None:
                    labels = generate_region_labels(capped)
                    record.update({
                        "capped_region": capped.__getstate__(),
                        "mask": build_atom_mask(capped, mode_config),
                        "region_labels": labels,
                        "region_codes": region_codes_for_labels(labels),
                    })
                    dft_forces = labels_rng.normal(0.0, 0.5, size=(len(capped.atoms), 3))
                    dataset_dir = out / "datasets" / str(event.event_id) / mode
                    filename = f"region{region_index:03d}.extxyz"
                    try:
                        write_allegro_dataset(
                            capped, forces=dft_forces, output_dir=dataset_dir, config=mode_config, filename=filename
                        )
                        record["dataset_error"] = None
                    except Exception as exc:  # noqa: BLE001 - the failure itself is reference behaviour
                        # ASE may leave a partial file behind; the reference outcome is the error.
                        (dataset_dir / filename).unlink(missing_ok=True)
                        record["dataset_error"] = {"type": type(exc).__name__, "message": str(exc)}
                payload.append(record)
            _dump_json(target, payload, compact=True)

    for scratch in ("_scratch_events", "_scratch_work"):
        shutil.rmtree(out / scratch, ignore_errors=True)
    _dump_json(out / "config.json", config | {"online": {**config["online"], "event_dir": None, "work_dir": None}})
    _dump_json(out / "scenario.json", {"name": name, "seed": SEED, **scenario})
    return sorted(path for path in out.rglob("*") if path.is_file())


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_all(out_root: Path, ref_root: Path, scenarios: list[str]) -> Path:
    import ase
    import scipy

    files: dict[str, str] = {}
    for name in scenarios:
        for path in capture_scenario(name, out_root, ref_root):
            files[path.relative_to(out_root).as_posix()] = _sha256(path)
    manifest = {
        "phal_commit": _git_head(ref_root),
        "capture_script_sha256": _sha256(Path(__file__)),
        "seed": SEED,
        "scenarios": scenarios,
        "determinism_patches": ["hotspot_al.buffer.rolling_buffer.uuid4 -> counter"],
        "mlip_backend": "precomputed committee forces replayed by step (input_frames.extxyz arrays committee_forces_<k>; model 0 is the primary model)",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "ase": ase.__version__,
        },
        "files": files,
    }
    _dump_json(out_root / "manifest.json", manifest)
    return out_root / "manifest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture PHAL golden outputs for parity tests.")
    parser.add_argument("--ref", type=Path, default=Path("ref/lammps_monitor"))
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/phal_reference"))
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS), default=None)
    args = parser.parse_args(argv)
    if not (args.ref / "src" / "hotspot_al").is_dir():
        parser.error(f"PHAL reference tree not found under {args.ref}")
    manifest = capture_all(args.out.resolve(), args.ref.resolve(), args.scenario or sorted(SCENARIOS))
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
