"""Vacuum-padded periodic MD box for an isolated alkane chain, with a checked non-interaction guarantee.

GMD needs a fully periodic orthorhombic box. The reference calculations stay isolated (CP2K
``PERIODIC NONE`` with the WAVELET solver); only the MD runs in a box. The box is cubic with edge

    L >= D + R_force + margin + stretch_allowance

where ``D`` is the chain's diameter (largest interatomic distance) and ``R_force`` the force receptive
field of the local GMD_SGT model, ``2 x n_blocks x local_cutoff``: per-atom energies see ``n_blocks``
message-passing hops of at most ``local_cutoff`` each, and a force collects every energy that depends
on the atom, so it sees twice that far (see ``receptive_field.py``). For *any* rigid rotation or
translation of the chain, every atom of a periodic image is then at least ``L - D >= R_force + margin``
from every atom of the molecule, so periodic images cannot change any model force or energy. This is
checked twice:

* when the box is built (rotation-invariant bound and the exact minimum image distance of the start
  structure);
* on every MD frame the controller monitors (:class:`ImageGuard`): the molecule is unwrapped along its
  bonds and the exact minimum distance between any atom and any atom of another image must exceed
  ``R_force``. A violation raises :class:`ImageInteractionError`; nothing is clipped or ignored.

A model with a global long-range term (``long_range_type`` other than ``none``) has no finite receptive
field, so a periodic box is refused for it.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gmd_active_learning.contracts import Frame, ModelArtifact
from gmd_active_learning.core.exceptions import ConfigurationError, UnsupportedCombinationError
from gmd_active_learning.utils.atomic_io import write_json_atomic

LOGGER = logging.getLogger(__name__)

SYMBOLS = {1: "H", 6: "C"}


class ImageInteractionError(ConfigurationError):
    """Periodic images of the molecule could interact through the model's receptive field."""


class MDBoxConfig(BaseModel):
    """``md_box`` section of the closed-loop YAML (distances in Angstrom)."""

    model_config = ConfigDict(extra="forbid")

    margin: float = Field(4.0, ge=0.0, description="required distance beyond the force receptive field")
    stretch_allowance: float = Field(1.0, ge=0.0, description="added to the chain diameter for stretching during MD")
    box_length: float | None = Field(None, gt=0.0, description="explicit cubic edge; null = derived (rounded up to 0.5 A)")
    frame_check: bool = Field(True, description="verify every monitored MD frame (raises on a violation)")
    receptive_field_override: float | None = Field(
        None, gt=0.0, description="use this radius instead of the derived one; only larger values are accepted")


@dataclass(frozen=True, slots=True)
class ReceptiveField:
    radius: float  # force receptive field (A)
    energy_radius: float  # per-atom energy receptive field (A)
    n_blocks: int
    local_cutoff: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"force_radius": self.radius, "energy_radius": self.energy_radius, "n_blocks": self.n_blocks,
                "local_cutoff": self.local_cutoff, "source": self.source}


def force_receptive_field(model_section: Mapping[str, Any], *, override: float | None = None) -> ReceptiveField:
    """Nominal force receptive field ``2 x n_blocks x local_cutoff`` of a local GMD_SGT model."""

    long_range = str(model_section.get("long_range_type", "none")).lower()
    if long_range not in ("none", "null", ""):
        raise UnsupportedCombinationError(
            f"model long_range_type={long_range!r} couples every atom to every periodic image; a vacuum-padded periodic box "
            "cannot isolate the chain for such a model")
    try:
        n_blocks = int(model_section["n_blocks"])
        cutoff = float(model_section["local_cutoff"])
    except KeyError as exc:
        raise ConfigurationError(f"training.model needs n_blocks and local_cutoff to derive the receptive field (missing {exc})") from exc
    if n_blocks < 1 or cutoff <= 0:
        raise ConfigurationError(f"training.model: n_blocks={n_blocks}, local_cutoff={cutoff} do not define a receptive field")
    nominal = 2.0 * n_blocks * cutoff
    if override is not None:
        if override < nominal:
            raise ConfigurationError(f"md_box.receptive_field_override={override} A is below the nominal {nominal} A; refused")
        return ReceptiveField(float(override), n_blocks * cutoff, n_blocks, cutoff, f"override ({override} A >= nominal {nominal} A)")
    return ReceptiveField(nominal, n_blocks * cutoff, n_blocks, cutoff, "nominal 2 x n_blocks x local_cutoff (training.model)")


def diameter(positions: np.ndarray) -> float:
    positions = np.asarray(positions, dtype=float)
    deltas = positions[:, None, :] - positions[None, :, :]
    return float(np.sqrt(np.max(np.sum(deltas**2, axis=-1))))


def min_image_distance(positions: np.ndarray, lengths: Sequence[float]) -> float:
    """Exact minimum distance between any atom and any atom of another periodic image.

    ``positions`` must be the unwrapped (whole) molecule; ``lengths`` the orthorhombic edges.
    """

    positions = np.asarray(positions, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    if lengths.shape != (3,) or np.any(lengths <= 0):
        raise ConfigurationError(f"box lengths must be three positive numbers, got {lengths.tolist()}")
    extent = np.ptp(positions, axis=0)
    ranges = [range(-(int(np.ceil(extent[k] / lengths[k])) + 1), int(np.ceil(extent[k] / lengths[k])) + 2) for k in range(3)]
    deltas = positions[:, None, :] - positions[None, :, :]
    best = np.inf
    for shift in itertools.product(*ranges):
        if shift == (0, 0, 0):
            continue
        vector = np.asarray(shift, dtype=float) * lengths
        distances = np.sqrt(np.sum((deltas + vector) ** 2, axis=-1))
        best = min(best, float(distances.min()))
    return best


@dataclass(frozen=True, slots=True)
class BoxSpec:
    lengths: tuple[float, float, float]
    positions: np.ndarray  # centred in the box
    diameter: float
    min_image_distance: float
    required_distance: float  # R_force + margin
    receptive_field: ReceptiveField
    margin: float
    stretch_allowance: float

    def to_dict(self) -> dict[str, Any]:
        return {"lengths": list(self.lengths), "diameter": self.diameter, "min_image_distance": self.min_image_distance,
                "rotation_invariant_bound": float(min(self.lengths) - self.diameter), "required_distance": self.required_distance,
                "receptive_field": self.receptive_field.to_dict(), "margin": self.margin, "stretch_allowance": self.stretch_allowance,
                "criterion": "L - D >= R_force + margin (any rigid rotation/translation) and exact min image distance > R_force + margin",
                "boundary_conditions": {"md": "periodic orthorhombic, vacuum padded (images non-interacting for the model)",
                                        "reference": "isolated (CP2K PERIODIC NONE, WAVELET); unchanged"}}


def build_box(positions: np.ndarray, receptive_field: ReceptiveField, config: MDBoxConfig) -> BoxSpec:
    """Cubic vacuum box around the molecule; raises :class:`ImageInteractionError` when too small."""

    positions = np.asarray(positions, dtype=float)
    size = diameter(positions)
    required = receptive_field.radius + config.margin
    needed = size + required + config.stretch_allowance
    length = config.box_length if config.box_length is not None else float(np.ceil(needed * 2.0) / 2.0)
    if length - size < required:
        raise ImageInteractionError(
            f"box edge {length:.3f} A minus chain diameter {size:.3f} A = {length - size:.3f} A is below the force receptive field "
            f"{receptive_field.radius:.3f} A + margin {config.margin:.3f} A; periodic images could interact through the model")
    centred = positions - positions.mean(axis=0) + length / 2.0
    lengths = (length, length, length)
    distance = min_image_distance(centred, lengths)
    if distance <= required:
        raise ImageInteractionError(f"minimum image distance {distance:.3f} A <= {required:.3f} A (receptive field + margin)")
    return BoxSpec(lengths, centred, size, distance, required, receptive_field, config.margin, config.stretch_allowance)


def write_gmd_xyz(path: str | Path, numbers: Sequence[int], positions: np.ndarray, lengths: Sequence[float]) -> Path:
    """GMD input: atom count, ``Lx Ly Lz``, then ``Symbol x y z`` lines (atom order = parent order)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(numbers)), " ".join(f"{value:.6f}" for value in lengths)]
    for number, (x, y, z) in zip(numbers, np.asarray(positions, dtype=float)):
        if int(number) not in SYMBOLS:
            raise ConfigurationError(f"unsupported element Z={number} for the alkane MD box")
        lines.append(f"{SYMBOLS[int(number)]} {x:.10f} {y:.10f} {z:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def orthorhombic_lengths(cell: np.ndarray | None) -> np.ndarray:
    if cell is None:
        raise ImageInteractionError("MD frame has no cell; the periodic-image check needs one")
    cell = np.asarray(cell, dtype=float)
    if np.any(np.abs(cell - np.diag(np.diag(cell))) > 1.0e-8):
        raise UnsupportedCombinationError("only orthorhombic MD boxes are supported by the image check")
    return np.diag(cell).copy()


def unwrap_molecule(positions: np.ndarray, lengths: np.ndarray, bonds: Sequence[tuple[int, int]]) -> np.ndarray:
    """Make the bonded molecule whole: walk the bond graph and apply the minimum image per bond."""

    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)
    out = positions.copy()
    seen = np.zeros(n, dtype=bool)
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            i = stack.pop()
            for j in adjacency[i]:
                if seen[j]:
                    continue
                delta = positions[j] - out[i]
                delta -= lengths * np.round(delta / lengths)
                out[j] = out[i] + delta
                seen[j] = True
                stack.append(j)
    if not seen.all():  # pragma: no cover - every atom is a root of its own component above
        raise ConfigurationError("unwrap failed")
    return out


class ImageGuard:
    """``FrameEnricher`` wrapper: checks every monitored frame for periodic-image interaction, then
    delegates to the wrapped enricher (committee forces). Statistics go to ``stats_path``."""

    def __init__(self, inner: Any, *, bonds: Sequence[tuple[int, int]], receptive_field: ReceptiveField, margin: float,
                 stats_path: str | Path, enabled: bool = True, write_every: int = 50) -> None:
        self.inner = inner
        self.bonds = list(bonds)
        self.receptive_field = receptive_field
        self.margin = float(margin)
        self.stats_path = Path(stats_path)
        self.enabled = enabled
        self.write_every = max(1, int(write_every))
        existing = json.loads(self.stats_path.read_text()) if self.stats_path.is_file() else {}
        self.stats: dict[str, Any] = {"n_frames_checked": int(existing.get("n_frames_checked", 0)),
                                      "min_image_distance": existing.get("min_image_distance"),
                                      "min_at": existing.get("min_at"), "below_margin": list(existing.get("below_margin", [])),
                                      "force_receptive_field": receptive_field.radius, "margin": self.margin, "enabled": enabled}

    def check(self, frame: Frame) -> float:
        lengths = orthorhombic_lengths(frame.cell)
        whole = unwrap_molecule(frame.positions, lengths, self.bonds)
        distance = min_image_distance(whole, lengths)
        self.stats["n_frames_checked"] += 1
        if self.stats["min_image_distance"] is None or distance < self.stats["min_image_distance"]:
            self.stats["min_image_distance"] = distance
            self.stats["min_at"] = {"run_id": frame.run_id, "step": int(frame.step), "model_version": frame.model_version}
            self._write()
        if distance <= self.receptive_field.radius:
            self._write()
            raise ImageInteractionError(
                f"frame {frame.run_id}:{frame.step}: minimum image distance {distance:.3f} A <= force receptive field "
                f"{self.receptive_field.radius:.3f} A; periodic images interact through the model")
        if distance <= self.receptive_field.radius + self.margin:
            self.stats["below_margin"].append({"run_id": frame.run_id, "step": int(frame.step), "distance": distance})
            LOGGER.warning("frame %s: image distance %.3f A is inside the safety margin", frame.step, distance)
        if self.stats["n_frames_checked"] % self.write_every == 0:
            self._write()
        return distance

    def _write(self) -> None:
        write_json_atomic(self.stats_path, self.stats)

    def flush(self) -> dict[str, Any]:
        self._write()
        return dict(self.stats)

    def enrich(self, frame: Frame, model: ModelArtifact) -> Frame:
        if self.enabled:
            self.check(frame)
        return frame if self.inner is None else self.inner.enrich(frame, model)


__all__ = [
    "BoxSpec",
    "ImageGuard",
    "ImageInteractionError",
    "MDBoxConfig",
    "ReceptiveField",
    "build_box",
    "diameter",
    "force_receptive_field",
    "min_image_distance",
    "orthorhombic_lengths",
    "unwrap_molecule",
    "write_gmd_xyz",
]
