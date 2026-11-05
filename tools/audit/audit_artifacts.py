# Ported from PHAL (scripts/audit_artifacts.py) at 96263b6; MIT License, see NOTICE.
"""Scan PHAL and GMD_active artefacts for corruption signatures and integrity failures.

Data produced before the PHAL source-audit fixes may be silently wrong, and GMD_active's stores are
only trustworthy while their recorded checksums still match. Rather than discarding everything,
this reports what is actually affected, grouped by how expensive the damage is:

``structure``
    DFT ran (or would run) on a structure that is not the intended one. Not recoverable.
``label``
    The structure is right but the calculation is not. Re-run those tasks.
``integrity``
    A GMD_active artefact no longer matches its record (checksum, identity, missing file).
    Restore it from a backup or re-publish/re-register it; never train on it.
``selection``
    The wrong region was chosen, but the DFT calculation on it is still valid. Such labels remain
    usable training data; only claims about selection efficiency are void.
``unknown``
    Could not be checked automatically; inspect by hand.

PHAL checks (ported): CP2K ``.inp`` MULTIPLICITY parity against the charge-adjusted electron count;
``.extxyz``/``.xyz`` minimum interatomic distance below 0.7 A (minimum image when periodic); CP2K
``.out`` with SCF-not-converged markers; LAMMPS ``.lammpstrj``/``.dump`` BOX BOUNDS with the
``mm``/``sm``/``ms`` styles PHAL once misread as periodic; PHAL ``event.json``/``task.json`` with
``ood.lj_fit_count == 0``.

GMD_active checks (new): extxyz frame headers whose ``pbc`` is implicit (ASE then assumes fully
periodic, the extxyz counterpart of the LAMMPS boundary misparse) or periodic along a degenerate
lattice vector; model registry entries (``<registry>/<model_id>/record.json``: every stored file
re-hashed against its recorded sha256); dataset versions (``<store>/registry/<v>.json``:
``tree_sha256`` of ``versions/<v>`` recomputed); labeling tasks (``task.json`` parses into
:class:`LabelTask`, ``task_id`` matches its directory, VALID tasks have a finite, converged
``result.json`` whose recorded output checksum still matches).

Usage::

    python tools/audit/audit_artifacts.py /path/to/cp2k_tasks /path/to/datasets --json findings.json

Exit status 1 when any ``structure``, ``label`` or ``integrity`` finding exists, else 0. ``--json``
is always written (an empty list when nothing was found).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import atomic_numbers

from gmd_active_learning.contracts import DatasetVersion, LabelResult, ModelArtifact, ValidationStatus
from gmd_active_learning.contracts.ids import file_sha256, tree_sha256
from gmd_active_learning.core.exceptions import GMDActiveLearningError

LOGGER = logging.getLogger("gmd_active_learning.tools.audit")

# Physical floor: nothing real sits below this. PHAL's slab extractor produced 0.56 A contacts, and
# CP2K either fails or returns nonsense for them.
MIN_PHYSICAL_DISTANCE = 0.7

# LAMMPS boundary styles that PHAL's dump parser once failed to recognise and read as periodic.
MISPARSED_BOUNDARY_STYLES = ("mm", "sm", "ms")

# A lattice vector shorter than this cannot carry a periodic direction.
DEGENERATE_LATTICE_LENGTH = 1.0e-6

SEVERITIES = ("structure", "label", "integrity", "selection", "unknown")
FAILING_SEVERITIES = frozenset({"structure", "label", "integrity"})
SEVERITY_ADVICE = {
    "structure": "DFT ran on a structure that is not the intended one -- discard and regenerate.",
    "label": "Structure is fine but the calculation is not -- re-run these tasks only.",
    "integrity": "Artefact no longer matches its record -- restore or re-publish it; do not train on it.",
    "selection": "Labels remain valid training data; claims about selection efficiency do not.",
    "unknown": "Could not be checked automatically -- inspect by hand.",
}

_CHARGE = re.compile(r"^\s*CHARGE\s+([-+]?\d+)", re.MULTILINE)
_MULTIPLICITY = re.compile(r"^\s*MULTIPLICITY\s+(\d+)", re.MULTILINE)
_COORD_BLOCK = re.compile(r"&COORD\s*\n(.*?)&END COORD", re.DOTALL)
_SCF_FAILED = ("SCF run NOT converged", "SCF not converged")

# GMD_active on-disk formats (kept in sync with the owning modules; see the tests).
LABEL_TASK_SCHEMA = "gmd_active_learning.label_task/1"
LABEL_RESULT_FILE = "result.json"
MODEL_RECORD_FILE = "record.json"


@dataclass
class Finding:
    """One corruption signature located in one artefact."""

    path: Path
    kind: str
    severity: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "kind": self.kind, "severity": self.severity, "detail": self.detail}


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    scanned: Counter = field(default_factory=Counter)

    def add(self, path: Path, kind: str, severity: str, detail: str) -> None:
        if severity not in SEVERITIES:
            raise ValueError(f"unknown severity {severity!r}")
        LOGGER.debug("%s [%s/%s] %s", path, severity, kind, detail)
        self.findings.append(Finding(path, kind, severity, detail))

    @property
    def failed(self) -> bool:
        return any(finding.severity in FAILING_SEVERITIES for finding in self.findings)


def _iter_files(roots: Iterable[Path], suffixes: tuple[str, ...]) -> Iterator[Path]:
    for root in roots:
        if root.is_file():
            if root.suffix in suffixes:
                yield root
            continue
        for suffix in suffixes:
            yield from sorted(root.rglob(f"*{suffix}"))


def _iter_named(roots: Iterable[Path], name: str) -> Iterator[Path]:
    for root in roots:
        if root.is_file():
            if root.name == name:
                yield root
            continue
        yield from sorted(root.rglob(name))


def _load_json(path: Path, report: Report, kind: str, severity: str) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        report.add(path, kind, severity, f"unreadable: {exc}")
        return None


# ---------------------------------------------------------------------------------------------
# PHAL checks


def check_cp2k_spin_states(roots: list[Path], report: Report) -> None:
    """Odd electron counts cannot be singlets; those inputs are invalid.

    Element numbers come from ``ase.data`` (every element; PHAL's table covered ~45).
    """

    for path in _iter_files(roots, (".inp",)):
        report.scanned["cp2k_inputs"] += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        charge_match = _CHARGE.search(text)
        multiplicity_match = _MULTIPLICITY.search(text)
        coord_match = _COORD_BLOCK.search(text)
        if not (charge_match and multiplicity_match and coord_match):
            continue
        charge = int(charge_match.group(1))
        multiplicity = int(multiplicity_match.group(1))
        electrons = -charge
        unknown: set[str] = set()
        for line in coord_match.group(1).splitlines():
            parts = line.split()
            if not parts or parts[0].upper().startswith(("SCALED", "UNIT")):
                continue
            number = atomic_numbers.get(parts[0], 0)
            if number > 0:
                electrons += number
            else:
                unknown.add(parts[0])
        if unknown:
            report.add(path, "spin", "unknown", f"unrecognised element(s) {sorted(unknown)}; not checked")
            continue
        if (multiplicity - 1) % 2 != electrons % 2:
            report.add(
                path,
                "spin",
                "label",
                f"MULTIPLICITY {multiplicity} impossible for {electrons} electrons "
                f"at charge {charge:+d} (lowest consistent: {1 + electrons % 2})",
            )


def check_structure_distances(roots: list[Path], report: Report) -> None:
    """Structures containing sub-bond-length contacts never described real matter."""

    from ase.io import read

    for path in _iter_files(roots, (".extxyz", ".xyz")):
        report.scanned["structures"] += 1
        try:
            frames = read(path, index=":")
        except Exception as exc:  # noqa: BLE001 - any reader failure is a finding, not a crash
            report.add(path, "structure", "unknown", f"unreadable: {exc}")
            continue
        frames = frames if isinstance(frames, list) else [frames]
        for index, atoms in enumerate(frames):
            if len(atoms) < 2:
                continue
            use_mic = bool(atoms.pbc.any()) and atoms.cell.rank == 3
            distances = atoms.get_all_distances(mic=use_mic)
            np.fill_diagonal(distances, np.inf)
            minimum = float(distances.min())
            if minimum < MIN_PHYSICAL_DISTANCE:
                report.add(
                    path,
                    "structure",
                    "structure",
                    f"frame {index}: minimum interatomic distance {minimum:.3f} A (below {MIN_PHYSICAL_DISTANCE} A)",
                )


def check_scf_convergence(roots: list[Path], report: Report) -> None:
    """Unconverged runs were previously accepted because ENERGY| was present."""

    for path in _iter_files(roots, (".out",)):
        report.scanned["cp2k_outputs"] += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker in text for marker in _SCF_FAILED):
            report.add(path, "scf", "label", "SCF did not converge; any parsed forces are invalid")


def check_dump_boundaries(roots: list[Path], report: Report) -> None:
    """Shrink-wrapped boundaries used to be silently read as periodic by PHAL."""

    for path in _iter_files(roots, (".lammpstrj", ".dump")):
        report.scanned["dumps"] += 1
        header = ""
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("ITEM: BOX BOUNDS"):
                    header = line.strip()
                    break
        if not header:
            continue
        styles = [token for token in header.split()[3:] if len(token) == 2]
        affected = [style for style in styles if style in MISPARSED_BOUNDARY_STYLES]
        if affected:
            report.add(
                path,
                "boundary",
                "structure",
                f"boundary styles {affected} in '{header}' were parsed as periodic by PHAL before the fix; "
                "neighbour lists, all OOD metrics and extracted geometry are affected",
            )


def check_selection_metadata(roots: list[Path], report: Report) -> None:
    """A constant LJ residual means the physics stage contributed nothing (PHAL event/task metadata)."""

    for name in ("event.json", "task.json"):
        for path in _iter_named(roots, name):
            payload = _load_json(path, report, "selection", "unknown")
            if payload is None:
                continue
            if isinstance(payload, dict) and payload.get("schema") == LABEL_TASK_SCHEMA:
                continue  # a GMD_active labeling task: check_label_tasks owns it
            report.scanned["event_metadata"] += 1
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            ood = metadata.get("ood", metadata) if isinstance(metadata, dict) else {}
            if isinstance(ood, dict) and ood.get("lj_fit_count") == 0:
                report.add(path, "selection", "selection", "lj_fit_count is 0: the physics stage evaluated no atoms for this event")


# ---------------------------------------------------------------------------------------------
# GMD_active checks


def _extxyz_comment_lines(path: Path) -> Iterator[tuple[int, str]]:
    """``(frame_index, comment_line)`` for every frame of an (ext)xyz file, without parsing atoms."""

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        frame = 0
        while True:
            count_line = handle.readline()
            if not count_line:
                return
            if not count_line.strip():
                continue
            try:
                n_atoms = int(count_line.split()[0])
            except ValueError:
                return  # not a plain (ext)xyz layout; the distance check reports unreadable files
            comment = handle.readline()
            yield frame, comment
            for _ in range(n_atoms):
                if not handle.readline():
                    return
            frame += 1


def check_extxyz_boundaries(roots: list[Path], report: Report) -> None:
    """The extxyz counterpart of the LAMMPS boundary check, for GMD frame files and exports.

    A frame with ``Lattice=`` but no ``pbc=`` is read by ASE as periodic in all three directions;
    a frame periodic along a (near-)zero lattice vector has no meaningful minimum image.
    """

    from ase.io.extxyz import key_val_str_to_dict

    for path in _iter_files(roots, (".extxyz", ".xyz")):
        report.scanned["extxyz_headers"] += 1
        implicit: list[int] = []
        for frame, comment in _extxyz_comment_lines(path):
            try:
                info = key_val_str_to_dict(comment)
            except Exception:  # noqa: BLE001 - plain xyz comments are free text
                continue
            if "Lattice" not in info:
                continue
            lattice = np.asarray(info["Lattice"], dtype=float).reshape(-1)
            if "pbc" not in info:
                implicit.append(frame)
                continue
            if lattice.size != 9:
                report.add(path, "boundary", "unknown", f"frame {frame}: Lattice has {lattice.size} values")
                continue
            pbc = np.broadcast_to(np.asarray(info["pbc"], dtype=bool), 3)
            lengths = np.linalg.norm(lattice.reshape(3, 3), axis=1)
            degenerate = [axis for axis in range(3) if pbc[axis] and lengths[axis] < DEGENERATE_LATTICE_LENGTH]
            if degenerate:
                report.add(
                    path,
                    "boundary",
                    "structure",
                    f"frame {frame}: periodic along axes {degenerate} whose lattice vectors have zero length",
                )
        if implicit:
            shown = ", ".join(str(index) for index in implicit[:5]) + (" ..." if len(implicit) > 5 else "")
            report.add(
                path,
                "boundary",
                "unknown",
                f"{len(implicit)} frame(s) ({shown}) have Lattice= but no pbc=; readers assume fully periodic -- "
                "confirm the run really was periodic in all directions",
            )


def _stored_file(recorded: str, record_dir: Path) -> tuple[Path | None, bool]:
    """The file a registry record points to; falls back to ``record_dir/<name>`` after a move."""

    path = Path(recorded)
    if path.is_file():
        return path, False
    relocated = record_dir / path.name
    if relocated.is_file():
        return relocated, True
    return None, False


def check_model_registry(roots: list[Path], report: Report) -> None:
    """Every file of a registered model must still hash to the sha256 recorded at registration."""

    required = {"model_id", "checkpoint_path", "checkpoint_sha256"}
    for path in _iter_named(roots, MODEL_RECORD_FILE):
        payload = _load_json(path, report, "model_registry", "integrity")
        if not isinstance(payload, dict) or not required <= set(payload):
            continue  # some other record.json
        report.scanned["model_records"] += 1
        try:
            data = dict(payload)
            data["species"] = tuple(data.get("species", ()))
            artifact = ModelArtifact(**{key: value for key, value in data.items() if key != "state"})
        except TypeError as exc:
            report.add(path, "model_registry", "integrity", f"record does not parse as ModelArtifact: {exc}")
            continue
        record_dir = path.parent
        if record_dir.name != artifact.model_id:
            report.add(path, "model_registry", "integrity", f"model_id {artifact.model_id} does not match directory {record_dir.name}")
        files = [("checkpoint", artifact.checkpoint_path, artifact.checkpoint_sha256)]
        if artifact.export_path is not None:
            files.append(("export", artifact.export_path, artifact.export_sha256))
        for name, extra in sorted((artifact.metadata.get("extra_files") or {}).items()):
            files.append((f"extra file {name}", extra.get("path", ""), extra.get("sha256")))
        for label, recorded, digest in files:
            stored, relocated = _stored_file(str(recorded), record_dir)
            if stored is None:
                report.add(path, "model_registry", "integrity", f"model {artifact.model_id}: {label} {recorded} is missing")
                continue
            if relocated:
                LOGGER.info("model %s: %s found at %s (registry was moved)", artifact.model_id, label, stored)
            actual = file_sha256(stored)
            if actual != digest:
                report.add(
                    path,
                    "model_registry",
                    "integrity",
                    f"model {artifact.model_id}: {label} {stored} sha256 {actual} != recorded {digest} (artifact modified)",
                )


def check_dataset_versions(roots: list[Path], report: Report) -> None:
    """``tree_sha256`` of every published ``versions/<v>`` must equal ``registry/<v>.json``'s sha256."""

    seen: set[Path] = set()
    for path in _iter_files(roots, (".json",)):
        if path.parent.name != "registry" or not (path.parent.parent / "versions").is_dir() or path in seen:
            continue
        seen.add(path)
        payload = _load_json(path, report, "dataset_version", "integrity")
        if not isinstance(payload, dict) or "version" not in payload or "sha256" not in payload:
            continue
        report.scanned["dataset_versions"] += 1
        try:
            data = dict(payload)
            data["sample_ids"] = tuple(data.get("sample_ids", ()))
            record = DatasetVersion(**data)
        except TypeError as exc:
            report.add(path, "dataset_version", "integrity", f"record does not parse as DatasetVersion: {exc}")
            continue
        if path.stem != record.version:
            report.add(path, "dataset_version", "integrity", f"version {record.version} recorded in {path.name}")
        store_root = path.parent.parent
        directory = store_root / "versions" / record.version
        if not directory.is_dir():
            recorded = Path(record.path)
            directory = recorded if recorded.is_dir() else directory
        if not directory.is_dir():
            report.add(path, "dataset_version", "integrity", f"version directory for {record.version} is missing")
            continue
        actual = tree_sha256(directory)
        if actual != record.sha256:
            report.add(
                path,
                "dataset_version",
                "integrity",
                f"dataset version {record.version}: tree sha256 {actual} != recorded {record.sha256} (version modified)",
            )
        index = store_root / "index.json"
        if index.is_file():
            listed = json.loads(index.read_text(encoding="utf-8")).get("versions", [])
            if record.version not in listed:
                report.add(path, "dataset_version", "integrity", f"version {record.version} is not listed in {index}")


def _finite(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(array)))


def check_label_tasks(roots: list[Path], report: Report) -> None:
    """GMD_active labeling tasks: parseable, correctly placed, and VALID only with a sound result."""

    from gmd_active_learning.labeling.task_store import task_from_dict

    for path in _iter_named(roots, "task.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue  # check_selection_metadata already reported it
        if not isinstance(payload, dict) or payload.get("schema") != LABEL_TASK_SCHEMA:
            continue
        report.scanned["label_tasks"] += 1
        try:
            task = task_from_dict(payload)
        except (GMDActiveLearningError, KeyError, TypeError, ValueError) as exc:
            report.add(path, "label_task", "integrity", f"task.json does not parse as LabelTask: {exc!r}")
            continue
        task_dir = path.parent
        if task.task_id != task_dir.name:
            report.add(path, "label_task", "integrity", f"task_id {task.task_id} does not match directory {task_dir.name}")
        if task.validation is not ValidationStatus.VALID:
            continue
        result_path = task_dir / str(task.metadata.get("result_file") or LABEL_RESULT_FILE)
        if not result_path.is_file():
            report.add(path, "label_task", "label", f"task {task.task_id} is VALID but {result_path.name} is missing")
            continue
        result = _load_json(result_path, report, "label_result", "label")
        if not isinstance(result, dict):
            continue
        _check_label_result(task.task_id, task.region_id, result_path, result, report)


def _check_label_result(task_id: str, region_id: str, path: Path, result: dict[str, Any], report: Report) -> None:
    if result.get("task_id") != task_id or result.get("region_id", region_id) != region_id:
        report.add(path, "label_result", "integrity", f"result belongs to task {result.get('task_id')}, not {task_id}")
        return
    if result.get("validation") != ValidationStatus.VALID.value:
        report.add(path, "label_result", "integrity", f"task.json says VALID but result.json says {result.get('validation')!r}")
        return
    forces = result.get("forces")
    energy = result.get("energy")
    problems: list[str] = []
    if forces is None or np.asarray(forces, dtype=object).size == 0:
        problems.append("no forces")
    elif not _finite(forces):
        problems.append("non-finite forces")
    if energy is not None and not (isinstance(energy, (int, float)) and math.isfinite(energy)):
        problems.append(f"non-finite energy {energy!r}")
    if result.get("stress") is not None and not _finite(result["stress"]):
        problems.append("non-finite stress")
    positions = result.get("positions")
    if forces is not None and positions is not None and len(positions) != len(forces):
        problems.append(f"{len(forces)} force rows for {len(positions)} atoms")
    if not problems:
        try:
            LabelResult(
                task_id=task_id,
                region_id=region_id,
                forces=np.asarray(forces, dtype=float),
                energy=energy,
                converged=bool(result.get("converged")),
                validation=ValidationStatus.VALID,
                stress=None if result.get("stress") is None else np.asarray(result["stress"], dtype=float),
            )
        except (GMDActiveLearningError, ValueError) as exc:
            problems.append(str(exc))
    if problems:
        report.add(path, "label_result", "label", f"VALID label for task {task_id} is unusable: {'; '.join(problems)}")
        return
    digest = result.get("output_sha256")
    output_file = result.get("output_file")
    if digest and output_file:
        output = path.parent / output_file
        if not output.is_file():
            report.add(path, "label_result", "unknown", f"engine output {output_file} is gone; checksum not verified")
        elif file_sha256(output) != digest:
            report.add(path, "label_result", "integrity", f"engine output {output_file} no longer matches output_sha256")


# ---------------------------------------------------------------------------------------------
# CLI

CHECKS = (
    check_cp2k_spin_states,
    check_structure_distances,
    check_extxyz_boundaries,
    check_scf_convergence,
    check_dump_boundaries,
    check_selection_metadata,
    check_model_registry,
    check_dataset_versions,
    check_label_tasks,
)


def audit(roots: list[Path]) -> Report:
    report = Report()
    for check in CHECKS:
        check(roots, report)
    return report


def render(report: Report) -> str:
    lines = ["scanned: " + (", ".join(f"{count} {name}" for name, count in sorted(report.scanned.items())) or "nothing"), ""]
    if not report.findings:
        lines.append("No corruption signatures found.")
        return "\n".join(lines)
    by_severity: dict[str, list[Finding]] = {}
    for finding in report.findings:
        by_severity.setdefault(finding.severity, []).append(finding)
    for severity in SEVERITIES:
        items = by_severity.get(severity)
        if not items:
            continue
        lines.append(f"[{severity.upper()}] {len(items)} finding(s) -- {SEVERITY_ADVICE[severity]}")
        for finding in items:
            lines.append(f"    {finding.path}")
            lines.append(f"        {finding.kind}: {finding.detail}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="directories or files holding PHAL / GMD_active artefacts")
    parser.add_argument("--json", type=Path, default=None, help="also write the findings as JSON (always written)")
    parser.add_argument("--log-level", default="WARNING", help="logging level (default WARNING)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.WARNING), format="%(levelname)s %(message)s")

    missing = [root for root in args.roots if not root.exists()]
    if missing:
        parser.error(f"path(s) not found: {', '.join(str(item) for item in missing)}")

    report = audit(list(args.roots))
    print(render(report))
    if args.json is not None:
        # PHAL returned before this point when nothing was found, so a clean audit left no JSON behind.
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([finding.to_dict() for finding in report.findings], indent=2) + "\n", encoding="utf-8")
        print(f"findings written to {args.json}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
