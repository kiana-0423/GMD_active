# Ported from PHAL (tests/unit/test_audit_artifacts.py) at 96263b6; MIT License, see NOTICE.
"""Artefact audit: PHAL corruption signatures plus GMD_active registry/dataset/label integrity.

The scanner decides whether expensive DFT data must be regenerated, so a false negative is costly
in exactly the way the original bugs were: it looks fine and is not. Each signature gets a positive
test and clean artefacts get a non-regression test. The GMD_active checks run against stores built
with the real GMD_active classes, which are then tampered with.
"""

from __future__ import annotations

import json
import stat
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from gmd_active_learning.contracts import ExecutionStatus, LabelTask, ValidationStatus
from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.datasets import VersionedDatasetStore
from gmd_active_learning.labeling import task_store
from gmd_active_learning.labeling.service import RESULT_FILE
from gmd_active_learning.retraining.registry import ModelRegistry
from gmd_active_learning.workflow.store import WorkflowStore

from datasets.helpers import sample_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCANNER_PATH = PROJECT_ROOT / "tools" / "audit" / "audit_artifacts.py"


def _load_scanner() -> ModuleType:
    spec = spec_from_file_location("gmd_audit_artifacts", SCANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_scanner()


# -- fixture builders -------------------------------------------------------


def _cp2k_input(path: Path, *, charge: int, multiplicity: int, symbols: list[str]) -> Path:
    coords = "\n".join(f"    {symbol}  {index * 1.1:.2f} 0.0 0.0" for index, symbol in enumerate(symbols))
    path.write_text(
        "&FORCE_EVAL\n  &DFT\n"
        f"    CHARGE {charge}\n    MULTIPLICITY {multiplicity}\n"
        "  &END DFT\n  &SUBSYS\n    &COORD\n"
        f"{coords}\n"
        "    &END COORD\n  &END SUBSYS\n&END FORCE_EVAL\n",
        encoding="utf-8",
    )
    return path


def _cp2k_output(path: Path, *, converged: bool) -> Path:
    marker = "*** SCF run converged in 11 steps ***" if converged else "*** SCF run NOT converged ***"
    path.write_text(f" {marker}\n ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:  -17.123456\n", encoding="utf-8")
    return path


def _dump(path: Path, *, bounds: str) -> Path:
    path.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        f"ITEM: BOX BOUNDS {bounds}\n0 10\n0 10\n0 30\n"
        "ITEM: ATOMS id type x y z\n1 1 1.0 1.0 1.0\n",
        encoding="utf-8",
    )
    return path


def _severities(report) -> set[str]:
    return {finding.severity for finding in report.findings}


def _scan(*checks, roots: list[Path]):
    report = audit.Report()
    for check in checks:
        check(roots, report)
    return report


def _make_writable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IWUSR)


# -- PHAL: positive detections ------------------------------------------------


def test_sub_bond_length_contact_is_structural(tmp_path: Path) -> None:
    atoms = Atoms("Cu3", positions=[[0, 0, 0], [0.56, 0, 0], [3, 0, 0]], cell=[20.0] * 3, pbc=False)
    write(tmp_path / "region.extxyz", atoms, format="extxyz")

    report = _scan(audit.check_structure_distances, roots=[tmp_path])

    assert _severities(report) == {"structure"}
    assert "0.560" in report.findings[0].detail


def test_close_contact_across_the_periodic_boundary_uses_minimum_image(tmp_path: Path) -> None:
    atoms = Atoms("H2", positions=[[0.1, 5, 5], [9.6, 5, 5]], cell=[10.0] * 3, pbc=True)
    write(tmp_path / "periodic.extxyz", atoms, format="extxyz")

    report = _scan(audit.check_structure_distances, roots=[tmp_path])

    assert _severities(report) == {"structure"}
    assert "0.500" in report.findings[0].detail


def test_shrink_wrapped_boundary_is_structural(tmp_path: Path) -> None:
    _dump(tmp_path / "traj.lammpstrj", bounds="pp pp mm")

    report = _scan(audit.check_dump_boundaries, roots=[tmp_path])

    assert _severities(report) == {"structure"}
    assert "mm" in report.findings[0].detail


@pytest.mark.parametrize("bounds", ["pp pp mm", "pp sm ff", "ms pp pp"])
def test_every_misparsed_boundary_style_is_caught(tmp_path: Path, bounds: str) -> None:
    _dump(tmp_path / "traj.lammpstrj", bounds=bounds)

    report = _scan(audit.check_dump_boundaries, roots=[tmp_path])

    assert report.findings, f"boundary {bounds!r} went undetected"


@pytest.mark.parametrize("suffix", [".lammpstrj", ".dump"])
def test_both_dump_suffixes_are_scanned(tmp_path: Path, suffix: str) -> None:
    _dump(tmp_path / f"traj{suffix}", bounds="pp pp sm")

    assert _scan(audit.check_dump_boundaries, roots=[tmp_path]).findings


def test_impossible_multiplicity_is_a_label_failure(tmp_path: Path) -> None:
    # OH carries 9 electrons and cannot be a singlet.
    _cp2k_input(tmp_path / "task_sp.inp", charge=0, multiplicity=1, symbols=["O", "H"])

    report = _scan(audit.check_cp2k_spin_states, roots=[tmp_path])

    assert _severities(report) == {"label"}
    assert "9 electrons" in report.findings[0].detail


def test_charge_is_included_in_the_electron_count(tmp_path: Path) -> None:
    # OH- carries 10 electrons, so a singlet is correct here.
    _cp2k_input(tmp_path / "anion_sp.inp", charge=-1, multiplicity=1, symbols=["O", "H"])

    assert _scan(audit.check_cp2k_spin_states, roots=[tmp_path]).findings == []


@pytest.mark.parametrize(
    ("symbols", "multiplicity", "flagged"),
    [
        (["Rh", "H"], 1, False),  # 46 electrons; Rh was missing from PHAL's ~45-element table
        (["Pd", "H"], 1, True),  # 47 electrons
        (["W", "Hf"], 1, False),  # 146 electrons
        (["U"], 1, False),  # 92 electrons
        (["Ir"], 2, False),  # 77 electrons -> doublet is consistent
    ],
)
def test_elements_beyond_phal_table_are_counted(tmp_path: Path, symbols: list[str], multiplicity: int, flagged: bool) -> None:
    _cp2k_input(tmp_path / "heavy_sp.inp", charge=0, multiplicity=multiplicity, symbols=symbols)

    report = _scan(audit.check_cp2k_spin_states, roots=[tmp_path])

    assert _severities(report) == ({"label"} if flagged else set())


def test_unconverged_scf_is_a_label_failure(tmp_path: Path) -> None:
    _cp2k_output(tmp_path / "task_sp.out", converged=False)

    report = _scan(audit.check_scf_convergence, roots=[tmp_path])

    assert _severities(report) == {"label"}


def test_empty_lj_evaluation_is_a_selection_finding(tmp_path: Path) -> None:
    (tmp_path / "event.json").write_text(json.dumps({"metadata": {"ood": {"lj_fit_count": 0}}}), encoding="utf-8")

    report = _scan(audit.check_selection_metadata, roots=[tmp_path])

    assert _severities(report) == {"selection"}


@pytest.mark.parametrize("symbol", ["Uuo", "X", "Qq"])
def test_unrecognised_elements_are_flagged_rather_than_miscounted(tmp_path: Path, symbol: str) -> None:
    _cp2k_input(tmp_path / "exotic_sp.inp", charge=0, multiplicity=1, symbols=[symbol, "H"])

    report = _scan(audit.check_cp2k_spin_states, roots=[tmp_path])

    assert _severities(report) == {"unknown"}


# -- PHAL: non-regression on clean artefacts ------------------------------------


def test_clean_artefacts_produce_no_findings(tmp_path: Path) -> None:
    _cp2k_input(tmp_path / "water_sp.inp", charge=0, multiplicity=1, symbols=["O", "H", "H"])
    _cp2k_output(tmp_path / "water_sp.out", converged=True)
    _dump(tmp_path / "traj.lammpstrj", bounds="pp pp pp")
    write(
        tmp_path / "region.extxyz",
        Atoms("Cu2", positions=[[0, 0, 0], [2.55, 0, 0]], cell=[20.0] * 3, pbc=False),
        format="extxyz",
    )
    (tmp_path / "event.json").write_text(json.dumps({"metadata": {"ood": {"lj_fit_count": 12}}}), encoding="utf-8")

    assert audit.main([str(tmp_path)]) == 0


def test_repository_fixtures_report_only_their_deliberate_defects() -> None:
    """GMD_active's fixtures contain two deliberate defects; anything else would be a false positive."""

    report = audit.audit([PROJECT_ROOT / "tests" / "fixtures"])

    found = sorted((Path(f.path).relative_to(PROJECT_ROOT).as_posix(), f.kind, f.severity) for f in report.findings)
    assert found == [
        ("tests/fixtures/cp2k/real_2026.2/h2_unconverged/sp.out", "scf", "label"),
        # ethane_box injects "close contact: H 107 placed 0.85 A from C 113" at frame 13 (scenario.json).
        ("tests/fixtures/phal_reference/ethane_box/input_frames.extxyz", "structure", "structure"),
    ]
    assert report.scanned["dumps"] >= 1 and report.scanned["cp2k_inputs"] >= 1


# -- exit codes and JSON output ---------------------------------------------------


def test_unrecoverable_findings_set_a_failing_exit_code(tmp_path: Path) -> None:
    write(tmp_path / "region.extxyz", Atoms("Cu2", positions=[[0, 0, 0], [0.4, 0, 0]], cell=[20.0] * 3, pbc=False), format="extxyz")

    assert audit.main([str(tmp_path)]) == 1


def test_selection_only_findings_do_not_fail_the_run(tmp_path: Path) -> None:
    """Selection findings leave labels usable, so they must not read as fatal."""

    (tmp_path / "event.json").write_text(json.dumps({"metadata": {"ood": {"lj_fit_count": 0}}}), encoding="utf-8")

    assert audit.main([str(tmp_path)]) == 0


def test_json_output_is_stable_and_parseable(tmp_path: Path) -> None:
    _cp2k_input(tmp_path / "task_sp.inp", charge=0, multiplicity=1, symbols=["O", "H"])
    _cp2k_output(tmp_path / "task_sp.out", converged=False)
    findings_path = tmp_path / "findings.json"

    audit.main([str(tmp_path), "--json", str(findings_path)])

    payload = json.loads(findings_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and payload
    for entry in payload:
        assert set(entry) == {"path", "kind", "severity", "detail"}
        assert entry["severity"] in set(audit.SEVERITIES)

    # Re-running must produce identical content, so findings can be diffed.
    audit.main([str(tmp_path), "--json", str(findings_path)])
    assert json.loads(findings_path.read_text(encoding="utf-8")) == payload


def test_json_is_written_even_without_findings(tmp_path: Path) -> None:
    """PHAL returned early on a clean audit and never wrote --json (OPS-001 notes)."""

    clean = tmp_path / "clean"
    clean.mkdir()
    _cp2k_output(clean / "ok.out", converged=True)
    findings_path = tmp_path / "out" / "findings.json"

    assert audit.main([str(clean), "--json", str(findings_path)]) == 0
    assert json.loads(findings_path.read_text(encoding="utf-8")) == []


def test_missing_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        audit.main([str(tmp_path / "does_not_exist")])


def test_report_groups_by_severity_in_cost_order(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "event.json").write_text(json.dumps({"metadata": {"ood": {"lj_fit_count": 0}}}), encoding="utf-8")
    _cp2k_output(tmp_path / "bad.out", converged=False)
    write(tmp_path / "close.extxyz", Atoms("H2", positions=[[0, 0, 0], [0.3, 0, 0]]), format="extxyz")

    assert audit.main([str(tmp_path)]) == 1

    out = capsys.readouterr().out
    assert out.index("[STRUCTURE]") < out.index("[LABEL]") < out.index("[SELECTION]")


# -- GMD_active: extxyz boundary counterpart ----------------------------------------


def test_extxyz_without_pbc_key_is_flagged_as_implicitly_periodic(tmp_path: Path) -> None:
    (tmp_path / "frames.extxyz").write_text(
        '2\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 format=gmd-frame-v1\n'
        "H 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )

    report = _scan(audit.check_extxyz_boundaries, roots=[tmp_path])

    assert _severities(report) == {"unknown"}
    assert "no pbc=" in report.findings[0].detail


def test_extxyz_periodic_along_a_zero_lattice_vector_is_structural(tmp_path: Path) -> None:
    (tmp_path / "frames.extxyz").write_text(
        '2\nLattice="10 0 0 0 10 0 0 0 0" Properties=species:S:1:pos:R:3 pbc="T T T"\nH 0 0 0\nH 0 0 1\n',
        encoding="utf-8",
    )

    report = _scan(audit.check_extxyz_boundaries, roots=[tmp_path])

    assert _severities(report) == {"structure"}
    assert "[2]" in report.findings[0].detail


def test_ase_written_slab_with_explicit_pbc_is_clean(tmp_path: Path) -> None:
    slab = Atoms("H2", positions=[[1, 1, 5], [1, 1, 6]], cell=[10.0, 10.0, 30.0], pbc=[True, True, False])
    write(tmp_path / "slab.extxyz", [slab, slab], format="extxyz")
    write(tmp_path / "molecule.xyz", Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]), format="xyz")

    assert _scan(audit.check_extxyz_boundaries, roots=[tmp_path]).findings == []


# -- GMD_active: model registry ---------------------------------------------------


def _build_registry(tmp_path: Path) -> tuple[ModelRegistry, str]:
    sources = tmp_path / "trained"
    sources.mkdir()
    (sources / "model.pt").write_bytes(b"checkpoint-bytes")
    (sources / "model.ts").write_bytes(b"torchscript-bytes")
    (sources / "member1.pt").write_bytes(b"committee-member")
    registry = ModelRegistry(tmp_path / "models", WorkflowStore(tmp_path / "workflow.sqlite"))
    artifact = registry.register(
        checkpoint=sources / "model.pt",
        export=sources / "model.ts",
        dataset_version=None,
        parent_model_id=None,
        species=[1, 6],
        cutoff=5.0,
        metrics={"force_rmse": 0.1},
        train_mode="finetune",
        backend="test",
        extra_files={"member1": sources / "member1.pt"},
    )
    return registry, artifact.model_id


def test_intact_model_registry_is_clean(tmp_path: Path) -> None:
    _unused, model_id = _build_registry(tmp_path)

    report = _scan(audit.check_model_registry, roots=[tmp_path / "models"])

    assert report.findings == []
    assert report.scanned["model_records"] == 1


@pytest.mark.parametrize("target", ["checkpoint.pt", "export.ts", "extra_member1.pt"])
def test_tampered_model_file_is_an_integrity_failure(tmp_path: Path, target: str) -> None:
    registry, model_id = _build_registry(tmp_path)
    tampered = tmp_path / "models" / model_id / target
    _make_writable(tampered)
    tampered.write_bytes(b"tampered")

    report = _scan(audit.check_model_registry, roots=[tmp_path])

    assert _severities(report) == {"integrity"}
    assert len(report.findings) == 1 and "sha256" in report.findings[0].detail
    with pytest.raises(Exception, match="checksum"):  # the registry itself agrees
        registry.verify(model_id)
    assert audit.main([str(tmp_path / "models")]) == 1


def test_missing_model_file_is_an_integrity_failure(tmp_path: Path) -> None:
    _unused, model_id = _build_registry(tmp_path)
    (tmp_path / "models" / model_id / "export.ts").unlink()

    report = _scan(audit.check_model_registry, roots=[tmp_path / "models"])

    assert _severities(report) == {"integrity"}
    assert "missing" in report.findings[0].detail


def test_relocated_model_registry_is_still_verified(tmp_path: Path) -> None:
    import shutil

    _unused, model_id = _build_registry(tmp_path)
    moved = tmp_path / "archive" / "models"
    shutil.copytree(tmp_path / "models", moved)
    shutil.rmtree(tmp_path / "models")  # recorded absolute paths now dangle

    assert _scan(audit.check_model_registry, roots=[moved]).findings == []
    stored = moved / model_id / "checkpoint.pt"
    _make_writable(stored)
    stored.write_bytes(b"different")
    assert _severities(_scan(audit.check_model_registry, roots=[moved])) == {"integrity"}


# -- GMD_active: dataset versions -------------------------------------------------


def _dataset_store(tmp_path: Path) -> tuple[VersionedDatasetStore, str]:
    store = VersionedDatasetStore(tmp_path / "store")
    samples = [sample_for(step=index, event_id=f"evt-{index}", task=f"task-{index}") for index in range(4)]
    version = store.publish([store.write_sample(sample) for sample in samples], parent=None, split_groups={})
    return store, version.version


def test_intact_dataset_store_is_clean(tmp_path: Path) -> None:
    _store, _version = _dataset_store(tmp_path)

    report = audit.audit([tmp_path / "store"])

    assert report.findings == []
    assert report.scanned["dataset_versions"] == 1


@pytest.mark.parametrize("target", ["train.extxyz", "samples.npz", "split.json"])
def test_tampered_dataset_version_is_an_integrity_failure(tmp_path: Path, target: str) -> None:
    store, version = _dataset_store(tmp_path)
    tampered = tmp_path / "store" / "versions" / version / target
    _make_writable(tampered)
    with tampered.open("ab") as handle:
        handle.write(b"\n")

    report = _scan(audit.check_dataset_versions, roots=[tmp_path / "store"])

    assert _severities(report) == {"integrity"}
    assert version in report.findings[0].detail
    with pytest.raises(Exception, match="modified"):  # the store's own verification agrees
        store.get(version)


def test_extra_file_in_a_dataset_version_is_detected(tmp_path: Path) -> None:
    _store, version = _dataset_store(tmp_path)
    directory = tmp_path / "store" / "versions" / version
    (directory / "notes.txt").write_text("added later\n", encoding="utf-8")

    assert _severities(_scan(audit.check_dataset_versions, roots=[tmp_path])) == {"integrity"}


def test_missing_dataset_version_directory_is_detected(tmp_path: Path) -> None:
    import shutil

    _store, version = _dataset_store(tmp_path)
    shutil.rmtree(tmp_path / "store" / "versions" / version)

    report = _scan(audit.check_dataset_versions, roots=[tmp_path / "store"])

    assert _severities(report) == {"integrity"} and "missing" in report.findings[0].detail


# -- GMD_active: labeling tasks ---------------------------------------------------


def _label_task(root: Path, *, task_id: str = "label-0123456789abcdef", validation: ValidationStatus = ValidationStatus.VALID) -> Path:
    work_dir = root / task_id
    attempt_dir = work_dir / "attempts" / "00-sp"
    attempt_dir.mkdir(parents=True)
    output = attempt_dir / "sp.out"
    output.write_text(" *** SCF run converged in 9 steps ***\n", encoding="utf-8")
    task = LabelTask(
        task_id=task_id,
        region_id="region-1",
        engine="cp2k",
        settings_hash="settings",
        work_dir=str(work_dir),
        attempt=1,
        execution=ExecutionStatus.COMPLETED,
        validation=validation,
        metadata={"stage": "done", "result_file": RESULT_FILE if validation is ValidationStatus.VALID else None},
    )
    task_store.save_task(task)
    if validation is ValidationStatus.VALID:
        payload = {
            "task_id": task_id,
            "region_id": "region-1",
            "validation": "valid",
            "validation_reasons": [],
            "converged": True,
            "energy": -31.25,
            "forces": [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
            "stress": None,
            "scf_steps": 9,
            "engine": "cp2k",
            "engine_version": "2026.2",
            "output_file": "attempts/00-sp/sp.out",
            "output_sha256": file_sha256(output),
            "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        }
        (work_dir / RESULT_FILE).write_text(json.dumps(payload), encoding="utf-8")
    return work_dir


def test_valid_label_task_is_clean_and_round_trips_through_the_real_task_store(tmp_path: Path) -> None:
    work_dir = _label_task(tmp_path / "tasks")
    assert task_store.load_task(work_dir).validation is ValidationStatus.VALID
    _label_task(tmp_path / "tasks", task_id="label-pending", validation=ValidationStatus.PENDING)

    report = audit.audit([tmp_path / "tasks"])

    assert report.findings == []
    assert report.scanned["label_tasks"] == 2
    assert report.scanned["event_metadata"] == 0  # GMD tasks are not mistaken for PHAL task metadata


def test_task_in_the_wrong_directory_is_an_integrity_failure(tmp_path: Path) -> None:
    import shutil

    work_dir = _label_task(tmp_path / "tasks")
    shutil.move(work_dir, tmp_path / "tasks" / "label-other")

    report = _scan(audit.check_label_tasks, roots=[tmp_path / "tasks"])

    assert _severities(report) == {"integrity"} and "does not match directory" in report.findings[0].detail


def test_corrupt_task_json_is_an_integrity_failure(tmp_path: Path) -> None:
    work_dir = _label_task(tmp_path / "tasks")
    data = json.loads((work_dir / "task.json").read_text(encoding="utf-8"))
    data["execution"] = "exploded"
    (work_dir / "task.json").write_text(json.dumps(data), encoding="utf-8")

    assert _severities(_scan(audit.check_label_tasks, roots=[tmp_path])) == {"integrity"}


def test_valid_task_without_result_is_a_label_failure(tmp_path: Path) -> None:
    work_dir = _label_task(tmp_path / "tasks")
    (work_dir / RESULT_FILE).unlink()

    report = _scan(audit.check_label_tasks, roots=[tmp_path])

    assert _severities(report) == {"label"} and "missing" in report.findings[0].detail
    assert audit.main([str(tmp_path)]) == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"forces": [[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]}, "non-finite forces"),
        ({"energy": float("inf")}, "non-finite energy"),
        ({"forces": None}, "no forces"),
        ({"forces": [[0.0, 0.0, 0.0]]}, "1 force rows for 2 atoms"),
        ({"converged": False}, "unconverged"),
    ],
)
def test_unusable_valid_result_is_a_label_failure(tmp_path: Path, change: dict, message: str) -> None:
    work_dir = _label_task(tmp_path / "tasks")
    result = json.loads((work_dir / RESULT_FILE).read_text(encoding="utf-8"))
    result.update(change)
    (work_dir / RESULT_FILE).write_text(json.dumps(result), encoding="utf-8")  # json writes NaN/Infinity tokens

    report = _scan(audit.check_label_tasks, roots=[tmp_path])

    assert _severities(report) == {"label"}
    assert message in report.findings[0].detail


def test_result_disagreeing_with_task_is_an_integrity_failure(tmp_path: Path) -> None:
    work_dir = _label_task(tmp_path / "tasks")
    result = json.loads((work_dir / RESULT_FILE).read_text(encoding="utf-8"))
    result["validation"] = "invalid"
    (work_dir / RESULT_FILE).write_text(json.dumps(result), encoding="utf-8")

    assert _severities(_scan(audit.check_label_tasks, roots=[tmp_path])) == {"integrity"}


def test_modified_engine_output_is_an_integrity_failure(tmp_path: Path) -> None:
    work_dir = _label_task(tmp_path / "tasks")
    (work_dir / "attempts" / "00-sp" / "sp.out").write_text("edited by hand\n", encoding="utf-8")

    report = _scan(audit.check_label_tasks, roots=[tmp_path])

    assert _severities(report) == {"integrity"} and "output_sha256" in report.findings[0].detail


def test_phal_task_json_still_gets_the_selection_check(tmp_path: Path) -> None:
    task_dir = tmp_path / "phal_task"
    task_dir.mkdir()
    (task_dir / "task.json").write_text(json.dumps({"metadata": {"lj_fit_count": 0}}), encoding="utf-8")

    report = audit.audit([tmp_path])

    assert _severities(report) == {"selection"}
    assert report.scanned["label_tasks"] == 0


def test_gmd_formats_match_the_owning_modules() -> None:
    """The audit hard-codes on-disk names so it can scan without instantiating services."""

    assert audit.LABEL_TASK_SCHEMA == task_store.SCHEMA
    assert audit.LABEL_RESULT_FILE == RESULT_FILE
    assert np.isclose(audit.MIN_PHYSICAL_DISTANCE, 0.7)
