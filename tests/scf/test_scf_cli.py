# Ported from PHAL (tests.scf.test_scf_cli) at 96263b6; MIT License, see NOTICE.
"""Offline tests for the ``phal cp2k-scf`` command group."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotspot_al.cli import build_parser, main
from gmd_active_learning.adapters.dft.scf.dataset import SCFDataset

from tests.scf.test_scf_dataset import make_sample


@pytest.fixture
def populated_dataset(tmp_path: Path) -> Path:
    dataset = SCFDataset.create(tmp_path / "scf_dataset")
    for index in range(4):
        dataset.add_sample(make_sample(f"task-{index:04d}"), split="train")
    dataset.add_sample(make_sample("task-9999"), split="validation")
    dataset.refresh_metadata()
    return dataset.root


def test_doctor_still_works_after_adding_the_group() -> None:
    # Regression guard: the existing CLI must not change behaviour.
    parser = build_parser()

    args = parser.parse_args(["doctor", "--strict"])

    assert args.command == "doctor"
    assert args.strict


def test_dataset_init_creates_the_layout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["cp2k-scf", "dataset", "init", "--root", str(tmp_path / "ds")])

    assert exit_code == 0
    assert (tmp_path / "ds" / "metadata.json").is_file()
    for split in ("train", "validation", "test"):
        assert (tmp_path / "ds" / split).is_dir()
    assert "initialised SCF dataset" in capsys.readouterr().out


def test_dataset_validate_passes_on_a_clean_dataset(
    populated_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["cp2k-scf", "dataset", "validate", "--root", str(populated_dataset), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["n_subjects"] == 5
    assert payload["n_failed"] == 0


def test_dataset_validate_fails_on_a_corrupt_record(
    populated_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (populated_dataset / "train" / "broken.json").write_text("{not json", encoding="utf-8")

    exit_code = main(["cp2k-scf", "dataset", "validate", "--root", str(populated_dataset)])

    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().out


def test_dataset_validate_strict_fails_on_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from gmd_active_learning.adapters.dft.scf.schema import SCFIterationRecord, SCFTrajectory

    dataset = SCFDataset.create(tmp_path / "ds")
    dataset.add_sample(
        make_sample(
            "stalled",
            trajectory=SCFTrajectory(
                iterations=(SCFIterationRecord(step=1, residual=1.0, energy=-1.0),),
                eps_scf=1.0e-6,
                max_scf=100,
                converged=False,
                initial_guess="ATOMIC",
            ),
        ),
        split="train",
    )

    lenient = main(["cp2k-scf", "dataset", "validate", "--root", str(dataset.root)])
    strict = main(["cp2k-scf", "dataset", "validate", "--root", str(dataset.root), "--strict"])
    capsys.readouterr()

    assert lenient == 0
    assert strict == 1


def test_dataset_validate_rejects_a_directory_that_is_not_a_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "empty").mkdir()

    exit_code = main(["cp2k-scf", "dataset", "validate", "--root", str(tmp_path / "empty")])

    assert exit_code == 2
    assert "not an SCF dataset" in capsys.readouterr().out


def test_dataset_inspect_summarises(populated_dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["cp2k-scf", "dataset", "inspect", "--root", str(populated_dataset), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["n_total"] == 5
    assert payload["counts"]["train"] == 4
    assert payload["converged_fraction"] == 1.0


def test_dataset_inspect_human_output(populated_dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["cp2k-scf", "dataset", "inspect", "--root", str(populated_dataset)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "samples: 5 total" in output
    assert "scf iterations" in output


def test_dataset_inspect_can_restrict_to_one_split(
    populated_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["cp2k-scf", "dataset", "inspect", "--root", str(populated_dataset), "--split", "validation", "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["n_decoded"] == 1


def test_dataset_export_writes_artifacts(
    populated_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "export"

    exit_code = main(
        ["cp2k-scf", "dataset", "export", "--root", str(populated_dataset), "--output", str(output), "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["counts"]["train"] == 4
    assert (output / "train.jsonl").is_file()
    assert (output / "train_features.npz").is_file()
    assert (output / "export_manifest.json").is_file()


def test_dataset_export_can_skip_features(
    populated_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "export"

    exit_code = main(
        [
            "cp2k-scf", "dataset", "export",
            "--root", str(populated_dataset),
            "--output", str(output),
            "--splits", "train",
            "--no-features",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert (output / "train.jsonl").is_file()
    assert not (output / "train_features.npz").exists()


def test_dataset_ingest_reports_what_it_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.scf.test_scf_dataset import converged_trajectory

    scf_root = tmp_path / "scf"
    (scf_root / "task-0001").mkdir(parents=True)
    (scf_root / "task-0001" / "scf_history.json").write_text(
        json.dumps(converged_trajectory().to_dict()), encoding="utf-8"
    )

    exit_code = main(
        [
            "cp2k-scf", "dataset", "ingest",
            "--root", str(tmp_path / "ds"),
            "--scf-root", str(scf_root),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["n_ingested"] == 0
    assert payload["skipped"][0]["task_id"] == "task-0001"


def test_capabilities_reports_a_missing_executable(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        ["cp2k-scf", "capabilities", "--executable", "definitely-not-a-real-cp2k-binary", "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["available"] is False
    assert payload["ml_scf_supported"] is False


def test_benchmark_prints_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from tools.scf.benchmark.modes import MODE_ML_FULL, MODE_NATIVE
    from tools.scf.benchmark.evaluate import SCFRun

    runs = [
        SCFRun(label=MODE_NATIVE, system_id="a", scf_iterations=24, converged=True, final_energy=-1.0),
        SCFRun(label=MODE_ML_FULL, system_id="a", scf_iterations=12, converged=True, final_energy=-1.0),
    ]
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps([run.to_dict() for run in runs]), encoding="utf-8")
    report_path = tmp_path / "report.json"

    exit_code = main(["cp2k-scf", "benchmark", "--runs", str(runs_path), "--output", str(report_path)])
    capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["speedup"] == pytest.approx(2.0)
    assert payload["converged"] is True


def test_benchmark_reports_a_bad_runs_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs_path = tmp_path / "runs.json"
    runs_path.write_text("{not json", encoding="utf-8")

    exit_code = main(["cp2k-scf", "benchmark", "--runs", str(runs_path)])

    assert exit_code == 1
    assert "invalid JSON" in capsys.readouterr().out
