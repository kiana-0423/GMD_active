# Ported from PHAL (tests.e2e.test_ml_scf_pipeline) at 96263b6; MIT License, see NOTICE.
"""End-to-end ML-SCF pipeline against a fake CP2K and a fake ML model.

Nothing real is executed. A fake CP2K output stands in for the fork's stdout,
including the ``ML-SCF|`` lines the Fortran writes, and a deliberately
mis-behaving predictor stands in for a model that has gone wrong. The point is
to exercise the whole loop -- predict, export, run, parse, ingest, benchmark --
and to prove the two failure modes that matter:

* a bad prediction never reaches CP2K, and
* a run that fell back is never counted as accelerated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gmd_active_learning.adapters.dft.scf.cp2k_ml_backend import CP2KMLSCFBackend
from tools.scf.benchmark.modes import MODE_ML_FULL, MODE_NATIVE, run_benchmark
from gmd_active_learning.adapters.dft.scf.dataset import SCFDataset, ingest_scf_root
from gmd_active_learning.adapters.dft.scf.density import (
    DummyDensityPredictor,
    SCFDensityPrediction,
    register_predictor,
)
from gmd_active_learning.adapters.dft.scf.density.exchange import read_density_exchange
from gmd_active_learning.adapters.dft.scf.density.registry import unregister_predictor
from gmd_active_learning.adapters.dft.scf.export import export_dataset
from tools.scf.benchmark.evaluate import SCFRun
from hotspot_al.models import ExtractedRegion
from gmd_active_learning.adapters.dft.scf.config import load_config

from tests.scf.fakes.fake_cp2k import write_fake_cp2k_force_output


def fake_ml_scf_output(*, accepted: bool, iterations: int) -> str:
    """Stand in for the fork's stdout, including its ML-SCF decision lines."""

    if accepted:
        decision = [
            " ML-SCF| accelerator enabled: ML initial guess accepted",
            " ML-SCF| density file: job.inp.ml_density",
        ]
    else:
        decision = [
            " ML-SCF| fallback used: ML density does not integrate to the requested electron count",
            " ML-SCF| continuing with SCF_GUESS ATOMIC",
        ]
    return "\n".join(
        [
            " CP2K| version string: CP2K version 2024.1 (phal-ml-scf)",
            " Source code revision git:deadbeef",
            *decision,
            *(
                f" SCF ITERATION {step} RESIDUAL {10.0 ** (-step - 1):.1E} ENERGY {-17.0 - 0.01 * step:.6f}"
                for step in range(1, iterations + 1)
            ),
            f" *** SCF run converged in {iterations} steps ***",
            " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -17.164200",
            "",
        ]
    )


def water_region() -> ExtractedRegion:
    atoms = Atoms(
        "OH2",
        positions=[(0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.24, 0.9266, 0.0)],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    return ExtractedRegion(
        atoms=atoms,
        original_indices=[0, 1, 2],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[2],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[0],
        region_labels=["core", "inner_buffer", "outer_buffer"],
        mask_weights=np.ones(3),
    )


def ml_config(tmp_path: Path, predictor_type: str = "dummy") -> dict:
    config = load_config("configs/default.yaml")
    config["backend"]["dft"]["engine"] = "cp2k_ml_scf"
    config["cp2k"]["ml_scf"] = {
        "enabled": True,
        "predictor": {"type": predictor_type, "options": {}},
        "prediction_dir": str(tmp_path / "predictions"),
    }
    config["cp2k_ml_scf"] = {
        "executable": "cp2k.psmp",
        "model_path": str(tmp_path / "model.pt"),
        "model_digest": None,
        "runtime_dir": str(tmp_path / "runtime"),
        "fallback_to_standard_scf": True,
        "require_final_scf_convergence": True,
        "controller_protocol": "phal-ml-scf-v1",
        "environment": {},
    }
    return config


class BrokenPredictor(DummyDensityPredictor):
    """A model that loses electrons -- the failure the gate exists to catch."""

    model_name = "broken"
    predictor_type = "broken"

    def predict(self, structure, electronic_state):
        good = super().predict(structure, electronic_state)
        return SCFDensityPrediction(
            model_name=self.model_name,
            model_version=self.model_version,
            model_digest=self.model_digest,
            structure_hash=good.structure_hash,
            electron_count=good.electron_count,
            spin_channels=good.spin_channels,
            density_format=good.density_format,
            values=np.asarray(good.values) * 0.5,
            predictor_type=self.predictor_type,
        )


@pytest.fixture
def broken_predictor():
    register_predictor("broken", BrokenPredictor)
    try:
        yield
    finally:
        unregister_predictor("broken")


# -- the happy path -----------------------------------------------------


def test_prediction_reaches_cp2k_and_the_run_is_recorded_as_accelerated(tmp_path: Path) -> None:
    backend = CP2KMLSCFBackend(config=ml_config(tmp_path))
    task_dir = tmp_path / "task-0001"

    written = backend.prepare_inputs(water_region(), task_dir, task_id="task-0001")
    input_file = Path(next(path for key, path in written.items() if key.endswith("_input")))
    density_file = Path(next(path for key, path in written.items() if key.endswith("_ml_density")))

    # The density CP2K would read is well-formed and electron-conserving.
    payload = read_density_exchange(density_file)
    assert payload["values"].sum() == pytest.approx(10.0)
    assert "SCF_GUESS ML" in input_file.read_text(encoding="utf-8")

    output = task_dir / "job.out"
    write_fake_cp2k_force_output(output, water_region().atoms)
    output.write_text(
        fake_ml_scf_output(accepted=True, iterations=12) + output.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = backend.parse_result(output, input_path=input_file)
    provenance = result.provenance()

    assert result.scf_converged
    assert result.scf_iterations == 12
    assert provenance["ml_scf_enabled"] is True
    assert provenance["fallback_used"] is False
    assert provenance["cp2k_commit"] == "deadbeef"
    assert provenance["ml_model_digest"] is None or provenance["ml_model_digest"].startswith("sha256:")


def test_a_fallback_run_is_not_counted_as_accelerated(tmp_path: Path) -> None:
    backend = CP2KMLSCFBackend(config=ml_config(tmp_path))
    task_dir = tmp_path / "task-0002"
    backend.prepare_inputs(water_region(), task_dir, task_id="task-0002")

    output = task_dir / "job.out"
    write_fake_cp2k_force_output(output, water_region().atoms)
    output.write_text(
        fake_ml_scf_output(accepted=False, iterations=24) + output.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    provenance = backend.parse_result(output).provenance()

    assert provenance["fallback_used"] is True
    assert provenance["scf_converged"] is True


# -- the failure the gate exists for ------------------------------------


def test_a_model_that_loses_electrons_never_reaches_cp2k(
    tmp_path: Path, broken_predictor: None
) -> None:
    backend = CP2KMLSCFBackend(config=ml_config(tmp_path, predictor_type="broken"))

    written = backend.prepare_inputs(water_region(), tmp_path / "task", task_id="task-0003")
    input_file = Path(next(path for key, path in written.items() if key.endswith("_input")))

    assert not any(key.endswith("_ml_density") for key in written)
    # Without a density the input must not claim SCF_GUESS ML.
    assert "SCF_GUESS ATOMIC" in input_file.read_text(encoding="utf-8")
    assert list(Path(tmp_path / "task").glob("*.ml_density")) == []


# -- dataset and benchmark loop -----------------------------------------


def test_recorded_runs_feed_the_dataset_and_the_benchmark(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history
    from gmd_active_learning.utils.atomic_io import write_json_atomic

    cp2k_input = "&SCF\n  MAX_SCF 100\n  EPS_SCF 1.0E-6\n  SCF_GUESS ML\n&END SCF\n"
    scf_root = tmp_path / "scf"
    for task_id, iterations in (("task-0001", 12), ("task-0002", 14)):
        directory = scf_root / task_id
        directory.mkdir(parents=True)
        from ase.io import write

        write(str(directory / "region.extxyz"), water_region().atoms, format="extxyz")
        history = parse_scf_history(
            fake_ml_scf_output(accepted=True, iterations=iterations), input_text=cp2k_input
        )
        write_json_atomic(directory / "scf_history.json", history.to_dict())

    dataset = SCFDataset.create(tmp_path / "scf_dataset")
    summary = ingest_scf_root(scf_root, dataset, config={"cp2k": {"charge": 0}})

    assert summary["n_ingested"] == 2
    assert summary["n_skipped"] == 0

    samples = dataset.samples()
    assert {sample.trajectory.initial_guess for sample in samples} == {"ML"}
    assert sorted(sample.trajectory.n_iterations for sample in samples) == [12, 14]

    export = export_dataset(dataset, tmp_path / "export")
    assert sum(export.counts.values()) == 2
    assert export.n_rejected == 0

    report = run_benchmark(
        [
            SCFRun(label=MODE_NATIVE, system_id="task-0001", scf_iterations=24, converged=True, final_energy=-17.1642),
            SCFRun(label=MODE_NATIVE, system_id="task-0002", scf_iterations=28, converged=True, final_energy=-17.1642),
            SCFRun(label=MODE_ML_FULL, system_id="task-0001", scf_iterations=12, converged=True, final_energy=-17.1642),
            SCFRun(label=MODE_ML_FULL, system_id="task-0002", scf_iterations=14, converged=True, final_energy=-17.1642),
        ]
    )
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["baseline_iterations"] == pytest.approx(26.0)
    assert payload["ml_iterations"] == pytest.approx(13.0)
    assert payload["speedup"] == pytest.approx(2.0)
    assert payload["energy_error"] == pytest.approx(0.0)
    assert payload["converged"] is True
