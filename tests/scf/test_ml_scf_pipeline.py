# Ported from PHAL (tests.e2e.test_ml_scf_pipeline) at 96263b6; MIT License, see NOTICE.
"""End-to-end ML-SCF pipeline against a fake CP2K output and a mis-behaving model.

Nothing real is executed. A fake CP2K output stands in for the fork's stdout,
including the ``ML-SCF|`` lines the reconstruction writes, and a deliberately
broken predictor stands in for a model that has gone wrong. The loop is
predict -> export -> "run" -> parse -> ingest -> export -> benchmark, and the two
failure modes that matter are proved: a bad prediction never reaches CP2K, and
a run that fell back is never counted as accelerated.

Deviations from PHAL's version: the fake output carries the v2026.2 banner and
the reconstruction's ``ml_guess_source`` / ``prediction_success`` lines as well
as the legacy ones (ledger SCF-040); electron counts are valence counts (F-23).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gmd_active_learning.adapters.dft.scf.dataset import SCFDataset, ingest_scf_root
from gmd_active_learning.adapters.dft.scf.density.exchange import read_density_exchange
from gmd_active_learning.adapters.dft.scf.export import export_dataset
from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history
from tests.scf.test_scf_ml_backend import prepared, water
from tools.scf.benchmark.evaluate import SCFRun
from tools.scf.benchmark.modes import MODE_ML_FULL, MODE_NATIVE, run_benchmark


def fake_ml_scf_output(*, accepted: bool, iterations: int) -> str:
    """Stand in for the fork's stdout, including its ML-SCF decision lines."""

    if accepted:
        decision = [
            " ML-SCF| accelerator enabled: ML initial guess accepted",
            " ML-SCF| ml_guess_source: artifact",
            " ML-SCF| prediction_success: T",
            " ML-SCF| electrons: 8.000000000",
        ]
    else:
        decision = [
            " ML-SCF| fallback used: ML density does not integrate to the requested electron count",
            " ML-SCF| ml_guess_source: fallback",
            " ML-SCF| prediction_success: F",
            " ML-SCF| continuing with SCF_GUESS ATOMIC",
        ]
    return "\n".join(
        [
            " CP2K| version string: CP2K version 2026.2",
            " Source code revision 67b5da8",
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


# -- the happy path -----------------------------------------------------


def test_prediction_reaches_cp2k_and_the_run_is_recorded_as_accelerated(tmp_path: Path) -> None:
    wrapper, written = prepared(tmp_path, name="task-0001")

    # The density CP2K would read is well-formed and conserves the valence count.
    payload = read_density_exchange(written["ml_density"])
    assert payload["values"].sum() == pytest.approx(8.0)
    assert "SCF_GUESS ML" in written["input"].read_text(encoding="utf-8")

    output = fake_ml_scf_output(accepted=True, iterations=12)
    outcome = wrapper.parse_outcome(output)
    history = parse_scf_history(output)
    provenance = wrapper.calculation_provenance(outcome, iterations=len(history.iterations))

    assert history.converged
    assert len(history.iterations) == 12
    assert history.cp2k_version is not None and "2026.2" in history.cp2k_version
    assert provenance["ml_scf_enabled"] is True
    assert provenance["fallback_used"] is False
    assert provenance["ml_guess_source"] == "artifact"
    assert provenance["cp2k_commit"] == "67b5da8"


def test_a_fallback_run_is_not_counted_as_accelerated(tmp_path: Path) -> None:
    wrapper, _ = prepared(tmp_path, name="task-0002")

    output = fake_ml_scf_output(accepted=False, iterations=24)
    provenance = wrapper.calculation_provenance(wrapper.parse_outcome(output))

    assert provenance["fallback_used"] is True
    assert provenance["ml_scf_enabled"] is False
    assert parse_scf_history(output).converged is True


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

        write(str(directory / "region.extxyz"), water(), format="extxyz")
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
