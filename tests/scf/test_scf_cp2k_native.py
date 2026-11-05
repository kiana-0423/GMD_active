"""Real CP2K runs of the SCF features an unmodified CP2K can exercise (marker ``cp2k``).

Uses the CP2K found via ``$GMD_SCF_CP2K``, ``$CP2K_BIN`` or ``cp2k.psmp`` on PATH;
skipped when none is installed. Every run pins ``OMP_NUM_THREADS=1`` because SCF
step counts are not thread-deterministic (cp2k-ml-scf RECONSTRUCTION.md 4g).

What is exercised for real here: the capability contract (probe + native
validation), SCF-history / converged-density / per-iteration-trajectory
recording from genuine &AO_MATRICES output, the deterministic retry ladder
rescuing MgO (PHAL's documented R_COND abort reproduced on native CP2K), and the
ML-SCF wrapper degrading to a native run on a binary without ML-SCF. Nothing
here can exercise the ML features: they need the modified cp2k-ml-scf binary.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.scf.capabilities import detect_cp2k_capabilities, run_native_validation

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "scf"


def _cp2k() -> str | None:
    for candidate in (os.environ.get("GMD_SCF_CP2K"), os.environ.get("CP2K_BIN"), shutil.which("cp2k.psmp")):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


CP2K = _cp2k()
pytestmark = [pytest.mark.cp2k, pytest.mark.skipif(CP2K is None, reason="no CP2K executable installed")]


def _data_dir() -> Path | None:
    configured = os.environ.get("CP2K_DATA_DIR")
    if configured and Path(configured).is_dir():
        return Path(configured)
    if CP2K is None:
        return None
    guess = Path(os.path.realpath(CP2K)).parents[1] / "share" / "cp2k" / "data"
    return guess if (guess / "BASIS_MOLOPT").is_file() else None


def run_cp2k(work_dir: Path, name: str, text: str) -> str:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / f"{name}.inp").write_text(text, encoding="utf-8")
    subprocess.run(
        [CP2K, "-i", f"{name}.inp", "-o", f"{name}.out"],
        cwd=work_dir,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    return (work_dir / f"{name}.out").read_text(encoding="utf-8", errors="ignore")


H2_INPUT = """&GLOBAL
  PROJECT h2
  RUN_TYPE ENERGY
  PRINT_LEVEL MEDIUM
&END GLOBAL
&FORCE_EVAL
  METHOD QUICKSTEP
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME POTENTIAL
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 50
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
  &SUBSYS
    &CELL
      ABC 10.0 10.0 10.0
    &END CELL
    &COORD
      H 5.0 5.0 4.63
      H 5.0 5.0 5.37
    &END COORD
    &KIND H
      BASIS_SET DZVP-MOLOPT-GTH
      POTENTIAL GTH-PBE-q1
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
"""


def test_the_native_binary_is_probed_and_validated_without_claiming_ml_features(tmp_path: Path) -> None:
    capabilities = detect_cp2k_capabilities(executable=CP2K)

    assert capabilities.available
    assert capabilities.version is not None
    assert capabilities.ml_scf_supported is False
    assert not capabilities.supports("ml_density_guess")
    assert not capabilities.supports("ml_mixing")

    validated = run_native_validation(capabilities, tmp_path)

    features = validated.to_dict()["features"]
    assert features["native_scf"]["validated"] is True
    assert features["ao_matrix_printing"]["validated"] is True
    assert "Tr(P S)=2.000000000" in features["ao_matrix_printing"]["evidence"]
    for name in ("ml_density_guess", "ml_inprocess_guess", "ml_mixing", "scf_controller", "pulay_recorder"):
        assert features[name] == {"advertised": False, "validated": False, "evidence": features[name]["evidence"]}


def test_scf_history_density_and_trajectory_are_recorded_from_a_real_run(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.recording.density import load_density
    from gmd_active_learning.adapters.dft.scf.recording.input import insert_ao_matrices_print
    from gmd_active_learning.adapters.dft.scf.recording.trajectory import load_trajectory
    from gmd_active_learning.adapters.dft.scf.scf_history import record_scf_history_text

    config = {
        "scf": {
            "recording": {
                "enabled": True,
                "output_dir": str(tmp_path / "scf"),
                "record_density": True,
                "record_trajectory": True,
                "trajectory_strict": True,
                "density_strict": True,
            }
        }
    }
    work = tmp_path / "run"
    output = run_cp2k(work, "h2", insert_ao_matrices_print(H2_INPUT, config))

    written = record_scf_history_text(output, input_path=work / "h2.inp", config=config, task_id="h2")

    history = json.loads(Path(written).read_text())
    assert history["converged"] is True
    assert len(history["iterations"]) == 10  # cp2k-ml-scf's recorded H2 control: 10 steps
    assert history["iterations"][-1]["energy"] == pytest.approx(-1.16593337453310, abs=1e-9)
    assert history["eps_scf"] == pytest.approx(1e-6)

    _, _, density_metadata = load_density(tmp_path / "scf" / "h2" / "density")
    assert density_metadata["electron_count_method"] == "trace_ps"
    assert density_metadata["electron_count"] == pytest.approx(2.0, abs=1e-6)

    densities, overlap, _ = load_trajectory(tmp_path / "scf" / "h2")
    counts = [float(np.sum(frame[0] * overlap)) for frame in densities]
    assert len(densities) == 11  # iterations + the post-SCF print
    assert np.allclose(counts, 2.0, atol=1e-6)

    # The same system as the committed PHAL fixture (recorded with the lost build):
    # the converged densities agree.
    fixture = np.load(FIXTURES / "h2_scf_trajectory.npz")
    assert np.allclose(overlap, fixture["overlap"], atol=1e-8)
    assert np.abs(densities[-1] - fixture["densities"][-1]).max() < 1e-5


def test_the_retry_ladder_rescues_mgo_on_native_cp2k(tmp_path: Path) -> None:
    """PHAL: MgO aborts with R_COND under NBUFFER 8 on native CP2K too; NBUFFER 4 converges in 10."""

    data_dir = _data_dir()
    if data_dir is None:
        pytest.skip("CP2K data directory not found (set CP2K_DATA_DIR)")
    from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history
    from gmd_active_learning.adapters.dft.scf.strategy import RetryLadder
    from tools.scf.difficulty import parse_abort
    from tools.scf.structures import read_structures, render_input, resolve_kinds

    structure = next(
        item
        for item in read_structures(REPO / "data/structures/pulay_sampling/periodic.extxyz")
        if item.label == "mgo-rocksalt"
    )
    base = render_input(
        (REPO / "configs/cp2k/strategy.inp.template").read_text(encoding="utf-8"),
        structure,
        resolve_kinds(structure.elements, data_dir=data_dir),
    )
    ladder = RetryLadder({"scf": {"retry": {"enabled": True}}})

    outcomes = []
    attempt = 0
    while ladder.strategy_for_attempt(attempt) is not None:
        output = run_cp2k(tmp_path / f"attempt{attempt}", "mgo", ladder.apply(base, attempt))
        history = parse_scf_history(output)
        outcomes.append((ladder.order[attempt], history.converged, len(history.iterations), parse_abort(output)))
        if history.converged:
            break
        attempt += 1

    (first, first_ok, _, first_abort), (second, second_ok, second_steps, _) = outcomes
    assert (first, second) == ("default", "nbuffer4")
    assert first_ok is False and first_abort is not None and "R_COND" in first_abort
    assert second_ok is True
    assert second_steps <= 12  # PHAL's table: 10


def test_the_ml_wrapper_degrades_to_a_native_run_on_a_native_binary(tmp_path: Path) -> None:
    from ase import Atoms

    from gmd_active_learning.adapters.dft.scf.cp2k_ml_backend import CP2KMLSCFBackend
    from gmd_active_learning.adapters.dft.scf.execution import JobState, LocalSchedulerBackend
    from gmd_active_learning.adapters.dft.scf.modes import MODE_NATIVE_FALLBACK
    from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history

    work = tmp_path / "task"
    work.mkdir()
    (work / "job.inp").write_text(H2_INPUT, encoding="utf-8")
    wrapper = CP2KMLSCFBackend({"scf": {"ml_scf": {"enabled": True, "executable": CP2K}}})
    h2 = Atoms("H2", positions=[(5.0, 5.0, 4.63), (5.0, 5.0, 5.37)], cell=[10, 10, 10], pbc=True)

    written = wrapper.prepare_input(work / "job.inp", h2, task_id="h2")
    request = wrapper.execution_request(written["input"], output_file=work / "job.out")

    assert "does not support ML-SCF" in (wrapper.degradation_reason() or "")
    assert request.metadata["scf_mode"] == MODE_NATIVE_FALLBACK
    assert not any(key.startswith("PHAL_ML_SCF") for key in request.environment)

    scheduler = LocalSchedulerBackend()
    job = scheduler.submit(
        type(request)(**{**{f: getattr(request, f) for f in request.__dataclass_fields__},
                         "environment": {**request.environment, "OMP_NUM_THREADS": "1"}})
    )
    job.native_handle.wait(timeout=900)
    assert scheduler.poll(job) is JobState.COMPLETED

    output = (work / "job.out").read_text(encoding="utf-8", errors="ignore")
    assert parse_scf_history(output).converged
    outcome = wrapper.parse_outcome(output)
    assert outcome.accelerator_enabled is False and outcome.ml_lines_present is False
    assert wrapper.calculation_provenance(outcome)["scf_mode"] == MODE_NATIVE_FALLBACK
