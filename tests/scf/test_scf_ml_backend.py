# Ported from PHAL (tests/unit/test_scf_density_exchange.py backend hookup, test_scf_production.py
# Stage 14) at 96263b6; MIT License, see NOTICE.
"""The ML initial-guess wrapper (engine ``cp2k_ml_scf``) without a CP2K binary.

PHAL's backend subclassed its standard CP2KBackend and wrote the input itself;
GMD_active's wrapper adds the ML guess to an input the labeling area wrote, so
these tests start from a small CP2K input. Capabilities are injected from the
banners the cp2k-ml-scf reconstruction records (VERSION_LOCK.md), so the
results do not depend on which CP2K happens to be on PATH.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gmd_active_learning.adapters.dft.scf.capabilities import (
    capabilities_from_banner,
    detect_cp2k_capabilities,
)
from gmd_active_learning.adapters.dft.scf.cp2k_ml_backend import (
    FORK_ENVIRONMENT,
    CP2KMLSCFBackend,
    parse_ml_scf_log,
)
from gmd_active_learning.adapters.dft.scf.density import DummyDensityPredictor, SCFDensityPrediction, register_predictor
from gmd_active_learning.adapters.dft.scf.density.exchange import read_density_exchange
from gmd_active_learning.adapters.dft.scf.density.registry import unregister_predictor
from gmd_active_learning.adapters.dft.scf.modes import MODE_ML_FULL, MODE_ML_GUESS, MODE_NATIVE, MODE_NATIVE_FALLBACK
from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.core.exceptions import CapabilityUnavailableError

ML_BANNER = (
    " CP2K version 2026.2\n Source code revision 67b5da8\n"
    " cp2kflags: omp libint fftw3 libxc elpa parallel scalapack mpi_f08 phal_ml_scf libtorch\n"
)
NATIVE_BANNER = (
    " CP2K version 2026.2\n Source code revision 67b5da8\n"
    " cp2kflags: omp libint fftw3 libxc elpa parallel scalapack mpi_f08\n"
)
#: The lost fork's banner: it advertised ML mixing too.
LOST_FORK_BANNER = ML_BANNER.replace("libtorch", "libtorch phal_ml_mixing")

WATER_INPUT = """&GLOBAL
  PROJECT job
  RUN_TYPE ENERGY_FORCE
&END GLOBAL
&FORCE_EVAL
  METHOD QUICKSTEP
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME POTENTIAL
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 120
      &MIXING
        METHOD PULAY_MIXING
        ALPHA 0.4
        NBUFFER 8
      &END MIXING
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
      O 0.0 0.0 0.0
      H 0.9572 0.0 0.0
      H -0.24 0.9266 0.0
    &END COORD
  &END SUBSYS
&END FORCE_EVAL
"""


def water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[(0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.24, 0.9266, 0.0)],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


def ml_config(tmp_path: Path, **ml_scf) -> dict:
    section = {
        "enabled": True,
        "executable": "cp2k.psmp",
        "predictor": {"type": "dummy", "options": {}},
        "prediction_dir": str(tmp_path / "predictions"),
        "model_path": str(tmp_path / "model.pt"),
        **ml_scf,
    }
    return {"scf": {"ml_scf": section}}


def backend(config: dict, banner: str = ML_BANNER) -> CP2KMLSCFBackend:
    return CP2KMLSCFBackend(config, capabilities=capabilities_from_banner(banner))


def write_input(directory: Path, text: str = WATER_INPUT) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "job.inp"
    path.write_text(text, encoding="utf-8")
    return path


def prepared(tmp_path: Path, config: dict | None = None, banner: str = ML_BANNER, name: str = "task"):
    wrapper = backend(config or ml_config(tmp_path), banner)
    written = wrapper.prepare_input(write_input(tmp_path / name), water(), task_id="task-0001")
    return wrapper, written


# -- input preparation ------------------------------------------------------


def test_backend_writes_a_density_guess_next_to_the_input(tmp_path: Path) -> None:
    _, written = prepared(tmp_path)

    density = written["ml_density"]
    assert density.name == "job.inp.ml_density"
    # Valence count: 6 + 1 + 1 = 8. PHAL wrote 10 = sum(Z) (finding F-23), which CP2K rejects.
    assert read_density_exchange(density)["electron_count"] == pytest.approx(8.0)


def test_backend_records_the_prediction_provenance(tmp_path: Path) -> None:
    prepared(tmp_path)

    entry = json.loads((tmp_path / "predictions" / "predictions.jsonl").read_text().strip().splitlines()[0])
    assert entry["task_id"] == "task-0001"
    assert entry["predictor_type"] == "dummy"
    assert entry["density_file"].endswith(".ml_density")


def test_scf_guess_ml_is_written_only_when_a_density_exists(tmp_path: Path) -> None:
    _, with_guess = prepared(tmp_path, name="with")
    disabled = ml_config(tmp_path)
    disabled["scf"]["ml_scf"]["enabled"] = False
    _, without_guess = prepared(tmp_path, disabled, name="without")

    assert "SCF_GUESS ML" in with_guess["input"].read_text()
    assert "SCF_GUESS ATOMIC" not in with_guess["input"].read_text()
    assert "ml_density" not in without_guess
    assert without_guess["input"].read_text() == WATER_INPUT


def test_a_broken_predictor_does_not_break_input_preparation(tmp_path: Path) -> None:
    config = ml_config(tmp_path, predictor={"type": "does-not-exist"})

    _, written = prepared(tmp_path, config)

    assert written["input"].read_text() == WATER_INPUT
    assert "ml_density" not in written


class BrokenPredictor(DummyDensityPredictor):
    """A model that loses electrons -- the failure the gate exists to catch."""

    predictor_type = "broken"

    def predict(self, structure, electronic_state):
        good = super().predict(structure, electronic_state)
        return SCFDensityPrediction(
            model_name="broken",
            model_version=good.model_version,
            model_digest=good.model_digest,
            structure_hash=good.structure_hash,
            electron_count=good.electron_count,
            spin_channels=good.spin_channels,
            density_format=good.density_format,
            values=np.asarray(good.values) * 0.5,
            predictor_type="broken",
        )


def test_a_model_that_loses_electrons_never_reaches_cp2k(tmp_path: Path) -> None:
    register_predictor("broken", BrokenPredictor)
    try:
        _, written = prepared(tmp_path, ml_config(tmp_path, predictor={"type": "broken"}))
    finally:
        unregister_predictor("broken")

    assert "ml_density" not in written
    assert "SCF_GUESS ATOMIC" in written["input"].read_text()
    assert list((tmp_path / "task").glob("*.ml_density")) == []


def test_eps_scf_and_max_scf_are_never_touched(tmp_path: Path) -> None:
    config = ml_config(tmp_path, experimental=True, mixing={"method": "ml", "history_length": 6})

    _, written = prepared(tmp_path, config, banner=LOST_FORK_BANNER)
    text = written["input"].read_text()

    assert "EPS_SCF 1.0E-6" in text
    assert "MAX_SCF 120" in text


def test_mixing_method_ml_is_experimental_and_replaces_the_existing_mixing(tmp_path: Path) -> None:
    from gmd_active_learning.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="experimental"):
        CP2KMLSCFBackend(ml_config(tmp_path, mixing={"method": "ml"}))

    _, default = prepared(tmp_path, name="pulay")
    config = ml_config(tmp_path, experimental=True, mixing={"method": "ml", "history_length": 6})
    _, ml = prepared(tmp_path, config, banner=LOST_FORK_BANNER, name="ml")
    ml_text = ml["input"].read_text()

    assert "METHOD ML" not in default["input"].read_text()
    assert "METHOD ML" in ml_text
    assert "NBUFFER 6" in ml_text
    # PHAL inserted a second &MIXING after &SCF without removing the existing one.
    assert ml_text.count("&MIXING") == 1
    assert "PULAY_MIXING" not in ml_text


def test_the_manifest_digest_matches_the_final_input(tmp_path: Path) -> None:
    _, written = prepared(tmp_path)

    payload = json.loads(written["ml_scf_manifest"].read_text())

    assert payload["input_digest"] == f"sha256:{file_sha256(written['input'])}"
    assert payload["ml_density_file"] == "job.inp.ml_density"


def test_the_manifest_carries_capability_provenance(tmp_path: Path) -> None:
    _, written = prepared(tmp_path)

    payload = json.loads(written["ml_scf_manifest"].read_text())

    assert payload["protocol"] == "phal-ml-scf-v2"
    for key in ("capability_contract", "cp2k_version", "cp2k_commit", "ml_model_digest", "ml_scf_enabled"):
        assert key in payload["provenance"]
    assert payload["provenance"]["cp2k_commit"] == "67b5da8"


# -- the density_gnn path (finding F-23: it could never deliver a guess) -----


@pytest.fixture
def gnn_checkpoint(tmp_path: Path) -> Path:
    pytest.importorskip("torch")
    from tests.scf.test_scf_density_model import WATER_BASIS, water_examples
    from tools.scf.training.density_gnn import DensityTrainingConfig, train_density_matrix_model

    checkpoint = tmp_path / "gnn.pt"
    train_density_matrix_model(
        water_examples(tmp_path / "train", count=2),
        None,
        checkpoint,
        basis=WATER_BASIS,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )
    return checkpoint


def test_density_gnn_with_an_overlap_writes_an_exchange_file_cp2k_can_check(
    tmp_path: Path, gnn_checkpoint: Path
) -> None:
    from tests.scf.test_scf_density_model import WATER_NAO

    overlap = np.eye(WATER_NAO) + 0.03 * (np.ones((WATER_NAO, WATER_NAO)) - np.eye(WATER_NAO))
    config = ml_config(tmp_path, predictor={"type": "density_gnn", "model_path": str(gnn_checkpoint)})
    wrapper = backend(config)

    written = wrapper.prepare_input(write_input(tmp_path / "task"), water(), task_id="t", overlap=overlap)

    payload = read_density_exchange(written["ml_density"])
    assert payload["density_format"] == "phal-density-matrix-v1"
    assert payload["overlap"] is not None and np.allclose(payload["overlap"], overlap)
    counted = float(sum(np.sum(channel * payload["overlap"]) for channel in payload["values"]))
    assert counted == pytest.approx(8.0, abs=1e-3)
    assert payload["electron_count"] == pytest.approx(counted, abs=1e-3)
    assert "SCF_GUESS ML" in written["input"].read_text()


def test_density_gnn_without_an_overlap_writes_nothing(tmp_path: Path, gnn_checkpoint: Path) -> None:
    config = ml_config(tmp_path, predictor={"type": "density_gnn", "model_path": str(gnn_checkpoint)})

    written = backend(config).prepare_input(write_input(tmp_path / "task"), water(), task_id="t")

    assert "ml_density" not in written
    assert "SCF_GUESS ATOMIC" in written["input"].read_text()


# -- execution and degradation ----------------------------------------------


def test_backend_exports_what_the_fork_reads(tmp_path: Path) -> None:
    config = ml_config(tmp_path, environment={"LD_LIBRARY_PATH": "/opt/toolchain/lib"}, guess_alpha=0.5)
    wrapper, written = prepared(tmp_path, config)

    request = wrapper.execution_request(written["input"], output_file=tmp_path / "task" / "job.out")
    environment = request.environment

    assert environment["PHAL_ML_SCF_ENABLED"] == "1"
    assert environment["PHAL_ML_SCF_PROTOCOL"] == "phal-ml-scf-v2"
    assert Path(environment["PHAL_ML_SCF_DENSITY"]).is_file()
    assert environment["PHAL_ML_SCF_STRUCTURE_DIGEST"] == read_density_exchange(written["ml_density"])["structure_hash"]
    assert environment["PHAL_ML_SCF_GUESS_ALPHA"] == "0.5"
    assert environment["LD_LIBRARY_PATH"] == "/opt/toolchain/lib"
    assert {key for key in environment if key.startswith("PHAL_ML_SCF")} <= set(FORK_ENVIRONMENT)
    assert request.metadata["scf_mode"] == MODE_ML_GUESS
    assert request.command == ("cp2k.psmp", "-i", "job.inp")


def test_a_disabled_wrapper_runs_native(tmp_path: Path) -> None:
    config = ml_config(tmp_path)
    config["scf"]["ml_scf"]["enabled"] = False
    wrapper, written = prepared(tmp_path, config)

    request = wrapper.execution_request(written["input"], output_file=tmp_path / "task" / "job.out")

    assert not any(key.startswith("PHAL_ML_SCF") for key in request.environment)
    assert request.metadata["scf_mode"] == MODE_NATIVE_FALLBACK


def test_a_missing_model_only_blocks_the_paths_that_need_it(tmp_path: Path) -> None:
    guess_only = backend(ml_config(tmp_path))
    with_mixing = backend(
        ml_config(tmp_path, experimental=True, mixing={"method": "ml"}), banner=LOST_FORK_BANNER
    )

    assert guess_only.degradation_reason() is None
    assert "neural mixing needs a deployed model" in (with_mixing.degradation_reason() or "")


def test_a_digest_mismatch_degrades_the_run(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"weights")

    reason = backend(ml_config(tmp_path, model_digest="sha256:notthedigest")).degradation_reason()

    assert reason is not None and "digest mismatch" in reason


def test_a_native_build_degrades_the_run(tmp_path: Path) -> None:
    reason = backend(ml_config(tmp_path), banner=NATIVE_BANNER).degradation_reason()

    assert reason is not None and "does not support ML-SCF" in reason


def test_ml_mixing_on_a_build_without_it_degrades(tmp_path: Path) -> None:
    """PHAL never checked ml_mixing_supported (finding F-23); the reconstruction does not advertise it."""

    (tmp_path / "model.pt").write_bytes(b"weights")
    config = ml_config(tmp_path, experimental=True, mixing={"method": "ml"})

    reason = backend(config, banner=ML_BANNER).degradation_reason()

    assert reason is not None and "phal_ml_mixing" in reason
    assert backend(config, banner=LOST_FORK_BANNER).degradation_reason() is None


def test_a_degraded_run_falls_back_to_standard_cp2k(tmp_path: Path) -> None:
    config = ml_config(tmp_path, environment={"LD_LIBRARY_PATH": "/opt/toolchain/lib"})
    wrapper, written = prepared(tmp_path, config)  # prepared against an ML build ...
    wrapper._capabilities = capabilities_from_banner(NATIVE_BANNER)  # ... then run on a native one

    request = wrapper.execution_request(written["input"], output_file=tmp_path / "task" / "job.out")

    assert request.metadata["scf_mode"] == MODE_NATIVE_FALLBACK
    assert request.metadata["fallback_used"] is True
    assert not any(key.startswith("PHAL_ML_SCF") for key in request.environment)
    # The binary still needs its loader path to start (cp2k-ml-scf VERSION_LOCK.md).
    assert request.environment["LD_LIBRARY_PATH"] == "/opt/toolchain/lib"
    native_input = Path(request.metadata["input_file"]).read_text()
    assert "SCF_GUESS ML" not in native_input
    assert "SCF_GUESS ATOMIC" in native_input
    assert "PULAY_MIXING" in native_input  # a native &MIXING section is kept


def test_fallback_can_be_disabled_so_a_broken_setup_fails_loudly(tmp_path: Path) -> None:
    wrapper, written = prepared(tmp_path, ml_config(tmp_path, fallback_to_standard_scf=False), NATIVE_BANNER)

    with pytest.raises(CapabilityUnavailableError, match="does not support ML-SCF"):
        wrapper.execution_request(written["input"], output_file=tmp_path / "task" / "job.out")


def test_scf_mode_reflects_what_will_actually_run(tmp_path: Path) -> None:
    guess_only = backend(ml_config(tmp_path))
    (tmp_path / "model.pt").write_bytes(b"weights")
    full = backend(ml_config(tmp_path, experimental=True, mixing={"method": "ml"}), LOST_FORK_BANNER)
    native = backend(ml_config(tmp_path), NATIVE_BANNER)

    assert guess_only.scf_mode(has_density=True) == MODE_ML_GUESS
    assert guess_only.scf_mode(has_density=False) == MODE_NATIVE
    assert full.scf_mode(has_density=True) == MODE_ML_FULL
    assert native.scf_mode(has_density=True) == MODE_NATIVE_FALLBACK


def test_failed_accelerated_runs_retry_natively_exactly_once(tmp_path: Path) -> None:
    wrapper, written = prepared(tmp_path)
    request = wrapper.execution_request(written["input"], output_file=tmp_path / "task" / "job.out")

    retry = wrapper.retry_request(request, output_file=tmp_path / "task" / "job_native.out")

    assert retry is not None
    assert retry.metadata["scf_mode"] == MODE_NATIVE_FALLBACK
    assert "retried once" in retry.metadata["fallback_reason"]
    assert "SCF_GUESS ATOMIC" in Path(retry.metadata["input_file"]).read_text()
    # Only once: the retry is native, so a second failure is not retried again.
    assert wrapper.retry_request(retry, output_file=tmp_path / "task" / "again.out") is None
    # And not when CP2K had already fallen back (the run *was* native).
    fell_back = " ML-SCF| fallback used: wrong electron count\n ML-SCF| prediction_success: F\n"
    assert wrapper.retry_request(request, output_file=tmp_path / "x.out", failed_output=fell_back) is None


def test_native_retry_can_be_switched_off(tmp_path: Path) -> None:
    wrapper, written = prepared(tmp_path, ml_config(tmp_path, retry_native_on_failure=False))
    request = wrapper.execution_request(written["input"], output_file=tmp_path / "task" / "job.out")

    assert wrapper.retry_request(request, output_file=tmp_path / "task" / "retry.out") is None


# -- what ran: CP2K's own ML-SCF| lines --------------------------------------

RECONSTRUCTION_ACCEPTED = """ ML-SCF| accelerator enabled: ML initial guess accepted
 ML-SCF| ml_guess_source: artifact
 ML-SCF| model_digest: artifact
 ML-SCF| prediction_success: T
 ML-SCF| electrons: 2.000000000
"""
RECONSTRUCTION_FALLBACK = """ ML-SCF| fallback used: density file electron count does not match
 ML-SCF| ml_guess_source: fallback
 ML-SCF| prediction_success: F
 ML-SCF| continuing with SCF_GUESS ATOMIC
"""


def test_an_accepted_guess_is_recorded_as_accelerated(tmp_path: Path) -> None:
    outcome = parse_ml_scf_log(RECONSTRUCTION_ACCEPTED)

    assert outcome.accelerator_enabled and not outcome.fallback_used
    assert outcome.guess_source == "artifact"
    assert outcome.electrons == pytest.approx(2.0)
    provenance = backend(ml_config(tmp_path)).calculation_provenance(outcome, iterations=1)
    assert provenance["scf_mode"] == MODE_ML_GUESS
    assert provenance["ml_scf_enabled"] is True


def test_a_fallback_run_is_not_counted_as_accelerated(tmp_path: Path) -> None:
    outcome = parse_ml_scf_log(RECONSTRUCTION_FALLBACK)
    legacy = parse_ml_scf_log(" ML-SCF| fallback used: ML density does not integrate\n")

    assert not outcome.accelerator_enabled and outcome.fallback_used
    assert outcome.fallback_reason == "density file electron count does not match"
    assert legacy.fallback_used and not legacy.accelerator_enabled
    provenance = backend(ml_config(tmp_path)).calculation_provenance(outcome)
    assert provenance["scf_mode"] == MODE_NATIVE_FALLBACK
    assert provenance["fallback_used"] is True


def test_a_run_without_ml_lines_is_not_accelerated(tmp_path: Path) -> None:
    """PHAL's parse_result set accelerator_enabled whenever the config was not degraded."""

    outcome = backend(ml_config(tmp_path)).parse_outcome(" SCF run converged in 10 steps\n")

    assert outcome.accelerator_enabled is False
    assert outcome.ml_lines_present is False


def test_a_degraded_configuration_is_never_accelerated(tmp_path: Path) -> None:
    outcome = backend(ml_config(tmp_path), NATIVE_BANNER).parse_outcome(RECONSTRUCTION_ACCEPTED)

    assert outcome.accelerator_enabled is False
    assert outcome.fallback_used is True


def test_calculation_provenance_has_the_documented_shape(tmp_path: Path) -> None:
    provenance = backend(ml_config(tmp_path)).calculation_provenance(runtime_seconds=12.5)

    for key in ("cp2k_version", "cp2k_commit", "ml_model_digest", "scf_mode", "iterations", "runtime", "fallback_used"):
        assert key in provenance, key
    assert provenance["runtime"] == pytest.approx(12.5)
    assert provenance["scf_mode"] in {MODE_NATIVE, MODE_ML_GUESS, MODE_ML_FULL, MODE_NATIVE_FALLBACK}


# -- the capability probe reaches the binary with the configured environment --


def fake_cp2k(tmp_path: Path) -> Path:
    """A 'binary' that, like the cp2k-ml-scf build, fails in the loader without its library path."""

    script = tmp_path / "cp2k.psmp"
    script.write_text(
        "#!/bin/sh\n"
        'case "$LD_LIBRARY_PATH" in\n'
        '  *toolchain*) printf "%s" "' + ML_BANNER.replace("\n", "\\n") + '"; exit 0;;\n'
        "esac\n"
        'echo "cp2k.psmp: error while loading shared libraries: libelpa_openmp.so.19: '
        'cannot open shared object file: No such file or directory" >&2\n'
        "exit 127\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_the_probe_uses_the_configured_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PHAL probed with the inherited environment only (cp2k_capabilities.py:246, finding F-22)."""

    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    executable = fake_cp2k(tmp_path)

    without = detect_cp2k_capabilities(executable=executable)
    configured = detect_cp2k_capabilities(
        executable=executable, config=ml_config(tmp_path, environment={"LD_LIBRARY_PATH": "/opt/toolchain/lib"})
    )

    assert without.loader_failure is True
    assert without.available is False
    assert any("dynamic linker" in detail for detail in without.details)
    assert configured.available is True
    assert configured.ml_scf_supported is True
    assert configured.supports("ml_density_guess")
    assert not configured.supports("ml_inprocess_guess")  # libtorch alone does not prove mode B


def test_the_wrapper_probes_with_its_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    executable = fake_cp2k(tmp_path)
    config = ml_config(tmp_path, executable=str(executable), environment={"LD_LIBRARY_PATH": "/x/toolchain/lib"})

    wrapper = CP2KMLSCFBackend(config)

    assert wrapper.degradation_reason() is None
    assert wrapper.capabilities().git_commit == "67b5da8"
    assert os.environ.get("LD_LIBRARY_PATH") is None
