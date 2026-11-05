# Ported from PHAL (tests.scf.test_scf_ml_regression) at 96263b6; MIT License, see NOTICE.
"""Regression guards: adding ML-SCF must not change standard CP2K behaviour.

These tests exist to make the project's central architectural promise
executable. Selecting ``backend.dft.engine: cp2k`` must produce exactly the
inputs, commands and results it produced before the ML-SCF work landed, and
must never load a model, write a prediction, or read ``cp2k.ml_scf``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gmd_active_learning.adapters.dft.scf.execution import BackendRole
from hotspot_al.backends.cp2k import CP2KBackend
from gmd_active_learning.adapters.dft.scf.cp2k_ml_backend import CP2KMLSCFBackend
from hotspot_al.backends.factory import backend_engine, create_backend
from hotspot_al.config_schema import validate_config
from hotspot_al.exceptions import ConfigError
from hotspot_al.models import ExtractedRegion
from gmd_active_learning.adapters.dft.scf.config import load_config


def base_config() -> dict:
    return load_config("configs/default.yaml")


def region() -> ExtractedRegion:
    atoms = Atoms("OH2", positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)], cell=[10, 10, 10], pbc=True)
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


# -- engine selection ---------------------------------------------------


def test_default_config_still_selects_the_standard_engine() -> None:
    config = base_config()

    assert backend_engine(config, BackendRole.DFT) == "cp2k"
    assert isinstance(create_backend(config, BackendRole.DFT), CP2KBackend)
    assert not isinstance(create_backend(config, BackendRole.DFT), CP2KMLSCFBackend)


def test_accelerated_engine_is_opt_in() -> None:
    config = base_config()
    config["backend"]["dft"]["engine"] = "cp2k_ml_scf"

    assert isinstance(create_backend(config, BackendRole.DFT), CP2KMLSCFBackend)


def test_standard_backend_ignores_the_ml_scf_config_block(tmp_path: Path) -> None:
    without = base_config()
    with_ml = copy.deepcopy(without)
    with_ml["cp2k"]["ml_scf"] = {
        "enabled": True,
        "predictor": {"type": "dummy"},
        "mixing": {"method": "ml"},
    }

    plain = CP2KBackend(config=without).prepare_inputs(region(), tmp_path / "a", task_id="t")
    polluted = CP2KBackend(config=with_ml).prepare_inputs(region(), tmp_path / "b", task_id="t")

    for key in plain:
        assert Path(plain[key]).read_text(encoding="utf-8") == Path(polluted[key]).read_text(encoding="utf-8")


def test_standard_backend_writes_no_ml_scf_manifest(tmp_path: Path) -> None:
    written = CP2KBackend(config=base_config()).prepare_inputs(region(), tmp_path, task_id="t")

    assert not any("ml_scf" in key for key in written)
    assert list(tmp_path.glob("*.ml_scf.json")) == []


def test_standard_backend_sets_no_ml_environment(tmp_path: Path) -> None:
    backend = CP2KBackend(config=base_config())
    written = backend.prepare_inputs(region(), tmp_path, task_id="t")
    input_file = Path(next(iter(written.values())))

    request = backend.execution_request(input_file, output_file=tmp_path / "job.out")

    assert not any(key.startswith("PHAL_ML_SCF") for key in request.environment)
    assert request.metadata["engine"] == "cp2k"


def test_standard_input_contains_no_ml_keywords(tmp_path: Path) -> None:
    written = CP2KBackend(config=base_config()).prepare_inputs(region(), tmp_path, task_id="t")

    for path in written.values():
        text = Path(path).read_text(encoding="utf-8").upper()
        assert "SCF_GUESS ML" not in text
        assert "MIXING_METHOD ML" not in text


# -- result provenance --------------------------------------------------


def test_standard_result_records_no_acceleration(tmp_path: Path) -> None:
    from tests.scf.fakes.fake_cp2k import fake_cp2k_scf_output, write_fake_cp2k_force_output

    output = tmp_path / "job.out"
    write_fake_cp2k_force_output(output, region().atoms)
    output.write_text(
        fake_cp2k_scf_output() + output.read_text(encoding="utf-8"), encoding="utf-8"
    )

    result = CP2KBackend(config=base_config()).parse_result(output)
    provenance = result.provenance()

    assert provenance["accelerator_enabled"] is False
    assert provenance["ml_scf_enabled"] is False
    assert provenance["ml_model_digest"] is None
    assert provenance["fallback_used"] is False


def test_provenance_carries_the_documented_ml_keys(tmp_path: Path) -> None:
    from tests.scf.fakes.fake_cp2k import write_fake_cp2k_force_output

    output = tmp_path / "job.out"
    write_fake_cp2k_force_output(output, region().atoms)
    output.write_text(
        " CP2K| version string: CP2K version 2024.1\n"
        " Source code revision git:DEADBEEF\n"
        " SCF run converged in 3 steps\n" + output.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    provenance = CP2KBackend(config=base_config()).parse_result(output).provenance()

    assert provenance["cp2k_version"] == "CP2K version 2024.1"
    assert provenance["cp2k_commit"] == "deadbeef"
    for key in ("cp2k_version", "cp2k_commit", "ml_model_digest", "ml_scf_enabled"):
        assert key in provenance


# -- configuration schema -----------------------------------------------


def test_config_without_ml_scf_still_validates() -> None:
    config = base_config()
    config["cp2k"].pop("ml_scf", None)

    assert validate_config(config) is config


def test_valid_ml_scf_block_is_accepted() -> None:
    config = base_config()
    config["cp2k"]["ml_scf"] = {
        "enabled": True,
        "predictor": {"type": "dummy", "model_path": None, "options": {}},
        "prediction_dir": "runtime/cp2k_ml/predictions",
        "electron_tolerance": 1.0e-3,
        "mixing": {"method": "ml", "fallback_method": "pulay", "history_length": 8},
    }

    assert validate_config(config) is config


def test_shipped_cp2k_override_validates() -> None:
    config = load_config("configs/cp2k.yaml")

    assert validate_config(config) is config
    assert config["cp2k"]["ml_scf"]["predictor"]["type"] == "dummy"


def test_unknown_ml_scf_key_is_rejected() -> None:
    config = base_config()
    config["cp2k"]["ml_scf"] = {"enabled": True, "typo_here": 1}

    with pytest.raises(ConfigError, match="typo_here"):
        validate_config(config)


def test_unknown_mixing_method_is_rejected() -> None:
    config = base_config()
    config["cp2k"]["ml_scf"] = {"mixing": {"method": "magic"}}

    with pytest.raises(ConfigError, match="mixing.method"):
        validate_config(config)


def test_ml_cannot_be_its_own_fallback() -> None:
    config = base_config()
    config["cp2k"]["ml_scf"] = {"mixing": {"method": "ml", "fallback_method": "ml"}}

    with pytest.raises(ConfigError, match="safety net"):
        validate_config(config)


def test_non_positive_electron_tolerance_is_rejected() -> None:
    config = base_config()
    config["cp2k"]["ml_scf"] = {"electron_tolerance": 0.0}

    with pytest.raises(ConfigError, match="electron_tolerance"):
        validate_config(config)


# -- Stage 14: the subsystems ML-SCF must not have touched ---------------


def test_every_other_backend_role_is_unchanged() -> None:
    """LAMMPS, Allegro and the schedulers must resolve exactly as before."""

    from hotspot_al.backends.allegro import RealAllegroBackend
    from hotspot_al.backends.lammps import LAMMPSBackend
    from gmd_active_learning.adapters.dft.scf.execution import LocalSchedulerBackend

    config = base_config()

    assert isinstance(create_backend(config, BackendRole.MD), LAMMPSBackend)
    assert isinstance(create_backend(config, BackendRole.MLIP), RealAllegroBackend)
    assert isinstance(create_backend(config, BackendRole.SCHEDULER), LocalSchedulerBackend)


def test_selecting_the_ml_engine_does_not_disturb_other_roles(tmp_path: Path) -> None:
    from hotspot_al.backends.lammps import LAMMPSBackend
    from gmd_active_learning.adapters.dft.scf.execution import LocalSchedulerBackend

    config = base_config()
    config["backend"]["dft"]["engine"] = "cp2k_ml_scf"
    config["cp2k"]["ml_scf"] = {"enabled": True, "predictor": {"type": "dummy"}}

    assert isinstance(create_backend(config, BackendRole.MD), LAMMPSBackend)
    assert isinstance(create_backend(config, BackendRole.SCHEDULER), LocalSchedulerBackend)


def test_lammps_input_generation_is_untouched(tmp_path: Path) -> None:
    """The MD input must not vary with any ML-SCF setting."""

    from hotspot_al.lammps.lammps_input import build_lammps_input

    plain = base_config()
    with_ml = copy.deepcopy(plain)
    with_ml["backend"]["dft"]["engine"] = "cp2k_ml_scf"
    with_ml["cp2k"]["ml_scf"] = {"enabled": True, "mixing": {"method": "ml"}}
    with_ml["cp2k"]["scf_recording"] = {
        "enabled": True,
        "output_dir": str(tmp_path),
        "record_density": True,
    }

    pair_style = "pair_style allegro\npair_coeff * * model.pth"
    assert build_lammps_input(pair_style, config=plain) == build_lammps_input(
        pair_style, config=with_ml
    )


def test_density_recording_does_not_change_the_standard_cp2k_input(tmp_path: Path) -> None:
    """Recording is opt-in; leaving it off must be byte-identical to before."""

    from hotspot_al.cp2k.cp2k_input import build_cp2k_input

    config = base_config()
    first = build_cp2k_input(region(), config=config, run_type="ENERGY_FORCE")

    config["cp2k"]["scf_recording"] = {"enabled": True, "output_dir": str(tmp_path)}
    second = build_cp2k_input(region(), config=config, run_type="ENERGY_FORCE")

    assert first == second


def test_scf_recording_without_density_writes_only_the_trajectory(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.scf_history import record_scf_history_text

    from tests.scf.fakes.fake_cp2k import fake_cp2k_scf_output

    input_path = tmp_path / "job.inp"
    input_path.write_text("&SCF\n  EPS_SCF 1.0E-6\n&END SCF\n", encoding="utf-8")
    config = {"cp2k": {"scf_recording": {"enabled": True, "output_dir": str(tmp_path / "scf")}}}

    record_scf_history_text(
        fake_cp2k_scf_output(), input_path=input_path, config=config, task_id="t"
    )

    assert (tmp_path / "scf" / "t" / "scf_history.json").is_file()
    assert not (tmp_path / "scf" / "t" / "density").exists()


def test_the_standard_result_parser_gained_no_ml_behaviour(tmp_path: Path) -> None:
    from hotspot_al.cp2k.cp2k_result import parse_cp2k_result

    from tests.scf.fakes.fake_cp2k import fake_cp2k_scf_output, write_fake_cp2k_force_output

    output = tmp_path / "job.out"
    write_fake_cp2k_force_output(output, region().atoms)
    output.write_text(fake_cp2k_scf_output() + output.read_text(encoding="utf-8"), encoding="utf-8")

    result = parse_cp2k_result(output)

    assert result.accelerator_enabled is False
    assert result.fallback_used is False
    assert result.model_digest is None


def test_ml_scf_block_of_the_wrong_type_is_rejected() -> None:
    config = base_config()
    config["cp2k"]["ml_scf"] = "yes please"

    with pytest.raises(ConfigError, match="ml_scf must be dict"):
        validate_config(config)
