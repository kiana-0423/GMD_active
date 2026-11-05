"""The packaged ``cp2k`` section validates, equals the model defaults and keeps PHAL's numeric defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig, FailureClass

SECTIONS = Path(__file__).resolve().parents[3] / "configs" / "sections"


def test_packaged_cp2k_section_equals_model_defaults() -> None:
    data = yaml.safe_load((SECTIONS / "cp2k.yaml").read_text())["cp2k"]
    assert CP2KConfig.model_validate(data) == CP2KConfig()


def test_phal_numeric_defaults_are_kept() -> None:
    config = CP2KConfig()
    assert (config.cutoff, config.rel_cutoff, config.max_scf, config.scf_eps) == (400.0, 60.0, 100, 1.0e-6)
    assert (config.basis, config.potential, config.functional, config.use_ot) == ("DZVP-MOLOPT-SR-GTH", "GTH-PBE", "PBE", True)
    assert (config.h_only_opt.enabled, config.h_only_opt.max_iter, config.uncapped_fragment_policy) == (True, 50, "error")
    assert (config.spin_policy, config.multiplicity, config.dispersion) == ("auto", None, False)
    assert config.poisson.non_periodic_solver == "WAVELET"  # PHAL default for PERIODIC NONE


def test_unknown_keys_and_phal_only_keys_are_rejected() -> None:
    for key in ("periodic", "charge", "submit_mode", "slurm_directives", "typo"):
        with pytest.raises(ValidationError):
            CP2KConfig.model_validate({key: "x"})


def test_every_failure_class_has_a_ladder() -> None:
    ladders = CP2KConfig().retry.ladders
    assert set(ladders) >= {cls for cls in FailureClass if cls is not FailureClass.ML_SCF_FAILED}
