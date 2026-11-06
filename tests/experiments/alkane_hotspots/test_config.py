"""The demo YAMLs validate; unsafe or inconsistent configurations are refused."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from ._helpers import CONFIGS
from tools.experiments.alkane_hotspots.config import AlkaneConfig, load_config


@pytest.mark.parametrize("name", ["smoke", "pilot", "campaign"])
def test_demo_configs_load(name: str) -> None:
    loaded = load_config(CONFIGS / f"{name}.yaml")
    assert loaded.name == name
    assert loaded.artifacts_root == (loaded.repo_root.parent / "alkane_artifacts" / name).resolve()


def test_smoke_is_flagged_and_pilot_is_production() -> None:
    smoke = load_config(CONFIGS / "smoke.yaml").config
    pilot = load_config(CONFIGS / "pilot.yaml").config
    assert smoke.cp2k.smoke_only and not pilot.cp2k.smoke_only
    assert pilot.systems.chain_lengths == [16, 32] and pilot.generation.n_parents == 24
    assert (pilot.cp2k.functional, pilot.cp2k.basis, pilot.cp2k.potential, pilot.cp2k.dispersion) == ("PBE", "DZVP-MOLOPT-SR-GTH", "GTH-PBE", True)
    assert pilot.cp2k.charge == 0 and pilot.cp2k.multiplicity == 1 and pilot.cp2k.poisson_solver == "WAVELET"


def test_campaign_long_chains_require_the_flag() -> None:
    data = yaml.safe_load((CONFIGS / "campaign.yaml").read_text())
    assert max(data["systems"]["chain_lengths"]) > 32 and data["systems"]["allow_long_chains"] is True
    data["systems"]["allow_long_chains"] = False
    with pytest.raises(ValidationError, match="allow_long_chains"):
        AlkaneConfig.model_validate(data)


def _pilot() -> dict:
    return yaml.safe_load((CONFIGS / "pilot.yaml").read_text())


def test_unknown_keys_and_inconsistent_counts_are_refused() -> None:
    data = _pilot()
    data["cp2k"]["cutof"] = 300
    with pytest.raises(ValidationError):
        AlkaneConfig.model_validate(data)
    data = _pilot()
    data["generation"]["n_parents"] = 25
    with pytest.raises(ValidationError, match="multiple"):
        AlkaneConfig.model_validate(data)
    data = _pilot()
    data["splits"]["fractions"]["calibration"] = 0.5
    with pytest.raises(ValidationError, match="sum to 1"):
        AlkaneConfig.model_validate(data)
    data = _pilot()
    data["cp2k"]["mpi_ranks"] = 4
    with pytest.raises(ValidationError, match="launcher"):
        AlkaneConfig.model_validate(data)
    data = _pilot()
    data["cp2k"]["multiplicity"] = 3
    with pytest.raises(ValidationError):
        AlkaneConfig.model_validate(data)


def test_hashes_separate_generation_from_budgets() -> None:
    base = AlkaneConfig.model_validate(_pilot())
    data = _pilot()
    data["budgets"]["max_total_cpu_core_hours"] = 1.0
    changed = AlkaneConfig.model_validate(data)
    assert changed.generation_hash() == base.generation_hash() and changed.config_hash() != base.config_hash()
