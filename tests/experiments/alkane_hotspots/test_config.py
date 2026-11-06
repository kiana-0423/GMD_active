"""The demo YAMLs validate; unsafe or inconsistent configurations are refused."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from ._helpers import CONFIGS
from tools.experiments.alkane_hotspots.config import AlkaneConfig, load_config


@pytest.mark.parametrize("name", ["smoke", "pilot", "pilot_c32", "campaign"])
def test_demo_configs_load(name: str) -> None:
    loaded = load_config(CONFIGS / f"{name}.yaml")
    assert loaded.name == name
    assert loaded.artifacts_root == (loaded.repo_root.parent / "alkane_artifacts" / name).resolve()


def test_smoke_is_flagged_and_pilot_is_production() -> None:
    smoke = load_config(CONFIGS / "smoke.yaml").config
    pilot = load_config(CONFIGS / "pilot.yaml").config
    assert smoke.cp2k.smoke_only and not pilot.cp2k.smoke_only
    # Sized from measured CP2K timings (docs/demos/alkane_hotspots/results/timing_probe.json): C16H34 pilot,
    # C32H66 as a separately budgeted size-transfer check.
    assert pilot.systems.chain_lengths == [16] and pilot.generation.n_parents == 16
    transfer = load_config(CONFIGS / "pilot_c32.yaml").config
    assert transfer.systems.chain_lengths == [32] and transfer.generation.n_parents == 2
    assert (transfer.cp2k.cutoff, transfer.cp2k.rel_cutoff) == (pilot.cp2k.cutoff, pilot.cp2k.rel_cutoff) == (300.0, 50.0)
    assert (pilot.cp2k.functional, pilot.cp2k.basis, pilot.cp2k.potential, pilot.cp2k.dispersion) == ("PBE", "DZVP-MOLOPT-SR-GTH", "GTH-PBE", True)
    assert pilot.cp2k.charge == 0 and pilot.cp2k.multiplicity == 1 and pilot.cp2k.poisson_solver == "WAVELET"


def test_campaign_long_chains_require_the_flag() -> None:
    data = yaml.safe_load((CONFIGS / "campaign.yaml").read_text())
    # stage 1 is C16 + C32 at the pilot's converged cutoff; C64/C128 need a separate opt-in (stage 2)
    assert data["systems"]["chain_lengths"] == [16, 32] and data["systems"]["allow_long_chains"] is False
    assert (data["cp2k"]["cutoff"], data["cp2k"]["rel_cutoff"]) == (300.0, 50.0)
    AlkaneConfig.model_validate(data)
    data["systems"]["chain_lengths"] = [16, 32, 64, 128]
    data["generation"]["n_parents"] = 128
    with pytest.raises(ValidationError, match="allow_long_chains"):
        AlkaneConfig.model_validate(data)
    data["systems"]["allow_long_chains"] = True
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
