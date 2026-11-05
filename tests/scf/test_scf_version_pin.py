"""The CP2K pin GMD_active records agrees with cp2k-ml-scf/VERSION_LOCK.md (authoritative)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gmd_active_learning.adapters.dft.scf.capabilities import CP2K_ML_SCF_DECLARED, parse_version_banner
from gmd_active_learning.adapters.dft.scf.config import EXCHANGE_PROTOCOL

LOCK = Path(__file__).resolve().parents[3] / "cp2k-ml-scf" / "VERSION_LOCK.md"


@pytest.mark.skipif(not LOCK.is_file(), reason="sibling cp2k-ml-scf repository not present")
def test_the_recorded_pin_matches_version_lock() -> None:
    text = LOCK.read_text(encoding="utf-8")

    version = re.search(r"\| Version \| `v([^`]+)` \|", text).group(1)
    commit = re.search(r"\| Commit \| `([0-9a-f]{40})` \|", text).group(1)
    protocol = re.search(r"\| Density exchange \| `([^`]+)` \|", text).group(1)

    assert CP2K_ML_SCF_DECLARED["upstream"] == {"version": version, "commit": commit}
    assert protocol == EXCHANGE_PROTOCOL == CP2K_ML_SCF_DECLARED["protocols"]["density_exchange"]
    for build in ("native", "ml"):
        assert CP2K_ML_SCF_DECLARED["banner_flags"][build] in text


def test_the_declared_ml_banner_is_classified_as_density_guess_only() -> None:
    flags = CP2K_ML_SCF_DECLARED["banner_flags"]["ml"]
    parsed = parse_version_banner(f" CP2K version 2026.2\n Source code revision 67b5da8\n cp2kflags: {flags}\n")

    assert parsed["ml_scf_supported"] is True
    assert parsed["ml_mixing_supported"] is False
    assert parsed["version"] == CP2K_ML_SCF_DECLARED["upstream"]["version"]
    assert CP2K_ML_SCF_DECLARED["validated_here"] is False
