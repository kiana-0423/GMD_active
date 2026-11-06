"""The experimental ML-SCF documentation keeps PHAL's measured results and makes no acceleration
claim (ledger SCF-041, SCF-043; findings F-21)."""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs" / "scf"


def _text(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def test_negative_results_are_recorded_verbatim() -> None:
    record = _text("negative_results.md")
    assert "no acceleration benefit was ever demonstrated" in record
    assert "**0.85**" in record and "| mean speedup |" in record
    assert "Status: negative result. Experimental only — not a production backend." in record
    assert "12/12" in record and "9/12" in record


def test_overview_states_the_measured_numbers_first() -> None:
    readme = _text("README.md")
    head = readme[: readme.index("\n\n", readme.index("Measured status first")) + 2000]
    assert "63 SCF steps against 53" in head and "speedup of 0.85" in head
    assert "No ML-SCF component has been shown to accelerate CP2K" in _text("architecture.md")


def test_no_scf_document_claims_a_speedup_above_one() -> None:
    claim = re.compile(r"(speed-?up|faster)[^.\n|]{0,40}\b([1-9]\d*(\.\d+)?)\s*[x×]", re.IGNORECASE)
    offenders = [f"{path.name}: {match.group(0)}" for path in DOCS.glob("*.md") for match in claim.finditer(path.read_text(encoding="utf-8"))]
    assert offenders == []
