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


def _model_has_path(model: type, path: list[str]) -> bool:
    from pydantic import BaseModel

    for index, name in enumerate(path):
        field = model.model_fields.get(name)
        if field is None:
            return False
        annotation = field.annotation
        if index == len(path) - 1:
            return True
        if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
            return False  # a dict-valued key (e.g. environment) cannot have named children
        model = annotation
    return True


def test_every_config_key_and_env_var_the_guides_name_exists() -> None:
    """SCF-043: README.md and architecture.md name only real ``scf.*`` keys and ``PHAL_ML_SCF_*`` variables
    (module paths such as ``…scf.recording.density`` are not config keys)."""

    from gmd_active_learning.adapters.dft.scf.config import ScfConfig

    text = _text("README.md") + _text("architecture.md")
    keys = sorted(set(re.findall(r"(?<![.…\w])scf\.((?:ml_scf|recording|retry|mixing|predictor|dataset|capabilities)(?:\.[a-z_]+)*)", text)))
    assert "ml_scf.enabled" in keys and len(keys) >= 5
    missing = [key for key in keys if not _model_has_path(ScfConfig, key.split("."))]
    assert missing == []
    source = "\n".join(path.read_text(encoding="utf-8") for root in ("src", "tools")
                       for path in (DOCS.parents[1] / root).rglob("*.py"))
    variables = sorted(set(re.findall(r"\bPHAL_ML_SCF_[A-Z_]+\b", text)))
    assert variables and [name for name in variables if name not in source] == []
