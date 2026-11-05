"""Strict configuration validation (ledger CFG-001), ported from PHAL tests/unit/test_config_schema.py.

PHAL's hand-written schema walker is replaced by strict pydantic section models validated as parsed
YAML: the same error classes are rejected and every error names the dotted path. Deliberate
difference: keys have documented defaults, so an absent *optional* key is not an error (absent
required sections and fields still are).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gmd_active_learning.config import build_config
from gmd_active_learning.config_phal import PHAL_KEY_MAP
from gmd_active_learning.core.exceptions import ConfigurationError

from .helpers import gmd_default, schema, set_path


def _build(raw: dict[str, Any], tmp_path: Path):  # noqa: ANN202
    return build_config(raw, base_dir=tmp_path)


def test_default_config_passes_validation(tmp_path: Path) -> None:
    loaded = _build(gmd_default(), tmp_path)
    assert loaded["workflow"].run_id == "default-run"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        # PHAL: ood_score.lj_lazy_threshold = "too-high" (nested type error)
        ("monitor.metrics.lj_residual.lazy_threshold", "too-high"),
        # PHAL: ood_score.weights.force = "high" (free-key dict values are typed)
        ("monitor.metrics.force.weight", "high"),
        ("capping.bond_lengths.C", "high"),
        # bool is never an int/float, a string never a number, an int never a bool (PHAL _matches_type)
        ("cp2k.max_scf", True),
        ("cp2k.cutoff", True),
        ("events.pre_trigger_frames", "10"),
        ("cp2k.use_ot", 1),
        ("workflow.segment_steps", 1.5),
        # unparsable stage interval (PHAL silently treated it as "triggered")
        ("monitor.stages.physics_interval", "sometimes"),
    ],
)
def test_type_errors_name_the_dotted_path(tmp_path: Path, path: str, value: Any) -> None:
    raw = set_path(gmd_default(), path, value)
    with pytest.raises(ConfigurationError, match=path.replace(".", r"\.")):
        _build(raw, tmp_path)


def test_integers_are_accepted_for_floats(tmp_path: Path) -> None:
    raw = set_path(gmd_default(), "cp2k.cutoff", 500)
    assert _build(raw, tmp_path)["cp2k"].cutoff == 500.0  # type: ignore[attr-defined]


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    raw = set_path(gmd_default(), "cp2k.typo_cutoff", 500)
    with pytest.raises(ConfigurationError, match=r"cp2k\.typo_cutoff: Extra inputs are not permitted"):
        _build(raw, tmp_path)


def test_unknown_section_and_missing_required_are_rejected(tmp_path: Path) -> None:
    raw = gmd_default()
    with pytest.raises(ConfigurationError, match=r"unknown config section\(s\) \['lammps'\]"):
        _build({**raw, "lammps": {}}, tmp_path)
    with pytest.raises(ConfigurationError, match=r"missing required config section\(s\) \['workflow'\]"):
        _build({key: value for key, value in raw.items() if key != "workflow"}, tmp_path)
    del raw["workflow"]["run_id"]
    with pytest.raises(ConfigurationError, match=r"workflow\.run_id: Field required"):
        _build(raw, tmp_path)
    with pytest.raises(ConfigurationError, match="must be a mapping"):
        _build({**gmd_default(), "cp2k": ["not", "a", "mapping"]}, tmp_path)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("cp2k.uncapped_fragment_policy", "sometimes"),  # PHAL _validate_choices
        ("capping.uncapped_policy", "sometimes"),
        ("extraction.block.min_atoms_policy", "sometimes"),
        ("extraction.mode", "sphere"),  # PHAL failed only at use time (CFG-001 notes)
        ("workflow.candidates.fingerprint", "cosine"),  # likewise
    ],
)
def test_bad_choices_are_rejected(tmp_path: Path, path: str, value: Any) -> None:
    with pytest.raises(ConfigurationError, match=path.replace(".", r"\.")):
        _build(set_path(gmd_default(), path, value), tmp_path)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("scf.ml_scf.electron_tolerance", 0),  # PHAL cp2k.ml_scf.electron_tolerance must be positive
        ("scf.ml_scf.mixing.method", "magic"),  # PHAL ML_SCF_MIXING_METHODS
        ("scf.ml_scf.mixing.history_length", 0),  # PHAL history_length must be positive
        ("scf.ml_scf.mixing.fallback_method", "ml"),  # PHAL rule; the key itself was never read, so it is unknown now
    ],
)
def test_ml_scf_rules(tmp_path: Path, path: str, value: Any) -> None:
    with pytest.raises(ConfigurationError, match=path.replace(".", r"\.")):
        _build(set_path(gmd_default(), path, value), tmp_path)


# PHAL positive-range paths whose GMD target accepts 0 on purpose (reason).
ZERO_ALLOWED = {
    "md.directives.time_step": "free-form GMD directive passed through to run.in; GMD validates it at launch",
    "events.pre_trigger_frames": "0 frames of context before the trigger is a valid capture policy",
    "events.post_trigger_frames": "0 closes the event at the trigger frame",
    "extraction.extract_radius": "0 extracts the hotspot atoms only; null means mlip_cutoff + buffer_radius",
    "extraction.min_atoms": "0 = no minimum size (padding is explicit via min_atoms_policy)",
    "extraction.block.min_atoms": "0 = no minimum size",
    "extraction.block.halo": "a block without halo is allowed (all atoms labeled as block core)",
    "extraction.block.buffer.inner": "zero-width buffer shells are allowed",
    "extraction.block.buffer.outer": "zero-width buffer shells are allowed",
    "extraction.block.max_merged_blocks": "0 = no merge limit",
    "workflow.candidates.max_per_cycle": "0 labels nothing in a cycle (selection still records why)",
}


@pytest.mark.parametrize("phal_path", schema()["positive_paths"])
def test_phal_positive_ranges(tmp_path: Path, phal_path: str) -> None:
    for target in PHAL_KEY_MAP[phal_path].split(" + "):
        raw = set_path(gmd_default(), target, 0)
        if target in ZERO_ALLOWED:
            _build(raw, tmp_path)
            raw = set_path(gmd_default(), target, -1)
            if target == "md.directives.time_step":
                continue
        with pytest.raises(ConfigurationError, match=target.replace(".", r"\.")):
            _build(raw, tmp_path)


def test_relative_paths_resolve_against_the_config_but_bare_executables_do_not(tmp_path: Path) -> None:
    raw = gmd_default()
    raw["md"].update(executable="gmd", initial_structure="box.xyz")
    raw["mlip"]["source_path"] = "../GMD_SGT"
    loaded = _build(raw, tmp_path)
    assert loaded["md"].executable == "gmd"  # type: ignore[attr-defined]  # PATH lookup
    assert loaded["md"].initial_structure == str((tmp_path / "box.xyz").resolve())  # type: ignore[attr-defined]
    assert loaded["mlip"].source_path == str((tmp_path / "../GMD_SGT").resolve())  # type: ignore[attr-defined]
    raw["md"]["executable"] = "bin/gmd"
    assert _build(raw, tmp_path)["md"].executable == str((tmp_path / "bin/gmd").resolve())  # type: ignore[attr-defined]
    (tmp_path / "gmd").write_text("#!/bin/sh\n")  # a bare name that exists next to the config is that file
    raw["md"]["executable"] = "gmd"
    assert _build(raw, tmp_path)["md"].executable == str((tmp_path / "gmd").resolve())  # type: ignore[attr-defined]
