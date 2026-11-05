"""Packaged default configuration (ledger CFG-003).

configs/default.yaml is generated from the section models (``default_config_dict``); it must
validate, equal the generated dict, and keep PHAL's numeric thresholds (OOD, buffer/hotspot,
extraction, CP2K, mask, candidate pool) unless a deliberate change is listed below with its reason.
"""

from __future__ import annotations

from pathlib import Path

from gmd_active_learning.config import SECTIONS, build_config, default_config_dict, load_config, write_default_config
from gmd_active_learning.config_phal import PHAL_KEY_MAP

from .helpers import DEFAULT_CONFIG, gmd_default, phal_default

_MISSING = object()

# PHAL leaf -> reason for a different GMD default (ledger references in brackets).
DELIBERATE_CHANGES = {
    "lammps.run_steps": "PHAL 0 emitted no `run` command (the MD never advanced); GMD runs segments of workflow.segment_steps",
    "online.run_steps": "same as lammps.run_steps",
    "monitor.rmin_threshold": "PHAL null scored nothing; rmin is scored below 0.8 A by default [MON, F-16]",
    "online.summary_interval": "driver summaries are opt-in (0); progress is recorded in the workflow store",
    "buffer.maxlen": "null = history sized to pre_trigger_frames + 1, the frames an event can use [EVT-001]",
    "extraction.extract_radius": "null = mlip_cutoff + buffer_radius, PHAL's own rule (= 10.0 A for these defaults)",
    "extraction.max_atoms": "size limits are opt-in with explicit trim/error policies; PHAL trimmed silently [EXT-001]",
    "extraction.min_atoms": "size limits are opt-in with explicit pad/error/allow policies; PHAL padded silently [EXT-001]",
    "extraction.block.max_atoms": "block size limits are opt-in [EXT-004]",
    "extraction.block.min_atoms": "block size limits are opt-in [EXT-004]",
    "cp2k.max_retries": "bounded per-failure retry ladders, 3 in total; PHAL never retried [LAB-009, F-25]",
    "training_mask.stress_weight": "stress is written only for labeled periodic regions and then weighted 1; otherwise weight 0",
    "candidate_pool.max_candidates_per_round": "per-cycle labeling budget of the closed loop (plus a per-event budget); "
                                               "PHAL's 50 was an offline batch pool size",
}
THRESHOLD_SECTIONS = ("monitor.", "ood_score.", "buffer.", "hotspot.", "extraction.", "cp2k.", "training_mask.",
                      "candidate_pool.", "online.", "lammps.", "retraining.", "h_capping.", "embedding.")


def _get(data: dict, dotted: str):  # noqa: ANN001, ANN202
    for part in dotted.split("."):
        if not isinstance(data, dict) or part not in data:
            return _MISSING
        data = data[part]
    return data


def test_packaged_default_validates_and_is_generated_from_the_models(tmp_path: Path) -> None:
    loaded = load_config(DEFAULT_CONFIG)
    assert set(loaded.sections) == set(SECTIONS)
    assert gmd_default() == default_config_dict(), "configs/default.yaml is stale; run python -m gmd_active_learning.config --write-default configs/default.yaml"
    regenerated = write_default_config(tmp_path / "default.yaml")
    assert regenerated.read_text() == DEFAULT_CONFIG.read_text()
    assert build_config(default_config_dict(), base_dir=tmp_path).config_hash == build_config(gmd_default(), base_dir=tmp_path).config_hash


def test_phal_numeric_thresholds_are_kept_unless_deliberately_changed() -> None:
    phal, gmd = phal_default(), gmd_default()
    compared, changed = 0, set()
    for path, target in PHAL_KEY_MAP.items():
        if target.startswith("dropped") or not path.startswith(THRESHOLD_SECTIONS):
            continue
        value = _get(phal, path)
        if path == "ood_score.weights":
            pairs = [(f"monitor.metrics.{name}.weight", weight) for name, weight in value.items()]
        elif isinstance(value, bool) or not isinstance(value, (int, float, type(None))):
            continue
        else:
            pairs = [(item, value) for item in target.split(" + ")]
        for item, expected in pairs:
            got = _get(gmd, item)
            assert got is not _MISSING, (path, item)
            if got != expected:
                assert path in DELIBERATE_CHANGES, f"{path} -> {item}: PHAL {expected!r}, GMD {got!r} (not a listed deliberate change)"
                changed.add(path)
            compared += 1
    assert compared > 60
    assert changed == set(DELIBERATE_CHANGES), f"stale deliberate-change entries: {set(DELIBERATE_CHANGES) - changed}"


def test_no_absolute_paths_in_the_default() -> None:
    def walk(node, path=""):  # noqa: ANN001, ANN202
        if isinstance(node, dict):
            for key, value in node.items():
                yield from walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, str):
            yield path, node

    absolute = [(path, value) for path, value in walk(gmd_default()) if value.startswith("/") and path != "scheduler.shell"
                and not path.endswith(".shell")]
    assert absolute == []
