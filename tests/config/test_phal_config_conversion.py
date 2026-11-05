"""PHAL configuration -> GMD_active configuration (ledger CFG-002).

The mapping table must cover 100% of PHAL's CONFIG_SCHEMA leaves (frozen in
tests/fixtures/phal_config/schema_leaves.json), every mapped target must exist in the GMD
configuration, and PHAL's packaged default.yaml must convert into a valid GMD configuration that
carries PHAL's thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gmd_active_learning.config import build_config, default_config_dict
from gmd_active_learning.config_phal import PHAL_KEY_MAP, convert_phal_config
from gmd_active_learning.core.exceptions import ConfigurationError

from .helpers import PHAL_FIXTURES, REF_ROOT, phal_default, schema, schema_leaves

_MISSING = object()


def _get(data: dict, dotted: str):  # noqa: ANN001, ANN202
    for part in dotted.split("."):
        if not isinstance(data, dict) or part not in data:
            return _MISSING
        data = data[part]
    return data


def test_mapping_table_covers_every_schema_leaf_exactly() -> None:
    leaves = schema_leaves()
    assert len(leaves) == 192
    assert sorted(set(leaves) - set(PHAL_KEY_MAP)) == []
    assert sorted(set(PHAL_KEY_MAP) - set(leaves)) == []
    for path, target in PHAL_KEY_MAP.items():
        assert target.startswith("dropped: ") and len(target) > len("dropped: ") + 10 or not target.startswith("dropped"), path


def test_every_mapped_target_exists_in_the_gmd_configuration() -> None:
    defaults = default_config_dict()
    for path, target in PHAL_KEY_MAP.items():
        if target.startswith("dropped"):
            continue
        for item in target.split(" + "):
            if "<metric>" in item:
                for metric in phal_default()["ood_score"]["weights"]:
                    assert _get(defaults, item.replace("<metric>", metric)) is not _MISSING, (path, item, metric)
                continue
            assert _get(defaults, item) is not _MISSING, (path, item)


def test_phal_default_converts_into_a_valid_config_with_phal_thresholds(tmp_path: Path) -> None:
    phal = phal_default()
    raw, report = convert_phal_config(phal, gmd_executable="/opt/gmd/bin/gmd", initial_structure="system.xyz")
    loaded = build_config(raw, base_dir=tmp_path)
    required = {leaf["path"] for leaf in schema()["leaves"] if not leaf["optional"]}
    assert required <= set(report.mapped) | set(report.dropped)  # every leaf PHAL requires is accounted for
    # every mapped scalar arrives unchanged at its target(s)
    checked = 0
    for path, target in report.mapped.items():
        value = _get(phal, path)
        if isinstance(value, dict) or path in {"project.name", "backend.dft.engine", "backend.md.engine",
                                               "backend.mlip.engine", "cp2k.poisson_solver", "cp2k.slurm_directives",
                                               "lammps.run_steps", "online.run_steps", "cp2k.max_unknown_polls"}:
            continue
        for item in target.split(" + "):
            if item == "scf.ml_scf.enabled":
                continue
            got = _get(raw, item)
            if got is _MISSING and value is None:  # null in PHAL = the GMD default (also null) applies
                got = _get(default_config_dict(), item)
            got = list(got) if isinstance(got, tuple) else got
            assert got == value, (path, item, value, got)
            checked += 1
    assert checked > 100
    weights = phal["ood_score"]["weights"]
    monitor = loaded["monitor"]
    assert {name: getattr(monitor.metrics, name).weight for name in weights} == weights  # type: ignore[attr-defined]
    assert loaded["workflow"].run_id == "hotspot_active_learning"  # type: ignore[attr-defined]
    assert loaded["backends"].md.engine == "gmd" and loaded["backends"].mlip.engine == "gmd_sgt"  # type: ignore[attr-defined]
    assert loaded["cp2k"].poisson.non_periodic_solver == "WAVELET"  # type: ignore[attr-defined]
    assert loaded["md"].executable == "/opt/gmd/bin/gmd"  # type: ignore[attr-defined]
    candidates = loaded["workflow"].candidates  # type: ignore[attr-defined]
    assert (candidates.max_per_cycle, candidates.diversity_threshold, candidates.fingerprint) == (50, 0.1, "pair_distance_histogram")
    for name, expected in (("histogram_normalisation", "density"), ("pair_distances", "plain")):
        if hasattr(candidates, name):  # PHAL threshold semantics are preserved where the selector offers the switch
            assert getattr(candidates, name) == expected
    # Dropped keys that carried real content are reported, not silently lost.
    assert {"lammps.type_map", "allegro.model_paths", "hotspot.threshold"} <= set(report.dropped_with_content)
    assert any("run_steps" in warning for warning in report.warnings)


def test_phal_aliases_engines_and_free_text_fields(tmp_path: Path) -> None:
    phal = phal_default()
    phal["candidate_pool"]["fingerprint"] = "weighted_pair"
    phal["backend"]["dft"]["engine"] = "cp2k_ml_scf"
    phal["cp2k"]["slurm_directives"] = "#SBATCH --qos=debug\n\n#SBATCH --exclusive\n"
    phal["cp2k"]["max_walltime_seconds"] = 3600
    phal["online"]["run_steps"] = 5000
    phal["online"]["progress_file"] = "progress.json"
    # every optional PHAL leaf (absent from PHAL's default.yaml) converts too
    phal["cp2k"]["scf_recording"].update(record_density=True, density_strict=True, record_trajectory=True, trajectory_strict=False)
    phal["cp2k"]["ml_scf"] = {
        "enabled": True, "predictor": {"type": "dummy", "model_path": None, "options": {}}, "prediction_dir": "pred",
        "dataset_dir": "scf_data", "electron_tolerance": 0.01,
        "mixing": {"method": "pulay", "fallback_method": "pulay", "history_length": 6, "alpha": 0.3},
    }
    raw, report = convert_phal_config(phal, run_id="explicit")
    assert {leaf["path"] for leaf in schema()["leaves"]} == set(report.mapped) | set(report.dropped)
    loaded = build_config(raw, base_dir=tmp_path)
    assert loaded["workflow"].candidates.fingerprint == "type_weighted_pair_distance_histogram"  # type: ignore[attr-defined]
    assert loaded["workflow"].segment_steps == 5000 and loaded["workflow"].run_id == "explicit"  # type: ignore[attr-defined]
    assert loaded["backends"].dft.engine == "cp2k" and loaded["scf"].ml_scf.enabled  # type: ignore[attr-defined]
    assert loaded["cp2k"].resources.extra_directives == ["#SBATCH --qos=debug", "#SBATCH --exclusive"]  # type: ignore[attr-defined]
    assert loaded["cp2k"].resources.walltime == 3600  # type: ignore[attr-defined]
    assert loaded["monitor"].driver.progress_file == "progress.json"  # type: ignore[attr-defined]
    scf = loaded["scf"]  # type: ignore[assignment]
    assert scf.recording.record_trajectory and scf.ml_scf.mixing.history_length == 6  # type: ignore[attr-defined]
    assert scf.ml_scf.electron_tolerance == 0.01 and scf.dataset.root == "scf_data"  # type: ignore[attr-defined]


def test_unknown_phal_keys_and_engines_are_errors() -> None:
    phal = phal_default()
    phal["cp2k"]["typo_cutoff"] = 1
    with pytest.raises(ConfigurationError, match="cp2k.typo_cutoff"):
        convert_phal_config(phal)
    phal = phal_default()
    phal["backend"]["md"]["engine"] = "gromacs"
    with pytest.raises(ConfigurationError, match="backend.md.engine"):
        convert_phal_config(phal)


def test_cli_converts_a_phal_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from gmd_active_learning.cli import main

    output = tmp_path / "out" / "run.yaml"
    assert main(["convert-phal-config", "--phal", str(PHAL_FIXTURES / "default.yaml"), "--output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mapped"] + summary["dropped"] == len([leaf for leaf in schema()["leaves"] if not leaf["optional"]]) + 2
    assert main(["validate-config", "--config", str(output)]) == 0
    assert yaml.safe_load(output.read_text())["cp2k"]["executable"] == "cp2k.popt"


@pytest.mark.skipif(not (REF_ROOT / "src" / "hotspot_al" / "config_schema.py").is_file(), reason="ref/lammps_monitor not present")
def test_frozen_fixtures_match_the_reference_tree() -> None:
    assert (PHAL_FIXTURES / "default.yaml").read_bytes() == (REF_ROOT / "src/hotspot_al/config/default.yaml").read_bytes()
    text = (REF_ROOT / "src/hotspot_al/config_schema.py").read_text()
    for leaf in schema_leaves():
        assert f'"{leaf.rsplit(".", 1)[-1]}"' in text, leaf
