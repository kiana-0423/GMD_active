"""Regenerate the frozen PHAL configuration fixtures from the reference tree (one-off).

    python tests/fixtures/phal_config/make_phal_config_fixture.py [REF_ROOT]

Writes ``default.yaml`` (byte copy of PHAL ``src/hotspot_al/config/default.yaml`` at 96263b6) and
``schema_leaves.json`` (every leaf of PHAL ``config_schema.CONFIG_SCHEMA`` with its accepted types
and whether it is optional, plus the positive-range paths and choice rules of ``validate_config``).
Only this generator reads ``ref/``; the tests read the frozen files, so they keep working after
``ref/`` is removed.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_REF = HERE.parents[2] / "ref" / "lammps_monitor"


def _type_names(expected: Any) -> list[str]:
    if isinstance(expected, tuple):
        return [name for item in expected for name in _type_names(item)]
    return ["None" if expected is type(None) else expected.__name__]


def _walk(schema: Mapping[str, Any], prefix: str, optional: bool, schema_mod: Any, out: list[dict[str, Any]]) -> None:
    for key, expected in schema.items():
        path = f"{prefix}.{key}" if prefix else key
        is_optional = optional
        if isinstance(expected, schema_mod.OptionalKey):
            is_optional, expected = True, expected.schema
        if isinstance(expected, Mapping):
            _walk(expected, path, is_optional, schema_mod, out)
        elif isinstance(expected, schema_mod.DictOf):
            out.append({"path": path, "types": [f"dict[{'|'.join(_type_names(expected.key_type))}, {'|'.join(_type_names(expected.value_type))}]"], "optional": is_optional})
        else:
            out.append({"path": path, "types": _type_names(expected), "optional": is_optional})


def main(ref_root: Path = DEFAULT_REF) -> None:
    sys.path.insert(0, str(ref_root / "src"))
    import hotspot_al.config_schema as schema_mod  # noqa: PLC0415 - generator only

    leaves: list[dict[str, Any]] = []
    _walk(schema_mod.CONFIG_SCHEMA, "", False, schema_mod, leaves)
    payload = {
        "source": "PHAL hotspot_al.config_schema at 96263b6 (MIT License, see NOTICE)",
        "leaves": leaves,
        "positive_paths": [
            "lammps.timestep_fs", "online.dump_freq", "online.monitor_freq", "online.max_errors",
            "buffer.pre_trigger_frames", "buffer.post_trigger_frames", "monitor.lj_cutoff",
            "extraction.extract_radius", "extraction.max_atoms", "extraction.min_atoms", "extraction.block.halo",
            "extraction.block.max_merged_blocks", "extraction.block.buffer.inner", "extraction.block.buffer.outer",
            "extraction.block.max_atoms", "extraction.block.min_atoms", "cp2k.max_scf", "cp2k.h_only_opt.max_iter",
            "retraining.min_new_samples", "retraining.interval_hours", "candidate_pool.max_candidates_per_round",
        ],
        "choices": {
            "cp2k.uncapped_fragment_policy": ["error", "warn", "allow"],
            "cp2k.ml_scf.mixing.method": list(schema_mod.ML_SCF_MIXING_METHODS),
            "cp2k.ml_scf.mixing.fallback_method": [m for m in schema_mod.ML_SCF_MIXING_METHODS if m != "ml"],
        },
    }
    (HERE / "schema_leaves.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    shutil.copyfile(ref_root / "src" / "hotspot_al" / "config" / "default.yaml", HERE / "default.yaml")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REF)
