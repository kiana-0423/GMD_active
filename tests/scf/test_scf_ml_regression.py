# Ported from PHAL (tests/unit/test_scf_ml_regression.py) at 96263b6; MIT License, see NOTICE.
"""Regression guards: the optional SCF subsystem must not change anything by default.

PHAL's version of this file checked its global config schema, backend factory,
LAMMPS and standard CP2K backends; those belong to other GMD_active areas. What
is kept is the SCF-side contract: every feature is off by default, the default
section file matches the model, configuration errors are loud, PHAL configs
translate, recording off writes nothing, and importing the package pulls in
neither torch nor the experimental tooling.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gmd_active_learning.adapters.dft.scf.config import ScfConfig, default_section_path, scf_config
from gmd_active_learning.core.exceptions import ConfigurationError

REPO = Path(__file__).resolve().parents[2]


def test_every_scf_feature_is_off_by_default() -> None:
    config = ScfConfig()

    assert config.recording.enabled is False
    assert config.recording.record_density is False
    assert config.recording.record_trajectory is False
    assert config.retry.enabled is False
    assert config.ml_scf.enabled is False
    assert config.ml_scf.experimental is False
    assert config.ml_scf.mixing.method == "pulay"


def test_the_shipped_section_file_is_the_model_default() -> None:
    payload = yaml.safe_load(default_section_path().read_text(encoding="utf-8"))

    assert set(payload) == {"scf"}
    assert ScfConfig.model_validate(payload["scf"]) == ScfConfig()


@pytest.mark.parametrize(
    ("section", "match"),
    [
        ({"ml_scf": {"unknown_key": 1}}, "unknown_key"),
        ({"ml_scf": {"mixing": {"method": "magic"}}}, "method"),
        ({"ml_scf": {"mixing": {"method": "ml"}}}, "experimental"),
        ({"ml_scf": {"predictor": {"type": "torch"}}}, "experimental"),
        ({"ml_scf": {"electron_tolerance": 0.0}}, "electron_tolerance"),
        ({"ml_scf": {"guess_alpha": 1.5}}, "guess_alpha"),
        ({"retry": {"order": []}}, "order"),
        ({"dataset": {"split_ratios": {"train": 0.5, "validation": 0.1, "test": 0.1}}}, "sum to 1"),
        ({"ml_scf": "yes please"}, "ml_scf"),
    ],
)
def test_bad_configuration_is_rejected(section: dict, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        scf_config({"scf": section})


def test_phal_engine_selection_translates() -> None:
    standard = scf_config({"backend": {"dft": {"engine": "cp2k"}}, "cp2k": {"ml_scf": {"enabled": True}}})
    accelerated = scf_config({"backend": {"dft": {"engine": "cp2k_ml_scf"}}, "cp2k": {"ml_scf": {}}})

    # PHAL: selecting engine cp2k never reads cp2k.ml_scf.
    assert standard.ml_scf.enabled is False
    assert accelerated.ml_scf.enabled is True


def test_phal_keys_that_were_never_read_are_dropped_loudly(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        config = scf_config(
            {
                "backend": {"dft": {"engine": "cp2k_ml_scf"}},
                "cp2k": {
                    "ml_scf": {
                        "dataset_dir": "custom/dataset",
                        "electron_tolerance": 2e-3,
                        "mixing": {"method": "pulay", "fallback_method": "pulay"},
                    },
                    "scf_recording": {"enabled": True, "trajectory_strict": True},
                },
                "cp2k_ml_scf": {"runtime_dir": "/runtime/cp2k_ml", "controller_protocol": "phal-ml-scf-v2"},
            }
        )

    assert "fallback_method" in caplog.text and "runtime_dir" in caplog.text
    # The other formerly unread keys now have readers.
    assert config.dataset.root == "custom/dataset"
    assert config.ml_scf.electron_tolerance == pytest.approx(2e-3)
    assert config.recording.trajectory_strict is True


def test_recording_off_writes_nothing(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.scf_history import record_scf_history_text
    from tests.scf.fakes.fake_cp2k import fake_cp2k_scf_output

    input_path = tmp_path / "job.inp"
    input_path.write_text("&SCF\n  EPS_SCF 1.0E-6\n&END SCF\n", encoding="utf-8")

    written = record_scf_history_text(
        fake_cp2k_scf_output(), input_path=input_path, config={"scf": {}}, task_id="t"
    )

    assert written is None
    assert list(tmp_path.iterdir()) == [input_path]


def test_scf_recording_without_density_writes_only_the_history(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.scf_history import record_scf_history_text
    from tests.scf.fakes.fake_cp2k import fake_cp2k_scf_output

    input_path = tmp_path / "job.inp"
    input_path.write_text("&SCF\n  EPS_SCF 1.0E-6\n&END SCF\n", encoding="utf-8")
    config = {"scf": {"recording": {"enabled": True, "output_dir": str(tmp_path / "scf")}}}

    record_scf_history_text(fake_cp2k_scf_output(), input_path=input_path, config=config, task_id="t")

    assert (tmp_path / "scf" / "t" / "scf_history.json").is_file()
    assert not (tmp_path / "scf" / "t" / "density").exists()
    assert not (tmp_path / "scf" / "t" / "trajectory").exists()


RUNTIME_MODULES = [
    "gmd_active_learning.adapters.dft.scf",
    "gmd_active_learning.adapters.dft.scf.scf_history",
    "gmd_active_learning.adapters.dft.scf.ao_matrices",
    "gmd_active_learning.adapters.dft.scf.pulay_matrix",
    "gmd_active_learning.adapters.dft.scf.recording.density",
    "gmd_active_learning.adapters.dft.scf.recording.trajectory",
    "gmd_active_learning.adapters.dft.scf.recording.input",
    "gmd_active_learning.adapters.dft.scf.schema",
    "gmd_active_learning.adapters.dft.scf.dataset",
    "gmd_active_learning.adapters.dft.scf.validation",
    "gmd_active_learning.adapters.dft.scf.features",
    "gmd_active_learning.adapters.dft.scf.export",
    "gmd_active_learning.adapters.dft.scf.strategy",
    "gmd_active_learning.adapters.dft.scf.capabilities",
    "gmd_active_learning.adapters.dft.scf.cp2k_ml_backend",
    "gmd_active_learning.adapters.dft.scf.density",
    "gmd_active_learning.adapters.dft.scf.density.exchange",
    "gmd_active_learning.adapters.dft.scf.density.gnn_model",
    "gmd_active_learning.adapters.dft.scf.density.gnn_predictor",
    "gmd_active_learning.adapters.dft.scf.deploy",
    "gmd_active_learning.adapters.dft.scf.cli",
    "gmd_active_learning.adapters.dft.scf._torch",
]


def test_importing_the_runtime_package_never_imports_torch_or_the_tooling() -> None:
    """SCF-024 / SCF-039: torch stays optional and lazily imported; runtime never imports tools/."""

    code = (
        "import importlib, sys\n"
        f"for name in {RUNTIME_MODULES!r}:\n"
        "    importlib.import_module(name)\n"
        "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.') or m.startswith('tools'))\n"
        "print(','.join(bad))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""


def test_the_scf_history_recorder_needs_neither_torch_nor_ase() -> None:
    """Core labeling uses SCF-history recording; it must import with torch and ASE unavailable."""

    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'ase'):\n"
        "            raise ImportError(f'blocked {name}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history, record_scf_history_text\n"
        "import gmd_active_learning.adapters.dft.scf as scf\n"
        "print(parse_scf_history(' SCF run converged in 3 steps\\n').converged)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "True"


def test_torch_only_entry_points_raise_a_typed_error_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import importlib

    from gmd_active_learning.adapters.dft.scf import _torch

    real_import = importlib.import_module

    def blocked(name: str, *args, **kwargs):
        if name == "torch":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_torch.importlib, "import_module", blocked)

    with pytest.raises(_torch.TorchUnavailableError):
        _torch.require_torch()
    assert issubclass(_torch.TorchUnavailableError, ImportError)
    assert builtins is not None
