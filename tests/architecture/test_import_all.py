# Ported from PHAL (tests/scripts/check_imports.py) at 96263b6; MIT License, see NOTICE.
"""Recursive import smoke test of every module under gmd_active_learning (ledger CORE-008).

PHAL's script imported every module unconditionally and therefore failed in minimal environments.
Here (1) every module is imported in the test environment, with modules that need an optional
dependency skipped explicitly when it is missing, and (2) a fresh interpreter with torch and
GMD_SGT *blocked* proves that only the declared torch modules need torch: the SCF tooling, the
GMD_SGT adapter and the CLI must stay importable (they load torch lazily, when used).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import gmd_active_learning

SRC = Path(__file__).resolve().parents[2] / "src"

# Modules that import torch at module level by design (optional reliability classifier).
TORCH_REQUIRED = frozenset(
    {
        "gmd_active_learning.reliability_model.dataset",
        "gmd_active_learning.reliability_model.model",
        "gmd_active_learning.reliability_model.predict",
        "gmd_active_learning.reliability_model.train",
        "gmd_active_learning.utils.torch_utils",
    }
)
OPTIONAL_DEPENDENCIES = {"torch": TORCH_REQUIRED}


def _all_modules() -> list[str]:
    errors: list[str] = []
    names = [
        info.name
        for info in pkgutil.walk_packages(gmd_active_learning.__path__, "gmd_active_learning.", onerror=errors.append)
    ]
    assert not errors, f"package walk failed for {errors}"
    return sorted({"gmd_active_learning", *names})


MODULES = _all_modules()


def test_walk_covers_every_source_file() -> None:
    package_root = SRC / "gmd_active_learning"
    expected = set()
    for path in package_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        expected.add(".".join(parts))
    assert expected == set(MODULES)


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    for dependency, needing in OPTIONAL_DEPENDENCIES.items():
        if module in needing and importlib.util.find_spec(dependency) is None:
            pytest.skip(f"{module} needs optional dependency {dependency!r}, which is not installed")
    importlib.import_module(module)


def test_only_declared_modules_need_torch_or_gmd_sgt() -> None:
    code = (
        "import importlib, json, sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['gmd_sgt'] = None\n"
        "failed = {}\n"
        f"for name in {MODULES!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        failed[name] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(failed))\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH", "")]))}
    env.pop("GMD_SGT_PATH", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    failed: dict[str, str] = json.loads(out.stdout.strip().splitlines()[-1])
    unexpected = {name: error for name, error in failed.items() if name not in TORCH_REQUIRED}
    assert not unexpected, f"modules that must import without torch/GMD_SGT failed: {unexpected}"
    # The declared list must stay minimal: every declared module really needs torch.
    assert set(failed) == TORCH_REQUIRED, f"TORCH_REQUIRED is stale; these import fine: {sorted(TORCH_REQUIRED - set(failed))}"
    assert all("torch" in error for error in failed.values())
