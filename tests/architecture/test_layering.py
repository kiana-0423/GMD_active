# Ported from PHAL (tests/architecture/test_core_domain_boundaries.py) at 96263b6; MIT License, see NOTICE.
"""Executable layering rules for gmd_active_learning (ledger CORE-007).

Layers (inner to outer):

* foundation  ``core``, ``utils``: imported by everyone, import nothing above them.
* contracts   typed data shared by every layer; may import only foundation code.
* domain      ``monitors events hotspots extraction datasets active_learning retraining``:
              backend-neutral algorithms and policy. They must not import adapters, the
              composition root (``workflow.assembly``), the CLI, labeling services or any
              backend (torch, gmd_sgt, subprocess, CP2K tooling).
* application ``workflow`` (controller, protocols, store) and ``labeling`` (the persistent label
              service): orchestrate the domain through protocols.
* infrastructure ``adapters`` (GMD, GMD_SGT, CP2K, schedulers, SCF tooling), ``importers``
              (legacy LAMMPS data), ``reliability_model`` (optional torch classifier).
* entry points ``cli``, ``config``, ``config_phal``, ``workflow.assembly``: may import anything.

Differences from PHAL's guard: relative imports are resolved (PHAL recorded ``node.module``
without the level, so ``from ..backends import x`` evaded the rule and ``from . import x`` was
skipped); ``utils`` is guarded instead of exempt; ``importlib.import_module("...")`` string
targets are checked as imports; every package must be classified.
"""

from __future__ import annotations

import importlib.util

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

# The package is located through the import system, so these checks also run on an installed
# wheel (clean-install gate), not only on a source checkout.
SRC = Path(importlib.util.find_spec("gmd_active_learning").origin).resolve().parents[1]  # src/ or site-packages
PACKAGE = "gmd_active_learning"
PACKAGE_ROOT = SRC / PACKAGE

FOUNDATION = ("core", "utils")
CONTRACTS = ("contracts",)
DOMAIN = ("monitors", "events", "hotspots", "extraction", "datasets", "active_learning", "retraining")
APPLICATION = ("workflow", "labeling")
INFRASTRUCTURE = ("adapters", "importers", "reliability_model")
ENTRY_POINTS = ("cli", "config", "config_phal", "__init__")

# Third-party backends and process launchers that domain code must reach only through adapters.
BACKEND_MODULES = ("torch", "gmd_sgt", "gmd_se3gnn", "subprocess", "cp2k", "lammps", "nequip", "allegro", "kubernetes")
# The project's own outer layers.
OUTER_FOR_DOMAIN = (
    f"{PACKAGE}.adapters",
    f"{PACKAGE}.workflow.assembly",
    f"{PACKAGE}.cli",
    f"{PACKAGE}.config_phal",
    f"{PACKAGE}.labeling",
    f"{PACKAGE}.importers",
    f"{PACKAGE}.reliability_model",
)
# Domain code may use the workflow's persistence and protocols, not its composition root.
ALLOWED_FOR_DOMAIN = (f"{PACKAGE}.workflow.store", f"{PACKAGE}.workflow.protocols")


@dataclass(frozen=True)
class ImportRef:
    module: str  # importing module, dotted
    target: str  # imported module, absolute dotted
    line: int


def module_name(path: Path, src: Path = SRC) -> str:
    parts = list(path.relative_to(src).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(current: str, is_package: bool, level: int, module: str | None) -> str:
    base = current.split(".")
    if not is_package:
        base = base[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, *([module] if module else [])])


def collect_imports(path: Path, src: Path = SRC) -> list[ImportRef]:
    """Every import of ``path`` as an absolute dotted name, relative imports resolved."""

    current = module_name(path, src)
    is_package = path.name == "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    refs: list[ImportRef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            refs.extend(ImportRef(current, alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = _resolve_relative(current, is_package, node.level, node.module)
            refs.append(ImportRef(current, base, node.lineno))
            # ``from pkg import submodule`` imports pkg.submodule as well.
            refs.extend(ImportRef(current, f"{base}.{alias.name}", node.lineno) for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if name in {"import_module", "__import__"} and node.args and isinstance(node.args[0], ast.Constant):
                if isinstance(node.args[0].value, str):
                    refs.append(ImportRef(current, node.args[0].value, node.lineno))
    return refs


def _matches(target: str, prefixes: tuple[str, ...]) -> bool:
    return any(target == prefix or target.startswith(prefix + ".") for prefix in prefixes)


def layer_of(module: str) -> str:
    parts = module.split(".")
    top = parts[1] if len(parts) > 1 else "__init__"
    if module == f"{PACKAGE}.workflow.assembly":
        return "entry"
    for layer, names in (
        ("foundation", FOUNDATION),
        ("contracts", CONTRACTS),
        ("domain", DOMAIN),
        ("application", APPLICATION),
        ("infrastructure", INFRASTRUCTURE),
        ("entry", ENTRY_POINTS),
    ):
        if top in names:
            return layer
    return "unclassified"


def violations_for(ref: ImportRef) -> list[str]:
    layer = layer_of(ref.module)
    target = ref.target
    problems: list[str] = []
    where = f"{ref.module}:{ref.line} imports {target}"
    if _matches(target, ("hotspot_al",)):
        problems.append(f"{where}: runtime code must never import PHAL")
    own = _matches(target, (PACKAGE,))
    if layer == "foundation":
        if own and not _matches(target, tuple(f"{PACKAGE}.{name}" for name in FOUNDATION)):
            problems.append(f"{where}: foundation code may import only core/utils")
    elif layer == "contracts":
        if own and not _matches(target, tuple(f"{PACKAGE}.{name}" for name in (*FOUNDATION, *CONTRACTS))):
            problems.append(f"{where}: contracts may import only core/utils/contracts")
        if _matches(target, BACKEND_MODULES):
            problems.append(f"{where}: contracts must not import a backend")
    elif layer == "domain":
        if _matches(target, BACKEND_MODULES):
            problems.append(f"{where}: domain code must not import a backend or process launcher")
        if own and _matches(target, OUTER_FOR_DOMAIN):
            problems.append(f"{where}: domain code must not import an adapter, service or entry point")
        if own and _matches(target, (f"{PACKAGE}.workflow",)) and not _matches(target, ALLOWED_FOR_DOMAIN):
            if target != f"{PACKAGE}.workflow":
                problems.append(f"{where}: domain code may use only workflow.store/protocols")
    elif layer == "application":
        if own and _matches(target, (f"{PACKAGE}.workflow.assembly", f"{PACKAGE}.cli")):
            problems.append(f"{where}: application code must not import the composition root or CLI")
        if ref.module.startswith(f"{PACKAGE}.workflow") and own and _matches(target, (f"{PACKAGE}.adapters",)):
            problems.append(f"{where}: the workflow reaches adapters only through workflow.assembly")
        if ref.module.startswith(f"{PACKAGE}.workflow") and _matches(target, BACKEND_MODULES):
            problems.append(f"{where}: the workflow must not import a backend")
    elif layer == "infrastructure":
        if own and _matches(target, (f"{PACKAGE}.workflow.assembly", f"{PACKAGE}.cli")):
            problems.append(f"{where}: adapters must not import the composition root or CLI")
    return problems


def source_files(root: Path = PACKAGE_ROOT) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def scan(root: Path = PACKAGE_ROOT, src: Path = SRC) -> list[str]:
    problems: list[str] = []
    for path in source_files(root):
        for ref in collect_imports(path, src):
            problems.extend(violations_for(ref))
    return problems


# ------------------------------------------------------------------------------------- tests
def test_every_module_respects_the_layering() -> None:
    problems = scan()
    assert not problems, "layering violations:\n" + "\n".join(problems)


def test_every_package_is_classified() -> None:
    classified = {*FOUNDATION, *CONTRACTS, *DOMAIN, *APPLICATION, *INFRASTRUCTURE, *ENTRY_POINTS}
    present = {path.name for path in PACKAGE_ROOT.iterdir() if path.is_dir() and (path / "__init__.py").exists()}
    present |= {path.stem for path in PACKAGE_ROOT.glob("*.py")}
    unclassified = sorted(present - classified)
    assert not unclassified, f"classify these packages/modules in test_layering.py: {unclassified}"
    stale = sorted(classified - present)
    assert not stale, f"classified names that no longer exist: {stale}"


def test_contracts_and_domain_import_no_backend_at_runtime() -> None:
    """Belt and braces: importing every domain package in a fresh interpreter loads no backend."""

    import subprocess  # the test (not the code under test) may launch processes
    import sys

    modules = [f"{PACKAGE}.{name}" for name in (*CONTRACTS, *DOMAIN)]
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}: importlib.import_module(name)\n"
        f"bad = sorted(m for m in sys.modules if m.split('.')[0] in {BACKEND_MODULES!r} or "
        f"m.startswith('{PACKAGE}.adapters') or m == '{PACKAGE}.workflow.assembly' or m.startswith('hotspot_al'))\n"
        "print('\\n'.join(bad))\n"
    )
    import os

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH", "")]))}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=SRC.parent, env=env)
    # ``subprocess`` itself is imported by the stdlib (e.g. via platform/ctypes helpers); only the
    # project's own direct imports of it are forbidden, which the AST scan checks.
    loaded = [name for name in out.stdout.split() if name != "subprocess"]
    assert loaded == [], f"importing contracts/domain loaded backend or adapter modules: {loaded}"


def test_no_source_file_reads_the_phal_reference_tree() -> None:
    offenders: list[str] = []
    for path in source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
                # Docstrings may cite PHAL sources; executable path literals may not point into ref/.
                if "lammps_monitor" in text and ("/" in text or "\\" in text) and "\n" not in text:
                    offenders.append(f"{module_name(path)}:{node.lineno}: {text!r}")
                if text.startswith(("ref/", "./ref/", "../ref/")):
                    offenders.append(f"{module_name(path)}:{node.lineno}: {text!r}")
    assert not offenders, "source code references the PHAL reference tree:\n" + "\n".join(offenders)


# --------------------------------------------------------------------- the guard catches violations
def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("relative", "source", "expected"),
    [
        ("monitors/bad.py", "from ..adapters import gmd\n", "must not import an adapter"),
        ("monitors/sub/bad.py", "from ...adapters.sgt import backend\n", "must not import an adapter"),
        ("events/bad.py", "import torch\n", "must not import a backend"),
        ("extraction/bad.py", "import subprocess\n", "must not import a backend"),
        ("datasets/bad.py", "from gmd_sgt.data import reader\n", "must not import a backend"),
        ("hotspots/bad.py", "from ..workflow import assembly\n", "must not import an adapter"),
        ("retraining/bad.py", "from gmd_active_learning.workflow.controller import Stage\n", "only workflow.store"),
        ("active_learning/bad.py", "import importlib\nimportlib.import_module('gmd_active_learning.adapters.registry')\n", "must not import an adapter"),
        ("contracts/bad.py", "from ..monitors import config\n", "contracts may import only"),
        ("utils/bad.py", "from gmd_active_learning.extraction import regions\n", "foundation code may import only"),
        ("workflow/bad.py", "from ..adapters.gmd import engine\n", "only through workflow.assembly"),
        ("adapters/bad.py", "import hotspot_al.monitor\n", "never import PHAL"),
    ],
)
def test_guard_flags_synthetic_violations(tmp_path: Path, relative: str, source: str, expected: str) -> None:
    src = tmp_path / "src"
    root = src / PACKAGE
    _write(root, "__init__.py", "")
    for package in Path(relative).parents:
        if str(package) != ".":
            _write(root, str(package / "__init__.py"), "")
    _write(root, relative, source)
    problems = scan(root, src)
    assert any(expected in problem for problem in problems), problems


def test_allowed_imports_pass(tmp_path: Path) -> None:
    src = tmp_path / "src"
    root = src / PACKAGE
    _write(root, "__init__.py", "")
    _write(root, "monitors/__init__.py", "from . import ok\n")
    _write(root, "monitors/ok.py", "import numpy\nfrom ..contracts import Frame\nfrom ..utils import periodic\n")
    _write(root, "retraining/__init__.py", "")
    _write(root, "retraining/ok.py", "from gmd_active_learning.workflow.store import WorkflowStore\n")
    _write(root, "workflow/__init__.py", "")
    _write(root, "workflow/assembly.py", "from ..adapters.gmd import engine\nimport torch\n")
    assert scan(root, src) == []
