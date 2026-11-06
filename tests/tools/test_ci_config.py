"""OPS-006 (CI workflow, pre-commit) and TST-007 (offline runner, tests/README) static checks."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
PRECOMMIT = ROOT / ".pre-commit-config.yaml"
RUNNER = ROOT / "tools" / "test" / "run_all_offline.sh"
TESTS_README = ROOT / "tests" / "README.md"
INTEGRATION_VARIABLES = ("GMD_EXECUTABLE", "GMD_SGT_PATH", "CP2K_EXECUTABLE", "CP2K_BIN")


def _declared_markers() -> set[str]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("markers = [", 1)[1].split("]", 1)[0]
    return {match.group(1) for match in re.finditer(r'"(\w+):', block)}


def test_workflow_runs_the_offline_suite_on_python_310_to_312() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["offline"]
    assert job["strategy"]["matrix"]["python-version"] == ["3.10", "3.11", "3.12"]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "bash tools/test/run_all_offline.sh" in commands
    assert "ruff check" in commands and "mypy" in commands
    assert "render_config_reference.py --check" in commands
    for program in ("cp2k.psmp", "apptainer", "qsub", "docker build", "GMD_EXECUTABLE"):
        assert program not in commands  # CI never needs an external program


def test_runner_excludes_external_markers_and_they_are_declared() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '-m "not integration and not cp2k and not slow"' in text
    assert {"integration", "cp2k", "slow"} <= _declared_markers()
    for variable in INTEGRATION_VARIABLES:
        assert variable in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_runner_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_precommit_config_parses_and_uses_the_ci_lint_gate() -> None:
    config = yaml.safe_load(PRECOMMIT.read_text(encoding="utf-8"))
    hooks = {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}
    assert hooks["ruff"]["args"] == ["--select", "E9,F63,F7,F82"]
    assert "render_config_reference.py --check" in hooks["config-reference"]["entry"]
    for repo in config["repos"]:
        if repo["repo"] != "local":
            assert re.fullmatch(r"v\d+\.\d+\.\d+", repo["rev"])  # pinned, not a branch


def test_documented_integration_variables_are_read_by_tests() -> None:
    """PHAL documented ALLEGRO_MODEL while its tests read ALLEGRO_MODEL_PATH; check ours match."""

    readme = TESTS_README.read_text(encoding="utf-8")
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "tests").rglob("*.py"))
    for variable in INTEGRATION_VARIABLES:
        assert f"`{variable}`" in readme, variable
        assert re.search(rf"environ(?:\.get)?\(\s*[\"']{variable}[\"']|environ\[[\"']{variable}[\"']\]|setenv\([\"']{variable}", sources), variable


def test_tests_readme_lists_every_test_directory_and_marker() -> None:
    readme = TESTS_README.read_text(encoding="utf-8")
    for directory in sorted(path.name for path in (ROOT / "tests").iterdir() if path.is_dir() and path.name != "__pycache__"):
        assert f"`{directory}/`" in readme, directory
    for marker in _declared_markers():
        assert f"`{marker}`" in readme, marker
    assert "tools/test/run_all_offline.sh" in readme
