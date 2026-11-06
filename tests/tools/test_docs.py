"""DOC-001/003/004/007 and the known-issues port: the documentation is consistent with the code."""

from __future__ import annotations

import importlib.util

import importlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
MKDOCS = ROOT / "tools" / "docs" / "mkdocs.yml"
OWN_PAGES = [
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "tests" / "README.md",
    DOCS / "index.md",
    DOCS / "architecture.md",
    DOCS / "operations.md",
    DOCS / "contributing.md",
    DOCS / "known-issues.md",
    DOCS / "runtime-data-contract.md",
    DOCS / "api.md",
    *sorted((DOCS / "adr").glob("*.md")),
    *sorted((DOCS / "containers").glob("*.md")),
    DOCS / "migration" / "phal_release_baseline" / "README.md",
]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _nav_files(node: object) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [item for child in node for item in _nav_files(child)]
    if isinstance(node, dict):
        return [item for child in node.values() for item in _nav_files(child)]
    return []


def _mkdocs_config() -> dict:
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", lambda loader, suffix, node: None)
    return yaml.load(MKDOCS.read_text(encoding="utf-8"), Loader=Loader)  # noqa: S506 - SafeLoader subclass


def test_mkdocs_nav_points_at_existing_pages() -> None:
    config = _mkdocs_config()
    assert (MKDOCS.parent / config["docs_dir"]).resolve() == DOCS
    missing = [page for page in _nav_files(config["nav"]) if not (DOCS / page).is_file()]
    assert not missing, missing
    titles = [next(iter(item)) for item in config["nav"] if isinstance(item, dict)]
    assert len(titles) == len(set(titles))  # PHAL's nav had two entries titled "Architecture" in one section


@pytest.mark.parametrize("page", OWN_PAGES, ids=lambda path: str(path.relative_to(ROOT)))
def test_relative_links_resolve(page: Path) -> None:
    broken = []
    for target in LINK.findall(page.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = (page.parent / target.split("#", 1)[0]).resolve()
        if not path.exists():
            broken.append(target)
    assert not broken, f"{page.relative_to(ROOT)}: broken links {broken}"


def test_api_page_targets_import() -> None:
    targets = re.findall(r"^::: ([\w.]+)$", (DOCS / "api.md").read_text(encoding="utf-8"), flags=re.M)
    assert targets
    for target in targets:
        importlib.import_module(target)


def test_readme_documents_every_cli_command() -> None:
    from gmd_active_learning.cli import build_parser

    parser = build_parser()
    commands = next(action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction").choices  # noqa: SLF001
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    undocumented = [name for name in commands if name not in {"train-reliability-model", "scf"} and name not in readme]
    assert not undocumented, undocumented
    for word in ("Install", "Quick start", "real_cycle", "Limitations", "Attribution", "NOTICE"):
        assert word in readme


def test_architecture_names_existing_modules_and_protocols() -> None:
    text = (DOCS / "architecture.md").read_text(encoding="utf-8")
    from gmd_active_learning.workflow import protocols
    from gmd_active_learning.workflow.controller import Stage

    for name in ("MDEngine", "FrameEnricher", "FrameMonitor", "RegionExtractor", "Labeler", "DatasetStore", "Trainer"):
        assert hasattr(protocols, name) and f"`{name}`" in text
    for stage in Stage:
        assert stage.name in text, stage
    for path in re.findall(r"`((?:adapters|workflow|monitors|labeling|extraction|datasets|retraining|importers)/[\w/]+\.py)`", text):
        assert (Path(importlib.util.find_spec("gmd_active_learning").origin).parent / path).is_file(), path


def test_known_issues_cover_phal_issues_and_new_findings() -> None:
    text = (DOCS / "known-issues.md").read_text(encoding="utf-8")
    for number in range(1, 7):
        assert re.search(rf"^### {number}\. ", text, flags=re.M), number
    for finding in ("F-33", "F-34", "F-35"):
        assert f"### {finding}" in text
    assert "WAVELET" in text and "never built" in text


def test_adr_004_records_the_gmd_decisions() -> None:
    text = (DOCS / "adr" / "004-distance-histogram-dedup.md").read_text(encoding="utf-8")
    for phrase in ("L2", "cosine", "minimum image", "density", "F-30"):
        assert phrase in text
    from gmd_active_learning.active_learning.candidates import CandidateConfig

    fields = CandidateConfig.model_fields
    for key in ("pair_distances", "histogram_normalisation", "compare_with_labeled", "tie_break"):
        assert key in fields and f"`{key}" in text


def test_changelog_has_the_release_entry() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    from importlib.metadata import PackageNotFoundError, version

    assert "## [v0.4.0]" in text
    try:
        assert version("GMD_active_learning") == "0.4.0"
    except PackageNotFoundError:
        assert 'version = "0.4.0"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("mkdocs") is None, reason="mkdocs is not installed (docs extra)")
def test_mkdocs_build_strict(tmp_path: Path) -> None:
    subprocess.run(["mkdocs", "build", "--strict", "-f", str(MKDOCS), "-d", str(tmp_path / "site")], check=True, cwd=ROOT)
