"""UTIL-005 (YAML loading + recursive merge) and UTIL-006 (logging configuration)."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from gmd_active_learning.config import build_config
from gmd_active_learning.core.exceptions import ConfigurationError
from gmd_active_learning.utils import logging as gmd_logging
from gmd_active_learning.utils.yaml import dump_yaml, load_yaml, merge_dicts, merge_yaml_files

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------- YAML
def test_load_yaml_empty_document_is_empty_dict(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("# only a comment\n", encoding="utf-8")
    assert load_yaml(path) == {}


def test_load_yaml_errors_are_configuration_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_yaml(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid YAML"):
        load_yaml(bad)
    listing = tmp_path / "list.yaml"
    listing.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mapping"):
        load_yaml(listing)


def test_merge_dicts_recurses_replaces_lists_and_deep_copies() -> None:
    base = {"a": {"b": 1, "c": [1, 2], "d": {"e": 1}}, "keep": {"x": [0]}}
    override = {"a": {"c": [3], "d": {"f": 2}, "g": None}, "new": {"y": [1]}}
    merged = merge_dicts(base, override)
    assert merged == {
        "a": {"b": 1, "c": [3], "d": {"e": 1, "f": 2}, "g": None},
        "keep": {"x": [0]},
        "new": {"y": [1]},
    }
    # No shared mutable state with either input (PHAL deep-copied as well).
    merged["keep"]["x"].append(9)
    merged["new"]["y"].append(9)
    merged["a"]["d"]["e"] = 99
    assert base == {"a": {"b": 1, "c": [1, 2], "d": {"e": 1}}, "keep": {"x": [0]}}
    assert override["new"] == {"y": [1]}


def test_mapping_replaces_scalar_and_scalar_replaces_mapping() -> None:
    assert merge_dicts({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}
    assert merge_dicts({"a": {"b": 2}}, {"a": 3}) == {"a": 3}


def test_merge_yaml_files_layers_left_to_right_and_feeds_the_central_config(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    dump_yaml({"workflow": {"run_id": "overlay-run", "max_cycles": 7}, "monitor": {"enabled": False}}, overlay)
    raw = merge_yaml_files(ROOT / "configs" / "default.yaml", overlay)
    assert raw["workflow"]["run_id"] == "overlay-run"
    assert raw["workflow"]["segment_steps"] == load_yaml(ROOT / "configs" / "default.yaml")["workflow"]["segment_steps"]
    config = build_config(raw, base_dir=tmp_path)
    assert config["workflow"].max_cycles == 7  # type: ignore[attr-defined]
    assert config["monitor"].enabled is False  # type: ignore[attr-defined]


def test_dump_yaml_round_trips_and_keeps_key_order(tmp_path: Path) -> None:
    data = {"z": 1, "a": {"y": [1, 2], "b": "x"}}
    path = dump_yaml(data, tmp_path / "sub" / "out.yaml")
    assert load_yaml(path) == data
    assert path.read_text(encoding="utf-8").splitlines()[0] == "z: 1"


# ------------------------------------------------------------------------------------ logging
@pytest.fixture()
def fresh_logger(monkeypatch: pytest.MonkeyPatch):
    for variable in (gmd_logging.ENV_LEVEL, gmd_logging.ENV_FILE, gmd_logging.ENV_FORMAT):
        monkeypatch.delenv(variable, raising=False)
    name = "gmd_active_learning_test_logging"
    logger = logging.getLogger(name)
    yield name
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


def _consoles(logger: logging.Logger) -> list[logging.Handler]:
    return [handler for handler in logger.handlers if getattr(handler, "_gmd_al_console", False)]


def test_reconfiguring_retunes_the_console_handler(fresh_logger: str) -> None:
    """Port of PHAL test_online_monitor.py:198: a second call changes the level, adds no handler."""

    logger = gmd_logging.configure_logging("WARNING", name=fresh_logger)
    assert [handler.level for handler in _consoles(logger)] == [logging.WARNING]
    logger = gmd_logging.configure_logging("DEBUG", name=fresh_logger)
    assert [handler.level for handler in _consoles(logger)] == [logging.DEBUG]
    assert logger.level == logging.DEBUG


def test_json_format_emits_one_parseable_object_per_line(fresh_logger: str) -> None:
    logger = gmd_logging.configure_logging("INFO", log_format="json", name=fresh_logger)
    stream = io.StringIO()
    _consoles(logger)[0].setStream(stream)  # type: ignore[attr-defined]
    logger.info("first %s", "message")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.exception("second")
    lines = stream.getvalue().splitlines()
    records = [json.loads(line) for line in lines if line.startswith("{")]
    assert [record["message"] for record in records] == ["first message", "second"]
    assert records[0]["level"] == "INFO" and records[0]["logger"] == fresh_logger
    assert "RuntimeError: boom" in records[1]["exception"]
    assert len(records) == 2 and all(line.startswith("{") for line in lines)


def test_file_handler_is_debug_and_deduplicated(fresh_logger: str, tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "run.log"
    logger = gmd_logging.configure_logging("WARNING", log_file=log_file, name=fresh_logger)
    gmd_logging.configure_logging("WARNING", log_file=log_file, name=fresh_logger)
    files = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
    assert len(files) == 1 and files[0].level == logging.DEBUG
    logger.debug("debug goes to the file only")
    files[0].flush()
    assert "debug goes to the file only" in log_file.read_text(encoding="utf-8")


def test_environment_overrides_config_and_arguments(
    fresh_logger: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(gmd_logging.ENV_LEVEL, "ERROR")
    monkeypatch.setenv(gmd_logging.ENV_FORMAT, "json")
    monkeypatch.setenv(gmd_logging.ENV_FILE, str(tmp_path / "env.log"))
    config = {"logging": {"level": "DEBUG", "format": "text", "file": str(tmp_path / "cfg.log")}}
    logger = gmd_logging.configure_logging("INFO", config=config, name=fresh_logger)
    console = _consoles(logger)[0]
    assert console.level == logging.ERROR  # PHAL ignored LOG_LEVEL whenever the config set a level
    assert isinstance(console.formatter, gmd_logging.JsonFormatter)
    assert (tmp_path / "env.log").exists() and not (tmp_path / "cfg.log").exists()


def test_config_section_is_used_without_environment(fresh_logger: str, tmp_path: Path) -> None:
    logger = gmd_logging.configure_logging(config={"level": "warning", "format": "json"}, name=fresh_logger)
    assert _consoles(logger)[0].level == logging.WARNING
    assert isinstance(_consoles(logger)[0].formatter, gmd_logging.JsonFormatter)


def test_invalid_level_and_format_are_rejected(fresh_logger: str) -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        gmd_logging.configure_logging("LOUD", name=fresh_logger)
    with pytest.raises(ValueError, match="unknown log format"):
        gmd_logging.configure_logging("INFO", log_format="xml", name=fresh_logger)


def test_records_still_propagate_to_root_by_default(fresh_logger: str, caplog: pytest.LogCaptureFixture) -> None:
    logger = gmd_logging.configure_logging("INFO", name=fresh_logger)
    child = gmd_logging.get_logger(f"{fresh_logger}.child")
    assert child.handlers == []  # module loggers carry no handlers
    with caplog.at_level(logging.INFO):
        child.info("visible to caplog")
    assert "visible to caplog" in caplog.text
    assert logger.propagate is True
    assert gmd_logging.configure_logging("INFO", name=fresh_logger, propagate=False).propagate is False


def test_get_logger_with_settings_configures_like_phal(fresh_logger: str) -> None:
    logger = gmd_logging.get_logger(fresh_logger, "INFO")
    logger = gmd_logging.get_logger(fresh_logger, "ERROR")
    assert [handler.level for handler in _consoles(logger)] == [logging.ERROR]
    assert logger.propagate is False


def test_existing_cli_call_signature_still_works(fresh_logger: str) -> None:
    # gmd-al calls configure_logging(<int level>) on the package logger.
    logger = gmd_logging.configure_logging(logging.WARNING, name=fresh_logger)
    assert _consoles(logger)[0].level == logging.WARNING
