# Ported from PHAL (hotspot_al.utils.logging) at 96263b6; MIT License, see NOTICE.
"""Project logging configuration (ledger UTIL-006).

Modules obtain loggers with ``get_logger(__name__)``; those are plain children of the package
logger ``gmd_active_learning`` and carry no handlers of their own. Applications (the ``gmd-al``
CLI, examples, tools) call :func:`configure_logging` once, which attaches to the package logger

* one console handler (stderr), re-tuned rather than duplicated when called again,
* optionally one file handler at DEBUG level (de-duplicated by resolved path),

both using a text or a one-line JSON formatter (:class:`JsonFormatter`).

Environment overrides (they win over arguments and config, so an operator can raise verbosity
without editing a run configuration):

``GMD_AL_LOG_LEVEL``   level name or number (PHAL: ``LOG_LEVEL``, which was ignored whenever the
                       config set a level, i.e. always with the default config)
``GMD_AL_LOG_FILE``    path of the DEBUG file log (PHAL: ``HOTSPOT_AL_LOG_FILE``)
``GMD_AL_LOG_FORMAT``  ``text`` or ``json`` (PHAL: ``HOTSPOT_AL_LOG_FORMAT``)

Deviation from PHAL: the package logger keeps ``propagate=True`` by default, so records still
reach root handlers (pytest's ``caplog``, an embedding application). Pass ``propagate=False`` for
PHAL's isolated behaviour when the root logger also prints.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PACKAGE_LOGGER = "gmd_active_learning"
ENV_LEVEL = "GMD_AL_LOG_LEVEL"
ENV_FILE = "GMD_AL_LOG_FILE"
ENV_FORMAT = "GMD_AL_LOG_FORMAT"
LOG_FORMATS = ("text", "json")
TEXT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_CONSOLE_TAG = "_gmd_al_console"


class JsonFormatter(logging.Formatter):
    """Format each record as one compact JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def resolve_level(level: int | str | None, default: int = logging.INFO) -> int:
    """Turn a level name/number into an ``int``; unknown names raise ``ValueError``."""

    if level is None or level == "":
        return default
    if isinstance(level, bool):
        raise ValueError(f"invalid log level {level!r}")
    if isinstance(level, int):
        return level
    text = str(level).strip()
    if text.isdigit():
        return int(text)
    value = logging.getLevelName(text.upper())
    if not isinstance(value, int):
        raise ValueError(f"unknown log level {level!r}")
    return value


def make_formatter(log_format: str | None) -> logging.Formatter:
    fmt = str(log_format or "text").strip().lower()
    if fmt not in LOG_FORMATS:
        raise ValueError(f"unknown log format {log_format!r}; expected one of {LOG_FORMATS}")
    return JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT)


def _has_file_handler(logger: logging.Logger, path: Path) -> bool:
    resolved = path.resolve()
    return any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved
        for handler in logger.handlers
    )


def _setup(
    logger: logging.Logger,
    level: int,
    formatter: logging.Formatter,
    log_file: str | Path | None,
    propagate: bool,
) -> logging.Logger:
    consoles = [handler for handler in logger.handlers if getattr(handler, _CONSOLE_TAG, False)]
    if consoles:
        # Re-tune instead of adding a second handler (PHAL test_online_monitor.py:198).
        for handler in consoles:
            handler.setLevel(level)
            handler.setFormatter(formatter)
    else:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        setattr(console, _CONSOLE_TAG, True)
        logger.addHandler(console)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(formatter)
    if log_file is not None and str(log_file) != "":
        path = Path(log_file).expanduser()
        if not _has_file_handler(logger, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    has_file = any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
    # The logger must let DEBUG through when a DEBUG file handler is attached.
    logger.setLevel(min(level, logging.DEBUG) if has_file else level)
    logger.propagate = propagate
    return logger


def configure_logging(
    level: int | str | None = None,
    *,
    log_file: str | Path | None = None,
    log_format: str | None = None,
    config: Mapping[str, Any] | None = None,
    name: str = PACKAGE_LOGGER,
    propagate: bool = True,
) -> logging.Logger:
    """Configure the package logger and return it.

    ``config`` may be a ``logging`` mapping (``{level, file, format}``) or a whole configuration
    containing one under ``"logging"``. Precedence per setting: environment variable, then the
    explicit argument, then ``config``, then the default (INFO, no file, text).

    Calling it again re-configures in place: the console handler's level and format change, a
    file handler is added once per path.
    """

    section: Mapping[str, Any] = {}
    if config:
        nested = config.get("logging")
        section = nested if isinstance(nested, Mapping) else config
    raw_level = os.environ.get(ENV_LEVEL) or (level if level is not None else section.get("level"))
    raw_file = os.environ.get(ENV_FILE) or (log_file if log_file is not None else section.get("file"))
    raw_format = os.environ.get(ENV_FORMAT) or (log_format if log_format is not None else section.get("format"))
    logger = logging.getLogger(name)
    return _setup(logger, resolve_level(raw_level), make_formatter(raw_format), raw_file, propagate)


def get_logger(
    name: str = PACKAGE_LOGGER,
    level: int | str | None = None,
    *,
    log_file: str | Path | None = None,
    log_format: str | None = None,
) -> logging.Logger:
    """Return a logger.

    With only a name this is ``logging.getLogger(name)`` (the module-logger idiom; handlers come
    from :func:`configure_logging`). With ``level``, ``log_file`` or ``log_format`` it configures
    that logger directly as PHAL's ``get_logger`` did (tagged console handler, re-tuned on
    reconfiguration; optional DEBUG file handler).
    """

    logger = logging.getLogger(name)
    if level is None and log_file is None and log_format is None:
        return logger
    return _setup(logger, resolve_level(level), make_formatter(log_format), log_file, propagate=False)


__all__ = [
    "ENV_FILE",
    "ENV_FORMAT",
    "ENV_LEVEL",
    "JsonFormatter",
    "PACKAGE_LOGGER",
    "configure_logging",
    "get_logger",
    "make_formatter",
    "resolve_level",
]
