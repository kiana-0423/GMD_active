"""CORE-005: PHAL's exception hierarchy exists additively under GMDActiveLearningError."""

from __future__ import annotations

import pytest

from gmd_active_learning.core import exceptions as exc


@pytest.mark.parametrize(
    ("name", "parents"),
    [
        ("BackendError", ("AdapterError",)),
        ("InferenceError", ("BackendError", "AdapterError")),
        ("MDRuntimeError", ("BackendError",)),
        ("DFTRuntimeError", ("BackendError",)),
        ("CP2KRuntimeError", ("DFTRuntimeError", "BackendError")),
        ("DataError", ("DataContractError",)),
        ("CooldownSkip", ()),
    ],
)
def test_phal_classes_are_gmd_errors(name: str, parents: tuple[str, ...]) -> None:
    cls = getattr(exc, name)
    assert issubclass(cls, exc.GMDActiveLearningError)
    for parent in parents:
        assert issubclass(cls, getattr(exc, parent))


def test_config_error_is_an_alias_and_cooldown_is_not_a_backend_failure() -> None:
    assert exc.ConfigError is exc.ConfigurationError
    assert not issubclass(exc.CooldownSkip, exc.AdapterError)
    assert not issubclass(exc.DataError, exc.AdapterError)


def test_existing_classes_are_unchanged() -> None:
    for name in (
        "ConfigurationError", "AdapterError", "MonitorError", "DataContractError", "UnsupportedCombinationError",
        "CapabilityUnavailableError", "ExtractionError", "LabelingError", "SchedulerError", "WorkflowError",
    ):
        cls = getattr(exc, name)
        assert cls.__bases__ == (exc.GMDActiveLearningError,)
