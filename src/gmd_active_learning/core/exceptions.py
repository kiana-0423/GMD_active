class GMDActiveLearningError(Exception):
    """Base exception for the project."""


class ConfigurationError(GMDActiveLearningError):
    """Raised when configuration is invalid."""


class AdapterError(GMDActiveLearningError):
    """Raised when adapter interaction fails."""


class MonitorError(GMDActiveLearningError):
    """Raised when monitoring fails."""
