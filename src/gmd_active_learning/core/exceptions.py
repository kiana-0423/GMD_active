class GMDActiveLearningError(Exception):
    """Base exception for the project."""


class ConfigurationError(GMDActiveLearningError):
    """Raised when configuration is invalid."""


class AdapterError(GMDActiveLearningError):
    """Raised when adapter interaction fails."""


class MonitorError(GMDActiveLearningError):
    """Raised when monitoring fails."""


class DataContractError(GMDActiveLearningError):
    """Raised when data violates a contract in ``gmd_active_learning.contracts``."""


class UnsupportedCombinationError(GMDActiveLearningError):
    """Raised for physical or runtime combinations the selected backend does not support."""


class CapabilityUnavailableError(GMDActiveLearningError):
    """Raised when a requested backend capability is not provided by the installed backend."""


class ExtractionError(GMDActiveLearningError):
    """Raised when a region cannot be extracted under the configured policy."""


class LabelingError(GMDActiveLearningError):
    """Raised when label preparation, execution or parsing fails."""


class SchedulerError(GMDActiveLearningError):
    """Raised when a scheduler command fails or returns an unexpected response."""


class WorkflowError(GMDActiveLearningError):
    """Raised when the persistent workflow cannot make progress safely."""
