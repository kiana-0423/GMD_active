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


# --- PHAL hierarchy (hotspot_al.exceptions at 96263b6), added for ledger CORE-005 ---------------
# PHAL: HotspotALError > {ConfigError, BackendError > {InferenceError, LAMMPSRuntimeError,
# CP2KRuntimeError}, DataError, CooldownSkip}. The GMD names are engine-neutral (LAMMPS -> MD) and
# sit under the existing GMD classes so ``except AdapterError`` / ``except DataContractError``
# keep catching them.

ConfigError = ConfigurationError
"""PHAL name of :class:`ConfigurationError` (alias, same class)."""


class BackendError(AdapterError):
    """An external backend (MD engine, MLIP, DFT code, scheduler) failed to set up or run."""


class InferenceError(BackendError):
    """MLIP inference (GMD_SGT forward pass, committee evaluation) failed."""


class MDRuntimeError(BackendError):
    """The MD engine (GMD) failed; replaces PHAL's ``LAMMPSRuntimeError``."""


class DFTRuntimeError(BackendError):
    """A DFT labeling run failed at the process level (not an invalid label)."""


class CP2KRuntimeError(DFTRuntimeError):
    """CP2K failed at the process level."""


class DataError(DataContractError):
    """Input or generated data is malformed (e.g. a non-finite metric or a corrupt file)."""


class CooldownSkip(GMDActiveLearningError):
    """Nothing to submit because every candidate region is still in cooldown.

    Expected throttling, not a failure. PHAL raised it as control flow from its CP2K submitter;
    GMD_active's extraction reports cooldown as a result (``extraction.cooldown``) and keeps this
    class only for callers that prefer the PHAL signalling style.
    """
