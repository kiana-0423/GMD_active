"""CP2K labeling adapter: input generation, output parsing, validation, backend.

Public API: :class:`CP2KBackend` (registry name ``dft:cp2k``), :class:`CP2KConfig` (``cp2k`` section),
:func:`build_cp2k_input` / :func:`write_cp2k_inputs`, :func:`parse_cp2k_output_file`,
:func:`validation_reasons`.
"""

from gmd_active_learning.adapters.dft.cp2k.backend import STAGE_HOPT, STAGE_SINGLE_POINT, CP2KBackend
from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig, FailureClass, RetryStep
from gmd_active_learning.adapters.dft.cp2k.input import build_cp2k_input, write_cp2k_inputs
from gmd_active_learning.adapters.dft.cp2k.parse import CP2KOutput, parse_cp2k_forces, parse_cp2k_output_file
from gmd_active_learning.adapters.dft.cp2k.validation import classify_failure, validation_reasons

__all__ = [
    "CP2KBackend",
    "CP2KConfig",
    "CP2KOutput",
    "FailureClass",
    "RetryStep",
    "STAGE_HOPT",
    "STAGE_SINGLE_POINT",
    "build_cp2k_input",
    "classify_failure",
    "parse_cp2k_forces",
    "parse_cp2k_output_file",
    "validation_reasons",
    "write_cp2k_inputs",
]
