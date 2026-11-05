# Ported from PHAL (hotspot_al.cp2k.cp2k_input: the &AO_MATRICES block) at 96263b6; MIT License, see NOTICE.
"""The CP2K input fragment that density/trajectory recording depends on.

Density recording reads the matrices CP2K prints under
``&DFT &PRINT &AO_MATRICES``. PHAL emitted that block inside its CP2K input
builder; GMD_active's input builder belongs to the CP2K labeling area, so the
block is provided here and either returned (:func:`ao_matrices_print_block`)
for the builder to place inside ``&DFT``, or inserted into an existing input
(:func:`insert_ao_matrices_print`). Both return nothing / the input unchanged
when recording is off: printing is O(n_ao^2) text per SCF iteration.
"""

from __future__ import annotations

import re
from typing import Any

from gmd_active_learning.adapters.dft.scf.config import scf_config
from gmd_active_learning.core.exceptions import ConfigurationError

_END_DFT = re.compile(r"^([ \t]*)&END[ \t]+DFT[ \t]*$", re.IGNORECASE | re.MULTILINE)


def ao_matrices_print_block(config: Any, *, indent: str = "    ") -> str:
    """The ``&PRINT &AO_MATRICES`` block for ``&DFT``, or ``""`` when not recording densities."""

    recording = scf_config(config).recording
    if not (recording.enabled and recording.record_density):
        return ""
    # CP2K prints this from inside the SCF loop when &EACH QS_SCF 1 is set: one
    # density per iteration plus a final one, which is what trajectories need.
    each = (
        [f"{indent}    &EACH", f"{indent}      QS_SCF 1", f"{indent}    &END EACH"]
        if recording.record_trajectory
        else []
    )
    return "\n".join(
        [
            f"{indent}&PRINT",
            f"{indent}  &AO_MATRICES",
            f"{indent}    DENSITY .TRUE.",
            # The overlap is what makes Tr(P S) computable.
            f"{indent}    OVERLAP .TRUE.",
            f"{indent}    NDIGITS {recording.ndigits}",
            *each,
            f"{indent}  &END AO_MATRICES",
            f"{indent}&END PRINT",
        ]
    )


def insert_ao_matrices_print(input_text: str, config: Any) -> str:
    """Insert the block just before ``&END DFT``; unchanged when recording is off.

    Refuses an input that already has a ``&PRINT`` section inside ``&DFT`` or no
    ``&END DFT`` at all, rather than producing an input CP2K would reject.
    """

    match = _END_DFT.search(input_text)
    block_probe = ao_matrices_print_block(config)
    if not block_probe:
        return input_text
    if match is None:
        raise ConfigurationError("the CP2K input has no '&END DFT' to insert &AO_MATRICES printing before")
    dft_start = input_text.rfind("&DFT", 0, match.start())
    if re.search(r"^[ \t]*&PRINT\b", input_text[dft_start : match.start()], re.IGNORECASE | re.MULTILINE):
        raise ConfigurationError("the &DFT section already has a &PRINT block; add AO_MATRICES there instead")
    indent = match.group(1) + "  "
    block = ao_matrices_print_block(config, indent=indent)
    return input_text[: match.start()] + block + "\n" + input_text[match.start() :]


__all__ = ["ao_matrices_print_block", "insert_ao_matrices_print"]
