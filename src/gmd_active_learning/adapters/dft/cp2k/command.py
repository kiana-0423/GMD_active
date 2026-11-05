# Ported from PHAL (hotspot_al.cp2k.cp2k_runner) at 96263b6; MIT License, see NOTICE.
"""CP2K command line and a synchronous runner.

Deviation from PHAL: the output file is passed with ``-o`` (CP2K writes it directly) instead of
holding stdout in memory, and an optional MPI launcher template (``cp2k.launcher``, e.g.
``["mpiexec", "-n", "{ntasks}"]``) is prepended so HPC jobs do not silently run single-rank.
There is no dry-run mode: callers that only want inputs call the input writer.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig
from gmd_active_learning.adapters.schedulers.config import SchedulerResources
from gmd_active_learning.core.exceptions import ConfigurationError, LabelingError

LOGGER = logging.getLogger(__name__)


def build_cp2k_command(
    config: CP2KConfig,
    input_name: str | Path,
    output_name: str | Path | None = None,
    *,
    resources: SchedulerResources | None = None,
) -> list[str]:
    """``[launcher..., executable, -i, input[, -o, output]]``."""

    values: dict[str, Any] = {"nodes": 1, "ntasks": 1, "cpus_per_task": 1, "ntasks_per_node": 1}
    if resources is not None:
        values.update({key: value for key, value in resources.model_dump().items() if value is not None and key in values})
    try:
        launcher = [part.format(**values) for part in config.launcher]
    except (KeyError, IndexError) as exc:
        raise ConfigurationError(f"cp2k.launcher has an unknown placeholder {exc}; use nodes/ntasks/cpus_per_task") from exc
    command = [*launcher, config.executable, "-i", str(input_name)]
    if output_name is not None:
        command += ["-o", str(output_name)]
    return command


def cp2k_environment(config: CP2KConfig) -> dict[str, str]:
    environment = dict(config.environment)
    if config.data_dir:
        environment.setdefault("CP2K_DATA_DIR", str(config.data_dir))
    return environment


def run_cp2k(
    input_path: str | Path,
    config: CP2KConfig,
    *,
    output_name: str | None = None,
    timeout: float | None = None,
    check: bool = True,
    history_hook: Callable[[Path, Path], Any] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run CP2K synchronously in the input's directory; returns (process, output path).

    ``history_hook(output_path, input_path)`` is an optional observer (e.g. SCF-history recording);
    its failure is logged and never fails the run (PHAL behaviour).
    """

    input_path = Path(input_path)
    output_path = input_path.with_name(output_name or f"{input_path.stem}.out")
    command = build_cp2k_command(config, input_path.name, output_path.name)
    try:
        process = subprocess.run(
            command,
            cwd=input_path.parent,
            env={**os.environ, **cp2k_environment(config)},
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise LabelingError(f"CP2K executable not found: {config.executable!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise LabelingError(f"CP2K run {input_path} exceeded {timeout} s") from exc
    if history_hook is not None:
        try:
            history_hook(output_path, input_path)
        except Exception as exc:  # noqa: BLE001 - observability must not invalidate a label
            LOGGER.warning("CP2K history hook failed for %s: %s", output_path, exc)
    if check and process.returncode != 0:
        raise LabelingError(f"CP2K exited with status {process.returncode} for {input_path}: {process.stderr.strip()[-500:]}")
    return process, output_path


def merged_environment(config: CP2KConfig, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    return {**cp2k_environment(config), **dict(extra or {})}


__all__ = ["build_cp2k_command", "cp2k_environment", "merged_environment", "run_cp2k"]
