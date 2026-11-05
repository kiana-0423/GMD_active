"""Online monitors: atom-wise staged OOD scoring (PHAL) and the frame-level global-LJ monitor (GMD).

Importing this package pulls in no MD/MLIP/DFT adapter and no torch.
"""

from gmd_active_learning.monitors.config import MonitorConfig, load_monitor_config, monitor_config_from_phal
from gmd_active_learning.monitors.coordination import coordination_deltas, smooth_coordination_numbers
from gmd_active_learning.monitors.ensemble_deviation import committee_force_deviation, compute_ensemble_deviation
from gmd_active_learning.monitors.force import delta_force_norms, force_norms
from gmd_active_learning.monitors.force_disagreement import (
    CommitteeForceProvider,
    ReferenceForceProvider,
    mlip_force_deviation,
)
from gmd_active_learning.monitors.geometry import displacement_norms, minimum_neighbor_distances
from gmd_active_learning.monitors.global_lj_monitor import GlobalLJMonitor
from gmd_active_learning.monitors.lj_local import LocalLJEstimator, LocalLJFit, compute_local_lj_residuals
from gmd_active_learning.monitors.online import (
    POLL_TIMEOUT,
    FrameMonitor,
    FrameSource,
    OnlineMonitorDriver,
    RunSummary,
    build_frame_monitor,
)
from gmd_active_learning.monitors.running_stats import RunningMetricStats
from gmd_active_learning.monitors.scoring import OODScorer, StageScore
from gmd_active_learning.monitors.staged_monitor import StagedEvaluation, StagedOODMonitor, decide_action, interval_due
from gmd_active_learning.utils.neighbors import MonitorNeighbors

__all__ = [
    "POLL_TIMEOUT",
    "CommitteeForceProvider",
    "FrameMonitor",
    "FrameSource",
    "GlobalLJMonitor",
    "LocalLJEstimator",
    "LocalLJFit",
    "MonitorConfig",
    "MonitorNeighbors",
    "OODScorer",
    "OnlineMonitorDriver",
    "ReferenceForceProvider",
    "RunSummary",
    "RunningMetricStats",
    "StageScore",
    "StagedEvaluation",
    "StagedOODMonitor",
    "build_frame_monitor",
    "committee_force_deviation",
    "compute_ensemble_deviation",
    "compute_local_lj_residuals",
    "coordination_deltas",
    "decide_action",
    "delta_force_norms",
    "displacement_norms",
    "force_norms",
    "interval_due",
    "load_monitor_config",
    "minimum_neighbor_distances",
    "mlip_force_deviation",
    "monitor_config_from_phal",
    "smooth_coordination_numbers",
]
