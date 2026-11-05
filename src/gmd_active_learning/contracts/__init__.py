"""Typed data contracts shared by every GMD_active subsystem.

Units throughout: positions and cells in Angstrom, energies in eV, forces in eV/Angstrom,
stress in eV/Angstrom^3 (ASE sign convention), time in fs. Adapters convert external
representations into these contracts; domain code never sees backend-specific formats.
"""

from gmd_active_learning.contracts.artifacts import DatasetVersion, ModelArtifact
from gmd_active_learning.contracts.events import EventRecord
from gmd_active_learning.contracts.frame import UNITS, Frame
from gmd_active_learning.contracts.ids import content_hash, stable_id
from gmd_active_learning.contracts.labels import (
    ExecutionStatus,
    LabelResult,
    LabelTask,
    SchedulerJob,
    ValidationStatus,
)
from gmd_active_learning.contracts.monitoring import (
    FrameAction,
    MetricStatus,
    MetricValues,
    MonitorResult,
    Stage,
)
from gmd_active_learning.contracts.regions import REGION_CODES, ExtractedRegion, RegionRole

__all__ = [
    "UNITS",
    "DatasetVersion",
    "EventRecord",
    "ExecutionStatus",
    "ExtractedRegion",
    "Frame",
    "FrameAction",
    "LabelResult",
    "LabelTask",
    "MetricStatus",
    "MetricValues",
    "ModelArtifact",
    "MonitorResult",
    "REGION_CODES",
    "RegionRole",
    "SchedulerJob",
    "Stage",
    "ValidationStatus",
    "content_hash",
    "stable_id",
]
