# Ported from PHAL (hotspot_al.extraction.block.BlockCooldownTracker) at 96263b6; MIT License, see NOTICE.
"""Block cooldown: suppress repeated labeling of the same spatial blocks.

PHAL kept this map in memory only, so a restart forgot every cooldown. The state here is plain
data (``to_dict`` / ``from_dict``) and can be persisted atomically with :meth:`save` / :meth:`load`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.utils.atomic_io import read_json_or_default, write_json_atomic

LOGGER = logging.getLogger(__name__)

BlockId = tuple[int, int, int]

#: Prune when the map grows beyond this many entries (PHAL constant).
PRUNE_SIZE = 10000
STATE_VERSION = 1


def _key(block_id: BlockId) -> str:
    return ",".join(str(int(value)) for value in block_id)


def _block(key: str) -> BlockId:
    parts = tuple(int(value) for value in key.split(","))
    if len(parts) != 3:
        raise DataContractError(f"invalid block id {key!r}")
    return parts  # type: ignore[return-value]


@dataclass(slots=True)
class BlockCooldownTracker:
    """``block_id -> last step it was scheduled for labeling``.

    ``should_skip`` is true when any block of the group was scheduled less than ``cooldown_steps``
    ago. ``update`` records the step and prunes entries older than ``2 * cooldown_steps`` when the
    map exceeds :data:`PRUNE_SIZE` entries or when ``step % cooldown_steps == 0`` (PHAL rule).
    As in PHAL the cooldown starts when a region is extracted, not when labeling succeeds; callers
    that want a failed label to release the blocks call :meth:`release`.
    """

    cooldown_steps: int
    last_labeled_step: dict[BlockId, int] = field(default_factory=dict)

    def should_skip(self, block_ids: Iterable[BlockId], step: int) -> bool:
        if self.cooldown_steps <= 0:
            return False
        for block_id in block_ids:
            last_step = self.last_labeled_step.get(tuple(block_id))  # type: ignore[arg-type]
            if last_step is not None and step - last_step < self.cooldown_steps:
                return True
        return False

    def update(self, block_ids: Iterable[BlockId], step: int) -> None:
        for block_id in block_ids:
            self.last_labeled_step[tuple(int(v) for v in block_id)] = int(step)  # type: ignore[index]
        should_prune = len(self.last_labeled_step) > PRUNE_SIZE or (
            self.cooldown_steps > 0 and step % self.cooldown_steps == 0
        )
        if should_prune:
            expired = [key for key, last in self.last_labeled_step.items() if step - last > self.cooldown_steps * 2]
            for key in expired:
                del self.last_labeled_step[key]

    def release(self, block_ids: Iterable[BlockId]) -> None:
        for block_id in block_ids:
            self.last_labeled_step.pop(tuple(int(v) for v in block_id), None)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "cooldown_steps": int(self.cooldown_steps),
            "last_labeled_step": {_key(key): int(value) for key, value in sorted(self.last_labeled_step.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlockCooldownTracker:
        if int(data.get("version", STATE_VERSION)) != STATE_VERSION:
            raise DataContractError(f"unsupported cooldown state version {data.get('version')!r}")
        return cls(
            cooldown_steps=int(data["cooldown_steps"]),
            last_labeled_step={_block(key): int(value) for key, value in data.get("last_labeled_step", {}).items()},
        )

    def save(self, path: str | Path) -> Path:
        return write_json_atomic(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path, cooldown_steps: int) -> BlockCooldownTracker:
        """Load persisted state; a missing file starts empty. ``cooldown_steps`` from config wins."""

        data = read_json_or_default(path, None, logger=LOGGER)
        if data is None:
            return cls(cooldown_steps=int(cooldown_steps))
        tracker = cls.from_dict(data)
        tracker.cooldown_steps = int(cooldown_steps)
        return tracker


__all__ = ["BlockCooldownTracker", "BlockId", "PRUNE_SIZE"]
