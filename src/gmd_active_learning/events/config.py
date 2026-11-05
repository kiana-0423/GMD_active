"""Configuration of event capture (YAML section ``events``; defaults in configs/sections/events.yaml)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gmd_active_learning.core.exceptions import ConfigurationError


class EventStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    directory: str | None = Field(None, description="event root directory; required when enabled")
    write_extxyz: bool = Field(True, description="also write frames.extxyz (frames.npz is always written, bit-exact)")


class EventsConfig(BaseModel):
    """Rolling pre/post-trigger capture (PHAL ``buffer`` section) plus cooldown and bounded history."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    pre_trigger_frames: int = Field(10, ge=0)
    post_trigger_frames: int = Field(5, ge=0)
    maxlen: int | None = Field(None, ge=1, description="rolling history length; null = pre_trigger_frames")
    cooldown_steps: int = Field(0, ge=0, description="MD steps after an event's last frame during which new triggers open no event")
    retain_recent: int = Field(16, ge=0, description="number of finished-event summaries kept in memory (bounded, F-07)")
    store: EventStoreConfig = Field(default_factory=EventStoreConfig)

    def history_length(self) -> int:
        return max(1, self.maxlen if self.maxlen is not None else self.pre_trigger_frames)


def load_events_config(section: dict[str, Any] | EventsConfig | None) -> EventsConfig:
    if isinstance(section, EventsConfig):
        return section
    try:
        config = EventsConfig.model_validate(section or {})
    except ValidationError as exc:
        raise ConfigurationError(f"invalid events configuration: {exc}") from exc
    if config.store.enabled and not config.store.directory:
        raise ConfigurationError("events.store.directory is required when events.store.enabled is true")
    return config


def events_config_from_phal(phal: dict[str, Any]) -> EventsConfig:
    """PHAL ``buffer`` section -> ``EventsConfig`` (PHAL had no cooldown: 0)."""

    buffer = phal.get("buffer", {})
    return load_events_config(
        {
            "pre_trigger_frames": int(buffer.get("pre_trigger_frames", 10)),
            "post_trigger_frames": int(buffer.get("post_trigger_frames", 5)),
            "maxlen": buffer.get("maxlen"),
        }
    )


__all__ = ["EventStoreConfig", "EventsConfig", "events_config_from_phal", "load_events_config"]
