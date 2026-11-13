from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SegmentMode(str, Enum):
    EVENT_BOUNDED = "event_bounded"
    FIXED_HORIZON = "fixed_horizon"


@dataclass(frozen=True)
class SegmentContract:
    mode: SegmentMode
    horizon_seconds: float
    min_seconds: float = 30.0
    max_seconds: float = 60.0

    def validate(self) -> None:
        if self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")
        if self.mode == SegmentMode.EVENT_BOUNDED and not (self.min_seconds <= self.horizon_seconds <= self.max_seconds):
            raise ValueError("event-bounded segment must use the configured 30-60s semantic range")

    def label_definition(self) -> str:
        self.validate()
        if self.mode == SegmentMode.EVENT_BOUNDED:
            return "objective_success before segment_finish, or failure/quit/timeout"
        return "survive_window and objective_progress within fixed horizon"


def segment_boundary(mode: SegmentMode, decision_time: float, horizon_seconds: float,
                     event_finish_time: float | None = None) -> tuple[float, float]:
    """Return [start, end] boundary used by feature/label generation."""
    if mode == SegmentMode.EVENT_BOUNDED:
        if event_finish_time is None or event_finish_time <= decision_time:
            raise ValueError("event-bounded segment needs a finish event after decision")
        return decision_time, event_finish_time
    if horizon_seconds <= 0:
        raise ValueError("fixed horizon must be positive")
    return decision_time, decision_time + horizon_seconds
