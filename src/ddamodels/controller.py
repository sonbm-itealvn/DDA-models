from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class ControllerState:
    cooldown_seconds: float = 0.0
    increase_streak: int = 0
    decrease_streak: int = 0
    current_difficulty: int = 2
    history: list[dict] = field(default_factory=list)


@dataclass
class Decision:
    applied_difficulty: int
    reason: str
    scores: dict[int, float]
    confidence: float


def decide(probabilities: Mapping[int, float], uncertainties: Mapping[int, float],
           current: int, state: ControllerState, *, p_target: float = .70,
           struggle_probability: float = 0.0, safe_checkpoint: bool = True,
           progress_ratio: float = 0.0, min_confidence: float = .80,
           cooldown_seconds: float = 60.0, dt_seconds: float = 1.0,
           horizon_seconds: float = 30.0) -> Decision:
    candidates = {int(k): float(v) for k, v in probabilities.items() if abs(int(k) - current) <= 1}
    if not candidates:
        return Decision(current, "NO_VALID_CANDIDATE", {}, 0.0)
    confidence = float(1.0 - min(1.0, np.mean([uncertainties.get(k, 1.0) for k in candidates])))
    scores = {k: -abs(p - p_target) - 0.25 * float(uncertainties.get(k, 1.0)) - 0.05 * abs(k-current)
              for k, p in candidates.items()}
    if state.cooldown_seconds > 0:
        state.cooldown_seconds = max(0.0, state.cooldown_seconds - max(0.0, dt_seconds))
        return Decision(current, "HOLD_COOLDOWN", scores, confidence)
    if not safe_checkpoint:
        return Decision(current, "HOLD_UNSAFE_CHECKPOINT", scores, confidence)
    if confidence < min_confidence:
        return Decision(current, "HOLD_LOW_CONFIDENCE", scores, confidence)
    if struggle_probability >= .75:
        state.decrease_streak += 1
        state.increase_streak = 0
    elif probabilities.get(current, 0.0) >= p_target and probabilities.get(current + 1, 0.0) >= p_target and progress_ratio >= .30:
        state.increase_streak += 1
        state.decrease_streak = 0
    else:
        state.increase_streak = state.decrease_streak = 0
    safe_cooldown = max(float(cooldown_seconds), 2.0 * float(horizon_seconds))
    if state.decrease_streak >= 2:
        state.cooldown_seconds = safe_cooldown
        state.decrease_streak = 0
        return Decision(max(0, current - 1), "DECREASE_STRUGGLE", scores, confidence)
    if state.increase_streak >= 2:
        state.cooldown_seconds = safe_cooldown
        state.increase_streak = 0
        return Decision(current + 1, "INCREASE_READY", scores, confidence)
    return Decision(current, "HOLD", scores, confidence)
