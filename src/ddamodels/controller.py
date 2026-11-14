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
           horizon_seconds: float = 30.0, increase_threshold: float = .80,
           max_difficulty: int | None = None) -> Decision:
    try:
        raw = {int(k): float(v) for k, v in probabilities.items()}
        unc = {int(k): float(uncertainties.get(k, 1.0)) for k in raw}
    except (TypeError, ValueError):
        return Decision(current, "HOLD_INVALID_INPUT", {}, 0.0)
    if not raw or any(not np.isfinite(v) or v < 0.0 or v > 1.0 for v in raw.values()):
        return Decision(current, "HOLD_INVALID_INPUT", {}, 0.0)
    if any(not np.isfinite(v) or v < 0.0 for v in unc.values()):
        return Decision(current, "HOLD_INVALID_INPUT", {}, 0.0)
    candidates = {k: v for k, v in raw.items() if abs(k - current) <= 1 and
                  (max_difficulty is None or 0 <= k <= max_difficulty)}
    if not candidates:
        return Decision(current, "NO_VALID_CANDIDATE", {}, 0.0)
    confidence = float(1.0 - min(1.0, np.mean([unc[k] for k in candidates])))
    scores = {k: -abs(p - p_target) - 0.25 * unc[k] - 0.05 * abs(k-current)
              for k, p in candidates.items()}

    # Consume elapsed real time before evaluating the gate.
    state.cooldown_seconds = max(0.0, state.cooldown_seconds - max(0.0, dt_seconds))
    if state.cooldown_seconds > 0:
        return Decision(current, "HOLD_COOLDOWN", scores, confidence)
    if not safe_checkpoint:
        return Decision(current, "HOLD_UNSAFE_CHECKPOINT", scores, confidence)
    if confidence < min_confidence:
        return Decision(current, "HOLD_LOW_CONFIDENCE", scores, confidence)

    best = max(scores, key=scores.get)
    if struggle_probability >= .75:
        direction = "decrease"
    elif (best == current + 1 and raw.get(current, 0.0) >= increase_threshold
          and raw.get(current + 1, 0.0) >= p_target and progress_ratio >= .30):
        direction = "increase"
    elif best == current - 1 and raw.get(current, 0.0) <= p_target:
        direction = "decrease"
    else:
        direction = "hold"

    if direction == "decrease":
        state.decrease_streak += 1; state.increase_streak = 0
    elif direction == "increase":
        state.increase_streak += 1; state.decrease_streak = 0
    else:
        state.increase_streak = state.decrease_streak = 0

    safe_cooldown = max(float(cooldown_seconds), 2.0 * float(horizon_seconds))
    if state.decrease_streak >= 2:
        state.cooldown_seconds = safe_cooldown; state.decrease_streak = 0
        return Decision(max(0, current - 1), "DECREASE_SCORE_CONFIRMED", scores, confidence)
    if state.increase_streak >= 2 and (max_difficulty is None or current < max_difficulty):
        state.cooldown_seconds = safe_cooldown; state.increase_streak = 0
        return Decision(current + 1, "INCREASE_SCORE_CONFIRMED", scores, confidence)
    return Decision(current, "HOLD", scores, confidence)
