from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


CANONICAL_FEATURES = [
    "progress_ratio", "progress_velocity", "time_budget_ratio", "tempo_rate",
    "vitality_ratio", "vitality_loss_rate", "critical_error_rate", "failure_cost_ratio",
    "resource_reserve_ratio", "resource_utilization_rate", "capability_ratio", "threat_pressure_ratio",
    "action_frequency", "invalid_action_rate", "idle_ratio", "reaction_latency",
    "error_slope", "recent_segment_win_rate", "efficiency_ratio",
]


@dataclass
class CanonicalState:
    values: np.ndarray
    mask: np.ndarray
    names: list[str]

    def as_model_input(self) -> tuple[np.ndarray, np.ndarray]:
        return self.values.astype(np.float32), self.mask.astype(np.float32)


def canonicalize(mapping: Mapping[str, float | None], available: Mapping[str, bool] | None = None,
                 names: Sequence[str] = CANONICAL_FEATURES) -> CanonicalState:
    """Convert adapter output into fixed semantic order plus availability mask.

    Missing values remain zero only as a tensor placeholder; the mask tells the
    model that the value is unavailable. This prevents missing from meaning zero.
    """
    values, mask = [], []
    available = available or {}
    for name in names:
        value = mapping.get(name)
        ok = bool(available.get(name, value is not None and np.isfinite(value)))
        values.append(float(value) if ok and value is not None and np.isfinite(value) else 0.0)
        mask.append(1.0 if ok else 0.0)
    return CanonicalState(np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=np.float32), list(names))


def session_relative_normalize(values: np.ndarray, mask: np.ndarray, reference: np.ndarray | None = None,
                               eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Causal relative normalization.

    `reference` must be computed from warm-up/previous observations only. The
    returned mask is unchanged and missing dimensions stay zero.
    """
    x = np.asarray(values, dtype=np.float32).copy()
    m = np.asarray(mask, dtype=np.float32)
    if reference is None:
        reference = np.ones(x.shape[-1], dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    safe = np.where(np.abs(ref) < eps, 1.0, np.abs(ref))
    x = x / safe
    x[m <= 0] = 0.0
    return x, m


def masked_feature_dropout(values: np.ndarray, mask: np.ndarray, drop_groups: Sequence[Sequence[int]],
                           group_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Utility for robustness evaluation, not for raw data collection."""
    x, m = np.asarray(values).copy(), np.asarray(mask).copy()
    for idx in drop_groups[group_index]:
        x[..., idx] = 0.0
        m[..., idx] = 0.0
    return x, m
