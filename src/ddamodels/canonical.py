from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


CANONICAL_FEATURES = [
    "progress_ratio", "progress_velocity", "time_budget_ratio", "tempo_rate",
    "vitality_ratio", "vitality_loss_rate", "critical_error_rate", "failure_cost_ratio",
    "resource_reserve_ratio", "resource_utilization_rate", "capability_ratio", "threat_pressure_ratio",
    "action_frequency", "invalid_action_rate", "idle_ratio", "reaction_latency",
    "error_slope", "recent_segment_win_rate", "recent_average_difficulty",
    "difficulty_adjusted_win_rate", "efficiency_ratio",
]

BOUNDED_FEATURES = {
    "progress_ratio", "vitality_ratio", "time_budget_ratio",
    "invalid_action_rate", "idle_ratio", "recent_segment_win_rate",
    "difficulty_adjusted_win_rate",
}

UNBOUNDED_FEATURES = set(CANONICAL_FEATURES) - BOUNDED_FEATURES


@dataclass
class CanonicalState:
    values: np.ndarray
    mask: np.ndarray
    names: list[str]

    def as_model_input(self) -> tuple[np.ndarray, np.ndarray]:
        return self.values.astype(np.float32), self.mask.astype(np.float32)


def canonicalize(mapping: Mapping[str, float | None], available: Mapping[str, bool] | None = None,
                 names: Sequence[str] = CANONICAL_FEATURES) -> CanonicalState:
    values, mask = [], []
    available = available or {}
    for name in names:
        value = mapping.get(name)
        finite = value is not None and np.isfinite(value)
        ok = bool(available.get(name, finite))
        values.append(float(value) if ok and finite else 0.0)
        mask.append(1.0 if ok and finite else 0.0)
    return CanonicalState(np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=np.float32), list(names))


def normalized_log_ratio_delta(candidate: float, current: float, max_step_ratio: float,
                               min_value: float = 1e-6) -> float:
    """Normalize a positive parameter change to [-1, 1] with a fixed contract."""
    if candidate <= 0 or current <= 0 or max_step_ratio <= 1:
        raise ValueError("candidate/current must be positive and max_step_ratio > 1")
    raw = np.log(max(candidate, min_value) / max(current, min_value)) / np.log(max_step_ratio)
    return float(np.clip(raw, -1.0, 1.0))


def build_difficulty_vector(relative_step: float, pressure_delta: float, forgiveness_delta: float,
                            resource_delta: float, *, empirical_prior: float | None = None,
                            prior_source: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build v2 difficulty input and mask without allowing confounded priors.

    empirical_prior is valid only when estimated from randomized fixed-difficulty
    data in the same domain/patch. Otherwise it is masked out, not imputed.
    """
    values = [float(np.clip(relative_step, -1, 1)), float(np.clip(pressure_delta, -1, 1)),
              float(np.clip(forgiveness_delta, -1, 1)), float(np.clip(resource_delta, -1, 1)),
              float(empirical_prior) if empirical_prior is not None else 0.0]
    mask = [1.0, 1.0, 1.0, 1.0, 1.0 if empirical_prior is not None and prior_source == "fixed_randomized_same_domain" else 0.0]
    if mask[-1] == 0:
        values[-1] = 0.0
    return np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=np.float32)


def compute_difficulty_adjusted_win_rate(recent_win_rate: float,
                                         empirical_prior: float | None) -> tuple[float, float]:
    """Return (adjusted value, availability mask) without inventing a prior."""
    if empirical_prior is None or not np.isfinite(empirical_prior):
        return 0.0, 0.0
    return float(np.clip(recent_win_rate - empirical_prior, -1.0, 1.0)), 1.0


@dataclass
class CausalSessionNormalizer:
    """Causal EMA/shrinkage normalizer with bounded ratios kept absolute."""
    game_mean: Mapping[str, float] = field(default_factory=dict)
    game_scale: Mapping[str, float] = field(default_factory=dict)
    prior_weight: float = 10.0
    min_scale: float = 0.05
    count: dict[str, int] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    m2: dict[str, float] = field(default_factory=dict)

    def transform(self, mapping: Mapping[str, float | None], available: Mapping[str, bool] | None = None):
        available = available or {}
        result, mask = {}, {}
        for name in CANONICAL_FEATURES:
            raw = mapping.get(name)
            ok = bool(available.get(name, raw is not None and np.isfinite(raw)))
            if not ok or raw is None or not np.isfinite(raw):
                result[name], mask[name] = 0.0, 0.0
                continue
            x = float(raw)
            if name in BOUNDED_FEATURES:
                # Preserve semantic absolute meaning; only clip to contract range.
                result[name] = float(np.clip(x, 0.0, 1.0))
            else:
                n = self.count.get(name, 0)
                game_mu = float(self.game_mean.get(name, 0.0))
                game_scale = max(float(self.game_scale.get(name, 1.0)), self.min_scale)
                session_mu = self.mean.get(name, game_mu)
                w = n / (self.prior_weight + n)
                mu = (1.0 - w) * game_mu + w * session_mu
                session_var = self.m2.get(name, 0.0) / max(n - 1, 1)
                scale = max(np.sqrt((1.0 - w) * game_scale ** 2 + w * session_var), self.min_scale)
                result[name] = float(np.clip((x - mu) / scale, -8.0, 8.0))
            mask[name] = 1.0
        return result, mask

    def update(self, mapping: Mapping[str, float | None], available: Mapping[str, bool] | None = None):
        available = available or {}
        for name, raw in mapping.items():
            if name in BOUNDED_FEATURES or name not in UNBOUNDED_FEATURES:
                continue
            if not available.get(name, raw is not None and np.isfinite(raw)):
                continue
            x = float(raw); n = self.count.get(name, 0) + 1
            old = self.mean.get(name, self.game_mean.get(name, 0.0))
            new = old + (x - old) / n
            self.m2[name] = self.m2.get(name, 0.0) + (x - old) * (x - new)
            self.mean[name] = new; self.count[name] = n


def session_relative_normalize(values: np.ndarray, mask: np.ndarray, reference: np.ndarray | None = None,
                               eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible safe ratio normalization for already canonical arrays."""
    x = np.asarray(values, dtype=np.float32).copy(); m = np.asarray(mask, dtype=np.float32)
    ref = np.ones(x.shape[-1], dtype=np.float32) if reference is None else np.asarray(reference, dtype=np.float32)
    safe = np.where(np.abs(ref) < eps, 1.0, np.abs(ref))
    x = np.clip(x / safe, -8.0, 8.0); x[m <= 0] = 0.0
    return x, m


def masked_feature_dropout(values: np.ndarray, mask: np.ndarray, drop_groups: Sequence[Sequence[int]],
                           group_index: int) -> tuple[np.ndarray, np.ndarray]:
    x, m = np.asarray(values).copy(), np.asarray(mask).copy()
    for idx in drop_groups[group_index]:
        x[..., idx] = 0.0; m[..., idx] = 0.0
    return x, m
