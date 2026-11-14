from __future__ import annotations

import numpy as np


class OnlineTemperatureCalibrator:
    """Bounded sliding-window MAP calibration over completed segments.

    The buffer stores logits directly. A temperature update occurs after
    `batch_size` new observations, using the recent `window_size` observations;
    an L2 prior and hard bounds prevent one noisy batch from destabilizing the
    remaining session.
    """
    def __init__(self, temperature: float = 1.0, lr: float = .02, batch_size: int = 20,
                 window_size: int = 60, min_temperature: float = .7,
                 max_temperature: float = 2.0, prior_strength: float = 10.0):
        if window_size < batch_size:
            raise ValueError("window_size must be >= batch_size")
        self.temperature = float(np.clip(temperature, min_temperature, max_temperature))
        self.base_temperature = self.temperature
        self.lr = lr
        self.batch_size = batch_size
        self.window_size = window_size
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.prior_strength = prior_strength
        self.buffer: list[tuple[float, int]] = []  # (logit, outcome)
        self.pending_since_update = 0
        self.updates = 0

    def transform(self, probability):
        p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p)) / self.temperature
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))

    def observe(self, probability: float, outcome: int) -> bool:
        """Buffer a completed-segment label; return True only after an update."""
        p = float(np.clip(probability, 1e-6, 1 - 1e-6))
        logit = float(np.log(p / (1 - p)))
        self.buffer.append((logit, int(outcome)))
        self.buffer = self.buffer[-self.window_size:]
        self.pending_since_update += 1
        if self.pending_since_update < self.batch_size:
            return False
        self.update_batch()
        return True

    def update_batch(self) -> float:
        if not self.buffer:
            return self.temperature
        logits = np.asarray([logit for logit, _ in self.buffer], dtype=float)
        y = np.asarray([outcome for _, outcome in self.buffer], dtype=float)
        t = self.temperature
        for _ in range(25):
            q = 1 / (1 + np.exp(-np.clip(logits / t, -40, 40)))
            grad_likelihood = np.mean((q - y) * (-logits / (t ** 2)))
            grad_prior = self.prior_strength * (t - self.base_temperature)
            t = t - self.lr * (grad_likelihood + grad_prior)
            t = float(np.clip(t, self.min_temperature, self.max_temperature))
        self.temperature = t
        self.pending_since_update = 0
        self.updates += 1
        return self.temperature

    def update(self, probability: float, outcome: int) -> float:
        """Compatibility alias; buffers and never forces a one-sample update."""
        self.observe(probability, outcome)
        return self.temperature
