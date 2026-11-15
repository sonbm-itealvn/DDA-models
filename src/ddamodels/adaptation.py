from __future__ import annotations

import numpy as np


class OnlineTemperatureCalibrator:
    """Bounded sliding-window MAP affine logit calibrator.

    It estimates ``logit(p_cal) = beta * logit(p_raw) + bias``. Temperature (tau)
    compatibility is preserved through ``tau = temperature = 1 / beta``. A prior
    keeps beta near 1 and bias near 0; both parameters are bounded to prevent
    few-shot calibration from changing decisions catastrophically.
    """
    def __init__(self, temperature: float = 1.0, tau: float | None = None, lr: float = .02, batch_size: int = 20,
                 window_size: int = 60, min_temperature: float = .7,
                 max_temperature: float = 2.0, prior_strength: float = 10.0,
                 bias_lr: float | None = None, min_bias: float = -1.0, max_bias: float = 1.0):
        if window_size < batch_size:
            raise ValueError("window_size must be >= batch_size")
        if tau is not None:
            temperature = tau
        self.temperature = float(np.clip(temperature, min_temperature, max_temperature))
        self.base_temperature = self.temperature
        self.beta = 1.0 / self.temperature
        self.base_beta = self.beta
        self.bias = 0.0
        self.lr = lr
        self.bias_lr = lr if bias_lr is None else bias_lr
        self.batch_size = batch_size; self.window_size = window_size
        self.min_temperature = min_temperature; self.max_temperature = max_temperature
        self.min_beta = 1.0 / max_temperature; self.max_beta = 1.0 / min_temperature
        self.min_bias = min_bias; self.max_bias = max_bias
        self.prior_strength = prior_strength
        self.buffer: list[tuple[float, int]] = []
        self.pending_since_update = 0; self.updates = 0

    @property
    def tau(self) -> float:
        """Alias for temperature (tau), matching theoretical documentation notation."""
        return self.temperature

    @tau.setter
    def tau(self, val: float):
        self.temperature = float(np.clip(val, self.min_temperature, self.max_temperature))
        self.beta = 1.0 / self.temperature

    def transform(self, probability):
        p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        logits = self.beta * np.log(p / (1 - p)) + self.bias
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))

    def observe(self, probability: float, outcome: int) -> bool:
        p = float(np.clip(probability, 1e-6, 1 - 1e-6))
        self.buffer.append((float(np.log(p / (1 - p))), int(outcome)))
        self.buffer = self.buffer[-self.window_size:]
        self.pending_since_update += 1
        if self.pending_since_update < self.batch_size:
            return False
        self.update_batch(); return True

    def update_batch(self) -> float:
        if not self.buffer:
            return self.temperature
        z = np.asarray([a for a, _ in self.buffer], dtype=float)
        y = np.asarray([b for _, b in self.buffer], dtype=float)
        beta, bias = self.beta, self.bias
        for _ in range(25):
            q = 1.0 / (1.0 + np.exp(-np.clip(beta * z + bias, -40, 40)))
            grad_beta = float(np.mean((q - y) * z) + self.prior_strength * (beta - self.base_beta))
            grad_bias = float(np.mean(q - y) + self.prior_strength * bias)
            beta = float(np.clip(beta - self.lr * grad_beta, self.min_beta, self.max_beta))
            bias = float(np.clip(bias - self.bias_lr * grad_bias, self.min_bias, self.max_bias))
        self.beta, self.bias = beta, bias
        self.temperature = float(np.clip(1.0 / beta, self.min_temperature, self.max_temperature))
        self.pending_since_update = 0; self.updates += 1
        return self.temperature

    def update(self, probability: float, outcome: int) -> float:
        self.observe(probability, outcome)
        return self.temperature
