from __future__ import annotations

import numpy as np


class OnlineTemperatureCalibrator:
    """Causal online calibration; update only with completed future labels."""
    def __init__(self, temperature: float = 1.0, lr: float = .01, min_temperature: float = .25, max_temperature: float = 4.0):
        self.temperature = float(temperature)
        self.lr = lr
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.updates = 0

    def transform(self, probability):
        p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1-1e-6)
        logits = np.log(p / (1-p)) / self.temperature
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))

    def update(self, probability: float, outcome: int) -> float:
        p = float(np.clip(probability, 1e-6, 1-1e-6))
        y = float(outcome)
        logit = np.log(p / (1-p))
        calibrated = 1 / (1 + np.exp(-np.clip(logit / self.temperature, -40, 40)))
        # d BCE / dT for logit/T; gradient descent with bounded T.
        grad = (calibrated - y) * (-logit / (self.temperature ** 2))
        self.temperature = float(np.clip(self.temperature - self.lr * grad,
                                         self.min_temperature, self.max_temperature))
        self.updates += 1
        return self.temperature
