from __future__ import annotations

import numpy as np


class OnlineTemperatureCalibrator:
    """Bounded batch-MAP calibration.

    Predictions are buffered after completed segment outcomes. Temperature is
    updated only when `batch_size` outcomes from the same domain/patch exist.
    The quadratic prior keeps T near `base_temperature` and prevents one bad
    wave from destabilizing the rest of a session.
    """
    def __init__(self, temperature: float = 1.0, lr: float = .02, batch_size: int = 20,
                 min_temperature: float = .7, max_temperature: float = 2.0,
                 prior_strength: float = 10.0):
        self.temperature = float(np.clip(temperature, min_temperature, max_temperature))
        self.base_temperature = self.temperature
        self.lr = lr; self.batch_size = batch_size
        self.min_temperature = min_temperature; self.max_temperature = max_temperature
        self.prior_strength = prior_strength
        self.buffer: list[tuple[float, int]] = []
        self.updates = 0

    def transform(self, probability):
        p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p)) / self.temperature
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))

    def observe(self, probability: float, outcome: int) -> bool:
        """Buffer a label; return True only when a calibration update occurred.

        Stores the logit (inverse sigmoid) directly to avoid numerical
        imprecision from round-tripping through probability space when
        p is near 0 or 1.
        """
        p = float(np.clip(probability, 1e-6, 1 - 1e-6))
        logit = float(np.log(p / (1 - p)))
        self.buffer.append((logit, int(outcome)))
        if len(self.buffer) < self.batch_size:
            return False
        self.update_batch()
        return True

    def update_batch(self) -> float:
        if not self.buffer:
            return self.temperature
        logits = np.asarray([l for l, _ in self.buffer], dtype=float)
        y = np.asarray([y for _, y in self.buffer], dtype=float)
        t = self.temperature
        for _ in range(25):
            q = 1 / (1 + np.exp(-np.clip(logits / t, -40, 40)))
            grad_likelihood = np.mean((q - y) * (-logits / (t ** 2)))
            grad_prior = self.prior_strength * (t - self.base_temperature)
            t = t - self.lr * (grad_likelihood + grad_prior)
            t = float(np.clip(t, self.min_temperature, self.max_temperature))
        self.temperature = t
        self.buffer.clear(); self.updates += 1
        return self.temperature

    # Backward-compatible alias: a call now buffers instead of mutating T.
    def update(self, probability: float, outcome: int) -> float:
        self.observe(probability, outcome)
        return self.temperature
