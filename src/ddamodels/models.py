from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class MajorityModel:
    def fit(self, y: np.ndarray) -> "MajorityModel":
        self.probability = float(np.mean(y))
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.full(n, self.probability, dtype=np.float32)


class NumpyLogisticRegression:
    """Small dependency-free fallback logistic model using gradient descent."""
    def __init__(self, lr: float = 1e-2, epochs: int = 500, l2: float = 1e-4):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mean_ = x.mean(0)
        self.std_ = x.std(0)
        self.std_[self.std_ < 1e-8] = 1.0
        z = (x - self.mean_) / self.std_
        z = np.c_[np.ones(len(z)), z]
        self.w_ = np.zeros(z.shape[1])
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-np.clip(z @ self.w_, -40, 40)))
            grad = z.T @ (p - y) / len(y)
            grad[1:] += self.l2 * self.w_[1:]
            self.w_ -= self.lr * grad
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x) - self.mean_) / self.std_
        z = np.c_[np.ones(len(z)), z]
        return 1 / (1 + np.exp(-np.clip(z @ self.w_, -40, 40)))


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


if nn is not None:
    class CandidateConditionedGRU(nn.Module):
        def __init__(self, feature_count: int, context_count: int, difficulty_count: int = 1,
                     hidden_size: int = 64, dropout: float = 0.2):
            super().__init__()
            self.gru = nn.GRU(feature_count, hidden_size, batch_first=True)
            self.context_net = nn.Sequential(nn.Linear(context_count, 16), nn.ReLU())
            self.difficulty_net = nn.Sequential(nn.Linear(difficulty_count, 16), nn.ReLU())
            self.head = nn.Sequential(
                nn.Linear(hidden_size + 16 + 16, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
            )

        def forward(self, history, context, difficulty):
            _, h = self.gru(history)
            h = h[-1]
            z = torch.cat([h, self.context_net(context), self.difficulty_net(difficulty)], dim=-1)
            return self.head(z).squeeze(-1)
else:
    CandidateConditionedGRU = None


@dataclass
class GRUTrainingResult:
    model: Any
    history: list[dict[str, float]]


def train_gru(history: np.ndarray, context: np.ndarray, candidate: np.ndarray, y: np.ndarray,
              epochs: int = 50, batch_size: int = 32, lr: float = 1e-3,
              seed: int = 42, class_weight: bool = True) -> GRUTrainingResult:
    if torch is None or CandidateConditionedGRU is None:
        raise RuntimeError("PyTorch is required for GRU training. Install requirements.txt")
    torch.manual_seed(seed)
    model = CandidateConditionedGRU(history.shape[-1], context.shape[-1], candidate.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    if class_weight:
        pos = max(float(y.sum()), 1.0)
        neg = max(float(len(y) - y.sum()), 1.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    h_t = torch.as_tensor(history, dtype=torch.float32)
    c_t = torch.as_tensor(context, dtype=torch.float32)
    d_t = torch.as_tensor(candidate, dtype=torch.float32)
    order = np.arange(len(y))
    records = []
    for epoch in range(epochs):
        np.random.default_rng(seed + epoch).shuffle(order)
        model.train()
        losses = []
        for start in range(0, len(order), batch_size):
            ix = order[start:start + batch_size]
            opt.zero_grad()
            logits = model(h_t[ix], c_t[ix], d_t[ix])
            loss = loss_fn(logits, y_t[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        records.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
    return GRUTrainingResult(model, records)


def gru_predict(model, history: np.ndarray, context: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for GRU inference")
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(history, dtype=torch.float32),
                       torch.as_tensor(context, dtype=torch.float32),
                       torch.as_tensor(candidate, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()


if nn is not None:
    class TokenizedFiLMGRU(nn.Module):
        """Masked canonical telemetry encoder with FiLM difficulty conditioning."""
        def __init__(self, feature_count: int, context_count: int, difficulty_count: int = 10,
                     latent_size: int = 64, token_size: int = 32, hidden_size: int = 64):
            super().__init__()
            self.feature_embedding = nn.Embedding(feature_count, token_size)
            self.value_net = nn.Sequential(nn.Linear(2, token_size), nn.ReLU())
            self.token_net = nn.Sequential(nn.Linear(token_size, latent_size), nn.ReLU())
            self.gate_net = nn.Linear(latent_size, 1)
            self.temporal = nn.GRU(latent_size, hidden_size, batch_first=True)
            self.context_net = nn.Sequential(nn.Linear(context_count, 16), nn.ReLU())
            self.difficulty_net = nn.Sequential(nn.Linear(difficulty_count, 32), nn.ReLU())
            self.gamma = nn.Linear(32, hidden_size)
            self.beta = nn.Linear(32, hidden_size)
            self.head = nn.Sequential(nn.Linear(hidden_size + 16 + 32, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, values, mask, context, difficulty, feature_ids=None):
            b, t, f = values.shape
            if feature_ids is None:
                feature_ids = torch.arange(f, device=values.device)
            emb = self.feature_embedding(feature_ids).view(1, 1, f, -1).expand(b, t, f, -1)
            vm = torch.stack([values, mask], dim=-1)
            tokens = self.value_net(vm) + emb
            tokens = self.token_net(tokens)
            # Static-shape gated pooling: multiplication by mask avoids -inf/
            # MaskedFill and is friendlier to FP16 ONNX/Sentis execution.
            gate_logits = self.gate_net(tokens).squeeze(-1)
            weights = torch.exp(torch.tanh(gate_logits)) * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            z = (tokens * weights.unsqueeze(-1)).sum(dim=-2)
            _, h = self.temporal(z)
            h = h[-1]
            de = self.difficulty_net(difficulty)
            h = self.gamma(de) * h + self.beta(de)
            out = torch.cat([h, self.context_net(context), de], dim=-1)
            return self.head(out).squeeze(-1)
else:
    TokenizedFiLMGRU = None


def train_tokenized_film_gru(values: np.ndarray, masks: np.ndarray, context: np.ndarray,
                             difficulty: np.ndarray, y: np.ndarray, epochs: int = 50,
                             batch_size: int = 32, lr: float = 1e-3, seed: int = 42,
                             empirical_prior_dropout: float = 0.35) -> GRUTrainingResult:
    if torch is None or TokenizedFiLMGRU is None:
        raise RuntimeError("PyTorch is required for TokenizedFiLMGRU training. Install requirements.txt")
    torch.manual_seed(seed)
    model = TokenizedFiLMGRU(values.shape[-1], context.shape[-1], difficulty.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    tensors = [torch.as_tensor(a, dtype=torch.float32) for a in (values, masks, context, difficulty, y)]
    order = np.arange(len(y)); records = []
    for epoch in range(epochs):
        np.random.default_rng(seed + epoch).shuffle(order)
        losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            ix = order[start:start + batch_size]
            opt.zero_grad()
            difficulty_batch = tensors[3][ix].clone()
            # Difficulty is packed as [five values, five availability bits].
            # Drop only empirical_prior so the network learns the cold-start path.
            if difficulty_batch.shape[-1] >= 10 and empirical_prior_dropout > 0:
                drop = torch.rand(len(ix)) < empirical_prior_dropout
                difficulty_batch[drop, 4] = 0.0
                difficulty_batch[drop, 9] = 0.0
            logits = model(tensors[0][ix], tensors[1][ix], tensors[2][ix], difficulty_batch)
            loss = loss_fn(logits, tensors[4][ix])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach()))
        records.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
    return GRUTrainingResult(model, records)


def export_tokenized_onnx(model, output_path: str, sequence_length: int,
                          feature_count: int, context_count: int, difficulty_count: int = 10) -> None:
    """Export static-shape model for early Unity Sentis compatibility checks.

    difficulty_count defaults to 10 (5 values + 5 mask) to match the v2
    data pipeline which concatenates difficulty values and their availability
    mask into a single tensor.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for ONNX export")
    model.eval()
    dummy = (torch.zeros(1, sequence_length, feature_count),
             torch.ones(1, sequence_length, feature_count),
             torch.zeros(1, context_count), torch.zeros(1, difficulty_count))
    torch.onnx.export(model, dummy, output_path, input_names=["values", "mask", "context", "difficulty"],
                      output_names=["logit"], dynamic_axes=None, opset_version=17)

def estimate_mask_uncertainty(mask, base_uncertainty: float = 0.05) -> float:
    """Estimate epistemic uncertainty from feature availability mask using group weighting.

    Instead of a flat average over all 21 features (which would under-penalize
    losing a small core group like vitality_error and over-penalize early-game
    unobserved historical stats), this calculates coverage across the 4 canonical
    business groups (progress, vitality, capability, cognitive-motor) with equal
    25% weight each.

    Returns a value in [base_uncertainty, 1.0].
    """
    m = np.asarray(mask, dtype=float)
    if m.ndim > 1:
        # Use average mask across the sequence window
        m = np.mean(m, axis=tuple(range(m.ndim - 1)))
    
    if len(m) == 21:
        # Canonical groups:
        # [0..3]: progress_tempo (4 features)
        # [4..7]: vitality_error (4 features)
        # [8..11]: capability_economy (4 features)
        # [12..20]: human_cognitive_motor (9 features)
        c_progress = float(np.mean(m[0:4]))
        c_vitality = float(np.mean(m[4:8]))
        c_capability = float(np.mean(m[8:12]))
        c_motor = float(np.mean(m[12:21]))
        coverage = 0.25 * c_progress + 0.25 * c_vitality + 0.25 * c_capability + 0.25 * c_motor
    else:
        coverage = float(np.mean(m))

    coverage = float(np.clip(coverage, 0.0, 1.0))
    return float(base_uncertainty + (1.0 - coverage) * (1.0 - base_uncertainty))


def estimate_struggle_probability(probabilities: dict,
                                  current_difficulty: int | None = None,
                                  recent_win_rate: float | None = None,
                                  win_rate_available: bool = True) -> float:
    """Estimate struggle probability from candidate success predictions and performance history.

    This is a principled heuristic struggle estimator:
    - Uses p(d_current) specifically rather than averaging across all candidate
      perturbations, avoiding lag when an easier candidate predicts high success.
    - Explicitly guards against unobserved early-game win rates (when mask=0):
      if recent_win_rate is missing or marked unavailable, it evaluates struggle
      strictly from failure probability at current difficulty: 1.0 - p(d_current).
    """
    if not probabilities:
        return 0.0

    if current_difficulty is not None and current_difficulty in probabilities:
        p_curr = float(probabilities[current_difficulty])
    else:
        # Fallback to mean if current_difficulty not specified or not in keys
        p_curr = sum(float(v) for v in probabilities.values()) / len(probabilities)

    # Early-game guard: if win rate is unavailable or unobserved (mask=0),
    # evaluate struggle strictly from model prediction at current difficulty.
    if not win_rate_available or recent_win_rate is None or not np.isfinite(recent_win_rate):
        struggle = 1.0 - p_curr
    else:
        struggle = (1.0 - p_curr) * 0.6 + (1.0 - float(np.clip(recent_win_rate, 0.0, 1.0))) * 0.4

    return float(np.clip(struggle, 0.0, 1.0))
