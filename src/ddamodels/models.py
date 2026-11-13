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
        def __init__(self, feature_count: int, context_count: int, difficulty_count: int = 5,
                     latent_size: int = 64, token_size: int = 32, hidden_size: int = 64):
            super().__init__()
            self.feature_embedding = nn.Embedding(feature_count, token_size)
            self.value_net = nn.Sequential(nn.Linear(2, token_size), nn.ReLU())
            self.token_net = nn.Sequential(nn.Linear(token_size, latent_size), nn.ReLU())
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
            denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            z = (tokens * mask.unsqueeze(-1)).sum(dim=-2) / denom
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
                             batch_size: int = 32, lr: float = 1e-3, seed: int = 42) -> GRUTrainingResult:
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
            logits = model(tensors[0][ix], tensors[1][ix], tensors[2][ix], tensors[3][ix])
            loss = loss_fn(logits, tensors[4][ix])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach()))
        records.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
    return GRUTrainingResult(model, records)
