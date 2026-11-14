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
        z = np.c_[np.ones(len(x)), (x - self.mean_) / self.std_]
        self.w_ = np.zeros(z.shape[1])
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-np.clip(z @ self.w_, -40, 40)))
            grad = z.T @ (p - y) / len(y)
            grad[1:] += self.l2 * self.w_[1:]
            self.w_ -= self.lr * grad
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = np.c_[np.ones(len(x)), (np.asarray(x) - self.mean_) / self.std_]
        return 1 / (1 + np.exp(-np.clip(z @ self.w_, -40, 40)))


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
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
            self.head = nn.Sequential(nn.Linear(hidden_size + 16 + 16, 32), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(32, 1))

        def forward(self, history, context, difficulty):
            _, h = self.gru(history)
            z = torch.cat([h[-1], self.context_net(context), self.difficulty_net(difficulty)], dim=-1)
            return self.head(z).squeeze(-1)
else:
    CandidateConditionedGRU = None


@dataclass
class GRUTrainingResult:
    model: Any
    history: list[dict[str, float]]


def train_gru(history: np.ndarray, context: np.ndarray, candidate: np.ndarray, y: np.ndarray,
              epochs: int = 50, batch_size: int = 32, lr: float = 1e-3,
              seed: int = 42, class_weight: bool = False) -> GRUTrainingResult:
    if torch is None or CandidateConditionedGRU is None:
        raise RuntimeError("PyTorch is required for GRU training. Install requirements.txt")
    torch.manual_seed(seed)
    model = CandidateConditionedGRU(history.shape[-1], context.shape[-1], candidate.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss()
    if class_weight:
        pos = max(float(y.sum()), 1.0); neg = max(float(len(y) - y.sum()), 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    h_t = torch.as_tensor(history, dtype=torch.float32)
    c_t = torch.as_tensor(context, dtype=torch.float32)
    d_t = torch.as_tensor(candidate, dtype=torch.float32)
    order = np.arange(len(y)); records = []
    for epoch in range(epochs):
        np.random.default_rng(seed + epoch).shuffle(order); losses = []; model.train()
        for start in range(0, len(order), batch_size):
            ix = order[start:start + batch_size]; opt.zero_grad()
            loss = loss_fn(model(h_t[ix], c_t[ix], d_t[ix]), y_t[ix]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach()))
        records.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
    return GRUTrainingResult(model, records)


class _StaticGRU(nn.Module if nn is not None else object):
    """A fixed-length GRU using only Linear/Sigmoid/Tanh/Mul/Add nodes.

    Unlike nn.GRU, this exports without an ONNX GRU operator. The loop is over
    a Python-known sequence length and is therefore unrolled by torch tracing.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 3 * hidden_size)
        self.hidden_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.hidden_size = hidden_size

    def forward(self, x):
        h = torch.zeros(x.shape[0], self.hidden_size, dtype=x.dtype, device=x.device)
        outputs = []
        for step in range(x.shape[1]):
            i_r, i_z, i_n = self.input_proj(x[:, step]).chunk(3, dim=-1)
            h_r, h_z, h_n = self.hidden_proj(h).chunk(3, dim=-1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1.0 - z) * n + z * h
            outputs.append(h)
        return torch.stack(outputs, dim=1), h.unsqueeze(0)


if nn is not None:
    class TokenizedFiLMGRU(nn.Module):
        """Masked canonical encoder with static GRU, safe FiLM and dropout."""
        def __init__(self, feature_count: int, context_count: int, difficulty_count: int = 10,
                     latent_size: int = 64, token_size: int = 32, hidden_size: int = 64,
                     dropout: float = 0.15):
            super().__init__()
            self.feature_embedding = nn.Embedding(feature_count, token_size)
            self.value_net = nn.Sequential(nn.Linear(2, token_size), nn.ReLU())
            self.token_net = nn.Sequential(nn.Linear(token_size, latent_size), nn.ReLU())
            self.gate_net = nn.Linear(latent_size, 1)
            self.temporal = _StaticGRU(latent_size, hidden_size)
            self.context_net = nn.Sequential(nn.Linear(context_count, 16), nn.ReLU())
            self.difficulty_net = nn.Sequential(nn.Linear(difficulty_count, 32), nn.ReLU())
            # Identity-preserving FiLM: h' starts as h rather than beta alone.
            self.gamma_delta = nn.Linear(32, hidden_size)
            self.beta = nn.Linear(32, hidden_size)
            nn.init.zeros_(self.gamma_delta.weight); nn.init.zeros_(self.gamma_delta.bias)
            nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)
            self.hidden_dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(hidden_size + 16 + 32, 32), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(32, 1))

        def forward(self, values, mask, context, difficulty, feature_ids=None):
            b, _, f = values.shape
            if feature_ids is None:
                feature_ids = torch.arange(f, device=values.device)
            emb = self.feature_embedding(feature_ids).view(1, 1, f, -1).expand(b, values.shape[1], f, -1)
            tokens = self.token_net(self.value_net(torch.stack([values, mask], dim=-1)) + emb)
            gates = self.gate_net(tokens).squeeze(-1)
            weights = torch.exp(torch.tanh(gates)) * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            z = (tokens * weights.unsqueeze(-1)).sum(dim=-2)
            _, h = self.temporal(z)
            h = h[-1]
            de = self.difficulty_net(difficulty)
            h = (1.0 + self.gamma_delta(de)) * h + self.beta(de)
            out = torch.cat([self.hidden_dropout(h), self.context_net(context), de], dim=-1)
            return self.head(out).squeeze(-1)
else:
    TokenizedFiLMGRU = None


def _drop_feature_groups(values, masks, probability: float, rng):
    """Randomly mask 0 or 1 canonical group while retaining at least 2 groups."""
    if probability <= 0 or values.shape[-1] != 21:
        return values, masks
    groups = ((0, 4), (4, 8), (8, 12), (12, 21))
    out_v, out_m = values.clone(), masks.clone()
    for row in range(len(out_v)):
        if rng.random() >= probability:
            continue
        group = int(rng.integers(0, len(groups)))
        start, end = groups[group]
        # Keep three groups; this makes the training corruption conservative.
        out_v[row, :, start:end] = 0.0
        out_m[row, :, start:end] = 0.0
    return out_v, out_m


def train_tokenized_film_gru(values: np.ndarray, masks: np.ndarray, context: np.ndarray,
                             difficulty: np.ndarray, y: np.ndarray, epochs: int = 50,
                             batch_size: int = 32, lr: float = 1e-3, seed: int = 42,
                             empirical_prior_dropout: float = 0.35,
                             feature_group_dropout: float = 0.15,
                             validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
                             patience: int = 8, min_delta: float = 1e-4,
                             dropout: float = 0.15) -> GRUTrainingResult:
    if torch is None or TokenizedFiLMGRU is None:
        raise RuntimeError("PyTorch is required for TokenizedFiLMGRU training. Install requirements.txt")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TokenizedFiLMGRU(values.shape[-1], context.shape[-1], difficulty.shape[-1], dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    tensors = [torch.as_tensor(a, dtype=torch.float32) for a in (values, masks, context, difficulty, y)]
    val_tensors = None if validation is None else [torch.as_tensor(a, dtype=torch.float32) for a in validation]
    order = np.arange(len(y)); records = []; best_state = None; best_val = float("inf"); bad_epochs = 0
    for epoch in range(epochs):
        rng.shuffle(order); losses = []; model.train()
        for start in range(0, len(order), batch_size):
            ix = order[start:start + batch_size]
            opt.zero_grad()
            v_batch, m_batch = _drop_feature_groups(tensors[0][ix], tensors[1][ix], feature_group_dropout, rng)
            d_batch = tensors[3][ix].clone()
            if d_batch.shape[-1] >= 10 and empirical_prior_dropout > 0:
                drop = torch.as_tensor(rng.random(len(ix)) < empirical_prior_dropout)
                d_batch[drop, 4] = 0.0; d_batch[drop, 9] = 0.0
            logits = model(v_batch, m_batch, tensors[2][ix], d_batch)
            loss = loss_fn(logits, tensors[4][ix]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        if val_tensors is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(*val_tensors[:4])
                val_loss = float(loss_fn(val_logits, val_tensors[4]).detach())
            record["validation_loss"] = val_loss
            if val_loss < best_val - min_delta:
                best_val = val_loss; bad_epochs = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                records.append(record); break
        records.append(record)
    if best_state is not None:
        model.load_state_dict(best_state)
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


def export_tokenized_onnx(model, output_path: str, sequence_length: int,
                          feature_count: int, context_count: int, difficulty_count: int = 10) -> None:
    """Export static-shape graph without ONNX GRU, using opset 15."""
    if torch is None:
        raise RuntimeError("PyTorch is required for ONNX export")
    model.eval()
    dummy = (torch.zeros(1, sequence_length, feature_count), torch.ones(1, sequence_length, feature_count),
             torch.zeros(1, context_count), torch.zeros(1, difficulty_count))
    torch.onnx.export(model, dummy, output_path, input_names=["values", "mask", "context", "difficulty"],
                      output_names=["logit"], dynamic_axes=None, opset_version=15)


def estimate_mask_uncertainty(mask, base_uncertainty: float = 0.05) -> float:
    m = np.asarray(mask, dtype=float)
    if m.ndim > 1:
        m = np.mean(m, axis=tuple(range(m.ndim - 1)))
    if len(m) == 21:
        coverage = .25 * (np.mean(m[0:4]) + np.mean(m[4:8]) + np.mean(m[8:12]) + np.mean(m[12:21]))
    else:
        coverage = float(np.mean(m))
    coverage = float(np.clip(coverage, 0.0, 1.0))
    return float(base_uncertainty + (1.0 - coverage) * (1.0 - base_uncertainty))


def estimate_struggle_probability(probabilities: dict, current_difficulty: int | None = None,
                                  recent_win_rate: float | None = None,
                                  win_rate_available: bool = True) -> float:
    if not probabilities:
        return 0.0
    p_curr = float(probabilities.get(current_difficulty, np.mean(list(probabilities.values()))))
    if not win_rate_available or recent_win_rate is None or not np.isfinite(recent_win_rate):
        return float(np.clip(1.0 - p_curr, 0.0, 1.0))
    return float(np.clip(.6 * (1.0 - p_curr) + .4 * (1.0 - np.clip(recent_win_rate, 0.0, 1.0)), 0.0, 1.0))
