from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import PreparedDataset, load_window_dataset
from .models import MajorityModel, NumpyLogisticRegression, train_gru, gru_predict


def group_split(groups: np.ndarray, seed: int = 42, train_frac: float = .70, valid_frac: float = .15):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = max(1, int(len(unique) * train_frac)) if len(unique) > 2 else max(1, len(unique) - 1)
    n_valid = max(0, int(len(unique) * valid_frac)) if len(unique) > 3 else 0
    train_g = set(unique[:n_train])
    valid_g = set(unique[n_train:n_train + n_valid])
    train = np.array([g in train_g for g in groups])
    valid = np.array([g in valid_g for g in groups])
    test = ~(train | valid)
    if test.sum() == 0:
        test = valid.copy(); valid[:] = False
    return train, valid, test, {"train": sorted(train_g), "validation": sorted(valid_g), "test": sorted(set(unique) - train_g - valid_g)}


def aggregate_history(x: np.ndarray) -> np.ndarray:
    last = x[:, -1, :]
    mean = x.mean(1)
    std = x.std(1)
    slope = (x[:, -1, :] - x[:, 0, :]) / max(x.shape[1] - 1, 1)
    return np.concatenate([last, mean, std, slope], axis=1)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p):
    p = np.clip(np.asarray(p), 1e-7, 1 - 1e-7)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p)))


def accuracy(y, p, threshold=.5):
    return float(np.mean((np.asarray(p) >= threshold) == np.asarray(y)))


def expected_calibration_error(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    result = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= lo) & (p <= hi if hi == 1 else p < hi)
        if mask.any():
            result += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


def metrics(y, p) -> dict[str, float]:
    out = {"brier": brier(y, p), "log_loss": log_loss(y, p),
           "accuracy": accuracy(y, p), "ece": expected_calibration_error(y, p)}
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
        if len(np.unique(y)) > 1:
            out["roc_auc"] = float(roc_auc_score(y, p))
            out["pr_auc"] = float(average_precision_score(y, p))
            out["balanced_accuracy"] = float(balanced_accuracy_score(y, p >= .5))
    except ImportError:
        pass
    return out


def _fit_temperature(y, logits, grid=None):
    """Small validation-only temperature search; avoids test leakage."""
    grid = grid or np.linspace(.25, 4.0, 32)
    best_t, best = 1.0, float("inf")
    for t in grid:
        p = 1 / (1 + np.exp(-np.clip(logits / t, -40, 40)))
        score = log_loss(y, p)
        if score < best:
            best, best_t = score, float(t)
    return best_t


def run_training(dataset_path: str | Path, output_dir: str | Path, epochs: int = 50, seed: int = 42) -> dict[str, Any]:
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    ds: PreparedDataset = load_window_dataset(dataset_path)
    tr, va, te, manifest = group_split(ds.groups, seed=seed)
    if tr.sum() == 0 or te.sum() == 0:
        raise ValueError("Need at least train and test samples after group split")
    np.savez(outdir / "normalized_data.npz", history=ds.history, context=ds.context,
             candidate=ds.candidate, y=ds.y)
    (outdir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    x = np.concatenate([aggregate_history(ds.history), ds.context, ds.candidate], axis=1)
    results: dict[str, Any] = {"dataset": {"samples": int(len(ds.y)), "history_shape": list(ds.history.shape),
                                             "context_shape": list(ds.context.shape), "positive_rate": float(ds.y.mean())},
                               "models": {}}
    majority = MajorityModel().fit(ds.y[tr])
    results["models"]["majority"] = metrics(ds.y[te], majority.predict_proba(te.sum()))

    logistic = NumpyLogisticRegression().fit(x[tr], ds.y[tr])
    lp = logistic.predict_proba(x[te])
    results["models"]["numpy_logistic"] = metrics(ds.y[te], lp)
    np.savez(outdir / "logistic_model.npz", mean=logistic.mean_, std=logistic.std_, weights=logistic.w_)

    try:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                    class_weight="balanced", random_state=seed, n_jobs=-1)
        rf.fit(x[tr], ds.y[tr])
        rp = rf.predict_proba(x[te])[:, 1]
        results["models"]["random_forest"] = metrics(ds.y[te], rp)
        import joblib
        joblib.dump(rf, outdir / "random_forest.joblib")
    except ImportError:
        results["models"]["random_forest"] = {"status": "skipped", "reason": "scikit-learn not installed"}

    try:
        from .models import torch
        if torch is None:
            raise ImportError
        gru_result = train_gru(ds.history[tr], ds.context[tr], ds.candidate[tr], ds.y[tr], epochs=epochs, seed=seed)
        torch.save(gru_result.model.state_dict(), outdir / "candidate_gru.pt")
        gp = gru_predict(gru_result.model, ds.history[te], ds.context[te], ds.candidate[te])
        results["models"]["candidate_gru"] = metrics(ds.y[te], gp)
        (outdir / "gru_training_history.json").write_text(json.dumps(gru_result.history, indent=2), encoding="utf-8")
    except (ImportError, RuntimeError) as exc:
        reason = str(exc) or "PyTorch is not installed"
        results["models"]["candidate_gru"] = {"status": "skipped", "reason": reason}

    contract = {"history_shape": list(ds.history.shape[1:]), "context_names": ds.context_names,
                "feature_names": ds.feature_names, "candidate_shape": list(ds.candidate.shape[1:]),
                "sequence_length": int(ds.history.shape[1]), "seed": seed}
    (outdir / "feature_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (outdir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
