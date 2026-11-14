#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ddamodels.data import load_tokenized_window_dataset
from ddamodels.models import export_tokenized_onnx, train_tokenized_film_gru, torch, TokenizedFiLMGRU
from ddamodels.train import group_split, metrics


def predict(model, values, masks, context, difficulty):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(values, dtype=torch.float32),
                       torch.as_tensor(masks, dtype=torch.float32),
                       torch.as_tensor(context, dtype=torch.float32),
                       torch.as_tensor(difficulty, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Train adaptive masked Tokenized FiLM GRU")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feature-group-dropout", type=float, default=.15)
    ap.add_argument("--empirical-prior-dropout", type=float, default=.35)
    args = ap.parse_args()
    if torch is None:
        raise SystemExit("PyTorch is required. Install requirements.txt")
    values, masks, context, difficulty, y, groups = load_tokenized_window_dataset(args.dataset)
    tr, va, te, manifest = group_split(groups, seed=args.seed)
    if tr.sum() == 0 or te.sum() == 0:
        raise SystemExit("Need train and test groups")
    validation = None if va.sum() == 0 else (values[va], masks[va], context[va], difficulty[va], y[va])
    result = train_tokenized_film_gru(
        values[tr], masks[tr], context[tr], difficulty[tr], y[tr], epochs=args.epochs,
        seed=args.seed, empirical_prior_dropout=args.empirical_prior_dropout,
        feature_group_dropout=args.feature_group_dropout, validation=validation,
        patience=args.patience)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), out / "tokenized_film_gru.pt")
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    test_p = predict(result.model, values[te], masks[te], context[te], difficulty[te])
    result_payload = {
        "dataset": {"samples": int(len(y)), "test_samples": int(te.sum()), "test_positive_rate": float(y[te].mean())},
        "metrics_test": metrics(y[te], test_p),
        "training_history": result.history,
        "model": "TokenizedFiLMGRU_static_unrolled",
        "input_shape": list(values.shape[1:]), "mask_shape": list(masks.shape[1:]),
        "difficulty_shape": list(difficulty.shape[1:]),
        "training": {"feature_group_dropout": args.feature_group_dropout,
                     "empirical_prior_dropout": args.empirical_prior_dropout,
                     "early_stopping_patience": args.patience},
    }
    (out / "metrics.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    try:
        export_tokenized_onnx(result.model, str(out / "tokenized_film_gru.onnx"),
                              values.shape[1], values.shape[2], context.shape[1], difficulty.shape[1])
        result_payload["onnx"] = {"status": "exported", "opset": 15, "gru": "static_unrolled"}
    except Exception as exc:
        result_payload["onnx"] = {"status": "skipped", "reason": str(exc)}
    (out / "metrics.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    main()
