#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ddamodels.data import load_tokenized_window_dataset
from ddamodels.models import train_tokenized_film_gru, torch
from ddamodels.train import group_split, metrics


def main():
    ap = argparse.ArgumentParser(description='Train adaptive masked Tokenized FiLM GRU')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if torch is None:
        raise SystemExit('PyTorch is required. Install requirements.txt')
    values, masks, context, difficulty, y, groups = load_tokenized_window_dataset(args.dataset)
    tr, va, te, manifest = group_split(groups, seed=args.seed)
    if tr.sum() == 0 or te.sum() == 0:
        raise SystemExit('Need train and test groups')
    result = train_tokenized_film_gru(values[tr], masks[tr], context[tr], difficulty[tr], y[tr], args.epochs, seed=args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), out / 'tokenized_film_gru.pt')
    (out / 'split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    result_payload = {'test_positive_rate': float(y[te].mean()), 'training_history': result.history,
                      'model': 'TokenizedFiLMGRU', 'input_shape': list(values.shape[1:]),
                      'mask_shape': list(masks.shape[1:]), 'difficulty_shape': list(difficulty.shape[1:])}
    (out / 'metrics.json').write_text(json.dumps(result_payload, indent=2), encoding='utf-8')
    print(json.dumps(result_payload, indent=2))

if __name__ == '__main__':
    main()
