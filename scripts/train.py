#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ddamodels.train import run_training


def main():
    ap = argparse.ArgumentParser(description='Train DDA telemetry models')
    ap.add_argument('--dataset', required=True, help='CSV/Parquet window dataset')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    result = run_training(args.dataset, args.output_dir, args.epochs, args.seed)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
