#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ddamodels.data import load_session, normalize_snapshot_frame, save_snapshot_features, build_window_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--snapshot-output')
    ap.add_argument('--sequence-length', type=int, default=12)
    ap.add_argument('--stride', type=int, default=3)
    args = ap.parse_args()
    session = load_session(args.input)
    df = normalize_snapshot_frame(session)
    if df.empty:
        raise SystemExit('No samples found in input')
    if args.snapshot_output:
        save_snapshot_features(df, args.snapshot_output)
    if 'segment_success' not in df.columns:
        raise SystemExit('No segment_success labels. Add segment-level outcome labels; session result must not be copied to every snapshot.')
    rows = build_window_rows(df, args.sequence_length, args.stride)
    if rows.empty:
        raise SystemExit('No valid windows created. Check snapshot interval, segment_id, labels and feature completeness.')
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.parquet': rows.to_parquet(out, index=False)
    else: rows.to_csv(out, index=False)
    print(f'created {len(rows)} windows -> {out}')

if __name__ == '__main__':
    main()
