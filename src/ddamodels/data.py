from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .canonical import CANONICAL_FEATURES, CausalSessionNormalizer

BASE_FEATURES = [
    "progress_ratio", "progress_velocity", "hp_ratio", "hp_loss_rate",
    "gold_normalized", "gold_spend_rate", "tower_count", "tower_build_rate",
    "normalized_segment_index", "wave_progress_rate", "enemy_alive", "kills_5s",
    "hits_taken_5s", "hits_taken_30s", "deaths_30s", "enemy_leaks_30s",
    "action_count_5s", "invalid_action_rate", "idle_ratio_5s",
    "recent_win_rate", "error_slope", "score_ema", "current_difficulty",
]

REQUIRED_SESSION = ["schemaVersion", "sessionId", "startedAtUtc"]


@dataclass
class PreparedDataset:
    history: np.ndarray
    context: np.ndarray
    candidate: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]
    context_names: list[str]


def _read_any(path: str | Path) -> Any:
    p = Path(path)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported input format: {p.suffix}")


def load_session(path: str | Path) -> dict[str, Any]:
    raw = _read_any(path)
    if not isinstance(raw, dict):
        raise ValueError("A raw session JSON must be an object")
    # Legacy loggame.json used matchId as the only session identifier.
    # Preserve compatibility while keeping the canonical output field sessionId.
    if "sessionId" not in raw and "matchId" in raw:
        raw["sessionId"] = raw["matchId"]
    missing = [k for k in REQUIRED_SESSION if k not in raw]
    if missing:
        raise ValueError(f"Missing session fields: {missing}")
    if not isinstance(raw.get("samples", []), list):
        raise ValueError("session.samples must be a list")
    return raw


def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def normalize_snapshot_frame(session: dict[str, Any]) -> pd.DataFrame:
    """Normalize legacy session samples into a snapshot frame.

    This function never fabricates events. It maps only fields actually present in
    the source log and leaves unavailable behavior features as NaN.
    """
    rows = session.get("samples", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    started = pd.to_datetime(session["startedAtUtc"], utc=True)
    if "recordedAtUtc" in df:
        df["recorded_at"] = pd.to_datetime(df["recordedAtUtc"], utc=True, errors="coerce")
        df["elapsed_seconds"] = (df["recorded_at"] - started).dt.total_seconds()
    elif "elapsedRealSeconds" in df:
        df["elapsed_seconds"] = pd.to_numeric(df["elapsedRealSeconds"], errors="coerce")
    else:
        raise ValueError("Samples need recordedAtUtc or elapsedRealSeconds")
    df = df.sort_values("elapsed_seconds").drop_duplicates(
        subset=["elapsed_seconds", "reason"] if "reason" in df else ["elapsed_seconds"]
    ).reset_index(drop=True)
    hp = pd.to_numeric(df["hp"] if "hp" in df else pd.Series(np.nan, index=df.index), errors="coerce")
    hp_max = pd.to_numeric(df["hpMax"] if "hpMax" in df else pd.Series(np.nan, index=df.index), errors="coerce")
    if hp_max.isna().all():
        hp_max = pd.Series(np.nan, index=df.index)
    df["hp_ratio"] = (hp / hp_max.replace(0, np.nan)).clip(0, 1)
    df["gold"] = pd.to_numeric(df.get("gold", np.nan), errors="coerce")
    df["tower_count"] = pd.to_numeric(df.get("towerCount", np.nan), errors="coerce")
    df["wave"] = pd.to_numeric(df.get("wave", np.nan), errors="coerce")
    df["enemies_leaked"] = pd.to_numeric(df.get("enemiesLeaked", np.nan), errors="coerce")
    df["combat_seconds"] = pd.to_numeric(df.get("combatSeconds", np.nan), errors="coerce")
    df["session_id"] = session["sessionId"]
    df["player_id_hash"] = session.get("playerIdHash", "unknown")
    df["segment_id"] = df.get("segmentId", pd.Series([None] * len(df)))
    df["objective_id"] = df.get("objectiveId", pd.Series([None] * len(df)))
    if "segmentIndex" in df:
        df["segment_index"] = pd.to_numeric(df["segmentIndex"], errors="coerce")
    elif "segmentId" in df and df["segment_id"].notna().any():
        numeric_seg = pd.to_numeric(df["segment_id"], errors="coerce")
        if numeric_seg.notna().any():
            df["segment_index"] = numeric_seg
        else:
            df["segment_index"] = pd.Series(pd.factorize(df["segment_id"])[0], index=df.index).astype(float)
    elif "wave" in df:
        df["segment_index"] = pd.to_numeric(df["wave"], errors="coerce")
    else:
        df["segment_index"] = pd.Series(np.nan, index=df.index)
    df["current_difficulty"] = pd.to_numeric(
        df.get("currentDifficulty", df.get("difficultyProfile", np.nan)), errors="coerce"
    )
    return add_derived_features(df)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6
    dt = out["elapsed_seconds"].diff()
    out["delta_seconds"] = dt
    out["progress_ratio"] = pd.to_numeric(out.get("progressRatio", np.nan), errors="coerce")
    out["progress_velocity"] = out["progress_ratio"].diff() / dt.replace(0, np.nan)
    out["hp_loss_rate"] = (-out["hp_ratio"].diff()).clip(lower=0) / dt.replace(0, np.nan)
    out["gold_normalized"] = out["gold"] / max(float(out["gold"].quantile(0.99) or 1), eps)
    out["gold_spend_rate"] = (-out["gold"].diff()).clip(lower=0) / dt.replace(0, np.nan)
    out["tower_build_rate"] = out["tower_count"].diff().clip(lower=0) / dt.replace(0, np.nan)
    seg_series = out["segment_index"] if ("segment_index" in out and out["segment_index"].notna().any()) else out.get("wave", pd.Series([1.0], index=out.index))
    total_segments = max(float(pd.to_numeric(seg_series, errors="coerce").max() or 1), 1.0)
    out["normalized_segment_index"] = (pd.to_numeric(seg_series, errors="coerce") / total_segments).clip(0.0, 1.0)
    # Maintain wave_normalized alias for backwards compatibility with legacy logs
    out["wave_normalized"] = out["normalized_segment_index"]
    out["wave_progress_rate"] = out["wave"].diff() / dt.replace(0, np.nan)
    out["enemy_alive"] = pd.to_numeric(out.get("enemyAlive", np.nan), errors="coerce")
    out["kills_5s"] = pd.to_numeric(out.get("kills5s", np.nan), errors="coerce")
    out["hits_taken_5s"] = pd.to_numeric(out.get("hitsTaken5s", np.nan), errors="coerce")
    out["hits_taken_30s"] = pd.to_numeric(out.get("hitsTaken30s", np.nan), errors="coerce")
    out["deaths_30s"] = pd.to_numeric(out.get("deaths30s", np.nan), errors="coerce")
    out["enemy_leaks_30s"] = pd.to_numeric(out.get("enemyLeaks30s", np.nan), errors="coerce")
    out["action_count_5s"] = pd.to_numeric(out.get("actionCount5s", np.nan), errors="coerce")
    out["invalid_action_rate"] = pd.to_numeric(out.get("invalidActionRate", np.nan), errors="coerce")
    out["idle_ratio_5s"] = pd.to_numeric(out.get("idleRatio5s", np.nan), errors="coerce")
    out["recent_win_rate"] = pd.to_numeric(out.get("recentWinRate", np.nan), errors="coerce")
    out["error_slope"] = out["enemy_leaks_30s"].diff() / dt.replace(0, np.nan)
    out["score_ema"] = pd.to_numeric(out.get("scoreEma", np.nan), errors="coerce")
    return out


def save_snapshot_features(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".parquet":
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


def _json_or_value(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return x
    return x


def load_window_dataset(path: str | Path, feature_names: Iterable[str] | None = None) -> PreparedDataset:
    raw = _read_any(path)
    df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    required = ["history_json", "context_json", "candidate_difficulty", "segment_success"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Training dataset missing columns: {missing}")
    histories = np.asarray([_json_or_value(x) for x in df["history_json"]], dtype=np.float32)
    if histories.ndim != 3:
        raise ValueError(f"history_json must decode to [N,T,F], got {histories.shape}")
    contexts_raw = [_json_or_value(x) for x in df["context_json"]]
    if contexts_raw and isinstance(contexts_raw[0], dict):
        names = sorted(set().union(*(x.keys() for x in contexts_raw)))
        contexts = np.asarray([[float(x.get(k, 0.0)) for k in names] for x in contexts_raw], dtype=np.float32)
    else:
        contexts = np.asarray(contexts_raw, dtype=np.float32)
        names = [f"context_{i}" for i in range(contexts.shape[1])]
    candidates = np.asarray(df["candidate_difficulty"], dtype=np.float32).reshape(-1, 1)
    y = np.asarray(df["segment_success"], dtype=np.float32)
    groups_col = "player_id_hash" if "player_id_hash" in df else "session_id"
    groups = df[groups_col].fillna("unknown").astype(str).to_numpy()
    return PreparedDataset(histories, contexts, candidates, y, groups,
                           list(feature_names or [f"x_{i}" for i in range(histories.shape[-1])]), names)


def load_tokenized_window_dataset(path: str | Path):
    """Load adaptive v2 rows with values, availability mask and difficulty vector.

    Required columns: values_json, mask_json, context_json, difficulty_vector_json,
    difficulty_mask_json, segment_success. Both masks are mandatory so missing
    does not become a fake 0 or an unobserved empirical prior.
    """
    raw = _read_any(path)
    df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    required = ["values_json", "mask_json", "context_json", "difficulty_vector_json", "difficulty_mask_json", "segment_success"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Adaptive dataset missing columns: {missing}")
    values = np.asarray([_json_or_value(x) for x in df["values_json"]], dtype=np.float32)
    masks = np.asarray([_json_or_value(x) for x in df["mask_json"]], dtype=np.float32)
    contexts_raw = [_json_or_value(x) for x in df["context_json"]]
    if contexts_raw and isinstance(contexts_raw[0], dict):
        context_names = sorted(set().union(*(x.keys() for x in contexts_raw)))
        contexts = np.asarray([[float(x.get(k, 0.0)) for k in context_names] for x in contexts_raw], dtype=np.float32)
    else:
        contexts = np.asarray(contexts_raw, dtype=np.float32)
    difficulty = np.asarray([_json_or_value(x) for x in df["difficulty_vector_json"]], dtype=np.float32)
    difficulty_mask = np.asarray([_json_or_value(x) for x in df["difficulty_mask_json"]], dtype=np.float32)
    y = np.asarray(df["segment_success"], dtype=np.float32)
    if values.shape != masks.shape or values.ndim != 3 or difficulty.shape != difficulty_mask.shape:
        raise ValueError(f"values/mask must both be [N,T,F], got {values.shape}/{masks.shape}")
    difficulty = np.concatenate([difficulty, difficulty_mask], axis=1)
    groups_col = "player_id_hash" if "player_id_hash" in df else "session_id"
    groups = df[groups_col].fillna("unknown").astype(str).to_numpy()
    return values, masks, contexts, difficulty, y, groups


def build_tokenized_window_rows(snapshot_df: pd.DataFrame, sequence_length: int = 12,
                                stride: int = 3) -> pd.DataFrame:
    """Build v2 rows from an already canonicalized snapshot frame.

    The adapter must provide the 21 canonical columns, causal difficulty values
    (`difficulty_vector_json`), their availability mask
    (`difficulty_mask_json`), and a future-aligned `segment_success`. This
    function never infers a difficulty prior or a match-level outcome.
    """
    required = set(CANONICAL_FEATURES) | {
        "session_id", "elapsed_seconds", "segment_success",
        "difficulty_vector_json", "difficulty_mask_json",
    }
    missing = sorted(required - set(snapshot_df.columns))
    if missing:
        raise ValueError(f"Canonical v2 frame missing columns: {missing}")
    rows = []
    for sid, group in snapshot_df.groupby("session_id", sort=False):
        g = group.sort_values("elapsed_seconds").reset_index(drop=True)
        for end in range(sequence_length - 1, len(g), stride):
            window = g.iloc[end - sequence_length + 1:end + 1]
            values = window[CANONICAL_FEATURES].to_numpy(dtype=np.float32, na_value=np.nan)
            mask = np.isfinite(values).astype(np.float32)
            values = np.nan_to_num(values, nan=0.0)
            if float(mask.mean()) == 0.0:
                continue
            last = g.iloc[end]
            context = {k: float(last[k]) if pd.notna(last[k]) else 0.0
                       for k in ("progress_ratio", "normalized_segment_index", "current_difficulty")
                       if k in g.columns}
            rows.append({
                "session_id": sid,
                "player_id_hash": last.get("player_id_hash", "unknown"),
                "decision_time": float(last["elapsed_seconds"]),
                "segment_id": last.get("segment_id"),
                "values_json": json.dumps(values.tolist()),
                "mask_json": json.dumps(mask.tolist()),
                "context_json": json.dumps(context),
                "difficulty_vector_json": last["difficulty_vector_json"],
                "difficulty_mask_json": last["difficulty_mask_json"],
                "segment_success": int(last["segment_success"]),
            })
    return pd.DataFrame(rows)


def causal_normalize_canonical_frame(snapshot_df: pd.DataFrame,
                                    game_mean: dict[str, float] | None = None,
                                    game_scale: dict[str, float] | None = None) -> pd.DataFrame:
    """Normalize each session causally: transform current, then update state.

    Missing values remain NaN so ``build_tokenized_window_rows`` emits an
    explicit availability mask. Unity adapters must reproduce this order.
    """
    missing = sorted(set(CANONICAL_FEATURES) - set(snapshot_df.columns))
    if missing:
        raise ValueError(f"Canonical frame missing columns: {missing}")
    out = snapshot_df.copy()
    ordered = out.sort_values(["session_id", "elapsed_seconds"])
    for _, group in ordered.groupby("session_id", sort=False):
        normalizer = CausalSessionNormalizer(game_mean=game_mean or {}, game_scale=game_scale or {})
        for row_idx in group.index:
            mapping = {name: out.at[row_idx, name] for name in CANONICAL_FEATURES}
            normalized, masks = normalizer.transform(mapping)
            for name in CANONICAL_FEATURES:
                out.at[row_idx, name] = normalized[name] if masks[name] else np.nan
            normalizer.update(mapping)
    return out


def build_window_rows(snapshot_df: pd.DataFrame, sequence_length: int = 12, stride: int = 3) -> pd.DataFrame:
    """Build only windows with an explicitly available future segment label.

    Expected columns include `segment_success` already aligned to the decision
    point, or `next_segment_success`. The function refuses to infer a session
    result as every window label.
    """
    if "segment_success" not in snapshot_df.columns:
        raise ValueError("Need segment_success aligned to each decision point; do not infer it from session result")
    features = [x for x in BASE_FEATURES if x in snapshot_df.columns]
    if len(features) < 5:
        raise ValueError("Too few usable features; check logger/schema")
    rows = []
    for sid, g in snapshot_df.groupby("session_id", sort=False):
        g = g.sort_values("elapsed_seconds").reset_index(drop=True)
        for end in range(sequence_length - 1, len(g), stride):
            window = g.iloc[end - sequence_length + 1:end + 1]
            # Distinguish temporal gaps (data quality) from systematic feature
            # unavailability (handled by mask=0 in canonical models).  Only count
            # NaN among features that are at least partially present in this window.
            available_features = [f for f in features if window[f].notna().any()]
            if len(available_features) < 5:
                continue
            if available_features and window[available_features].isna().mean().mean() > 0.35:
                continue
            last = g.iloc[end]
            rows.append({
                "session_id": sid,
                "player_id_hash": last.get("player_id_hash", "unknown"),
                "decision_time": float(last["elapsed_seconds"]),
                "segment_id": last.get("segment_id"),
                "current_difficulty": float(last.get("current_difficulty", 0)),
                "candidate_difficulty": float(last.get("candidate_difficulty", last.get("current_difficulty", 0))),
                "history_json": json.dumps(window[features].fillna(0).to_numpy().tolist()),
                "context_json": json.dumps({k: float(last[k]) if pd.notna(last[k]) else 0.0 for k in ["progress_ratio", "normalized_segment_index", "current_difficulty"] if k in last}),
                "segment_success": int(last["segment_success"]),
            })
    return pd.DataFrame(rows)
