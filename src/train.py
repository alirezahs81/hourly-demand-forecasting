"""Train demand models with strict temporal validation and held-out test."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.baselines import SeasonalNaive
from src.config import (
    META_FILE,
    METRICS_FILE,
    PIPELINE_FILE,
    RANDOM_STATE,
    TARGET_COL,
    TIME_COL,
)
from src.data import build_dataset
from src.features import build_feature_frame, feature_columns
from src.metrics import regression_metrics
from src.split import temporal_split


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    cols = feature_columns(frame)
    clean = frame.dropna(subset=cols + [TARGET_COL]).copy()
    X = clean[cols]
    y = clean[TARGET_COL].to_numpy(dtype=float)
    return X, y, cols


def train(*, persist: bool = True) -> dict[str, Any]:
    raw = build_dataset()
    featured = build_feature_frame(raw)
    train_df, valid_df, test_df = temporal_split(featured)

    X_train, y_train, cols = _xy(train_df)
    X_valid, y_valid, _ = _xy(valid_df)
    X_test, y_test, _ = _xy(test_df)

    # --- Baselines on validation (aligned lags) ---
    baseline_scores: dict[str, dict[str, float]] = {}
    for season, name in [(24, "naive_24h"), (168, "naive_168h")]:
        pred = valid_df.loc[X_valid.index, f"lag_{season}"].to_numpy(dtype=float)
        # X_valid index may not match if dropna changed rows — re-align properly
        valid_clean = valid_df.dropna(subset=cols + [TARGET_COL])
        pred = valid_clean[f"lag_{season}"].to_numpy(dtype=float)
        y_v = valid_clean[TARGET_COL].to_numpy(dtype=float)
        baseline_scores[name] = regression_metrics(y_v, pred)

    candidates: dict[str, Any] = {
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "hgb": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=200,
            random_state=RANDOM_STATE,
        ),
    }

    valid_table: dict[str, dict[str, float]] = {}
    best_name = None
    best_model = None
    best_mae = float("inf")

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred_v = model.predict(X_valid)
        scores = regression_metrics(y_valid, pred_v)
        valid_table[name] = scores
        print(f"{name} valid: {scores}")
        if scores["mae"] < best_mae:
            best_mae = scores["mae"]
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None

    # Refit on train+valid for final test evaluation
    train_valid = pd.concat([train_df, valid_df], axis=0)
    X_tv, y_tv, cols = _xy(train_valid)
    final_model = candidates[best_name].__class__(**best_model.get_params())
    final_model.fit(X_tv, y_tv)

    test_clean = test_df.dropna(subset=cols + [TARGET_COL])
    X_test = test_clean[cols]
    y_test = test_clean[TARGET_COL].to_numpy(dtype=float)
    pred_test = final_model.predict(X_test)
    test_scores = regression_metrics(y_test, pred_test)

    # Baseline on test for comparison
    test_baseline = regression_metrics(
        y_test, test_clean["lag_24"].to_numpy(dtype=float)
    )

    metrics = {
        "best_model": best_name,
        "valid_scores": valid_table,
        "baseline_valid": baseline_scores,
        "test": test_scores,
        "test_naive_24h": test_baseline,
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_test": int(len(X_test)),
        "feature_count": len(cols),
    }
    print("\nTest:", test_scores)
    print("Test naive-24h:", test_baseline)

    artifact = {
        "model": final_model,
        "features": cols,
        "best_model_name": best_name,
    }

    if persist:
        joblib.dump(artifact, PIPELINE_FILE)
        METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        # Keep tail history for recursive multi-step API forecasts
        history = raw[[TIME_COL, TARGET_COL]].tail(168 * 3).copy()
        meta = {
            "features": cols,
            "best_model": best_name,
            "last_timestamp": str(raw[TIME_COL].iloc[-1]),
        }
        META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        history_path = PIPELINE_FILE.parent / "history.csv"
        history.to_csv(history_path, index=False)
        print(f"Saved model -> {PIPELINE_FILE}")

    return {"artifact": artifact, "metrics": metrics}


if __name__ == "__main__":
    train()
