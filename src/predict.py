"""Load trained model and produce multi-step hourly forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import PIPELINE_FILE, TARGET_COL, TIME_COL
from src.features import build_feature_frame


class DemandForecaster:
    def __init__(self, model_path: Path | None = None):
        path = model_path or PIPELINE_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run: python -m src.train"
            )
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.features = artifact["features"]
        history_path = path.parent / "history.csv"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history file: {history_path}")
        self.history = pd.read_csv(history_path, parse_dates=[TIME_COL])

    def forecast(self, horizon: int = 24) -> list[dict[str, Any]]:
        if horizon < 1 or horizon > 168 * 2:
            raise ValueError("horizon must be between 1 and 336 hours")

        hist = self.history.copy()
        results: list[dict[str, Any]] = []

        for step in range(horizon):
            featured = build_feature_frame(hist)
            row = featured.iloc[[-1]].copy()
            # For the last row, lag features are based on history; after append they update
            # We need a *next* timestamp row: append placeholder then take last complete features
            next_ts = hist[TIME_COL].iloc[-1] + pd.Timedelta(hours=1)
            placeholder = pd.DataFrame(
                {TIME_COL: [next_ts], TARGET_COL: [np.nan]}
            )
            extended = pd.concat([hist, placeholder], ignore_index=True)
            feat_ext = build_feature_frame(extended)
            x_row = feat_ext.iloc[[-1]][self.features]
            if x_row.isna().any(axis=None):
                # Should not happen if history is long enough
                raise RuntimeError("Insufficient history for lag features")
            yhat = float(self.model.predict(x_row)[0])
            yhat = max(yhat, 0.0)
            results.append(
                {
                    "timestamp": next_ts.isoformat(),
                    "forecast": round(yhat, 3),
                    "step": step + 1,
                }
            )
            hist = pd.concat(
                [
                    hist,
                    pd.DataFrame({TIME_COL: [next_ts], TARGET_COL: [yhat]}),
                ],
                ignore_index=True,
            )

        return results
