"""Simple seasonal naive predictors for honest baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_COL


class SeasonalNaive:
    """Predict y[t] = y[t - season] (default 24h or 168h)."""

    def __init__(self, season: int = 24):
        self.season = season
        self.history_: np.ndarray | None = None

    def fit(self, y: pd.Series | np.ndarray):
        self.history_ = np.asarray(y, dtype=float)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.history_ is None:
            raise RuntimeError("Call fit first")
        hist = list(self.history_)
        preds = []
        for _ in range(horizon):
            preds.append(hist[-self.season])
            hist.append(hist[-self.season])
        return np.asarray(preds, dtype=float)

    def predict_aligned(self, frame: pd.DataFrame) -> np.ndarray:
        """In-sample style: use lag column if present, else shift."""
        col = f"lag_{self.season}"
        if col in frame.columns:
            return frame[col].to_numpy(dtype=float)
        return frame[TARGET_COL].shift(self.season).to_numpy(dtype=float)
