"""Feature engineering for hourly demand — lag-aware, no future leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LAGS, ROLL_WINDOWS, TARGET_COL, TIME_COL


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[TIME_COL])
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    # Cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    return out


def add_lag_features(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    out = df.copy()
    for lag in LAGS:
        out[f"lag_{lag}"] = out[target].shift(lag)
    for w in ROLL_WINDOWS:
        # Rolling stats use past window only (shifted by 1)
        rolled = out[target].shift(1).rolling(window=w, min_periods=max(3, w // 4))
        out[f"roll_mean_{w}"] = rolled.mean()
        out[f"roll_std_{w}"] = rolled.std()
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix; rows with NaN lags are dropped by caller."""
    out = add_calendar_features(df)
    out = add_lag_features(out)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {TIME_COL, TARGET_COL}
    return [c for c in df.columns if c not in skip]
