"""Generate / load hourly demand series with daily & weekly seasonality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATASET_FILE, RANDOM_STATE, TARGET_COL, TIME_COL


def _synthesize(n_days: int = 400, start: str = "2023-01-01") -> pd.DataFrame:
    """Deterministic synthetic load curve (MW-scale) for CI and demos."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = pd.date_range(start=start, periods=n_days * 24, freq="h")
    hours = np.arange(len(idx))

    # Components
    trend = 100 + 0.002 * hours
    daily = 25 * np.sin(2 * np.pi * (idx.hour - 8) / 24)
    weekly = 12 * np.sin(2 * np.pi * (idx.dayofweek - 1) / 7)
    # Peak evenings on weekdays
    evening_peak = np.where(
        (idx.dayofweek < 5) & (idx.hour >= 17) & (idx.hour <= 21),
        18.0,
        0.0,
    )
    # Mild summer bump (months 6-8)
    summer = np.where(idx.month.isin([6, 7, 8]), 10.0, 0.0)
    noise = rng.normal(0, 4.0, size=len(idx))

    demand = trend + daily + weekly + evening_peak + summer + noise
    demand = np.clip(demand, 40.0, None)

    return pd.DataFrame({TIME_COL: idx, TARGET_COL: demand.astype(float)})


def build_dataset(force: bool = False, n_days: int = 400) -> pd.DataFrame:
    if DATASET_FILE.exists() and not force:
        df = pd.read_csv(DATASET_FILE, parse_dates=[TIME_COL])
    else:
        df = _synthesize(n_days=n_days)
        DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_FILE, index=False)

    df = df.sort_values(TIME_COL).reset_index(drop=True)
    if df[TIME_COL].duplicated().any():
        raise ValueError("Duplicate timestamps in demand series")
    return df


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATASET_FILE
    if not path.exists():
        return build_dataset()
    return pd.read_csv(path, parse_dates=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
