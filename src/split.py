"""Chronological splits — never shuffle time series."""

from __future__ import annotations

import pandas as pd

from src.config import TEST_HOURS, VALID_HOURS


def temporal_split(
    df: pd.DataFrame,
    *,
    test_hours: int = TEST_HOURS,
    valid_hours: int = VALID_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) <= test_hours + valid_hours + 200:
        raise ValueError("Series too short for configured temporal hold-outs")

    test = df.iloc[-test_hours:].copy()
    valid = df.iloc[-(test_hours + valid_hours) : -test_hours].copy()
    train = df.iloc[: -(test_hours + valid_hours)].copy()
    return train, valid, test
