import pandas as pd

from src.data import build_dataset
from src.features import build_feature_frame, feature_columns
from src.split import temporal_split
from src.config import TIME_COL, TARGET_COL


def test_dataset_monotonic_time():
    df = build_dataset()
    assert df[TIME_COL].is_monotonic_increasing
    assert len(df) > 5000


def test_lags_do_not_look_ahead():
    df = build_dataset()
    feat = build_feature_frame(df)
    # lag_1 at t equals demand at t-1
    assert (feat[TARGET_COL].shift(1).iloc[10:50].values == feat["lag_1"].iloc[10:50].values).all()


def test_temporal_split_order():
    df = build_feature_frame(build_dataset())
    train, valid, test = temporal_split(df)
    assert train[TIME_COL].max() < valid[TIME_COL].min()
    assert valid[TIME_COL].max() < test[TIME_COL].min()
    assert len(test) == 24 * 14


def test_feature_columns_exclude_target():
    df = build_feature_frame(build_dataset())
    cols = feature_columns(df)
    assert TARGET_COL not in cols
    assert TIME_COL not in cols
    assert "lag_24" in cols
