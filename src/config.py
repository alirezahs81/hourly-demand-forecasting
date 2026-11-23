from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"

for _d in (RAW_DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_FILE = RAW_DATA_DIR / "hourly_demand.csv"
PIPELINE_FILE = MODELS_DIR / "forecast_pipeline.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
META_FILE = MODELS_DIR / "meta.json"

RANDOM_STATE = 42

# Temporal hold-out sizes (hours)
TEST_HOURS = 24 * 14  # last 2 weeks
VALID_HOURS = 24 * 14  # 2 weeks before test (for model selection)

TARGET_COL = "demand"
TIME_COL = "timestamp"

# Feature engineering
LAGS = [1, 2, 3, 24, 48, 168]  # hours; 168 = 1 week
ROLL_WINDOWS = [24, 168]
