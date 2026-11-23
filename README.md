# Hourly Demand Forecasting

[![CI](https://github.com/alirezahs81/hourly-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/hourly-demand-forecasting/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Leakage-free hourly demand forecasting** with chronological train / validation / test splits, lag & calendar features, seasonal-naive baselines, and a Gradient Boosting / Ridge model selection loop.

FastAPI multi-step forecasts · Docker · pytest · GitHub Actions.

---

## Why this project

Most tabular “forecasting” demos shuffle rows and leak the future into the past. This repo is built the opposite way:

1. **Time-ordered splits only** (last 2 weeks test, 2 weeks valid, rest train).
2. **Lags & rolling stats** computed with `shift` — no future values.
3. **Seasonal naive (24h / 168h)** reported so ML gains are honest.
4. **Recursive multi-step API** for horizons up to 336 hours.

The bundled series is a **deterministic synthetic hourly load** (daily + weekly seasonality, weekday peaks, summer bump) so CI stays offline-reliable. Swap `data/raw/hourly_demand.csv` for real meter / traffic / sales data without changing the pipeline.

---

## Features

| Component | Detail |
|-----------|--------|
| Calendar | hour, dow, month, weekend, sin/cos encodings |
| Lags | 1, 2, 3, 24, 48, 168 h |
| Rolling | mean/std over 24h & 168h (shifted) |
| Models | Ridge, HistGradientBoosting; pick by valid MAE |
| Baselines | Seasonal naive 24h & 168h |
| Metrics | MAE, RMSE, MAPE |

---

## Layout

```text
hourly-demand-forecasting/
├── src/
│   ├── data.py          # synthetic series builder
│   ├── features.py      # lags, rolling, calendar
│   ├── split.py         # temporal hold-outs
│   ├── baselines.py     # seasonal naive
│   ├── train.py         # fit + select + persist
│   ├── predict.py       # recursive multi-step forecast
│   ├── metrics.py
│   └── api.py           # FastAPI
├── tests/
├── main.py
└── Dockerfile
```

---

## Quick start

```bash
git clone https://github.com/alirezahs81/hourly-demand-forecasting.git
cd hourly-demand-forecasting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py
pytest tests/ -v
uvicorn src.api:app --reload --port 8000
```

```bash
curl -s 'http://localhost:8000/forecast?horizon=24' | head
```

### Docker

```bash
docker build -t demand-forecast-api .
docker run --rm -p 8000:8000 demand-forecast-api
```

---

## Methodology

```text
raw series
  → lag / rolling / calendar features
  → chronological train | valid | test
  → fit candidates on train, rank by valid MAE
  → refit winner on train+valid
  → report test MAE/RMSE/MAPE vs naive-24h
  → persist model + trailing history for API recursion
```

---

## License

[MIT](LICENSE) © 2026 Alireza Hasheminasab
