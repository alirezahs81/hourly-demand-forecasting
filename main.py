#!/usr/bin/env python3
"""Train the forecasting pipeline and print a short horizon demo."""

from __future__ import annotations

from src.predict import DemandForecaster
from src.train import train


def main() -> None:
    print("=== Hourly Demand Forecasting ===\n")
    result = train(persist=True)
    m = result["metrics"]
    print("\nBest model:", m["best_model"])
    print("Test metrics:", m["test"])
    print("Naive-24h test:", m["test_naive_24h"])

    fc = DemandForecaster()
    pts = fc.forecast(horizon=12)
    print("\nNext 12h forecast (sample):")
    for p in pts:
        print(f"  {p['timestamp']} -> {p['forecast']:.2f}")

    print("\nAPI: uvicorn src.api:app --reload --port 8000")


if __name__ == "__main__":
    main()
