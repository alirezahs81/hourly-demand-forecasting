"""FastAPI service for hourly demand forecasts."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.predict import DemandForecaster

app = FastAPI(
    title="Hourly Demand Forecasting API",
    description="Multi-step hourly demand forecasts from a leakage-free ML pipeline.",
    version="1.0.0",
)

_forecaster: DemandForecaster | None = None


def get_forecaster() -> DemandForecaster:
    global _forecaster
    if _forecaster is None:
        try:
            _forecaster = DemandForecaster()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
    return _forecaster


class ForecastRequest(BaseModel):
    horizon: int = Field(24, ge=1, le=336)


class ForecastResponse(BaseModel):
    horizon: int
    forecasts: list[dict]


@app.get("/health")
def health():
    ready = True
    try:
        get_forecaster()
    except HTTPException:
        ready = False
    return {"status": "ok" if ready else "model_missing", "model_ready": ready}


@app.get("/forecast", response_model=ForecastResponse)
def forecast_get(horizon: int = Query(24, ge=1, le=336)):
    fc = get_forecaster()
    try:
        points = fc.forecast(horizon=horizon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"horizon": horizon, "forecasts": points}


@app.post("/forecast", response_model=ForecastResponse)
def forecast_post(body: ForecastRequest):
    return forecast_get(horizon=body.horizon)
