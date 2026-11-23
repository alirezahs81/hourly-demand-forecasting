import pytest
from fastapi.testclient import TestClient

from src.config import PIPELINE_FILE
from src.train import train


@pytest.fixture(scope="module")
def client():
    if not PIPELINE_FILE.exists():
        train(persist=True)
    from src.api import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_ready"] is True


def test_forecast_get(client):
    r = client.get("/forecast", params={"horizon": 6})
    assert r.status_code == 200
    body = r.json()
    assert body["horizon"] == 6
    assert len(body["forecasts"]) == 6


def test_forecast_post(client):
    r = client.post("/forecast", json={"horizon": 3})
    assert r.status_code == 200
    assert len(r.json()["forecasts"]) == 3
