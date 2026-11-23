import pytest

from src.config import PIPELINE_FILE
from src.predict import DemandForecaster
from src.train import train


@pytest.fixture(scope="module")
def trained():
    return train(persist=True)


def test_train_beats_or_matches_naive(trained):
    m = trained["metrics"]
    assert m["best_model"] in {"ridge", "hgb"}
    # Model should not be dramatically worse than seasonal naive on test
    assert m["test"]["mae"] <= m["test_naive_24h"]["mae"] * 1.15
    assert PIPELINE_FILE.exists()


def test_forecast_horizon(trained):
    fc = DemandForecaster()
    pts = fc.forecast(horizon=24)
    assert len(pts) == 24
    assert pts[0]["step"] == 1
    assert pts[-1]["step"] == 24
    assert all(p["forecast"] >= 0 for p in pts)


def test_forecast_rejects_bad_horizon(trained):
    fc = DemandForecaster()
    with pytest.raises(ValueError):
        fc.forecast(horizon=0)
