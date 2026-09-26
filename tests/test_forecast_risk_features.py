import numpy as np

from benchmark.forecast_risk_features import build_forecast_risk_batch
from benchmark.human_forecast_dataset import ForecastOutput


def _out(points, sigma=0.1):
    pts = np.asarray(points, dtype=np.float32)[None, ...]
    sig = np.full_like(pts, sigma)
    mask = np.ones(pts.shape[:2], dtype=bool)
    return ForecastOutput(pts, sig, mask)


def test_approaching_and_uncertain_predictions_raise_risk():
    robot = np.array([0.0, 0.0], dtype=np.float32)
    vel = np.array([1.0, 0.0], dtype=np.float32)
    route = np.array([1.0, 0.0], dtype=np.float32)
    approaching = _out([[2.0, 0.1], [1.5, 0.1], [1.0, 0.1], [0.5, 0.1]], sigma=0.1)
    diverging = _out([[2.0, 1.0], [2.0, 1.5], [2.0, 2.0], [2.0, 2.5]], sigma=0.1)
    a = build_forecast_risk_batch(robot, vel, route, approaching, forecast_dt=0.5, observation_ages=[0.0])
    d = build_forecast_risk_batch(robot, vel, route, diverging, forecast_dt=0.5, observation_ages=[0.0])
    assert a.features.shape == (1, 6)
    assert np.all(np.isfinite(a.features))
    assert np.all((a.features >= 0.0) & (a.features <= 1.0))
    assert a.features[0, 2] > d.features[0, 2]
    more_uncertain = _out([[2.0, 0.1], [1.5, 0.1], [1.0, 0.1], [0.5, 0.1]], sigma=0.8)
    u = build_forecast_risk_batch(robot, vel, route, more_uncertain, forecast_dt=0.5, observation_ages=[0.0])
    assert u.features[0, 2] >= a.features[0, 2]


def test_empty_forecast_batch_is_valid():
    out = ForecastOutput(np.zeros((0, 4, 2), np.float32), np.zeros((0, 4, 2), np.float32), np.zeros((0, 4), bool))
    batch = build_forecast_risk_batch([0, 0], [0, 0], [1, 0], out, forecast_dt=0.5, observation_ages=[])
    assert batch.features.shape == (0, 6)
    assert batch.mask.shape == (0,)
