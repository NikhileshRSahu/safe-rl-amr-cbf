import numpy as np

from benchmark.constant_velocity_forecaster import ConstantVelocityForecaster
from benchmark.human_history import HumanTrackHistory


def test_constant_velocity_predicts_straight_line():
    h = HumanTrackHistory(history_len=4)
    h.update("human-0", 0.0, [0.0, 0.0], [1.0, 0.0])
    h.update("human-0", 0.5, [0.5, 0.0], [1.0, 0.0])
    seq = h.sequence("human-0", now=0.5)
    pred = ConstantVelocityForecaster(horizon_seconds=2.0, steps=4).predict([seq])
    assert pred.mean_xy.shape == (1, 4, 2)
    assert pred.sigma_xy.shape == (1, 4, 2)
    assert pred.mask.shape == (1, 4)
    assert np.allclose(pred.mean_xy[0, :, 0], [1.0, 1.5, 2.0, 2.5], atol=1e-6)
    assert np.allclose(pred.mean_xy[0, :, 1], 0.0)
    assert np.isfinite(pred.sigma_xy).all()


def test_constant_velocity_handles_empty_and_stationary_history():
    h = HumanTrackHistory(history_len=3)
    empty = h.sequence("missing", now=0.0)
    pred = ConstantVelocityForecaster().predict([empty])
    assert not pred.mask.any()
    h.update("human-1", 0.0, [2.0, -1.0], [0.0, 0.0])
    stationary = ConstantVelocityForecaster(steps=3).predict([h.sequence("human-1", now=0.0)])
    assert np.allclose(stationary.mean_xy[0], [[2.0, -1.0]] * 3)
