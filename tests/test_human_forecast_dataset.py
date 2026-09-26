import numpy as np
import pytest

from benchmark.human_forecast_dataset import ForecastSample, assert_training_seed_allowed, make_forecast_sample


def test_forecast_sample_separates_past_from_future():
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    xy = np.stack([times, np.zeros_like(times)], axis=-1)
    sample = make_forecast_sample(times, xy, t0_index=2, history_len=3, horizon_steps=2, scenario="cross", seed=10100)
    assert isinstance(sample, ForecastSample)
    assert sample.history.shape == (3, 5)
    assert sample.future_xy.shape == (2, 2)
    assert np.all(sample.history[:, 4][sample.history_mask] <= 1.0 + 1e-9)
    assert np.allclose(sample.future_xy, [[1.5, 0.0], [2.0, 0.0]])
    assert sample.forecast_dt == pytest.approx(0.5)


def test_future_stride_creates_true_long_horizon_without_future_input_leakage():
    times = np.arange(0.0, 3.1, 0.1, dtype=np.float32)
    xy = np.stack([times, np.zeros_like(times)], axis=-1)
    sample = make_forecast_sample(times, xy, t0_index=8, history_len=8, horizon_steps=6, future_stride=3, scenario="hesitation", seed=10100)
    assert sample.forecast_dt == pytest.approx(0.3, abs=1e-5)
    assert np.allclose(sample.future_xy[:, 0], [1.1, 1.4, 1.7, 2.0, 2.3, 2.6], atol=1e-5)
    assert float(sample.future_xy[-1, 0] - times[8]) == pytest.approx(1.8, abs=1e-5)
    assert np.max(sample.history[:, 4][sample.history_mask]) <= times[8] + 1e-7


def test_training_seed_guard_rejects_non_training_partitions():
    assert_training_seed_allowed(10100)
    assert_training_seed_allowed(10115)
    for seed in (10116, 10300, 11100, 13100):
        with pytest.raises(ValueError):
            assert_training_seed_allowed(seed)
