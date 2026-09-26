import numpy as np
import pytest

from benchmark.human_forecast_dataset import (
    ForecastSample,
    assert_training_seed_allowed,
    make_forecast_sample,
)


def test_forecast_sample_separates_past_from_future():
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    xy = np.stack([times, np.zeros_like(times)], axis=-1)
    sample = make_forecast_sample(times, xy, t0_index=2, history_len=3, horizon_steps=2, scenario="cross", seed=10100)
    assert isinstance(sample, ForecastSample)
    assert sample.history.shape == (3, 5)
    assert sample.future_xy.shape == (2, 2)
    assert np.all(sample.history[:, 4][sample.history_mask] <= 1.0 + 1e-9)
    assert np.allclose(sample.future_xy, [[1.5, 0.0], [2.0, 0.0]])


def test_training_seed_guard_rejects_non_training_partitions():
    assert_training_seed_allowed(10100)
    assert_training_seed_allowed(10115)
    for seed in (10116, 10300, 11100, 13100):
        with pytest.raises(ValueError):
            assert_training_seed_allowed(seed)
