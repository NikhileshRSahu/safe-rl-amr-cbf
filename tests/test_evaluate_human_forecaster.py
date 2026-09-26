import numpy as np

from benchmark.evaluate_human_forecaster import evaluate_constant_velocity
from benchmark.human_forecast_dataset import ForecastSample


def _sample(scenario, x0, vx):
    history = np.zeros((4, 5), dtype=np.float32)
    history[:, 0] = np.array([x0 - 0.3 * vx, x0 - 0.2 * vx, x0 - 0.1 * vx, x0], dtype=np.float32)
    history[:, 2] = vx
    history[:, 4] = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    future = np.array([[x0 + 0.3 * vx, 0.0], [x0 + 0.6 * vx, 0.0]], dtype=np.float32)
    return ForecastSample(history, np.ones(4, bool), future, np.ones(2, bool), scenario, 10116, 0.3)


def test_constant_velocity_evaluation_reports_per_scenario_metrics():
    samples = [_sample("human_crossing", 1.0, 1.0), _sample("human_reversal", 2.0, 0.5)]
    result = evaluate_constant_velocity(samples)
    assert set(result["per_scenario"]) == {"human_crossing", "human_reversal"}
    assert result["per_scenario"]["human_crossing"]["ade"] == 0.0
    assert result["per_scenario"]["human_reversal"]["fde"] == 0.0
