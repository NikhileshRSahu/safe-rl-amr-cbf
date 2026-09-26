import numpy as np
import torch

from benchmark.evaluate_human_forecaster import displacement_metrics, evaluate_forecaster
from benchmark.human_forecast_dataset import make_forecast_sample
from benchmark.human_forecaster import TorchForecastOutput
from benchmark.train_human_forecaster import (
    constant_velocity_difficulty_weights,
    forecast_training_loss,
    load_forecaster_checkpoint,
    save_forecaster_checkpoint,
    train_forecaster,
)


def _samples():
    rows = []
    for seed in (10100, 10101):
        t = np.arange(0.0, 3.0, 0.25, dtype=np.float32)
        xy = np.stack([0.4 * t, 0.1 * t], axis=-1)
        for t0 in (4, 5, 6):
            rows.append(make_forecast_sample(t, xy, t0_index=t0, history_len=5, horizon_steps=4, scenario="linear", seed=seed))
    return rows


def test_train_forecaster_is_finite_and_roundtrips(tmp_path):
    model, meta = train_forecaster(_samples(), epochs=2, hidden_dim=16)
    assert np.isfinite(meta["final_loss"])
    assert meta["mean_loss_weight"] > 0.0
    assert meta["hard_example_weight"] >= 0.0
    assert set(meta["training_seeds"]) == {10100, 10101}
    path = tmp_path / "forecast.pt"
    digest = save_forecaster_checkpoint(path, model, meta)
    assert len(digest) == 64
    loaded, loaded_meta = load_forecaster_checkpoint(path)
    assert loaded_meta["architecture"] == "gru_human_forecaster_v1"
    report = evaluate_forecaster(loaded, _samples())
    assert np.isfinite(report["ade"])
    assert np.isfinite(report["fde"])


def test_forecast_training_loss_adds_horizon_weighted_mean_accuracy_term():
    target = torch.zeros(1, 3, 2)
    mask = torch.ones(1, 3, dtype=torch.bool)
    sigma = torch.ones(1, 3, 2)
    early_error = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    late_error = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]])
    early = TorchForecastOutput(early_error, sigma, mask)
    late = TorchForecastOutput(late_error, sigma, mask)
    early_total, early_parts = forecast_training_loss(early, target, mask, mean_loss_weight=2.0)
    late_total, late_parts = forecast_training_loss(late, target, mask, mean_loss_weight=2.0)
    assert early_parts["mean_loss"] > 0.0
    assert late_parts["mean_loss"] > early_parts["mean_loss"]
    assert late_total > early_total


def test_cv_difficulty_weights_prioritize_stop_or_reversal_without_eval_labels():
    # Two training samples with identical current state. Sample 0 keeps moving,
    # sample 1 stops. The nonlinear stop should receive more training weight.
    history = torch.zeros(2, 4, 5)
    history[:, -1, :2] = torch.tensor([1.0, 0.0])
    history[:, -1, 2:4] = torch.tensor([1.0, 0.0])
    history_mask = torch.ones(2, 4, dtype=torch.bool)
    future = torch.tensor([
        [[1.3, 0.0], [1.6, 0.0], [1.9, 0.0]],
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    ])
    future_mask = torch.ones(2, 3, dtype=torch.bool)
    weights = constant_velocity_difficulty_weights(
        history, history_mask, future, future_mask, forecast_dt=0.3, strength=2.0
    )
    assert weights.shape == (2,)
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5)
    assert weights[1] > weights[0]


def test_displacement_metrics_exact_case():
    target = np.array([[[1.0, 0.0], [2.0, 0.0]]], np.float32)
    pred = np.array([[[1.0, 0.0], [1.0, 0.0]]], np.float32)
    metrics = displacement_metrics(pred, target, np.array([[True, True]]))
    assert metrics["ade"] == 0.5
    assert metrics["fde"] == 1.0