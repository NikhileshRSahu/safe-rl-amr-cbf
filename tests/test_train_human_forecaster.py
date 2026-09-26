import numpy as np
import torch

from benchmark.evaluate_human_forecaster import displacement_metrics, evaluate_forecaster
from benchmark.human_forecast_dataset import make_forecast_sample
from benchmark.train_human_forecaster import load_forecaster_checkpoint, save_forecaster_checkpoint, train_forecaster


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
    assert set(meta["training_seeds"]) == {10100, 10101}
    path = tmp_path / "forecast.pt"
    digest = save_forecaster_checkpoint(path, model, meta)
    assert len(digest) == 64
    loaded, loaded_meta = load_forecaster_checkpoint(path)
    assert loaded_meta["architecture"] == "gru_human_forecaster_v1"
    report = evaluate_forecaster(loaded, _samples())
    assert np.isfinite(report["ade"])
    assert np.isfinite(report["fde"])


def test_displacement_metrics_exact_case():
    target = np.array([[[1.0, 0.0], [2.0, 0.0]]], np.float32)
    pred = np.array([[[1.0, 0.0], [1.0, 0.0]]], np.float32)
    metrics = displacement_metrics(pred, target, np.array([[True, True]]))
    assert metrics["ade"] == 0.5
    assert metrics["fde"] == 1.0
