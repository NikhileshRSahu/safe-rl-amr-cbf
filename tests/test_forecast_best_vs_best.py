import numpy as np

from benchmark.best_vs_best_protocol import local_human_navigation_catalog, paired_world_fingerprint
from benchmark.best_vs_best_runner import _interaction_metrics, run_three_way_development_screen
from benchmark.human_forecaster import GRUHumanForecaster
from benchmark.spatiotemporal_policy import STASACActor
from benchmark.stasac_warehouse import EGO_DIM, FORECAST_ENTITY_DIM
from benchmark.warehouse_interaction_features import FEATURE_DIM


def test_interaction_metrics_are_finite_and_detect_reversals():
    actions = [
        np.array([[0.0, 0.5]], np.float32),
        np.array([[0.0, -0.5]], np.float32),
        np.array([[-1.0, 0.5]], np.float32),
    ]
    m = _interaction_metrics(actions)
    assert m["commitment_reversals"] >= 2
    assert 0.0 <= m["stop_yield_fraction"] <= 1.0
    assert np.isfinite(m["angular_oscillation"])


def test_three_way_evaluator_uses_same_scenario_seed_and_returns_all_controllers():
    spec = next(s for s in local_human_navigation_catalog() if s.name == "human_crossing")
    seed = 10300
    before = paired_world_fingerprint(spec, seed)
    base_actor = STASACActor(EGO_DIM, hidden_dim=16, entity_dim=FEATURE_DIM)
    forecast_actor = STASACActor(EGO_DIM, hidden_dim=16, entity_dim=FORECAST_ENTITY_DIM)
    forecaster = GRUHumanForecaster(hidden_dim=16, steps=4)
    result = run_three_way_development_screen(base_actor, forecast_actor, forecaster, [spec], [seed], max_steps=2)
    assert set(result) == {"AP-ORCA+CBF", "ST-SAC+CBF", "Forecast-ST-SAC+CBF"}
    for payload in result.values():
        row = payload["episodes"][0]
        assert row["seed"] == seed
        assert row["scenario"] == spec.name
        assert "cbf_intervention_rate" in row
        assert "stop_yield_fraction" in row
    assert paired_world_fingerprint(spec, seed) == before
