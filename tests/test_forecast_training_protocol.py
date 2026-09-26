from benchmark.forecast_training_protocol import (
    FORECAST_GATE_SCENARIOS,
    FORECAST_REQUIRED_WIN_SCENARIOS,
    forecaster_eval_seeds,
    forecaster_training_seeds,
    learned_forecaster_beats_cv,
)


def test_forecast_train_eval_seed_splits_are_disjoint_and_do_not_touch_validation_holdout():
    train = forecaster_training_seeds()
    evaluate = forecaster_eval_seeds()
    assert train == tuple(range(10100, 10116))
    assert evaluate == tuple(range(10116, 10120))
    assert set(train).isdisjoint(evaluate)
    assert all(seed < 11100 for seed in train + evaluate)


def test_forecast_gate_requires_key_human_behavior_families():
    assert set(FORECAST_REQUIRED_WIN_SCENARIOS) == {
        "human_crossing",
        "human_hesitation",
        "human_reversal",
        "dense_human_flow",
    }
    assert set(FORECAST_REQUIRED_WIN_SCENARIOS) <= set(FORECAST_GATE_SCENARIOS)
    assert "mixed_local_traffic" in FORECAST_GATE_SCENARIOS


def test_forecast_gate_rejects_model_that_only_wins_in_aggregate():
    learned = {
        "ade": 0.20,
        "fde": 0.30,
        "per_scenario": {
            "human_crossing": {"ade": 0.10, "fde": 0.10},
            "human_hesitation": {"ade": 0.25, "fde": 0.35},
            "human_reversal": {"ade": 0.10, "fde": 0.10},
            "dense_human_flow": {"ade": 0.10, "fde": 0.10},
        },
    }
    cv = {
        "ade": 0.30,
        "fde": 0.40,
        "per_scenario": {
            "human_crossing": {"ade": 0.20, "fde": 0.20},
            "human_hesitation": {"ade": 0.20, "fde": 0.30},
            "human_reversal": {"ade": 0.20, "fde": 0.20},
            "dense_human_flow": {"ade": 0.20, "fde": 0.20},
        },
    }
    ok, report = learned_forecaster_beats_cv(learned, cv)
    assert not ok
    assert "human_hesitation" in report["failed_scenarios"]


def test_forecast_gate_accepts_consistent_ade_and_fde_improvement():
    scenarios = {
        name: {"ade": 0.15, "fde": 0.20}
        for name in FORECAST_REQUIRED_WIN_SCENARIOS
    }
    cv_scenarios = {
        name: {"ade": 0.20, "fde": 0.25}
        for name in FORECAST_REQUIRED_WIN_SCENARIOS
    }
    learned = {"ade": 0.15, "fde": 0.20, "per_scenario": scenarios}
    cv = {"ade": 0.20, "fde": 0.25, "per_scenario": cv_scenarios}
    ok, report = learned_forecaster_beats_cv(learned, cv)
    assert ok
    assert report["failed_scenarios"] == []
