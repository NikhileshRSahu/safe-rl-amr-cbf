import numpy as np
import torch

from benchmark.best_vs_best_protocol import local_human_navigation_catalog
from benchmark.human_forecaster import GRUHumanForecaster
from benchmark.stasac_warehouse import (
    BASE_OBSERVATION_VERSION,
    FORECAST_ENTITY_DIM,
    FORECAST_OBSERVATION_VERSION,
    WarehouseObservationBuilder,
    make_training_world,
)
from benchmark.warehouse_interaction_features import FEATURE_DIM


def test_forecast_builder_extends_entities_without_breaking_base_contract():
    spec = next(s for s in local_human_navigation_catalog() if s.name == "human_crossing")
    world = make_training_world(spec, 10100)
    base = WarehouseObservationBuilder(world, perception_range=100.0, use_forecast=False)
    ego0, b0 = base.observe(world, 0)
    assert b0.features.shape[1] == FEATURE_DIM
    assert base.observation_version == BASE_OBSERVATION_VERSION

    model = GRUHumanForecaster(hidden_dim=16, steps=4)
    model.eval()
    forecast = WarehouseObservationBuilder(world, perception_range=100.0, use_forecast=True, forecaster=model, forecast_dt=0.5)
    ego1, b1 = forecast.observe(world, 0)
    assert np.allclose(ego0, ego1)
    assert b1.features.shape[1] == FORECAST_ENTITY_DIM
    assert np.isfinite(b1.features).all()
    assert forecast.observation_version == FORECAST_OBSERVATION_VERSION
    assert tuple(b0.entity_ids) == tuple(b1.entity_ids)
    assert np.allclose(b0.features, b1.features[:, :FEATURE_DIM])


def test_forecast_builder_requires_explicit_forecaster():
    spec = local_human_navigation_catalog()[0]
    world = make_training_world(spec, 10100)
    import pytest
    with pytest.raises(ValueError):
        WarehouseObservationBuilder(world, use_forecast=True, forecaster=None)
