import torch

from benchmark.spatiotemporal_policy import STASACActor
from benchmark.stasac_warehouse import EGO_DIM, FORECAST_ENTITY_DIM
from benchmark.warehouse_interaction_features import FEATURE_DIM


def _run(entity_dim, n):
    torch.manual_seed(11)
    actor = STASACActor(EGO_DIM, hidden_dim=32, entity_dim=entity_dim)
    ego = torch.randn(2, EGO_DIM)
    entities = torch.randn(2, n, entity_dim)
    mask = torch.ones(2, n, dtype=torch.bool)
    action, _, hidden, weights = actor.sample(ego, entities, mask, deterministic=True)
    assert action.shape == (2, 2)
    assert hidden.shape == (2, 32)
    assert torch.isfinite(action).all()
    assert torch.max(torch.abs(action)) <= 1.0 + 1e-6
    return actor, action


def test_base_and_forecast_entity_dimensions_share_same_actor_contract():
    _run(FEATURE_DIM, 0)
    _run(FEATURE_DIM, 24)
    _run(FORECAST_ENTITY_DIM, 0)
    actor, action = _run(FORECAST_ENTITY_DIM, 24)
    loss = action.square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
