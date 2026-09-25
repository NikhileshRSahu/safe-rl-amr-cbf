import numpy as np
import torch

from benchmark.best_vs_best_protocol import ScenarioSpec
from benchmark.stasac_warehouse import (
    EGO_DIM,
    WarehouseObservationBuilder,
    make_training_world,
    rollout_untrained_actor_smoke,
)
from benchmark.spatiotemporal_policy import STASACActor
from benchmark.warehouse_interaction_features import FEATURE_DIM


def test_observation_builder_has_fixed_ego_width_and_keeps_all_nearby_entities():
    spec = ScenarioSpec("mixed_behavior", "mixed", humans=12, n_amr=4, randomness_level="high")
    world = make_training_world(spec, seed=10101)
    builder = WarehouseObservationBuilder(world, perception_range=100.0)
    ego, entity_batch = builder.observe(world, 0, now=0.0)
    assert ego.shape == (EGO_DIM,)
    assert entity_batch.features.shape[1] == FEATURE_DIM
    # 3 peer AMRs + all 12 humans: no fixed four-human bottleneck.
    assert entity_batch.features.shape[0] == 15
    assert entity_batch.mask.all()
    assert np.isfinite(ego).all()
    assert np.isfinite(entity_batch.features).all()


def test_observation_history_is_causal_and_acceleration_channels_change_after_motion():
    spec = ScenarioSpec("hesitation", "intent", humans=6, n_amr=4, randomness_level="high")
    world = make_training_world(spec, seed=10102)
    builder = WarehouseObservationBuilder(world, perception_range=100.0)
    _, first = builder.observe(world, 0, now=0.0)
    actions = np.tile(np.array([-1.0, 0.0], np.float32), (4, 1))
    world.step(actions, use_cbf=False)
    _, second = builder.observe(world, 0, now=0.1)
    assert first.features.shape[1] == second.features.shape[1] == FEATURE_DIM
    assert np.isfinite(second.features[:, 9:13]).all()
    assert np.any(np.abs(second.features[:, 9:13]) > 0.0)


def test_same_seed_builds_same_initial_policy_observation():
    spec = ScenarioSpec("cross_intersection", "intent", humans=8, n_amr=4, randomness_level="medium")
    w1 = make_training_world(spec, seed=10103)
    w2 = make_training_world(spec, seed=10103)
    b1 = WarehouseObservationBuilder(w1, perception_range=100.0)
    b2 = WarehouseObservationBuilder(w2, perception_range=100.0)
    e1, x1 = b1.observe(w1, 2, now=0.0)
    e2, x2 = b2.observe(w2, 2, now=0.0)
    np.testing.assert_allclose(e1, e2)
    np.testing.assert_allclose(x1.features, x2.features)
    assert x1.entity_ids == x2.entity_ids


def test_untrained_actor_real_world_cbf_smoke_has_finite_actions_and_state():
    torch.manual_seed(5)
    spec = ScenarioSpec("cross_intersection", "intent", humans=6, n_amr=4, randomness_level="medium")
    world = make_training_world(spec, seed=10104)
    actor = STASACActor(ego_dim=EGO_DIM)
    result = rollout_untrained_actor_smoke(actor, world, steps=16)
    assert result["steps"] > 0
    assert result["finite_actions"]
    assert result["finite_world_state"]
    assert result["max_abs_action"] <= 1.0 + 1e-6
