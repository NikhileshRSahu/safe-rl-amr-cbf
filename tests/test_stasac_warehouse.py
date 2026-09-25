import numpy as np
import torch

from benchmark.best_vs_best_protocol import ScenarioSpec
from benchmark.shared_warehouse_perception import visible_with_shelves
from benchmark.stasac_warehouse import (
    EGO_DIM,
    WarehouseObservationBuilder,
    make_training_world,
    rollout_untrained_actor_smoke,
)
from benchmark.spatiotemporal_policy import STASACActor
from benchmark.train_multi_agent_research import SHELVES
from benchmark.warehouse_interaction_features import FEATURE_DIM


def test_observation_builder_has_fixed_ego_width_and_keeps_all_visible_entities_without_slot_cap():
    spec = ScenarioSpec("mixed_behavior", "mixed", humans=12, n_amr=4, randomness_level="high")
    world = make_training_world(spec, seed=10101)
    builder = WarehouseObservationBuilder(world, perception_range=100.0)
    ego, entity_batch = builder.observe(world, 0, now=0.0)
    assert ego.shape == (EGO_DIM,)
    assert entity_batch.features.shape[1] == FEATURE_DIM

    visible_humans = sum(
        visible_with_shelves(
            world.p[0], world.hp[j], SHELVES,
            max_range=100.0, shelf_padding=0.02,
        )
        for j in range(world.nppl)
    )
    expected_entities = (world.n - 1) + visible_humans
    assert entity_batch.features.shape[0] == expected_entities
    assert entity_batch.mask.all()
    assert np.isfinite(ego).all()
    assert np.isfinite(entity_batch.features).all()


def test_observation_builder_has_no_fixed_four_human_bottleneck_when_six_are_visible():
    spec = ScenarioSpec("density_06", "density", humans=6, n_amr=4, randomness_level="baseline")
    world = make_training_world(spec, seed=10111)
    # Place six humans in the open horizontal corridor, all visible from AMR 1.
    world.p[1] = np.array([-9.0, 0.0], dtype=np.float32)
    world.hp[:] = np.array([
        [-7.8, -1.2], [-7.2, -0.7], [-6.7, 0.0],
        [-5.9, 0.6], [-5.2, 1.1], [-4.5, -1.0],
    ], dtype=np.float32)
    world.hv[:] = 0.0
    builder = WarehouseObservationBuilder(world, perception_range=10.0)
    _, entity_batch = builder.observe(world, 1, now=0.0)
    human_ids = [eid for eid in entity_batch.entity_ids if eid.startswith("human-")]
    assert len(human_ids) == 6


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
