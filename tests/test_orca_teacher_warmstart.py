import inspect

import numpy as np
import pytest
import torch

from benchmark.orca_teacher import (
    behavior_cloning_loss,
    bc_coefficient,
    normalize_teacher_action,
)
from benchmark.spatiotemporal_policy import STASACActor, TwinRecurrentQ
from benchmark.train_stasac_cbf import SequenceReplay, recurrent_sac_update
from benchmark.warehouse_interaction_features import FEATURE_DIM


def test_bc_coefficient_decays_linearly_and_stays_zero_after_first_15_percent():
    total = 1000
    assert bc_coefficient(0, total) == pytest.approx(1.0)
    assert bc_coefficient(75, total) == pytest.approx(0.5)
    assert bc_coefficient(150, total) == pytest.approx(0.0)
    assert bc_coefficient(151, total) == pytest.approx(0.0)
    assert bc_coefficient(1000, total) == pytest.approx(0.0)


def test_teacher_physical_commands_are_clipped_and_normalized_to_actor_bounds():
    a = normalize_teacher_action(v=2.0, omega=-3.0, v_max=1.0, omega_max=1.5)
    np.testing.assert_allclose(a, np.array([1.0, -1.0], np.float32))
    stopped = normalize_teacher_action(v=0.0, omega=0.0, v_max=1.0, omega_max=1.5)
    np.testing.assert_allclose(stopped, np.array([-1.0, 0.0], np.float32))


def test_behavior_cloning_loss_is_zero_when_actor_matches_teacher_and_weight_zero_disables_it():
    actor_action = torch.tensor([[0.2, -0.4], [0.0, 0.5]])
    teacher_action = actor_action.clone()
    assert behavior_cloning_loss(actor_action, teacher_action, coefficient=1.0).item() == pytest.approx(0.0)
    mismatch = teacher_action + 0.5
    assert behavior_cloning_loss(actor_action, mismatch, coefficient=0.0).item() == pytest.approx(0.0)
    assert behavior_cloning_loss(actor_action, mismatch, coefficient=1.0).item() > 0.0


def _teacher_step(t, done=False):
    n = 2
    return {
        "ego": np.full((12,), t, np.float32),
        "entities": np.full((n, FEATURE_DIM), t, np.float32),
        "entity_mask": np.ones((n,), np.bool_),
        "action": np.array([0.0, 0.0], np.float32),
        "teacher_action": np.array([0.8, -0.6], np.float32),
        "reward": 0.01,
        "next_ego": np.full((12,), t + 1, np.float32),
        "next_entities": np.full((n, FEATURE_DIM), t + 1, np.float32),
        "next_entity_mask": np.ones((n,), np.bool_),
        "done": bool(done),
    }


def _run_update_with_bc(coefficient):
    torch.manual_seed(21)
    replay = SequenceReplay(capacity_episodes=2, burn_in=1, train_len=3)
    replay.add_episode([_teacher_step(t, done=(t == 5)) for t in range(6)])
    batch = replay.sample(1, rng=np.random.default_rng(3))
    actor = STASACActor(ego_dim=12)
    q = TwinRecurrentQ(actor.hidden_dim)
    target_q = TwinRecurrentQ(actor.hidden_dim)
    target_q.load_state_dict(q.state_dict())
    metrics = recurrent_sac_update(
        batch,
        actor,
        q,
        target_q,
        torch.optim.Adam(actor.parameters(), 1e-3),
        torch.optim.Adam(q.parameters(), 1e-3),
        bc_coeff=coefficient,
    )
    return metrics


def test_recurrent_sac_update_applies_teacher_only_when_bc_coefficient_positive():
    early = _run_update_with_bc(1.0)
    late = _run_update_with_bc(0.0)
    assert early["bc_loss"] > 0.0
    assert late["bc_loss"] == pytest.approx(0.0, abs=1e-12)


def test_deployment_actor_contains_no_orca_runtime_dependency():
    source = inspect.getsource(STASACActor)
    assert "AStarORCA" not in source
    assert "orca_teacher" not in source
