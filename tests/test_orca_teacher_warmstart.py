import inspect

import numpy as np
import pytest
import torch

from benchmark.orca_teacher import (
    behavior_cloning_loss,
    bc_coefficient,
    normalize_teacher_action,
)
from benchmark.spatiotemporal_policy import STASACActor


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


def test_deployment_actor_contains_no_orca_runtime_dependency():
    source = inspect.getsource(STASACActor)
    assert "AStarORCA" not in source
    assert "orca_teacher" not in source
