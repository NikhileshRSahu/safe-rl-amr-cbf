import pytest

from benchmark.train_stasac_cbf import actor_objective_weights


def test_full_teacher_phase_is_pure_behavior_cloning():
    sac_weight, bc_weight = actor_objective_weights(1.0)
    assert sac_weight == pytest.approx(0.0)
    assert bc_weight == pytest.approx(1.0)


def test_sac_ramps_in_as_teacher_support_ramps_out():
    sac_weight, bc_weight = actor_objective_weights(0.6)
    assert sac_weight == pytest.approx(0.4)
    assert bc_weight == pytest.approx(0.6)


def test_independent_phase_is_pure_sac():
    sac_weight, bc_weight = actor_objective_weights(0.0)
    assert sac_weight == pytest.approx(1.0)
    assert bc_weight == pytest.approx(0.0)


def test_teacher_coefficient_is_clamped_for_objective_weights():
    assert actor_objective_weights(1.5) == pytest.approx((0.0, 1.0))
    assert actor_objective_weights(-0.5) == pytest.approx((1.0, 0.0))
