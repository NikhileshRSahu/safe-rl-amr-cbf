import math

import numpy as np
import pytest

from benchmark.warehouse_interaction_features import (
    EntityObservation,
    ObservationHistory,
    build_entity_batch,
    compute_risk_features,
    compute_ttc_cpa,
)


def test_stationary_entity_has_horizon_cpa_and_finite_values():
    t_cpa, d_cpa = compute_ttc_cpa(
        rel_pos=np.array([2.0, 0.0]),
        rel_vel=np.array([0.0, 0.0]),
        horizon=4.0,
    )
    assert math.isfinite(t_cpa)
    assert math.isfinite(d_cpa)
    assert t_cpa == pytest.approx(0.0)
    assert d_cpa == pytest.approx(2.0)


def test_closing_trajectory_has_near_future_cpa_and_higher_risk_than_diverging():
    closing_t, closing_d = compute_ttc_cpa(
        rel_pos=np.array([2.0, 0.0]),
        rel_vel=np.array([-1.0, 0.0]),
        horizon=4.0,
    )
    diverging_t, diverging_d = compute_ttc_cpa(
        rel_pos=np.array([2.0, 0.0]),
        rel_vel=np.array([1.0, 0.0]),
        horizon=4.0,
    )
    assert closing_t == pytest.approx(2.0, abs=1e-6)
    assert closing_d == pytest.approx(0.0, abs=1e-6)
    assert diverging_t == pytest.approx(0.0, abs=1e-6)
    assert diverging_d == pytest.approx(2.0, abs=1e-6)

    closing = compute_risk_features(np.array([2.0, 0.0]), np.array([-1.0, 0.0]))
    diverging = compute_risk_features(np.array([2.0, 0.0]), np.array([1.0, 0.0]))
    assert closing["ttc_risk"] > diverging["ttc_risk"]


def test_causal_history_derives_acceleration_and_heading_rate_without_future_samples():
    history = ObservationHistory(maxlen=4)
    history.update("human-0", 0.0, np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    history.update("human-0", 0.5, np.array([0.5, 0.0]), np.array([1.0, 1.0]))
    kin = history.kinematics("human-0", now=0.5)
    np.testing.assert_allclose(kin.acceleration, np.array([0.0, 2.0]), atol=1e-6)
    assert kin.heading_rate == pytest.approx(math.pi / 2.0, rel=1e-6)
    assert kin.observation_age == pytest.approx(0.0)

    with pytest.raises(ValueError, match="future observation"):
        history.update("human-0", 0.4, np.array([0.4, 0.0]), np.array([1.0, 0.0]))


def test_build_entity_batch_handles_zero_and_twenty_four_humans_with_fixed_feature_width():
    empty = build_entity_batch(
        ego_pos=np.zeros(2),
        ego_vel=np.zeros(2),
        entities=[],
        now=0.0,
    )
    assert empty.features.shape == (0, empty.feature_dim)
    assert empty.mask.shape == (0,)

    entities = [
        EntityObservation(
            entity_id=f"human-{i}",
            entity_type="human",
            position=np.array([float(i + 1), 0.1 * i]),
            velocity=np.array([-0.1, 0.0]),
            timestamp=1.0,
            visible=True,
        )
        for i in range(24)
    ]
    batch = build_entity_batch(
        ego_pos=np.zeros(2),
        ego_vel=np.zeros(2),
        entities=entities,
        now=1.0,
    )
    assert batch.features.shape[0] == 24
    assert batch.features.shape[1] == empty.feature_dim
    assert batch.mask.dtype == np.bool_
    assert batch.mask.all()
    assert np.isfinite(batch.features).all()


def test_entity_order_is_deterministic_by_risk_then_distance_then_id():
    entities = [
        EntityObservation("human-b", "human", np.array([3.0, 0.0]), np.array([-0.5, 0.0]), 0.0),
        EntityObservation("human-a", "human", np.array([3.0, 0.0]), np.array([-0.5, 0.0]), 0.0),
        EntityObservation("human-c", "human", np.array([8.0, 0.0]), np.array([0.0, 0.0]), 0.0),
    ]
    a = build_entity_batch(np.zeros(2), np.zeros(2), entities, now=0.0)
    b = build_entity_batch(np.zeros(2), np.zeros(2), list(reversed(entities)), now=0.0)
    assert a.entity_ids == b.entity_ids
    assert a.entity_ids[:2] == ("human-a", "human-b")
    np.testing.assert_allclose(a.features, b.features)
