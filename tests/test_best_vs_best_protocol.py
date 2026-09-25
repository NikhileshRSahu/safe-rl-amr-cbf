import numpy as np
import pytest

from benchmark.adaptive_predictive_orca import (
    AdaptiveORCAConfig,
    adaptive_time_horizon,
    uncertainty_inflation,
    human_responsibility,
)
from benchmark.best_vs_best_protocol import (
    ScenarioSpec,
    paired_world_fingerprint,
    scenario_catalog,
    split_seed_sets,
)


def test_adaptive_horizon_is_bounded_and_uncertainty_reduces_long_prediction():
    cfg = AdaptiveORCAConfig(horizon_min=1.0, horizon_max=4.0)
    low_uncertainty = adaptive_time_horizon(
        ttc=1.0, uncertainty=0.05, density=4, yield_streak=0, config=cfg
    )
    high_uncertainty = adaptive_time_horizon(
        ttc=1.0, uncertainty=1.5, density=4, yield_streak=0, config=cfg
    )
    deadlocked = adaptive_time_horizon(
        ttc=1.0, uncertainty=0.05, density=4, yield_streak=40, config=cfg
    )
    assert cfg.horizon_min <= high_uncertainty <= cfg.horizon_max
    assert cfg.horizon_min <= low_uncertainty <= cfg.horizon_max
    assert cfg.horizon_min <= deadlocked <= cfg.horizon_max
    assert low_uncertainty > high_uncertainty
    assert low_uncertainty > deadlocked


def test_uncertainty_inflation_is_monotonic_and_never_shrinks_physical_radius():
    base = 0.72
    r0 = uncertainty_inflation(base, sigma=0.0, gain=0.35, max_extra=0.40)
    r1 = uncertainty_inflation(base, sigma=0.5, gain=0.35, max_extra=0.40)
    r2 = uncertainty_inflation(base, sigma=2.0, gain=0.35, max_extra=0.40)
    assert r0 == pytest.approx(base)
    assert base <= r0 < r1 <= r2 <= base + 0.40


def test_humans_are_nonreciprocal_full_robot_responsibility():
    assert human_responsibility(observed_yield_probability=0.0) == pytest.approx(1.0)
    assert human_responsibility(observed_yield_probability=1.0) >= 0.75
    assert human_responsibility(observed_yield_probability=0.2) > human_responsibility(0.9)


def test_scenario_catalog_contains_real_warehouse_failure_modes_and_density_ladder():
    specs = scenario_catalog()
    names = {s.name for s in specs}
    assert {"cross_intersection", "shelf_corner", "hesitation", "mixed_behavior"} <= names
    density = sorted(s.humans for s in specs if s.family == "density")
    assert density == [6, 12, 18, 24]
    assert all(isinstance(s, ScenarioSpec) for s in specs)


def test_paired_world_fingerprint_is_identical_for_same_seed_and_changes_for_other_seed():
    spec = ScenarioSpec("cross_intersection", "intent", humans=12, n_amr=4)
    a = paired_world_fingerprint(spec, seed=12101)
    b = paired_world_fingerprint(spec, seed=12101)
    c = paired_world_fingerprint(spec, seed=12102)
    assert a == b
    assert a != c


def test_seed_splits_are_disjoint_and_holdout_is_not_returned_before_freeze():
    dev, validation, holdout = split_seed_sets(frozen=False)
    assert set(dev).isdisjoint(validation)
    assert holdout == ()
    dev2, validation2, holdout2 = split_seed_sets(frozen=True)
    assert dev2 == dev and validation2 == validation
    assert len(holdout2) >= 30
    assert set(holdout2).isdisjoint(dev2)
    assert set(holdout2).isdisjoint(validation2)
