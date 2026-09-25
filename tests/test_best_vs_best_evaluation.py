import pytest

from benchmark.best_vs_best_evaluation import (
    ControllerAggregate,
    compare_best_vs_best,
    select_validation_candidate,
    wilson_interval,
)


def agg(success, collision, timeout, fleet, throughput, n=100):
    return ControllerAggregate(
        agents=n,
        episodes=n // 4,
        success_rate=success,
        collision_rate=collision,
        timeout_rate=timeout,
        fleet_success_rate=fleet,
        throughput_per_min=throughput,
    )


def test_wilson_interval_is_bounded_and_contains_empirical_rate():
    lo, hi = wilson_interval(93, 100)
    assert 0.0 <= lo <= 0.93 <= hi <= 1.0
    lo0, hi0 = wilson_interval(0, 30)
    assert lo0 == pytest.approx(0.0)
    assert hi0 > 0.0


def test_rl_win_requires_safety_parity_and_better_liveness():
    orca = agg(.84, .02, .14, .52, 4.5)
    rl = agg(.93, .02, .05, .76, 5.1)
    result = compare_best_vs_best(rl, orca, collision_tolerance=.01)
    assert result.rl_wins
    assert result.safety_parity
    assert result.success_delta == pytest.approx(.09)
    assert result.fleet_success_delta == pytest.approx(.24)


def test_higher_rl_collision_rate_blocks_win_even_with_higher_success():
    orca = agg(.82, .00, .18, .48, 4.2)
    rl = agg(.96, .04, .00, .88, 5.7)
    result = compare_best_vs_best(rl, orca, collision_tolerance=.01)
    assert not result.rl_wins
    assert not result.safety_parity


def test_equal_safety_but_no_liveness_improvement_is_not_a_win():
    orca = agg(.90, .01, .09, .70, 5.0)
    rl = agg(.90, .01, .09, .70, 5.0)
    assert not compare_best_vs_best(rl, orca).rl_wins


def test_validation_selector_prioritizes_safety_then_fleet_success_then_timeout_then_throughput():
    candidates = {
        "unsafe_fast": agg(.98, .04, .00, .92, 6.0),
        "safe_low_fleet": agg(.90, .01, .09, .60, 4.8),
        "safe_high_fleet": agg(.91, .01, .08, .76, 4.7),
        "safe_high_fleet_fast": agg(.91, .01, .08, .76, 5.2),
    }
    assert select_validation_candidate(candidates) == "safe_high_fleet_fast"
