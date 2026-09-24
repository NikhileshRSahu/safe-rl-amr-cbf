import json
from dataclasses import asdict
from pathlib import Path

from benchmark.beast_classical import BeastORCAConfig
from benchmark.beast_config import (
    DEV_SEEDS_BY_N,
    VALIDATION_SEEDS_BY_N,
    LEGACY_TEST_SEEDS_BY_N,
    TEST_SEEDS_BY_N,
    load_beast_config,
    save_beast_config,
)


def test_seed_sets_are_pairwise_disjoint():
    for n in (2, 4, 6):
        d = set(DEV_SEEDS_BY_N[n])
        v = set(VALIDATION_SEEDS_BY_N[n])
        t = set(TEST_SEEDS_BY_N[n])
        legacy = set(LEGACY_TEST_SEEDS_BY_N[n])
        assert not d & v
        assert not d & t
        assert not v & t
        assert not legacy & t


def test_legacy_test_seeds_preserve_existing_protocol():
    assert LEGACY_TEST_SEEDS_BY_N[2] == tuple(range(5200, 5230))
    assert LEGACY_TEST_SEEDS_BY_N[4] == tuple(range(5400, 5430))
    assert LEGACY_TEST_SEEDS_BY_N[6] == tuple(range(5600, 5630))


def test_new_final_holdout_is_fresh_and_versioned():
    assert TEST_SEEDS_BY_N[2] == tuple(range(6200, 6230))
    assert TEST_SEEDS_BY_N[4] == tuple(range(6400, 6430))
    assert TEST_SEEDS_BY_N[6] == tuple(range(6600, 6630))


def test_config_roundtrip(tmp_path: Path):
    cfg = BeastORCAConfig(
        time_horizon=3.0,
        command_speed_samples=9,
        arc_horizon=1.0,
    )
    p = tmp_path / "cfg.json"
    save_beast_config(cfg, p)
    loaded = load_beast_config(p)
    assert asdict(loaded) == asdict(cfg)


def test_config_rejects_unknown_field(tmp_path: Path):
    cfg = asdict(BeastORCAConfig())
    cfg["surprise"] = 123
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(cfg))
    try:
        load_beast_config(p)
    except ValueError as exc:
        assert "unknown" in str(exc).lower()
    else:
        raise AssertionError("unknown field should fail")


def test_ranking_prioritizes_safety_then_liveness():
    from benchmark.tune_beast_classical import rank_key

    safe_slow = dict(
        collision_rate=0.0,
        timeout_rate=0.2,
        fleet_success_rate=0.5,
        success_rate=0.8,
        traversal_time_success_mean=40.0,
        path_length_success_mean=30.0,
    )
    unsafe_fast = dict(
        collision_rate=0.01,
        timeout_rate=0.0,
        fleet_success_rate=1.0,
        success_rate=0.99,
        traversal_time_success_mean=20.0,
        path_length_success_mean=20.0,
    )
    assert rank_key(safe_slow) < rank_key(unsafe_fast)

    fewer_timeouts = dict(safe_slow, timeout_rate=0.1)
    assert rank_key(fewer_timeouts) < rank_key(safe_slow)


def test_tuner_has_multiple_nonidentical_candidates():
    from benchmark.tune_beast_classical import candidate_configs

    configs = candidate_configs()
    assert len(configs) >= 12
    serial = {tuple(sorted(asdict(c).items())) for c in configs}
    assert len(serial) == len(configs)


def test_tuner_source_does_not_reference_final_test_seeds():
    import inspect
    import benchmark.tune_beast_classical as tuner

    assert "TEST_SEEDS_BY_N" not in inspect.getsource(tuner)


def test_evaluator_protocol_uses_fresh_holdout_seeds():
    from benchmark.evaluate_beast_controllers import test_seeds_for_n

    assert test_seeds_for_n(2) == tuple(range(6200, 6230))
    assert test_seeds_for_n(4) == tuple(range(6400, 6430))
    assert test_seeds_for_n(6) == tuple(range(6600, 6630))


def test_config_digest_is_stable_and_sensitive():
    from benchmark.evaluate_beast_controllers import config_digest

    a = BeastORCAConfig()
    b = BeastORCAConfig(time_horizon=a.time_horizon + 0.1)
    assert config_digest(a) == config_digest(a)
    assert config_digest(a) != config_digest(b)


def test_successive_halving_uses_full_validation_only_for_finalists():
    from benchmark.tune_beast_classical import successive_halving_schedule

    stages = successive_halving_schedule()
    assert stages[0]["seeds_per_n"] == 1
    assert stages[-1]["seeds_per_n"] == 15
    assert stages[-1]["keep"] == 1
    assert all(
        stages[i + 1]["seeds_per_n"] > stages[i]["seeds_per_n"]
        for i in range(len(stages) - 1)
    )
    assert all(
        stages[i + 1]["keep"] <= stages[i]["keep"]
        for i in range(len(stages) - 1)
    )


def test_successive_halving_budget_is_below_old_tuner_budget():
    from benchmark.tune_beast_classical import estimated_episode_budget

    old_budget = 16 * 5 * 3 + 4 * 15 * 3
    assert estimated_episode_budget() < old_budget * 0.45
