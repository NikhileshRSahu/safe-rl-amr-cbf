import numpy as np

from benchmark.intent_shift_benchmark import (
    FINAL_SEEDS,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    IntentShiftWorld,
    AdaptiveORCADD,
)
from benchmark.beast_config import load_beast_config


def test_seed_partitions_are_disjoint_and_fresh():
    assert set(TRAIN_SEEDS).isdisjoint(VALIDATION_SEEDS)
    assert set(TRAIN_SEEDS).isdisjoint(FINAL_SEEDS)
    assert set(VALIDATION_SEEDS).isdisjoint(FINAL_SEEDS)
    assert min(TRAIN_SEEDS) >= 12000
    assert min(VALIDATION_SEEDS) >= 13000
    assert min(FINAL_SEEDS) >= 14000


def test_intent_shift_world_is_deterministic_for_same_seed():
    a = IntentShiftWorld(4, 12, 13000)
    b = IntentShiftWorld(4, 12, 13000)
    oa, ob = a.reset(), b.reset()
    assert np.allclose(a.hp, b.hp)
    assert np.allclose(a.hv, b.hv)
    assert np.array_equal(a.intent_mode, b.intent_mode)
    zero = np.tile(np.array([-1.0, 0.0], np.float32), (4, 1))
    for _ in range(80):
        a.step(zero, False)
        b.step(zero, False)
    assert np.allclose(a.hp, b.hp)
    assert np.allclose(a.hv, b.hv)
    assert np.array_equal(a.intent_mode, b.intent_mode)
    assert a.intent_change_count > 0


def test_adaptive_orca_increases_prediction_horizon_for_uncertain_human():
    cfg = load_beast_config('benchmark/frozen_peak_orca_config.json')
    w = IntentShiftWorld(4, 12, 13001)
    w.reset()
    c = AdaptiveORCADD(w, 0, cfg)
    base = cfg.time_horizon
    calm = c._human_time_horizon(w, 0)
    w.human_uncertainty[0] = 1.0
    uncertain = c._human_time_horizon(w, 0)
    assert calm >= base
    assert uncertain > calm
    assert uncertain <= c.max_human_time_horizon
