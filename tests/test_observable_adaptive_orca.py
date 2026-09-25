import numpy as np

from benchmark.beast_config import load_beast_config
from benchmark.intent_shift_benchmark import AdaptiveORCADD, IntentShiftWorld


def test_adaptive_orca_does_not_depend_on_hidden_uncertainty_scalar():
    cfg = load_beast_config('benchmark/frozen_peak_orca_config.json')
    w = IntentShiftWorld(4, 12, 13000)
    w.reset()
    c = AdaptiveORCADD(w, 0, cfg)

    # First observation establishes history.
    c._update_observed_human_uncertainty(w)
    before = c._observed_human_uncertainty.copy()

    # Changing hidden environment uncertainty alone must not affect ORCA's estimate.
    w.human_uncertainty[:] = 1.0
    c._update_observed_human_uncertainty(w)
    assert np.allclose(c._observed_human_uncertainty, before)


def test_observed_velocity_change_increases_adaptive_orca_uncertainty():
    cfg = load_beast_config('benchmark/frozen_peak_orca_config.json')
    w = IntentShiftWorld(4, 12, 13001)
    w.reset()
    c = AdaptiveORCADD(w, 0, cfg)
    c._update_observed_human_uncertainty(w)
    calm = float(c._observed_human_uncertainty[0])

    # A visible abrupt velocity change should raise the estimate.
    w.hv[0] = -w.hv[0]
    c._update_observed_human_uncertainty(w)
    changed = float(c._observed_human_uncertainty[0])
    assert changed > calm
    assert c._human_time_horizon(w, 0) > cfg.time_horizon
