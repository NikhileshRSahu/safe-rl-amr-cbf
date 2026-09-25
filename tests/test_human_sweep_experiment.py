import benchmark.human_sweep_experiment as sweep


def test_human_count_sweep_increases_by_about_25_percent_and_keeps_other_factors_fixed():
    scenarios = sweep.build_count_scenarios()
    assert [s.humans for s in scenarios] == [12, 15, 19, 24]
    assert {s.speed_scale for s in scenarios} == {1.0}
    assert {s.randomness_level for s in scenarios} == {"baseline"}


def test_speed_sweep_changes_only_speed():
    scenarios = sweep.build_speed_scenarios()
    assert [s.speed_scale for s in scenarios] == [1.0, 1.25, 1.5, 1.75]
    assert {s.humans for s in scenarios} == {12}
    assert {s.randomness_level for s in scenarios} == {"baseline"}


def test_randomness_sweep_changes_only_motion_randomness():
    scenarios = sweep.build_randomness_scenarios()
    assert [s.randomness_level for s in scenarios] == ["baseline", "low", "medium", "high"]
    assert {s.humans for s in scenarios} == {12}
    assert {s.speed_scale for s in scenarios} == {1.0}


def test_randomness_profiles_are_monotonic():
    profiles = [sweep.RANDOMNESS_PROFILES[k] for k in ["baseline", "low", "medium", "high"]]
    assert [p.heading_sigma for p in profiles] == sorted(p.heading_sigma for p in profiles)
    assert [p.stop_probability for p in profiles] == sorted(p.stop_probability for p in profiles)
    assert [p.reverse_probability for p in profiles] == sorted(p.reverse_probability for p in profiles)
