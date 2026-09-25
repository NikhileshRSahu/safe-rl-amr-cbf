from benchmark.evaluate_progressive_ladder import CLASSICAL_SCENARIOS, PROGRESSION_SEEDS


def test_progressive_ladder_uses_same_seeds_for_every_classical_method():
    assert PROGRESSION_SEEDS == tuple(range(9100, 9110))


def test_reciprocal_crossing_is_four_amrs_without_humans():
    s = CLASSICAL_SCENARIOS["reciprocal_crossing"]
    assert s["n_amr"] == 4
    assert s["humans"] == 0


def test_dense_reciprocal_is_six_amrs_without_humans():
    s = CLASSICAL_SCENARIOS["dense_reciprocal"]
    assert s["n_amr"] == 6
    assert s["humans"] == 0


def test_classical_scenarios_change_density_not_robot_dynamics():
    assert CLASSICAL_SCENARIOS["reciprocal_crossing"]["world_kind"] == "base"
    assert CLASSICAL_SCENARIOS["dense_reciprocal"]["world_kind"] == "base"
