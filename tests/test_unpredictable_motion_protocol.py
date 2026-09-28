from benchmark.best_vs_best_protocol import (
    local_human_navigation_catalog,
    split_seed_sets,
    unpredictable_motion_catalog,
    unpredictable_motion_holdout_seeds,
    make_paired_worlds,
)


def test_unpredictable_motion_suite_is_evaluation_only_and_disjoint_from_training():
    training_names = {s.name for s in local_human_navigation_catalog()}
    evaluation = unpredictable_motion_catalog()
    evaluation_names = {s.name for s in evaluation}

    assert evaluation_names == {
        "surprise_stop_go",
        "lateral_cut_in",
        "compound_nonreciprocal",
    }
    assert training_names.isdisjoint(evaluation_names)
    assert all(s.n_amr == 1 for s in evaluation)
    assert all(s.randomness_level == "high" for s in evaluation)

    dev, validation, frozen_holdout = split_seed_sets(frozen=True)
    unpredictable = set(unpredictable_motion_holdout_seeds())
    assert unpredictable
    assert unpredictable.isdisjoint(dev)
    assert unpredictable.isdisjoint(validation)
    assert unpredictable.isdisjoint(frozen_holdout)


def test_unpredictable_motion_worlds_are_paired_for_fair_controller_comparison():
    for spec in unpredictable_motion_catalog():
        left, right = make_paired_worlds(spec, unpredictable_motion_holdout_seeds()[0])
        assert left.scenario_name == right.scenario_name == spec.name
        assert (left.p == right.p).all()
        assert (left.g == right.g).all()
        assert (left.hp == right.hp).all()
        assert (left.hv == right.hv).all()
