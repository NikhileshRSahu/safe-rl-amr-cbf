import numpy as np

from benchmark.best_vs_best_protocol import ScenarioSpec
from benchmark.shared_warehouse_perception import visible_with_shelves
from benchmark.train_multi_agent_research import SHELVES
from benchmark.warehouse_scenario_world import make_scenario_world


def test_cross_intersection_places_pedestrians_at_real_aisle_conflict_zone():
    spec = ScenarioSpec("cross_intersection", "intent", humans=12, n_amr=4)
    world = make_scenario_world(spec, 10100)
    world.reset()
    center = np.array([-4.5, 0.0], dtype=np.float32)
    near = np.linalg.norm(world.hp - center, axis=1) < 2.2
    assert int(near.sum()) >= 4
    headings = np.arctan2(world.hv[near, 1], world.hv[near, 0])
    assert np.any(np.abs(np.cos(headings)) > 0.8)
    assert np.any(np.abs(np.sin(headings)) > 0.8)


def test_shelf_corner_contains_initially_occluded_emerging_human():
    spec = ScenarioSpec("shelf_corner", "occlusion", humans=12, n_amr=4)
    world = make_scenario_world(spec, 10100)
    world.reset()
    # AMR 1 starts in the left horizontal corridor; human 0 is staged behind
    # the lower shelf corner and moves toward that corridor.
    observer = world.p[1]
    assert not visible_with_shelves(observer, world.hp[0], SHELVES, max_range=8.0, shelf_padding=0.02)
    assert world.hv[0, 1] < -0.1


def test_named_scenarios_are_not_just_aliases_of_same_initial_human_world():
    a = make_scenario_world(ScenarioSpec("cross_intersection", "intent", 12), 10105)
    b = make_scenario_world(ScenarioSpec("shelf_corner", "occlusion", 12), 10105)
    a.reset(); b.reset()
    assert not np.allclose(a.hp, b.hp)
    assert not np.allclose(a.hv, b.hv)
