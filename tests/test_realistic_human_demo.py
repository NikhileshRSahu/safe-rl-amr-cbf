import math

import numpy as np

import benchmark.render_shared_success_case as demo


def _rect_distance(point, rect):
    x, y = float(point[0]), float(point[1])
    x0, y0, x1, y1 = rect
    cx = min(max(x, x0), x1)
    cy = min(max(y, y0), y1)
    return math.hypot(x - cx, y - cy)


def test_realistic_human_demo_world_keeps_people_out_of_geometry_and_each_other():
    assert hasattr(demo, "RealisticHumanWorld"), (
        "demo renderer must provide a RealisticHumanWorld instead of using "
        "straight-line pedestrians"
    )

    from benchmark.train_multi_agent_research import SHELVES, WORLD

    world = demo.RealisticHumanWorld(4, 12, seed=7712)
    world.reset()
    human_radius = demo.HUMAN_RADIUS
    min_pair_distance = 2.0 * human_radius - 1e-5

    stopped = np.tile(np.array([-1.0, 0.0], dtype=np.float32), (4, 1))

    for _ in range(250):
        world.step(stopped, use_cbf=False)

        assert np.all(np.abs(world.hp[:, 0]) <= WORLD - human_radius + 1e-6)
        assert np.all(np.abs(world.hp[:, 1]) <= WORLD - human_radius + 1e-6)

        for person in world.hp:
            for rect in SHELVES:
                assert _rect_distance(person, rect) >= human_radius - 1e-5

        for i in range(world.nppl):
            for j in range(i + 1, world.nppl):
                assert np.linalg.norm(world.hp[i] - world.hp[j]) >= min_pair_distance


def test_realistic_humans_keep_personal_space_from_stopped_amr():
    from benchmark.train_multi_agent_research import ROBOT_R

    world = demo.RealisticHumanWorld(1, 1, seed=7781)
    world.reset()

    # Put one pedestrian directly in front of the stationary AMR and make the
    # current velocity point toward it. A realistic pedestrian should steer or
    # stop instead of walking through the robot body.
    world.hp[0] = world.p[0] + np.array([1.00, 0.0], dtype=np.float32)
    world.hv[0] = np.array([-0.50, 0.0], dtype=np.float32)
    world._human_preferred_speed[0] = 0.50

    stopped = np.array([[-1.0, 0.0]], dtype=np.float32)
    required = ROBOT_R + demo.HUMAN_RADIUS + 0.05

    for _ in range(30):
        world.step(stopped, use_cbf=False)
        assert np.linalg.norm(world.hp[0] - world.p[0]) >= required - 1e-5
