import math
import numpy as np

from benchmark.beast_classical import AStarORCADD, BeastORCAConfig
from benchmark.train_multi_agent_research import DT, ROBOT_R, VMAX, WMAX, wrap


class SimWorld:
    def __init__(self, p, g, th, humans=None):
        self.p = np.asarray(p, dtype=float)
        self.g = np.asarray(g, dtype=float)
        self.th = np.asarray(th, dtype=float)
        self.n = len(self.p)
        self.v = np.zeros(self.n)
        self.w = np.zeros(self.n)
        self.done = np.zeros(self.n, dtype=bool)
        self.priority = (
            np.linspace(0, 1, self.n)
            if self.n > 1
            else np.array([0.5])
        )
        humans = humans or []
        self.nppl = len(humans)
        self.hp = (
            np.asarray([x[0] for x in humans], dtype=float).reshape((-1, 2))
            if humans
            else np.zeros((0, 2))
        )
        self.hv = (
            np.asarray([x[1] for x in humans], dtype=float).reshape((-1, 2))
            if humans
            else np.zeros((0, 2))
        )

    def tick(self, ctrls):
        acts = np.asarray([c.action(self) for c in ctrls])
        assert np.all(np.isfinite(acts))
        assert np.max(np.abs(acts)) <= 1.00001

        p0 = self.p.copy()
        th0 = self.th.copy()
        new_v = (acts[:, 0] + 1.0) * 0.5 * VMAX
        new_w = acts[:, 1] * WMAX
        new_th = np.asarray(
            [wrap(th0[i] + new_w[i] * DT) for i in range(self.n)]
        )
        self.p = (
            p0
            + np.c_[np.cos(new_th), np.sin(new_th)]
            * new_v[:, None]
            * DT
        )
        self.th = new_th
        self.v = new_v
        self.w = new_w
        self.hp += self.hv * DT
        return acts


def min_pair_distance(p):
    best = float("inf")
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            best = min(best, float(np.linalg.norm(p[i] - p[j])))
    return best


def test_head_on_corridor_stays_separated_and_progresses():
    w = SimWorld(
        [[-1.5, 0], [1.5, 0]],
        [[3, 0], [-3, 0]],
        [0, math.pi],
    )
    cs = [AStarORCADD(w, i, BeastORCAConfig()) for i in range(2)]
    start = np.linalg.norm(w.g - w.p, axis=1)
    min_d = 99.0
    for _ in range(30):
        w.tick(cs)
        min_d = min(min_d, min_pair_distance(w.p))

    assert min_d >= 2 * ROBOT_R - 1e-6
    assert np.all(np.linalg.norm(w.g - w.p, axis=1) < start)


def test_perpendicular_crossing_stays_collision_free():
    w = SimWorld(
        [[-1.2, 0], [0, -1.2]],
        [[2, 0], [0, 2]],
        [0, math.pi / 2],
    )
    cs = [AStarORCADD(w, i, BeastORCAConfig()) for i in range(2)]
    min_d = 99.0
    for _ in range(30):
        w.tick(cs)
        min_d = min(min_d, min_pair_distance(w.p))

    assert min_d >= 2 * ROBOT_R - 1e-6


def test_pedestrian_crossing_avoids_contact():
    w = SimWorld(
        [[-1.2, 0]],
        [[2, 0]],
        [0],
        humans=[([0, -0.8], [0, 0.5])],
    )
    c = AStarORCADD(w, 0, BeastORCAConfig())
    min_d = 99.0
    for _ in range(35):
        w.tick([c])
        min_d = min(
            min_d,
            float(np.linalg.norm(w.p[0] - w.hp[0])),
        )

    assert min_d >= 2 * ROBOT_R - 1e-6


def test_six_amr_congestion_emits_finite_safe_commands():
    p = [
        [-2, -2], [-2, 0], [-2, 2],
        [2, -2], [2, 0], [2, 2],
    ]
    g = [
        [2, 2], [2, 0], [2, -2],
        [-2, 2], [-2, 0], [-2, -2],
    ]
    th = [
        math.atan2(g[i][1] - p[i][1], g[i][0] - p[i][0])
        for i in range(6)
    ]
    w = SimWorld(p, g, th)
    cs = [AStarORCADD(w, i, BeastORCAConfig()) for i in range(6)]
    min_d = 99.0
    for _ in range(20):
        w.tick(cs)
        min_d = min(min_d, min_pair_distance(w.p))

    assert min_d >= 2 * ROBOT_R - 1e-6
