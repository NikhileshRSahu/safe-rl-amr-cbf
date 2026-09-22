import math
import numpy as np

from benchmark.beast_classical import AStarORCADD, BeastORCAConfig


class World:
    def __init__(self, positions, goals, headings, velocities=None, humans=None):
        self.p = np.asarray(positions, dtype=float)
        self.g = np.asarray(goals, dtype=float)
        self.th = np.asarray(headings, dtype=float)
        self.n = len(self.p)
        self.v = np.zeros(self.n) if velocities is None else np.asarray(velocities, dtype=float)
        self.w = np.zeros(self.n)
        self.done = np.zeros(self.n, dtype=bool)
        self.priority = np.linspace(0, 1, self.n) if self.n > 1 else np.array([0.5])
        humans = humans or []
        self.nppl = len(humans)
        self.hp = np.asarray([h[0] for h in humans], dtype=float).reshape((-1, 2)) if humans else np.zeros((0, 2))
        self.hv = np.asarray([h[1] for h in humans], dtype=float).reshape((-1, 2)) if humans else np.zeros((0, 2))


def test_clear_path_moves_forward():
    w = World([[-5, 0]], [[-4, 0]], [0.0])
    c = AStarORCADD(w, 0, BeastORCAConfig(command_speed_samples=5, command_omega_samples=7))
    a = c.action(w)
    assert a.shape == (2,)
    assert a[0] > -0.5
    assert np.all(np.isfinite(a))
    assert np.all(np.abs(a) <= 1.00001)


def test_head_on_commands_are_not_both_full_speed_straight():
    w = World([[-1, 0], [1, 0]], [[4, 0], [-4, 0]], [0.0, math.pi], velocities=[1.0, 1.0])
    cfg = BeastORCAConfig(command_speed_samples=6, command_omega_samples=9, time_horizon=2.0)
    c0 = AStarORCADD(w, 0, cfg)
    c1 = AStarORCADD(w, 1, cfg)
    a0 = c0.action(w)
    a1 = c1.action(w)
    full_straight = a0[0] > 0.95 and abs(a0[1]) < 0.05 and a1[0] > 0.95 and abs(a1[1]) < 0.05
    assert not full_straight


def test_pedestrian_crossing_triggers_avoidance_or_yield():
    w = World([[0, 0]], [[5, 0]], [0.0], velocities=[1.0], humans=[([1.0, -0.5], [0.0, 0.6])])
    cfg = BeastORCAConfig(command_speed_samples=6, command_omega_samples=9, time_horizon=2.0)
    c = AStarORCADD(w, 0, cfg)
    a = c.action(w)
    assert not (a[0] > 0.95 and abs(a[1]) < 0.05)
    assert c.diagnostics()["orca_constraints_total"] >= 1


def test_close_noncolliding_pair_can_choose_separating_motion():
    w = World(
        [[0.0, -1.5], [0.65, -1.5]],
        [[-2.0, -1.5], [2.0, -1.5]],
        [math.pi, 0.0],
    )
    cfg = BeastORCAConfig(
        peer_margin=0.08,
        command_speed_samples=6,
        command_omega_samples=9,
    )
    c0 = AStarORCADD(w, 0, cfg)
    a0 = c0.action(w)
    assert a0[0] > -0.9
