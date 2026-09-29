import numpy as np

from compare_controllers import (
    EpisodeResult,
    PairedResult,
    clone_snapshot,
    path_length,
    rank_aegis_wins,
    write_side_by_side_video,
)


def _episode(seed, controller, outcome, time_s, path_m, clearance, smoothness=1.0):
    return EpisodeResult(
        seed=seed,
        controller=controller,
        outcome=outcome,
        steps=int(round(time_s / 0.1)),
        sim_time=time_s,
        path_length=path_m,
        min_clearance=clearance,
        mean_abs_omega=0.2,
        smoothness=smoothness,
        collision_type="none" if outcome != "collision" else "dynamic_obstacle",
        final_distance=0.1 if outcome == "success" else 2.0,
        cbf_interventions=4 if controller == "aegis" else 0,
    )


def test_path_length_accumulates_segments():
    assert path_length([(0.0, 0.0), (3.0, 4.0), (3.0, 8.0)]) == 9.0


def test_clone_snapshot_is_deep_copy():
    state = {
        "robot_state": np.array([1.0, 2.0]),
        "obs_buffer": [{"lidar": np.array([0.1, 0.2])}],
        "rng_state": {"state": {"x": 7}},
    }
    cloned = clone_snapshot(state)
    cloned["robot_state"][0] = 99
    cloned["obs_buffer"][0]["lidar"][0] = 99
    cloned["rng_state"]["state"]["x"] = 99
    assert state["robot_state"][0] == 1.0
    assert state["obs_buffer"][0]["lidar"][0] == 0.1
    assert state["rng_state"]["state"]["x"] == 7


def test_rank_prioritizes_success_vs_failure_then_clearance_win():
    pairs = [
        PairedResult(
            seed=10,
            aegis=_episode(10, "aegis", "success", 18.0, 11.0, 0.42),
            baseline=_episode(10, "astar_dwa", "collision", 8.0, 5.0, -0.01),
        ),
        PairedResult(
            seed=11,
            aegis=_episode(11, "aegis", "success", 20.0, 12.0, 0.55),
            baseline=_episode(11, "astar_dwa", "success", 19.0, 11.8, 0.25),
        ),
        PairedResult(
            seed=12,
            aegis=_episode(12, "aegis", "timeout", 50.0, 10.0, 0.5),
            baseline=_episode(12, "astar_dwa", "success", 15.0, 9.0, 0.3),
        ),
    ]
    ranked = rank_aegis_wins(pairs, top_k=3)
    assert [r.seed for r in ranked] == [10, 11]
    assert "succeeds" in ranked[0].reason.lower()
    assert "clearance" in ranked[1].reason.lower()


class _FakeWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(np.asarray(frame))

    def close(self):
        self.closed = True


def test_video_writer_pairs_frames_and_holds_last_frame(tmp_path):
    left = [
        np.zeros((20, 30, 3), dtype=np.uint8),
        np.ones((20, 30, 3), dtype=np.uint8) * 20,
    ]
    right = [np.ones((20, 30, 3), dtype=np.uint8) * 100]
    fake = _FakeWriter()
    out = write_side_by_side_video(
        left,
        right,
        tmp_path / "demo.mp4",
        fps=10,
        writer_factory=lambda *_args, **_kwargs: fake,
        label="Selected representative benchmark scenario",
    )
    assert out == tmp_path / "demo.mp4"
    assert fake.closed
    assert len(fake.frames) == 2
    assert fake.frames[0].shape[1] > 60
    assert np.mean(fake.frames[1][:, fake.frames[1].shape[1] // 2 :, :]) > 50
