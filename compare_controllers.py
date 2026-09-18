"""Paired Aegis-RL vs A*+DWA benchmark and representative video exporter."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from classical_baseline import AStarDWAController, DWAConfig


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    controller: str
    outcome: str
    steps: int
    sim_time: float
    path_length: float
    min_clearance: float
    mean_abs_omega: float
    smoothness: float
    collision_type: str
    final_distance: float
    cbf_interventions: int = 0


@dataclass(frozen=True)
class PairedResult:
    seed: int
    aegis: EpisodeResult
    baseline: EpisodeResult


@dataclass(frozen=True)
class RepresentativeWin:
    seed: int
    category: str
    reason: str
    score: Tuple[float, ...]
    clearance_gain: float
    time_gain: float
    path_gain: float
    smoothness_gain: float


@dataclass
class EpisodeTrace:
    frames: List[np.ndarray]
    labels: List[str]
    result: EpisodeResult


def clone_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def path_length(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1).sum())


def rank_aegis_wins(
    pairs: Sequence[PairedResult],
    *,
    top_k: int = 3,
    clearance_threshold: float = 0.15,
    time_threshold: float = 1.0,
    path_threshold: float = 0.50,
) -> List[RepresentativeWin]:
    ranked: list[RepresentativeWin] = []

    for pair in pairs:
        a = pair.aegis
        b = pair.baseline
        clearance_gain = a.min_clearance - b.min_clearance
        time_gain = b.sim_time - a.sim_time
        path_gain = b.path_length - a.path_length
        smoothness_gain = b.smoothness - a.smoothness

        if a.outcome == "success" and b.outcome != "success":
            category = "success_vs_failure"
            reason = f"Aegis succeeds while A*+DWA ends in {b.outcome}."
            failure_rank = {
                "collision": 3.0,
                "planner_failure": 2.5,
                "timeout": 2.0,
            }.get(b.outcome, 1.0)
            score = (
                4.0,
                failure_rank,
                clearance_gain,
                time_gain,
                path_gain,
                smoothness_gain,
            )
        elif (
            a.outcome == "success"
            and b.outcome == "success"
            and clearance_gain >= clearance_threshold
        ):
            category = "clearance"
            reason = (
                "Both succeed; Aegis minimum clearance is higher by "
                f"{clearance_gain:.2f} m."
            )
            score = (
                3.0,
                clearance_gain,
                time_gain,
                path_gain,
                smoothness_gain,
            )
        elif (
            a.outcome == "success"
            and b.outcome == "success"
            and (
                time_gain >= time_threshold
                or path_gain >= path_threshold
            )
        ):
            category = "efficiency"
            reason = (
                "Both succeed; Aegis is more efficient "
                f"(time gain {time_gain:.1f} s, "
                f"path gain {path_gain:.2f} m)."
            )
            score = (
                2.0,
                max(
                    time_gain / max(time_threshold, 1e-9),
                    path_gain / max(path_threshold, 1e-9),
                ),
                clearance_gain,
                smoothness_gain,
            )
        elif (
            a.outcome == "success"
            and b.outcome == "success"
            and smoothness_gain > 0.25
        ):
            category = "smoothness"
            reason = (
                "Both succeed; Aegis has lower control variation by "
                f"{smoothness_gain:.2f}."
            )
            score = (
                1.0,
                smoothness_gain,
                clearance_gain,
                time_gain,
                path_gain,
            )
        else:
            continue

        ranked.append(
            RepresentativeWin(
                seed=pair.seed,
                category=category,
                reason=reason,
                score=tuple(float(value) for value in score),
                clearance_gain=float(clearance_gain),
                time_gain=float(time_gain),
                path_gain=float(path_gain),
                smoothness_gain=float(smoothness_gain),
            )
        )

    ranked.sort(key=lambda row: row.score, reverse=True)
    return ranked[: max(0, int(top_k))]


def _compose_frame(
    left: np.ndarray,
    right: np.ndarray,
    *,
    label: str,
    left_text: str = "A* + DWA",
    right_text: str = "AEGIS-RL (SAC + CBF)",
) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12.8, 6.8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.10, 0.90],
        hspace=0.02,
        wspace=0.02,
    )

    header = fig.add_subplot(grid[0, :])
    header.axis("off")
    header.text(
        0.5,
        0.62,
        label,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    header.text(0.25, 0.08, left_text, ha="center", va="bottom", fontsize=11)
    header.text(0.75, 0.08, right_text, ha="center", va="bottom", fontsize=11)

    for axis, frame in (
        (fig.add_subplot(grid[1, 0]), left),
        (fig.add_subplot(grid[1, 1]), right),
    ):
        axis.imshow(np.asarray(frame, dtype=np.uint8))
        axis.axis("off")

    canvas.draw()
    output = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
    fig.clear()
    return output


def _summary_frame(
    summary: str,
    *,
    width: int = 1280,
    height: int = 680,
) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axis = fig.add_subplot(111)
    axis.axis("off")
    axis.text(
        0.5,
        0.78,
        "Measured result",
        ha="center",
        fontsize=22,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.52,
        summary,
        ha="center",
        va="center",
        fontsize=16,
        linespacing=1.6,
    )
    axis.text(
        0.5,
        0.16,
        "Selected representative benchmark scenario",
        ha="center",
        fontsize=11,
    )
    canvas.draw()
    output = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
    fig.clear()
    return output


def write_side_by_side_video(
    left_frames: Sequence[np.ndarray],
    right_frames: Sequence[np.ndarray],
    output_path: Path | str,
    *,
    fps: int,
    writer_factory: Optional[Callable[..., Any]] = None,
    label: str = "Selected representative benchmark scenario",
    left_labels: Optional[Sequence[str]] = None,
    right_labels: Optional[Sequence[str]] = None,
    final_summary: Optional[str] = None,
) -> Path:
    if not left_frames or not right_frames:
        raise ValueError("both frame streams must contain at least one frame")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if writer_factory is None:
        import imageio.v2 as imageio

        writer_factory = imageio.get_writer

    writer = writer_factory(str(output_path), fps=int(fps))
    try:
        frame_count = max(len(left_frames), len(right_frames))
        for index in range(frame_count):
            left = left_frames[min(index, len(left_frames) - 1)]
            right = right_frames[min(index, len(right_frames) - 1)]
            left_text = (
                left_labels[min(index, len(left_labels) - 1)]
                if left_labels
                else "A* + DWA"
            )
            right_text = (
                right_labels[min(index, len(right_labels) - 1)]
                if right_labels
                else "AEGIS-RL (SAC + CBF)"
            )
            writer.append_data(
                _compose_frame(
                    left,
                    right,
                    label=label,
                    left_text=left_text,
                    right_text=right_text,
                )
            )

        if final_summary:
            card = _summary_frame(final_summary)
            for _ in range(max(1, int(fps) * 3)):
                writer.append_data(card)
    finally:
        writer.close()

    return output_path


def _load_config() -> dict[str, Any]:
    from config import (
        ACCEL_MAX,
        ALPHA_MAX,
        DT,
        DYNAMIC_OBS_RADIUS,
        MAP_MAX_X,
        MAP_MAX_Y,
        MAP_MIN_X,
        MAP_MIN_Y,
        OMEGA_MAX,
        OMEGA_MIN,
        ROBOT_RADIUS,
        SHELVES,
        V_MAX,
        V_MIN,
    )

    return locals()


def make_baseline_controller(
    *,
    dwa_config: Optional[DWAConfig] = None,
    grid_resolution: float = 0.25,
) -> AStarDWAController:
    config = _load_config()
    return AStarDWAController(
        map_bounds=(
            config["MAP_MIN_X"],
            config["MAP_MAX_X"],
            config["MAP_MIN_Y"],
            config["MAP_MAX_Y"],
        ),
        shelves=config["SHELVES"],
        robot_radius=config["ROBOT_RADIUS"],
        dynamic_obstacle_radius=config["DYNAMIC_OBS_RADIUS"],
        v_min=config["V_MIN"],
        v_max=config["V_MAX"],
        omega_min=config["OMEGA_MIN"],
        omega_max=config["OMEGA_MAX"],
        accel_max=config["ACCEL_MAX"],
        alpha_max=config["ALPHA_MAX"],
        dt=config["DT"],
        config=dwa_config,
        grid_resolution=grid_resolution,
    )


def load_aegis_agent(
    checkpoint_path: Path | str,
    *,
    device: str = "cpu",
) -> Any:
    import torch
    from environment import AMRWarehouseEnv
    from safe_sac import SafeSACAgent, SafeSACConfig

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    env = AMRWarehouseEnv()
    try:
        agent = SafeSACAgent(
            policy_config=checkpoint["policy_config"],
            observation_space=env.observation_space,
            action_dim=2,
            sac_config=SafeSACConfig(device=device),
            replay_buffer_size=10,
        )
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.eval()
        return agent
    finally:
        env.close()


def _initial_obs_from_snapshot(
    snapshot: dict[str, Any],
) -> dict[str, np.ndarray]:
    buffer = snapshot.get("obs_buffer") or []
    if not buffer:
        raise ValueError("snapshot does not contain an observation buffer")
    return {
        key: np.copy(value)
        for key, value in buffer[0].items()
    }


def _episode_result(
    *,
    seed: int,
    controller: str,
    outcome: str,
    env: Any,
    omegas: Sequence[float],
    smoothness: float,
    collision_type: str,
    min_clearance: float,
    cbf_interventions: int,
) -> EpisodeResult:
    config = _load_config()
    final_distance = float(
        np.linalg.norm(
            np.asarray(env.robot_state[:2])
            - np.asarray(env.goal_pos)
        )
    )

    return EpisodeResult(
        seed=int(seed),
        controller=controller,
        outcome=outcome,
        steps=int(env.step_count),
        sim_time=float(env.step_count * config["DT"]),
        path_length=path_length(list(env.trajectory_history)),
        min_clearance=float(
            min_clearance
            if np.isfinite(min_clearance)
            else 0.0
        ),
        mean_abs_omega=(
            float(np.mean(np.abs(omegas)))
            if omegas
            else 0.0
        ),
        smoothness=float(smoothness),
        collision_type=collision_type,
        final_distance=final_distance,
        cbf_interventions=int(cbf_interventions),
    )


def _rollout_aegis(
    agent: Any,
    env: Any,
    snapshot: dict[str, Any],
    *,
    seed: int,
    capture_frames: bool,
) -> EpisodeTrace:
    env.set_state(clone_snapshot(snapshot))
    obs = _initial_obs_from_snapshot(snapshot)

    frames: list[np.ndarray] = []
    labels: list[str] = []
    min_clearance = float("inf")
    cbf_interventions = 0
    omegas: list[float] = []
    smoothness = 0.0
    previous = np.array([0.0, 0.0], dtype=float)
    collision_type = "none"
    outcome = "timeout"
    dt = _load_config()["DT"]

    if capture_frames:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
            labels.append("AEGIS-RL | t=0.0s")

    while True:
        action = np.asarray(
            agent.select_action(obs, deterministic=True),
            dtype=np.float32,
        )
        obs, _, terminated, truncated, info = env.step(action)

        v_actual = float(info.get("v_actual", 0.0))
        omega_actual = float(info.get("omega_actual", 0.0))
        omegas.append(omega_actual)

        current = np.array(
            [v_actual, omega_actual],
            dtype=float,
        )
        smoothness += float(
            np.abs(current - previous).sum()
        )
        previous = current

        min_clearance = min(
            min_clearance,
            float(info.get("barrier_value", float("inf"))),
        )
        cbf_interventions += int(
            info.get("cbf_intervened", False)
        )

        if capture_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                labels.append(
                    "AEGIS-RL | "
                    f"t={env.step_count * dt:.1f}s | "
                    f"CBF={cbf_interventions}"
                )

        if info.get("goal_reached", False):
            outcome = "success"
            break
        if info.get("collision", False):
            outcome = "collision"
            collision_type = str(
                info.get("collision_type", "unknown")
            )
            break
        if terminated or truncated:
            outcome = "timeout"
            break

    result = _episode_result(
        seed=seed,
        controller="aegis",
        outcome=outcome,
        env=env,
        omegas=omegas,
        smoothness=smoothness,
        collision_type=collision_type,
        min_clearance=min_clearance,
        cbf_interventions=cbf_interventions,
    )
    return EpisodeTrace(
        frames=frames,
        labels=labels,
        result=result,
    )


def _rollout_baseline(
    env: Any,
    snapshot: dict[str, Any],
    *,
    seed: int,
    capture_frames: bool,
    dwa_config: Optional[DWAConfig],
    grid_resolution: float,
) -> EpisodeTrace:
    env.set_state(clone_snapshot(snapshot))
    controller = make_baseline_controller(
        dwa_config=dwa_config,
        grid_resolution=grid_resolution,
    )

    if not controller.reset(
        tuple(env.robot_state[:2]),
        tuple(env.goal_pos),
    ):
        result = _episode_result(
            seed=seed,
            controller="astar_dwa",
            outcome="planner_failure",
            env=env,
            omegas=[],
            smoothness=0.0,
            collision_type="none",
            min_clearance=float("inf"),
            cbf_interventions=0,
        )
        frame = env.render() if capture_frames else None
        return EpisodeTrace(
            frames=[frame] if frame is not None else [],
            labels=(
                ["A* + DWA | planner failure"]
                if frame is not None
                else []
            ),
            result=result,
        )

    frames: list[np.ndarray] = []
    labels: list[str] = []
    min_clearance = float("inf")
    omegas: list[float] = []
    smoothness = 0.0
    previous = np.array([0.0, 0.0], dtype=float)
    collision_type = "none"
    outcome = "timeout"
    dt = _load_config()["DT"]

    if capture_frames:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
            labels.append("A* + DWA | t=0.0s")

    while True:
        action = controller.command(
            env.robot_state,
            env.dynamic_obstacles,
        )
        _, _, terminated, truncated, info = env.step(action)

        v_actual = float(info.get("v_actual", 0.0))
        omega_actual = float(info.get("omega_actual", 0.0))
        omegas.append(omega_actual)

        current = np.array(
            [v_actual, omega_actual],
            dtype=float,
        )
        smoothness += float(
            np.abs(current - previous).sum()
        )
        previous = current

        min_clearance = min(
            min_clearance,
            float(info.get("barrier_value", float("inf"))),
        )

        if capture_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                labels.append(
                    f"A* + DWA | t={env.step_count * dt:.1f}s"
                )

        if info.get("goal_reached", False):
            outcome = "success"
            break
        if info.get("collision", False):
            outcome = "collision"
            collision_type = str(
                info.get("collision_type", "unknown")
            )
            break
        if terminated or truncated:
            outcome = "timeout"
            break

    result = _episode_result(
        seed=seed,
        controller="astar_dwa",
        outcome=outcome,
        env=env,
        omegas=omegas,
        smoothness=smoothness,
        collision_type=collision_type,
        min_clearance=min_clearance,
        cbf_interventions=0,
    )
    return EpisodeTrace(
        frames=frames,
        labels=labels,
        result=result,
    )


def run_paired_seed(
    agent: Any,
    seed: int,
    *,
    capture_frames: bool = False,
    dwa_config: Optional[DWAConfig] = None,
    grid_resolution: float = 0.25,
) -> tuple[
    PairedResult,
    Optional[EpisodeTrace],
    Optional[EpisodeTrace],
]:
    from environment import AMRWarehouseEnv

    render_mode = (
        "rgb_array"
        if capture_frames
        else None
    )
    aegis_env = AMRWarehouseEnv(
        render_mode=render_mode,
        use_cbf_filter=True,
    )
    baseline_env = AMRWarehouseEnv(
        render_mode=render_mode,
        use_cbf_filter=False,
    )

    try:
        aegis_env.reset(seed=int(seed))
        snapshot = aegis_env.get_state()
        baseline_env.reset(seed=int(seed))

        aegis_trace = _rollout_aegis(
            agent,
            aegis_env,
            snapshot,
            seed=seed,
            capture_frames=capture_frames,
        )
        baseline_trace = _rollout_baseline(
            baseline_env,
            snapshot,
            seed=seed,
            capture_frames=capture_frames,
            dwa_config=dwa_config,
            grid_resolution=grid_resolution,
        )

        pair = PairedResult(
            seed=int(seed),
            aegis=aegis_trace.result,
            baseline=baseline_trace.result,
        )
        return (
            pair,
            aegis_trace if capture_frames else None,
            baseline_trace if capture_frames else None,
        )
    finally:
        aegis_env.close()
        baseline_env.close()


def write_benchmark_outputs(
    pairs: Sequence[PairedResult],
    ranked: Sequence[RepresentativeWin],
    output_dir: Path | str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pair in pairs:
        rows.append(asdict(pair.aegis))
        rows.append(asdict(pair.baseline))

    if rows:
        with (
            output_dir / "episode_results.csv"
        ).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    rank_rows = []
    for row in ranked:
        encoded = asdict(row)
        encoded["score"] = json.dumps(list(row.score))
        rank_rows.append(encoded)

    with (
        output_dir / "representative_aegis_wins.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        if rank_rows:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rank_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rank_rows)
        else:
            handle.write("seed,category,reason\n")

    def aggregate(controller: str) -> dict[str, float]:
        results = [
            (
                pair.aegis
                if controller == "aegis"
                else pair.baseline
            )
            for pair in pairs
        ]
        denominator = max(1, len(results))
        successes = [
            result
            for result in results
            if result.outcome == "success"
        ]
        return {
            "episodes": len(results),
            "success_rate": (
                sum(
                    result.outcome == "success"
                    for result in results
                )
                / denominator
            ),
            "collision_rate": (
                sum(
                    result.outcome == "collision"
                    for result in results
                )
                / denominator
            ),
            "timeout_rate": (
                sum(
                    result.outcome == "timeout"
                    for result in results
                )
                / denominator
            ),
            "mean_success_time": (
                float(
                    np.mean(
                        [
                            result.sim_time
                            for result in successes
                        ]
                    )
                )
                if successes
                else float("nan")
            ),
            "mean_success_path": (
                float(
                    np.mean(
                        [
                            result.path_length
                            for result in successes
                        ]
                    )
                )
                if successes
                else float("nan")
            ),
            "mean_min_clearance": (
                float(
                    np.mean(
                        [
                            result.min_clearance
                            for result in results
                        ]
                    )
                )
                if results
                else float("nan")
            ),
            "mean_smoothness": (
                float(
                    np.mean(
                        [
                            result.smoothness
                            for result in results
                        ]
                    )
                )
                if results
                else float("nan")
            ),
        }

    with (
        output_dir / "aggregate_results.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "aegis": aggregate("aegis"),
                "astar_dwa": aggregate("astar_dwa"),
            },
            handle,
            indent=2,
            allow_nan=True,
        )


def _final_summary(
    pair: PairedResult,
    reason: str,
) -> str:
    a = pair.aegis
    b = pair.baseline
    return (
        "A* + DWA: "
        f"{b.outcome} | {b.sim_time:.1f}s | "
        f"path {b.path_length:.2f}m | "
        f"min clearance {b.min_clearance:.2f}m\n"
        "AEGIS-RL: "
        f"{a.outcome} | {a.sim_time:.1f}s | "
        f"path {a.path_length:.2f}m | "
        f"min clearance {a.min_clearance:.2f}m | "
        f"CBF {a.cbf_interventions}\n\n"
        f"Why selected: {reason}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Aegis-RL against A* + DWA "
            "on identical warehouse scenarios."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_results"),
    )
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--dwa-horizon",
        type=float,
        default=1.5,
    )
    args = parser.parse_args()

    agent = load_aegis_agent(
        args.checkpoint,
        device=args.device,
    )
    dwa_config = DWAConfig(
        horizon=float(args.dwa_horizon)
    )

    pairs: list[PairedResult] = []
    for seed in range(
        args.seed_start,
        args.seed_start + args.num_seeds,
    ):
        pair, _, _ = run_paired_seed(
            agent,
            seed,
            capture_frames=False,
            dwa_config=dwa_config,
            grid_resolution=args.grid_resolution,
        )
        pairs.append(pair)
        print(
            f"seed={seed} "
            f"aegis={pair.aegis.outcome} "
            f"astar_dwa={pair.baseline.outcome}"
        )

    ranked = rank_aegis_wins(
        pairs,
        top_k=args.render_top_k,
    )
    write_benchmark_outputs(
        pairs,
        ranked,
        args.output_dir,
    )

    for index, win in enumerate(ranked, start=1):
        pair, aegis_trace, baseline_trace = run_paired_seed(
            agent,
            win.seed,
            capture_frames=True,
            dwa_config=dwa_config,
            grid_resolution=args.grid_resolution,
        )
        assert aegis_trace is not None
        assert baseline_trace is not None

        write_side_by_side_video(
            baseline_trace.frames,
            aegis_trace.frames,
            args.output_dir
            / f"selected_{index:02d}_seed_{win.seed}.mp4",
            fps=args.fps,
            left_labels=baseline_trace.labels,
            right_labels=aegis_trace.labels,
            final_summary=_final_summary(
                pair,
                win.reason,
            ),
        )
        print(
            f"video seed={win.seed}: {win.reason}"
        )

    if not ranked:
        print(
            "No Aegis-win scenario met the configured "
            "representative thresholds; aggregate results "
            "were still saved."
        )


if __name__ == "__main__":
    main()
