"""visualizer.py

Standalone, publication-quality plotting utilities for the Safe RL AMR
project. Kept independent of Gymnasium/PyTorch so it can be imported from a
notebook or a lightweight post-processing script without pulling in the
full training stack.

Two entry points:
    * ``plot_trajectory``: draws the warehouse layout, the robot's realized
      path, dynamic-obstacle snapshot positions, and the goal, for a single
      evaluation episode.
    * ``plot_training_curves``: reads a TensorBoard event-file directory
      (as written by ``train.py``'s ``SummaryWriter``) and plots Success
      Rate and mean CBF interventions over training, on twin y-axes.

Both functions use ``config.VISUALIZATION_CONFIG`` for colors and figure
sizing so rendered figures stay visually consistent with
``AMRWarehouseEnv.render()``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from config import (
    DYNAMIC_OBS_RADIUS,
    GOAL_TOLERANCE,
    MAP_MAX_X,
    MAP_MAX_Y,
    MAP_MIN_X,
    MAP_MIN_Y,
    ROBOT_RADIUS,
    VISUALIZATION_CONFIG,
)

RectBounds = Tuple[float, float, float, float]
Point2D = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Trajectory plotting
# --------------------------------------------------------------------------- #

def plot_trajectory(
    trajectory_history: Sequence[Point2D],
    obstacles: Optional[Sequence[Sequence[float]]],
    shelves: Sequence[RectBounds],
    goal: Point2D,
    save_path: Union[str, Path],
    title: str = "AMR Trajectory",
    dpi: int = 300,
) -> None:
    """Renders a single episode's trajectory over the warehouse layout.

    Draws the map boundary, static shelf racks, the goal (with its
    tolerance radius), a snapshot of dynamic-obstacle positions, and the
    robot's realized path (start marked with a circle, end with a star).
    Saved directly to disk as a high-resolution PNG or PDF (chosen from
    ``save_path``'s suffix) -- no interactive window is opened, so this is
    safe to call from a headless training/evaluation script.

    Args:
        trajectory_history: Sequence of ``(x, y)`` robot positions for one
            episode, in chronological order (e.g.
            ``env.trajectory_history`` after an episode completes).
        obstacles: Optional sequence of dynamic-obstacle rows (each at
            least ``[x, y, ...]``), representing a snapshot of obstacle
            positions to draw (e.g. their final positions at episode end,
            or positions at a specific timestep). Pass ``None`` or an empty
            sequence to omit obstacles entirely.
        shelves: Sequence of static shelf rectangles as
            ``(xmin, ymin, xmax, ymax)`` (e.g. ``config.SHELVES``).
        goal: ``(x, y)`` goal position for this episode.
        save_path: Destination file path. The extension (``.png``,
            ``.pdf``, ``.svg``, ...) determines the saved format.
        title: Figure title.
        dpi: Resolution (dots per inch) for raster output formats.
    """
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.FIGURE_SIZE)

    ax.set_xlim(MAP_MIN_X - 0.5, MAP_MAX_X + 0.5)
    ax.set_ylim(MAP_MIN_Y - 0.5, MAP_MAX_Y + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=13, fontweight="bold")

    # -- Map boundary. -- #
    ax.add_patch(
        Rectangle(
            (MAP_MIN_X, MAP_MIN_Y),
            MAP_MAX_X - MAP_MIN_X,
            MAP_MAX_Y - MAP_MIN_Y,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
        )
    )

    # -- Static shelves. -- #
    for rect in shelves:
        ax.add_patch(
            Rectangle(
                (rect[0], rect[1]),
                rect[2] - rect[0],
                rect[3] - rect[1],
                facecolor=VISUALIZATION_CONFIG.COLOR_SHELF,
                edgecolor="black",
                alpha=0.85,
                zorder=2,
            )
        )

    # -- Goal + tolerance ring. -- #
    ax.add_patch(
        Circle(
            goal,
            GOAL_TOLERANCE,
            facecolor=VISUALIZATION_CONFIG.COLOR_GOAL,
            edgecolor="darkgreen",
            alpha=0.6,
            zorder=3,
            label="Goal",
        )
    )

    # -- Dynamic obstacle snapshot. -- #
    if obstacles is not None and len(obstacles) > 0:
        for idx, obs in enumerate(obstacles):
            ax.add_patch(
                Circle(
                    (obs[0], obs[1]),
                    DYNAMIC_OBS_RADIUS,
                    facecolor=VISUALIZATION_CONFIG.COLOR_DYNAMIC_OBS,
                    edgecolor="maroon",
                    alpha=0.7,
                    zorder=4,
                    label="Dynamic obstacle" if idx == 0 else None,
                )
            )

    # -- Robot trajectory. -- #
    if len(trajectory_history) > 0:
        traj = np.asarray(trajectory_history, dtype=np.float64)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=VISUALIZATION_CONFIG.COLOR_TRAJECTORY,
            linewidth=2.0,
            alpha=0.9,
            zorder=5,
            label="Robot path",
        )
        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            color=VISUALIZATION_CONFIG.COLOR_ROBOT,
            edgecolor="navy",
            s=120,
            marker="o",
            zorder=6,
            label="Start",
        )
        ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            color=VISUALIZATION_CONFIG.COLOR_ROBOT,
            edgecolor="navy",
            s=180,
            marker="*",
            zorder=6,
            label="End",
        )
        ax.add_patch(
            Circle(
                (traj[-1, 0], traj[-1, 1]),
                ROBOT_RADIUS,
                fill=False,
                edgecolor=VISUALIZATION_CONFIG.COLOR_SAFETY_BARRIER,
                linestyle=":",
                linewidth=1.5,
                zorder=6,
            )
        )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Training-curve plotting
# --------------------------------------------------------------------------- #

def _load_scalar_series(log_dir: Union[str, Path], tag: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a single scalar tag's (step, value) series from TensorBoard event files.

    Uses ``tensorboard``'s bundled ``EventAccumulator`` (installed as a
    dependency of ``torch.utils.tensorboard``, which ``train.py`` already
    requires -- no extra package needed).

    Args:
        log_dir: Directory containing TensorBoard event files (the same
            path passed to ``SummaryWriter`` in ``train.py``).
        tag: Scalar tag name (e.g. ``"eval/success_rate"``).

    Returns:
        Tuple ``(steps, values)`` as 1-D NumPy arrays, sorted by step.
        Empty arrays if the tag is not present in the logs.

    Raises:
        ImportError: If the ``tensorboard`` package is not installed.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise ImportError(
            "Reading TensorBoard event files requires the 'tensorboard' package "
            "(installed alongside torch.utils.tensorboard). Install it with "
            "`pip install tensorboard`."
        ) from exc

    accumulator = EventAccumulator(
        str(log_dir), size_guidance={"scalars": 0}  # 0 == load all scalar events.
    )
    accumulator.Reload()

    available_tags = accumulator.Tags().get("scalars", [])
    if tag not in available_tags:
        return np.array([]), np.array([])

    events = accumulator.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.int64)
    values = np.array([e.value for e in events], dtype=np.float64)
    order = np.argsort(steps)
    return steps[order], values[order]


def plot_training_curves(
    log_dir: Union[str, Path],
    save_path: Union[str, Path],
    success_tag: str = "eval/success_rate",
    cbf_tag: str = "eval/mean_cbf_interventions",
    title: str = "Training Progress: Success Rate vs. CBF Interventions",
    dpi: int = 300,
) -> None:
    """Plots evaluation Success Rate and mean CBF interventions over training.

    Reads the scalar tags logged by ``train.py``'s ``evaluate_agent`` calls
    (``eval/success_rate`` and ``eval/mean_cbf_interventions`` by default)
    directly from the TensorBoard event files in ``log_dir``, and plots
    them on twin y-axes against the shared environment-step x-axis.

    Args:
        log_dir: Directory containing the run's TensorBoard event files
            (i.e. the ``log_dir`` passed to ``SummaryWriter`` in
            ``train.py``).
        save_path: Destination file path for the saved figure.
        success_tag: TensorBoard scalar tag for success rate.
        cbf_tag: TensorBoard scalar tag for mean CBF interventions.
        title: Figure title.
        dpi: Resolution (dots per inch) for raster output formats.

    Raises:
        ImportError: If the ``tensorboard`` package is not installed.
        FileNotFoundError: If neither requested tag has any logged data
            (nothing to plot).
    """
    success_steps, success_values = _load_scalar_series(log_dir, success_tag)
    cbf_steps, cbf_values = _load_scalar_series(log_dir, cbf_tag)

    if success_steps.size == 0 and cbf_steps.size == 0:
        raise FileNotFoundError(
            f"No data found for tags {success_tag!r} or {cbf_tag!r} under {log_dir}. "
            "Confirm this is a valid train.py TensorBoard log directory."
        )

    fig, ax_left = plt.subplots(figsize=VISUALIZATION_CONFIG.FIGURE_SIZE)
    ax_right = ax_left.twinx()

    if success_steps.size > 0:
        ax_left.plot(
            success_steps,
            success_values * 100.0,
            color=VISUALIZATION_CONFIG.COLOR_GOAL,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Success rate (%)",
        )
    ax_left.set_xlabel("Environment steps")
    ax_left.set_ylabel("Success rate (%)", color=VISUALIZATION_CONFIG.COLOR_GOAL)
    ax_left.set_ylim(0.0, 100.0)
    ax_left.tick_params(axis="y", labelcolor=VISUALIZATION_CONFIG.COLOR_GOAL)

    if cbf_steps.size > 0:
        ax_right.plot(
            cbf_steps,
            cbf_values,
            color=VISUALIZATION_CONFIG.COLOR_SAFETY_BARRIER,
            linewidth=2.0,
            marker="s",
            markersize=4,
            linestyle="--",
            label="Mean CBF interventions / episode",
        )
    ax_right.set_ylabel(
        "Mean CBF interventions / episode", color=VISUALIZATION_CONFIG.COLOR_SAFETY_BARRIER
    )
    ax_right.tick_params(axis="y", labelcolor=VISUALIZATION_CONFIG.COLOR_SAFETY_BARRIER)

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="lower right", fontsize=9)

    ax_left.set_title(title, fontsize=13, fontweight="bold")
    ax_left.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Optional CLI (quick ad-hoc plotting without writing a script)
# --------------------------------------------------------------------------- #

def _parse_cli_args() -> argparse.Namespace:
    """Parses command-line arguments for ad-hoc training-curve plotting.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Plot Success Rate vs. CBF interventions from a train.py TensorBoard log dir."
    )
    parser.add_argument("--log-dir", type=str, required=True, help="TensorBoard event-file directory.")
    parser.add_argument(
        "--save-path", type=str, default="training_curves.png", help="Output figure path."
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_cli_args()
    plot_training_curves(cli_args.log_dir, cli_args.save_path)
    print(f"[visualizer.py] Saved training curves to {cli_args.save_path}")