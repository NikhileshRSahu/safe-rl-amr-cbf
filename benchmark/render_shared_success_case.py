from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmark.beast_classical import AStarORCADD
from benchmark.beast_config import load_beast_config
from benchmark.train_multi_agent_research import DT, SHELVES, WORLD, World


HUMAN_RADIUS = 0.28
HUMAN_MIN_SEPARATION = 2.0 * HUMAN_RADIUS + 0.08
HUMAN_MAX_SPEED = 0.60
HUMAN_MIN_SPEED = 0.20


def _rect_distance(point, rect):
    x, y = float(point[0]), float(point[1])
    x0, y0, x1, y1 = rect
    cx = min(max(x, x0), x1)
    cy = min(max(y, y0), y1)
    return math.hypot(x - cx, y - cy)


def _rotate(vec, angle):
    c, s = math.cos(angle), math.sin(angle)
    x, y = float(vec[0]), float(vec[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


class RealisticHumanWorld(World):
    """Demo-only World with collision-aware, shelf-aware pedestrian motion.

    The AMR dynamics, controller interfaces, collision definitions, and reward
    logic remain inherited from the research World.  Only pedestrian spawning
    and the velocity chosen immediately before the base-world pedestrian update
    are changed.  This keeps the thesis benchmark untouched while making demo
    pedestrians behave like finite-radius people instead of point particles
    moving through warehouse geometry or one another.
    """

    def reset(self):
        obs = super().reset()
        self._spawn_realistic_humans()
        # Human positions are part of every AMR observation, so rebuild the
        # observations after replacing the base world's random pedestrian set.
        return [self.obs(i) for i in range(self.n)]

    def _human_point_is_free(self, point, extra=0.0):
        radius = HUMAN_RADIUS + float(extra)
        if abs(float(point[0])) > WORLD - radius:
            return False
        if abs(float(point[1])) > WORLD - radius:
            return False
        return all(_rect_distance(point, rect) >= radius for rect in SHELVES)

    def _spawn_realistic_humans(self):
        placed = []
        for idx in range(self.nppl):
            chosen = None
            for _ in range(3000):
                q = self.rng.uniform(-8.5, 8.5, size=2).astype(np.float32)
                if not self._human_point_is_free(q, extra=0.08):
                    continue
                if any(np.linalg.norm(q - p) < HUMAN_MIN_SEPARATION + 0.18 for p in placed):
                    continue
                chosen = q
                break
            if chosen is None:
                raise RuntimeError(f"could not place realistic pedestrian {idx}")
            placed.append(chosen)

        self.hp = np.asarray(placed, dtype=np.float32)
        angles = self.rng.uniform(-math.pi, math.pi, self.nppl)
        speeds = self.rng.uniform(0.28, 0.52, self.nppl)
        self.hv = np.c_[np.cos(angles) * speeds, np.sin(angles) * speeds].astype(np.float32)
        self._human_preferred_speed = speeds.astype(np.float32)

    def _candidate_is_safe(self, person_idx, next_point, planned_points):
        if not self._human_point_is_free(next_point, extra=0.03):
            return False

        max_step = HUMAN_MAX_SPEED * DT
        for j in range(self.nppl):
            if j == person_idx:
                continue
            if j < len(planned_points) and planned_points[j] is not None:
                other = planned_points[j]
                required = HUMAN_MIN_SEPARATION
            else:
                # The unplanned pedestrian can still move toward us by at most
                # one max-speed step, so reserve that distance now.
                other = self.hp[j]
                required = HUMAN_MIN_SEPARATION + max_step
            if np.linalg.norm(next_point - other) < required:
                return False
        return True

    def _choose_human_velocities(self):
        planned_points = [None] * self.nppl
        new_velocities = np.zeros_like(self.hv)

        # A small correlated heading drift gives natural walking rather than
        # perfectly straight scripted particles while remaining deterministic
        # under the episode seed.
        angle_offsets = (0.0, 0.28, -0.28, 0.55, -0.55, 0.85, -0.85, 1.15, -1.15, 1.57, -1.57, math.pi)
        speed_scales = (1.0, 0.82, 0.62, 0.42, 0.0)

        for i in range(self.nppl):
            current = np.asarray(self.hv[i], dtype=np.float32)
            current_speed = float(np.linalg.norm(current))
            if current_speed < 1e-6:
                heading = float(self.rng.uniform(-math.pi, math.pi))
                current = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
            else:
                current = current / current_speed

            wander = float(self.rng.normal(0.0, 0.045))
            desired_dir = _rotate(current, wander)

            # Human-human personal-space repulsion biases the preferred heading
            # before the hard non-overlap check below.
            repulse = np.zeros(2, dtype=np.float32)
            for j in range(self.nppl):
                if i == j:
                    continue
                delta = self.hp[i] - self.hp[j]
                dist = float(np.linalg.norm(delta))
                if 1e-6 < dist < 1.35:
                    repulse += (delta / dist) * ((1.35 - dist) / 1.35)
            desired = desired_dir + 0.75 * repulse
            norm = float(np.linalg.norm(desired))
            if norm > 1e-6:
                desired /= norm
            else:
                desired = desired_dir

            target_speed = float(np.clip(self._human_preferred_speed[i], HUMAN_MIN_SPEED, HUMAN_MAX_SPEED))
            best_velocity = np.zeros(2, dtype=np.float32)
            best_point = self.hp[i].copy()
            best_score = -float("inf")

            for angle in angle_offsets:
                direction = _rotate(desired, angle)
                for scale in speed_scales:
                    speed = target_speed * scale
                    velocity = direction * speed
                    next_point = self.hp[i] + velocity * DT
                    if not self._candidate_is_safe(i, next_point, planned_points):
                        continue
                    # Prefer the intended heading and normal human walking speed;
                    # still allow slowing/stopping when an aisle interaction is tight.
                    alignment = float(np.dot(direction, desired))
                    score = 2.0 * alignment + 0.75 * scale
                    if score > best_score:
                        best_score = score
                        best_velocity = velocity.astype(np.float32)
                        best_point = next_point.astype(np.float32)

            new_velocities[i] = best_velocity
            planned_points[i] = best_point

        self.hv = new_velocities

    def step(self, actions, use_cbf=True):
        self._choose_human_velocities()
        return super().step(actions, use_cbf)


def _snapshot(w):
    return {
        "p": np.asarray(w.p, dtype=np.float32).copy(),
        "th": np.asarray(w.th, dtype=np.float32).copy(),
        "hp": np.asarray(w.hp, dtype=np.float32).copy(),
        "hv": np.asarray(w.hv, dtype=np.float32).copy(),
        "done": np.asarray(w.done, dtype=bool).copy(),
        "hit": np.asarray(w.hit, dtype=bool).copy(),
    }


def _make_world(humans: int, seed: int, realistic_humans: bool):
    cls = RealisticHumanWorld if realistic_humans else World
    return cls(4, humans, seed)


def _run_sac(actor, seed: int, humans: int, realistic_humans: bool):
    import torch

    w = _make_world(humans, seed, realistic_humans)
    obs = w.reset()
    frames = [_snapshot(w)]
    for _ in range(600):
        with torch.no_grad():
            action, _ = actor.sample(
                torch.tensor(np.asarray(obs), dtype=torch.float32), True
            )
        obs, _, done = w.step(action.numpy(), True)
        frames.append(_snapshot(w))
        if np.all(done):
            break
    return w, frames


def _run_orca(config, seed: int, humans: int, realistic_humans: bool):
    w = _make_world(humans, seed, realistic_humans)
    w.reset()
    ctrls = [AStarORCADD(w, i, config) for i in range(4)]
    frames = [_snapshot(w)]
    for _ in range(600):
        acts = np.asarray(
            [ctrls[i].action(w) if not w.done[i] else [-1.0, 0.0] for i in range(4)],
            dtype=np.float32,
        )
        _, _, done = w.step(acts, False)
        frames.append(_snapshot(w))
        if np.all(done):
            break
    return w, frames, [c.diagnostics() for c in ctrls]


def _summary(w, diagnostics=None):
    success = w.done & ~w.hit
    finite = w.min_clearance[np.isfinite(w.min_clearance)]
    out = {
        "fleet_success": bool(success.all()),
        "success": int(success.sum()),
        "collision": int(w.hit.sum()),
        "steps": int(w.steps),
        "min_clearance": float(np.min(finite)) if finite.size else None,
        "path_length_success_mean": float(w.path_length[success].mean()) if success.any() else None,
        "interventions": int(w.interventions.sum()),
    }
    if diagnostics is not None:
        out["orca_constraints_total"] = int(sum(d.get("orca_constraints_total", 0) for d in diagnostics))
        out["stop_yield_ticks"] = int(sum(d.get("stop_yield_ticks", 0) for d in diagnostics))
        out["recovery_count"] = int(sum(d.get("recovery_count", 0) for d in diagnostics))
    return out


def _render(frames, goals, title: str, outfile: Path, stride: int = 3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Rectangle

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.set_xlim(-WORLD, WORLD)
    ax.set_ylim(-WORLD, WORLD)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, linewidth=0.35, alpha=0.35)

    for x0, y0, x1, y1 in SHELVES:
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor="0.35", edgecolor="0.15", alpha=0.85))

    markers = ["o", "s", "^", "D"]
    for i, g in enumerate(goals):
        ax.scatter([g[0]], [g[1]], marker="*", s=130, label=f"Goal {i+1}")

    robot_pts = [ax.plot([], [], markers[i], markersize=10, label=f"AMR {i+1}")[0] for i in range(4)]
    heading_lines = [ax.plot([], [], linewidth=2)[0] for _ in range(4)]
    trails = [ax.plot([], [], linewidth=1.3, alpha=0.7)[0] for _ in range(4)]
    human_pts = ax.scatter([], [], marker="o", s=45, label="Humans")
    human_heading = [ax.plot([], [], linewidth=0.8, alpha=0.75)[0] for _ in range(len(frames[0]["hp"]))]
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    status_text = ax.text(0.02, 0.94, "", transform=ax.transAxes, va="top")
    ax.legend(loc="lower right", fontsize=7, ncol=2)

    history = [[] for _ in range(4)]
    writer = FFMpegWriter(fps=10, metadata={"title": title}, bitrate=2600)
    chosen = list(range(0, len(frames), max(1, stride)))
    if chosen[-1] != len(frames)-1:
        chosen.append(len(frames)-1)

    with writer.saving(fig, str(outfile), dpi=120):
        for k in chosen:
            f = frames[k]
            for i in range(4):
                p = f["p"][i]
                robot_pts[i].set_data([p[0]], [p[1]])
                hx = p[0] + 0.55 * math.cos(float(f["th"][i]))
                hy = p[1] + 0.55 * math.sin(float(f["th"][i]))
                heading_lines[i].set_data([p[0], hx], [p[1], hy])
                history[i].append((float(p[0]), float(p[1])))
                arr = np.asarray(history[i], dtype=float)
                trails[i].set_data(arr[:, 0], arr[:, 1])

            hp = f["hp"]
            hv = f["hv"]
            human_pts.set_offsets(hp)
            for j in range(len(hp)):
                speed = float(np.linalg.norm(hv[j]))
                if speed > 1e-5:
                    direction = hv[j] / speed
                    end = hp[j] + 0.35 * direction
                else:
                    end = hp[j]
                human_heading[j].set_data([hp[j, 0], end[0]], [hp[j, 1], end[1]])

            t = k * DT
            time_text.set_text(f"sim time: {t:5.1f} s")
            done = int(np.sum(f["done"] & ~f["hit"]))
            status_text.set_text(f"successful AMRs: {done}/4 | humans: {len(hp)}")
            writer.grab_frame()
    plt.close(fig)


def main():
    import torch
    from benchmark.train_multi_agent_research import Actor

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--humans", type=int, default=6)
    ap.add_argument("--realistic-humans", action="store_true")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.humans < 1:
        raise ValueError("--humans must be >= 1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_beast_config(args.config)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = Actor()
    actor.load_state_dict(ck["actor"])
    actor.eval()

    sac_w, sac_frames = _run_sac(actor, args.seed, args.humans, args.realistic_humans)
    orca_w, orca_frames, diagnostics = _run_orca(config, args.seed, args.humans, args.realistic_humans)
    sac = _summary(sac_w)
    orca = _summary(orca_w, diagnostics)
    shared = bool(sac["fleet_success"] and orca["fleet_success"])

    if shared:
        clearance = min(float(sac["min_clearance"]), float(orca["min_clearance"]))
        complexity = (
            -clearance,
            max(int(sac["steps"]), int(orca["steps"])),
            int(sac["interventions"]) + int(orca.get("orca_constraints_total", 0)),
        )
    else:
        complexity = None

    meta = {
        "seed": args.seed,
        "humans": args.humans,
        "human_model": "realistic_collision_aware" if args.realistic_humans else "legacy_linear",
        "human_radius": HUMAN_RADIUS if args.realistic_humans else None,
        "human_min_separation": HUMAN_MIN_SEPARATION if args.realistic_humans else None,
        "shared_fleet_success": shared,
        "sac_cbf": sac,
        "peak_orca_dd": orca,
        "complexity_key": complexity,
        "config": asdict(config),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)

    if not shared:
        return

    human_label = "realistic humans" if args.realistic_humans else "humans"
    _render(
        sac_frames,
        sac_w.g,
        f"4-AMR SAC+CBF — {args.humans} {human_label} — seed {args.seed}",
        out / "sac_cbf.mp4",
    )
    _render(
        orca_frames,
        orca_w.g,
        f"4-AMR Peak ORCA-DD — {args.humans} {human_label} — seed {args.seed}",
        out / "peak_orca_dd.mp4",
    )


if __name__ == "__main__":
    main()
