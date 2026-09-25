from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmark.beast_classical import AStarORCADD
from benchmark.beast_config import load_beast_config


def _snapshot(w):
    return {
        "p": np.asarray(w.p, dtype=np.float32).copy(),
        "th": np.asarray(w.th, dtype=np.float32).copy(),
        "hp": np.asarray(w.hp, dtype=np.float32).copy(),
        "done": np.asarray(w.done, dtype=bool).copy(),
        "hit": np.asarray(w.hit, dtype=bool).copy(),
    }


def _run_sac(actor, seed: int, humans: int):
    import torch
    from benchmark.train_multi_agent_research import World

    w = World(4, humans, seed)
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


def _run_orca(config, seed: int, humans: int):
    from benchmark.train_multi_agent_research import World

    w = World(4, humans, seed)
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
    from benchmark.train_multi_agent_research import SHELVES, WORLD, DT

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
    human_pts = ax.plot([], [], "x", markersize=6, label="Humans")[0]
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
            human_pts.set_data(hp[:, 0], hp[:, 1])
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

    sac_w, sac_frames = _run_sac(actor, args.seed, args.humans)
    orca_w, orca_frames, diagnostics = _run_orca(config, args.seed, args.humans)
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

    _render(
        sac_frames,
        sac_w.g,
        f"4-AMR SAC+CBF — {args.humans} humans — seed {args.seed}",
        out / "sac_cbf.mp4",
    )
    _render(
        orca_frames,
        orca_w.g,
        f"4-AMR Peak ORCA-DD — {args.humans} humans — seed {args.seed}",
        out / "peak_orca_dd.mp4",
    )


if __name__ == "__main__":
    main()
