from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.beast_classical import AStarORCADD
from benchmark.beast_config import load_beast_config
from benchmark.human_sweep_experiment import SweepHumanWorld, SweepScenario, _episode_metrics
from benchmark.render_shared_success_case import _render, _snapshot


def run_sac(actor, seed: int, humans: int, speed_scale: float):
    import torch

    world = SweepHumanWorld(
        4,
        humans,
        seed,
        speed_scale=speed_scale,
        randomness_level="baseline",
    )
    obs = world.reset()
    frames = [_snapshot(world)]
    for _ in range(600):
        with torch.no_grad():
            action, _ = actor.sample(
                torch.tensor(np.asarray(obs), dtype=torch.float32), True
            )
        obs, _, done = world.step(action.numpy(), True)
        frames.append(_snapshot(world))
        if np.all(done):
            break
    return world, frames, _episode_metrics(world)


def run_orca(config, seed: int, humans: int, speed_scale: float):
    world = SweepHumanWorld(
        4,
        humans,
        seed,
        speed_scale=speed_scale,
        randomness_level="baseline",
    )
    world.reset()
    ctrls = [AStarORCADD(world, i, config) for i in range(4)]
    frames = [_snapshot(world)]
    for _ in range(600):
        actions = np.asarray(
            [
                ctrls[i].action(world) if not world.done[i] else [-1.0, 0.0]
                for i in range(4)
            ],
            dtype=np.float32,
        )
        _, _, done = world.step(actions, False)
        frames.append(_snapshot(world))
        if np.all(done):
            break
    diagnostics = [c.diagnostics() for c in ctrls]
    return world, frames, _episode_metrics(world, diagnostics)


def main():
    import torch
    from benchmark.train_multi_agent_research import Actor

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=8100)
    ap.add_argument("--humans", type=int, default=12)
    ap.add_argument("--speed-scale", type=float, default=1.5)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = load_beast_config(args.config)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = Actor()
    actor.load_state_dict(ck["actor"])
    actor.eval()

    sac_w, sac_frames, sac_metrics = run_sac(
        actor, args.seed, args.humans, args.speed_scale
    )
    orca_w, orca_frames, orca_metrics = run_orca(
        config, args.seed, args.humans, args.speed_scale
    )

    metadata = {
        "proof_type": "same_seed_real_simulator_rollout",
        "seed": args.seed,
        "n_amr": 4,
        "humans": args.humans,
        "human_speed_scale": args.speed_scale,
        "human_model": "realistic_collision_aware",
        "sac_cbf": sac_metrics,
        "peak_orca_dd": orca_metrics,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)

    _render(
        sac_frames,
        sac_w.g,
        f"PROOF: SAC+CBF | 4 AMRs | {args.humans} humans | {args.speed_scale:.2f}x | seed {args.seed}",
        out / "sac_cbf_proof.mp4",
    )
    _render(
        orca_frames,
        orca_w.g,
        f"PROOF: Peak ORCA-DD | 4 AMRs | {args.humans} humans | {args.speed_scale:.2f}x | seed {args.seed}",
        out / "peak_orca_proof.mp4",
    )


if __name__ == "__main__":
    main()
