from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from benchmark.beast_classical import AStarORCADD, BeastORCAConfig
from benchmark.beast_config import VALIDATION_SEEDS_BY_N, save_beast_config


def rank_key(summary):
    def val(name, default=float("inf")):
        x = summary.get(name)
        return default if x is None else float(x)

    return (
        val("collision_rate"),
        val("timeout_rate"),
        -val("fleet_success_rate", 0.0),
        -val("success_rate", 0.0),
        val("traversal_time_success_mean"),
        val("path_length_success_mean"),
    )


def candidate_configs():
    base = BeastORCAConfig()
    profiles = [
        {},
        dict(
            time_horizon=1.8,
            neighbor_distance=3.5,
            peer_margin=0.06,
            human_margin=0.10,
        ),
        dict(
            time_horizon=2.2,
            neighbor_distance=4.0,
            peer_margin=0.08,
            human_margin=0.12,
        ),
        dict(
            time_horizon=3.0,
            neighbor_distance=4.5,
            peer_margin=0.10,
            human_margin=0.14,
        ),
        dict(
            time_horizon=3.5,
            neighbor_distance=5.0,
            peer_margin=0.12,
            human_margin=0.16,
        ),
        dict(
            preferred_speed=0.8,
            time_horizon=2.5,
            static_margin=0.10,
            planning_margin=0.12,
        ),
        dict(
            preferred_speed=0.9,
            time_horizon=3.0,
            peer_margin=0.10,
            human_margin=0.15,
        ),
        dict(
            preferred_speed=1.0,
            time_horizon=2.0,
            peer_margin=0.05,
            human_margin=0.10,
        ),
        dict(
            arc_horizon=0.6,
            time_horizon=2.5,
            replan_interval=10,
            lookahead_distance=1.3,
        ),
        dict(
            arc_horizon=1.0,
            time_horizon=3.0,
            replan_interval=12,
            lookahead_distance=1.8,
        ),
        dict(
            command_speed_samples=9,
            command_omega_samples=17,
            time_horizon=2.5,
            neighbor_distance=4.5,
        ),
        dict(
            command_speed_samples=7,
            command_omega_samples=19,
            time_horizon=3.0,
            preferred_speed=0.9,
        ),
        dict(
            smoothness_weight=0.06,
            progress_weight=4.5,
            time_horizon=2.5,
        ),
        dict(
            smoothness_weight=0.20,
            progress_weight=3.5,
            preferred_speed=0.9,
            time_horizon=3.0,
        ),
        dict(
            stuck_ticks=16,
            recovery_ticks=16,
            replan_interval=10,
            time_horizon=2.5,
        ),
        dict(
            stuck_ticks=28,
            recovery_ticks=24,
            replan_interval=20,
            time_horizon=3.0,
        ),
    ]
    return [replace(base, **profile) for profile in profiles]


def _episode(config, n, seed):
    from benchmark.train_multi_agent_research import World, DT

    w = World(n, max(3, n + 2), seed)
    w.reset()
    ctrls = [AStarORCADD(w, i, config) for i in range(n)]
    for _ in range(600):
        acts = np.asarray(
            [
                ctrls[i].action(w) if not w.done[i] else [-1.0, 0.0]
                for i in range(n)
            ],
            dtype=np.float32,
        )
        _, _, done = w.step(acts, False)
        if np.all(done):
            break

    success = w.done & ~w.hit
    collision = w.hit.copy()
    timeout = ~w.done
    return {
        "n": n,
        "seed": seed,
        "success": int(success.sum()),
        "collision": int(collision.sum()),
        "timeout": int(timeout.sum()),
        "fleet_success": bool(success.all()),
        "traversal_times": [
            float(w.finish_step[i] * DT) for i in range(n) if success[i]
        ],
        "path_lengths": [float(w.path_length[i]) for i in range(n) if success[i]],
        "diagnostics": [c.diagnostics() for c in ctrls],
    }


def summarize(rows):
    agents = sum(r["n"] for r in rows)
    successes = sum(r["success"] for r in rows)
    collisions = sum(r["collision"] for r in rows)
    timeouts = sum(r["timeout"] for r in rows)
    fleets = len(rows)
    times = [x for r in rows for x in r["traversal_times"]]
    paths = [x for r in rows for x in r["path_lengths"]]
    return {
        "collision_rate": collisions / agents,
        "timeout_rate": timeouts / agents,
        "fleet_success_rate": sum(r["fleet_success"] for r in rows) / fleets,
        "success_rate": successes / agents,
        "traversal_time_success_mean": float(np.mean(times)) if times else None,
        "path_length_success_mean": float(np.mean(paths)) if paths else None,
    }


def evaluate_config(config, seeds_by_n):
    rows = []
    for n in (2, 4, 6):
        for seed in seeds_by_n[n]:
            rows.append(_episode(config, n, seed))
    return summarize(rows), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/beast_tuning")
    ap.add_argument("--coarse-seeds", type=int, default=5)
    ap.add_argument("--finalists", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    configs = candidate_configs()
    coarse_seeds = {
        n: VALIDATION_SEEDS_BY_N[n][: args.coarse_seeds] for n in (2, 4, 6)
    }

    coarse = []
    for idx, cfg in enumerate(configs):
        summary, _ = evaluate_config(cfg, coarse_seeds)
        coarse.append(
            {
                "index": idx,
                "config": asdict(cfg),
                "summary": summary,
                "rank_key": rank_key(summary),
            }
        )
        print("coarse", idx, json.dumps(summary), flush=True)
    coarse.sort(key=lambda x: tuple(x["rank_key"]))

    finalists = []
    for item in coarse[: args.finalists]:
        cfg = BeastORCAConfig(**item["config"])
        summary, rows = evaluate_config(cfg, VALIDATION_SEEDS_BY_N)
        finalists.append(
            {
                "index": item["index"],
                "config": item["config"],
                "summary": summary,
                "rank_key": rank_key(summary),
                "rows": rows,
            }
        )
        print("finalist", item["index"], json.dumps(summary), flush=True)
    finalists.sort(key=lambda x: tuple(x["rank_key"]))

    best = finalists[0]
    save_beast_config(
        BeastORCAConfig(**best["config"]),
        out / "best_config.json",
    )
    (out / "validation_results.json").write_text(
        json.dumps({"coarse": coarse, "finalists": finalists}, indent=2)
    )
    print("best", best["index"], json.dumps(best["summary"]), flush=True)


if __name__ == "__main__":
    main()
