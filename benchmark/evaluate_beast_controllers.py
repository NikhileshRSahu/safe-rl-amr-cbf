from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmark.beast_classical import AStarORCADD, BeastORCAConfig
from benchmark.beast_config import TEST_SEEDS_BY_N, load_beast_config


def test_seeds_for_n(n: int):
    return tuple(TEST_SEEDS_BY_N[int(n)])


def config_digest(config: BeastORCAConfig) -> str:
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def wilson(k, n, z=1.96):
    if n <= 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def summarize_episode(w, n, seed, controller, compute_s, diagnostics=None):
    from benchmark.train_multi_agent_research import DT

    success = w.done & ~w.hit
    collision = w.hit.copy()
    timeout = ~w.done
    effective = np.where(w.finish_step > 0, w.finish_step, 600)
    types = {"amr_amr": 0, "human": 0, "static": 0}
    for collision_type in w.collision_type:
        if collision_type in types:
            types[collision_type] += 1

    finite_clearance = w.min_clearance[np.isfinite(w.min_clearance)]
    min_clearance = (
        float(np.min(finite_clearance))
        if finite_clearance.size
        else float("inf")
    )

    row = {
        "seed": seed,
        "controller": controller,
        "n": n,
        "steps": int(w.steps),
        "success": int(success.sum()),
        "collision": int(collision.sum()),
        "timeout": int(timeout.sum()),
        "fleet_success": bool(success.all()),
        "fleet_collision": bool(collision.any()),
        "path_length_success_mean": (
            float(w.path_length[success].mean()) if success.any() else None
        ),
        "traversal_time_success_mean": (
            float((w.finish_step[success] * DT).mean()) if success.any() else None
        ),
        "min_clearance": min_clearance,
        "interventions": int(w.interventions.sum()),
        "deadlock": int(w.deadlock.sum()),
        "intervention_fraction": float(
            w.interventions.sum() / max(1, effective.sum())
        ),
        "throughput_per_min": float(
            success.sum() / (max(1, w.steps) * DT) * 60.0
        ),
        "collision_types": types,
        "controller_compute_seconds": float(compute_s),
        "controller_ms_per_agent_step": float(
            1000.0 * compute_s / max(1, w.steps * n)
        ),
    }

    if diagnostics is not None:
        keys = (
            "orca_constraints_total",
            "infeasible_command_events",
            "stop_yield_ticks",
            "replan_count",
            "recovery_count",
            "candidate_commands_evaluated",
        )
        totals = {
            key: sum(int(d.get(key, 0)) for d in diagnostics)
            for key in keys
        }
        totals["stop_yield_fraction"] = float(
            totals["stop_yield_ticks"] / max(1, w.steps * n)
        )
        row["classical_diagnostics"] = totals

    return row


def aggregate(rows, n):
    agents = n * len(rows)
    succ = sum(r["success"] for r in rows)
    coll = sum(r["collision"] for r in rows)
    tout = sum(r["timeout"] for r in rows)
    fs = sum(1 for r in rows if r["fleet_success"])
    fc = sum(1 for r in rows if r["fleet_collision"])

    def mean_nonnull(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    collision_types = {
        key: sum(r["collision_types"][key] for r in rows)
        for key in ("amr_amr", "human", "static")
    }

    out = {
        "episodes": len(rows),
        "agents": agents,
        "success_rate": succ / agents,
        "success_ci95": wilson(succ, agents),
        "collision_rate": coll / agents,
        "collision_ci95": wilson(coll, agents),
        "timeout_rate": tout / agents,
        "timeout_ci95": wilson(tout, agents),
        "fleet_success_rate": fs / len(rows),
        "fleet_success_ci95": wilson(fs, len(rows)),
        "fleet_collision_rate": fc / len(rows),
        "fleet_collision_ci95": wilson(fc, len(rows)),
        "path_length_success_mean": mean_nonnull(
            "path_length_success_mean"
        ),
        "traversal_time_success_mean": mean_nonnull(
            "traversal_time_success_mean"
        ),
        "episode_min_clearance_mean": float(
            np.mean([r["min_clearance"] for r in rows])
        ),
        "worst_min_clearance": float(
            np.min([r["min_clearance"] for r in rows])
        ),
        "mean_intervention_fraction": float(
            np.mean([r["intervention_fraction"] for r in rows])
        ),
        "mean_throughput_per_min": float(
            np.mean([r["throughput_per_min"] for r in rows])
        ),
        "collision_type_counts": collision_types,
        "mean_controller_ms_per_agent_step": float(
            np.mean([r["controller_ms_per_agent_step"] for r in rows])
        ),
    }

    if rows and "classical_diagnostics" in rows[0]:
        keys = rows[0]["classical_diagnostics"].keys()
        out["classical_diagnostics_mean"] = {
            key: float(
                np.mean([r["classical_diagnostics"][key] for r in rows])
            )
            for key in keys
        }

    return out


def run_actor(actor, n, seed):
    import torch
    from benchmark.train_multi_agent_research import World

    w = World(n, max(3, n + 2), seed)
    obs = w.reset()
    compute = 0.0
    for _ in range(600):
        t0 = time.perf_counter()
        with torch.no_grad():
            action, _ = actor.sample(
                torch.tensor(np.asarray(obs), dtype=torch.float32),
                True,
            )
        compute += time.perf_counter() - t0
        obs, _, done = w.step(action.numpy(), True)
        if np.all(done):
            break
    return summarize_episode(w, n, seed, "sac_cbf", compute)


def run_orca(config, n, seed):
    from benchmark.train_multi_agent_research import World

    w = World(n, max(3, n + 2), seed)
    w.reset()
    ctrls = [AStarORCADD(w, i, config) for i in range(n)]
    compute = 0.0
    for _ in range(600):
        t0 = time.perf_counter()
        acts = np.asarray(
            [
                ctrls[i].action(w) if not w.done[i] else [-1.0, 0.0]
                for i in range(n)
            ],
            np.float32,
        )
        compute += time.perf_counter() - t0
        _, _, done = w.step(acts, False)
        if np.all(done):
            break
    return summarize_episode(
        w,
        n,
        seed,
        "astar_orca_dd",
        compute,
        [c.diagnostics() for c in ctrls],
    )


def main():
    import torch
    from benchmark.train_multi_agent_research import Actor

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/beast_eval")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = load_beast_config(args.config)
    digest = config_digest(config)

    ck = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    actor = Actor()
    actor.load_state_dict(ck["actor"])
    actor.eval()

    summary = {}
    all_rows = []
    for n in (2, 4, 6):
        summary[str(n)] = {}
        seeds = test_seeds_for_n(n)
        for name, runner in (
            ("sac_cbf", lambda sd: run_actor(actor, n, sd)),
            ("astar_orca_dd", lambda sd: run_orca(config, n, sd)),
        ):
            rows = [runner(sd) for sd in seeds]
            all_rows.extend(rows)
            summary[str(n)][name] = aggregate(rows, n)
            print(
                n,
                name,
                json.dumps(summary[str(n)][name]),
                flush=True,
            )

    summary["_provenance"] = {
        "config_sha256": digest,
        "test_seeds": {
            str(n): list(test_seeds_for_n(n))
            for n in (2, 4, 6)
        },
    }

    (out / "beast_eval_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    (out / "beast_eval_rows.json").write_text(
        json.dumps(all_rows, indent=2)
    )
    (out / "beast_eval_config.json").write_text(
        json.dumps(
            {"sha256": digest, "config": asdict(config)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
