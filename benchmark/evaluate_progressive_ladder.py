from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmark.beast_classical import AStarORCADD
from benchmark.beast_config import load_beast_config
from benchmark.evaluate_beast_controllers import aggregate, summarize_episode
from benchmark.progressive_velocity_baselines import AStarRVODD, AStarVODD
from benchmark.train_multi_agent_research import World


PROGRESSION_SEEDS = tuple(range(9100, 9110))

CLASSICAL_SCENARIOS = {
    "reciprocal_crossing": {
        "n_amr": 4,
        "humans": 0,
        "world_kind": "base",
        "purpose": "VO to RVO: mutual collision-avoidance responsibility",
    },
    "dense_reciprocal": {
        "n_amr": 6,
        "humans": 0,
        "world_kind": "base",
        "purpose": "RVO to ORCA: dense reciprocal feasibility and liveness",
    },
}

CONTROLLERS = {
    "vo": AStarVODD,
    "rvo": AStarRVODD,
    "orca": AStarORCADD,
}


def _sign_nonzero(x: float, eps: float = 0.08) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def run_classical(controller_name: str, scenario_name: str, config, seed: int):
    scenario = CLASSICAL_SCENARIOS[scenario_name]
    n = int(scenario["n_amr"])
    humans = int(scenario["humans"])
    controller_cls = CONTROLLERS[controller_name]

    w = World(n, humans, seed)
    w.reset()
    ctrls = [controller_cls(w, i, config) for i in range(n)]

    last_turn_sign = np.zeros(n, dtype=np.int8)
    omega_sign_flips = np.zeros(n, dtype=np.int32)
    compute_s = 0.0

    for _ in range(600):
        t0 = time.perf_counter()
        acts = np.asarray(
            [
                ctrls[i].action(w) if not w.done[i] else [-1.0, 0.0]
                for i in range(n)
            ],
            dtype=np.float32,
        )
        compute_s += time.perf_counter() - t0

        for i in range(n):
            if w.done[i]:
                continue
            sign = _sign_nonzero(float(acts[i, 1]))
            if sign and last_turn_sign[i] and sign != last_turn_sign[i]:
                omega_sign_flips[i] += 1
            if sign:
                last_turn_sign[i] = sign

        _, _, done = w.step(acts, False)
        if np.all(done):
            break

    diagnostics = [c.diagnostics() for c in ctrls]
    row = summarize_episode(
        w,
        n,
        seed,
        controller_name,
        compute_s,
        diagnostics,
    )
    row["scenario"] = scenario_name
    row["omega_sign_flips_total"] = int(omega_sign_flips.sum())
    row["omega_sign_flips_per_agent"] = float(omega_sign_flips.mean())
    row["target_infeasible_events"] = int(
        sum(d.get("velocity_target_infeasible_events", 0) for d in diagnostics)
    )
    row["velocity_candidates_evaluated"] = int(
        sum(d.get("velocity_candidates_evaluated", 0) for d in diagnostics)
    )
    return row


def aggregate_ladder(rows, n: int):
    out = aggregate(rows, n)
    out["mean_omega_sign_flips_per_agent"] = float(
        np.mean([r["omega_sign_flips_per_agent"] for r in rows])
    )
    out["mean_target_infeasible_events"] = float(
        np.mean([r["target_infeasible_events"] for r in rows])
    )
    out["mean_velocity_candidates_evaluated"] = float(
        np.mean([r["velocity_candidates_evaluated"] for r in rows])
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=sorted(CLASSICAL_SCENARIOS))
    ap.add_argument("--controller", required=True, choices=sorted(CONTROLLERS))
    ap.add_argument("--config", default="benchmark/frozen_peak_orca_config.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = load_beast_config(args.config)
    scenario = CLASSICAL_SCENARIOS[args.scenario]
    rows = [
        run_classical(args.controller, args.scenario, config, seed)
        for seed in PROGRESSION_SEEDS
    ]
    summary = aggregate_ladder(rows, int(scenario["n_amr"]))

    payload = {
        "scenario": args.scenario,
        "scenario_spec": scenario,
        "controller": args.controller,
        "seeds": list(PROGRESSION_SEEDS),
        "aggregate": summary,
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
