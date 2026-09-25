from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark.beast_config import load_beast_config
from benchmark.evaluate_beast_controllers import (
    aggregate,
    run_actor,
    run_orca,
    test_seeds_for_n,
)
from benchmark.train_multi_agent_research import Actor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, choices=(2, 4, 6), required=True)
    ap.add_argument(
        "--controller",
        choices=("sac_cbf", "astar_orca_dd"),
        required=True,
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = int(args.n)
    seeds = test_seeds_for_n(n)

    if args.controller == "sac_cbf":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for sac_cbf")
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        actor = Actor()
        actor.load_state_dict(ck["actor"])
        actor.eval()
        rows = [run_actor(actor, n, seed) for seed in seeds]
    else:
        config = load_beast_config(args.config)
        rows = [run_orca(config, n, seed) for seed in seeds]

    payload = {
        "n": n,
        "controller": args.controller,
        "seeds": list(seeds),
        "aggregate": aggregate(rows, n),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
