from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path

from benchmark.beast_classical import BeastORCAConfig
from benchmark.beast_config import save_beast_config
from benchmark.tune_beast_classical import (
    _extend_rows,
    candidate_configs,
    estimated_episode_budget,
    rank_key,
    successive_halving_schedule,
    summarize,
)


def _score_item(payload):
    """Evaluate one candidate for one successive-halving stage.

    This is intentionally a top-level function so ProcessPoolExecutor can
    pickle it. It calls the exact same _extend_rows/summarize implementation
    as the serial tuner; only the scheduling is parallelized.
    """
    item, seeds_per_n = payload
    cfg = BeastORCAConfig(**item["config"])
    rows = _extend_rows(cfg, list(item["rows"]), int(seeds_per_n))
    summary = summarize(rows)
    return {
        "index": item["index"],
        "config": item["config"],
        "summary": summary,
        "rank_key": rank_key(summary),
        "rows": rows,
    }


def run_successive_halving_parallel(configs=None, workers=4):
    configs = candidate_configs() if configs is None else list(configs)
    active = [
        {
            "index": idx,
            "config": asdict(cfg),
            "rows": [],
        }
        for idx, cfg in enumerate(configs)
    ]
    stage_results = []
    workers = max(1, int(workers))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for stage_no, stage in enumerate(successive_halving_schedule(), start=1):
            seeds_per_n = int(stage["seeds_per_n"])
            # executor.map preserves input ordering. Therefore Python's stable
            # sort below has the same tie behaviour as the serial tuner.
            scored = list(
                pool.map(
                    _score_item,
                    [(item, seeds_per_n) for item in active],
                )
            )

            for item in scored:
                print(
                    f"stage{stage_no}",
                    item["index"],
                    json.dumps(item["summary"]),
                    flush=True,
                )

            scored.sort(key=lambda x: tuple(x["rank_key"]))
            stage_results.append(
                {
                    "stage": stage_no,
                    "seeds_per_n": seeds_per_n,
                    "keep": int(stage["keep"]),
                    "ranked": scored,
                }
            )
            active = scored[: int(stage["keep"])]

    return active[0], stage_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/beast_tuning_parallel")
    ap.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="CPU processes; changes scheduling only, not candidates/seeds/ranking.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best, stages = run_successive_halving_parallel(workers=args.workers)

    save_beast_config(
        BeastORCAConfig(**best["config"]),
        out / "best_config.json",
    )
    (out / "validation_results.json").write_text(
        json.dumps(
            {
                "schedule": successive_halving_schedule(),
                "estimated_episode_budget": estimated_episode_budget(),
                "workers": int(args.workers),
                "stages": stages,
                "best": best,
            },
            indent=2,
        )
    )
    print(
        "best",
        best["index"],
        json.dumps(best["summary"]),
        "episode_budget",
        estimated_episode_budget(),
        "workers",
        args.workers,
        flush=True,
    )


if __name__ == "__main__":
    main()
