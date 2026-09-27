from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from benchmark.adaptive_predictive_orca import AdaptiveORCAConfig
from benchmark.best_vs_best_evaluation import select_validation_candidate
from benchmark.best_vs_best_protocol import local_human_navigation_catalog, split_seed_sets
from benchmark.best_vs_best_runner import aggregate_episode_rows, peak_orca_config, run_ap_orca_episode


# Predeclared causal AP-ORCA candidates. All candidates use the already-frozen
# Peak ORCA base configuration; only the observable adaptive/predictive layer
# varies. No candidate reads hidden human intent or future state.
ADAPTIVE_CANDIDATES: dict[str, AdaptiveORCAConfig] = {
    "default": AdaptiveORCAConfig(),
    "balanced_long": AdaptiveORCAConfig(
        horizon_min=1.0,
        horizon_max=5.0,
        uncertainty_gain=0.40,
        max_uncertainty_extra=0.45,
        base_uncertainty=0.02,
        acceleration_scale=0.25,
        heading_rate_scale=0.15,
        density_scale=0.04,
        yield_relief_ticks=40.0,
        min_human_responsibility=0.80,
        occlusion_memory_seconds=2.0,
        occlusion_uncertainty_rate=0.28,
    ),
    "safety_long": AdaptiveORCAConfig(
        horizon_min=1.2,
        horizon_max=5.0,
        uncertainty_gain=0.45,
        max_uncertainty_extra=0.50,
        base_uncertainty=0.025,
        acceleration_scale=0.22,
        heading_rate_scale=0.14,
        density_scale=0.05,
        yield_relief_ticks=45.0,
        min_human_responsibility=0.85,
        occlusion_memory_seconds=2.0,
        occlusion_uncertainty_rate=0.30,
    ),
    "intent_reactive": AdaptiveORCAConfig(
        horizon_min=0.8,
        horizon_max=4.5,
        uncertainty_gain=0.45,
        max_uncertainty_extra=0.40,
        base_uncertainty=0.02,
        acceleration_scale=0.30,
        heading_rate_scale=0.18,
        density_scale=0.05,
        yield_relief_ticks=35.0,
        min_human_responsibility=0.80,
        occlusion_memory_seconds=1.5,
        occlusion_uncertainty_rate=0.30,
    ),
    "dense_conservative": AdaptiveORCAConfig(
        horizon_min=1.0,
        horizon_max=4.8,
        uncertainty_gain=0.40,
        max_uncertainty_extra=0.45,
        base_uncertainty=0.02,
        acceleration_scale=0.22,
        heading_rate_scale=0.12,
        density_scale=0.02,
        yield_relief_ticks=45.0,
        min_human_responsibility=0.82,
        occlusion_memory_seconds=1.8,
        occlusion_uncertainty_rate=0.28,
    ),
    "liveness_balanced": AdaptiveORCAConfig(
        horizon_min=0.8,
        horizon_max=4.0,
        uncertainty_gain=0.30,
        max_uncertainty_extra=0.35,
        base_uncertainty=0.015,
        acceleration_scale=0.18,
        heading_rate_scale=0.10,
        density_scale=0.08,
        yield_relief_ticks=25.0,
        min_human_responsibility=0.70,
        occlusion_memory_seconds=1.2,
        occlusion_uncertainty_rate=0.20,
    ),
}


def select_peak(out_dir: Path, *, max_steps: int = 600) -> dict:
    development_seeds, validation_seeds, holdout_seeds = split_seed_sets(frozen=True)
    scenarios = local_human_navigation_catalog()
    aggregates = {}
    episode_rows = {}
    for name, adaptive in ADAPTIVE_CANDIDATES.items():
        rows = []
        for spec in scenarios:
            for seed in validation_seeds:
                rows.append(
                    run_ap_orca_episode(
                        spec,
                        seed=int(seed),
                        max_steps=int(max_steps),
                        adaptive_config=adaptive,
                    )
                )
        aggregates[name] = aggregate_episode_rows(rows)
        episode_rows[name] = rows
        print(json.dumps({"candidate": name, "aggregate": aggregates[name].__dict__}), flush=True)

    selected_name = select_validation_candidate(aggregates)
    selected_config = ADAPTIVE_CANDIDATES[selected_name]
    report = {
        "status": "selected_and_frozen",
        "controller": "Peak AP-ORCA+CBF",
        "selection_protocol": "validation_only_safety_first_lexicographic_v1",
        "perception_contract": "causal_observable_only_no_hidden_intent_no_future_state",
        "base_peak_orca_config": peak_orca_config().__dict__,
        "validation_seeds": list(validation_seeds),
        "development_seeds_not_used_for_selection": list(development_seeds),
        "holdout_seeds_used": [],
        "holdout_seed_count_reserved": len(holdout_seeds),
        "scenarios": [s.__dict__ for s in scenarios],
        "adaptive_candidates": {
            name: {
                "config": asdict(cfg),
                "validation_aggregate": aggregates[name].__dict__,
            }
            for name, cfg in ADAPTIVE_CANDIDATES.items()
        },
        "selected_candidate": selected_name,
        "adaptive_config": asdict(selected_config),
        "selected_validation_aggregate": aggregates[selected_name].__dict__,
        "validation_episode_rows": episode_rows,
        "freeze_policy": (
            "Do not retune Peak AP-ORCA base or adaptive parameters after this validation selection. "
            "The final holdout must use exactly this frozen configuration."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "peak_ap_orca_selection.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()
    report = select_peak(Path(args.out), max_steps=int(args.max_steps))
    print(json.dumps({
        "selected_candidate": report["selected_candidate"],
        "adaptive_config": report["adaptive_config"],
        "selected_validation_aggregate": report["selected_validation_aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
