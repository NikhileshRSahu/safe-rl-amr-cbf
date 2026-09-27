from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark.best_vs_best_evaluation import compare_best_vs_best
from benchmark.best_vs_best_protocol import (
    local_human_navigation_catalog,
    paired_world_fingerprint,
    split_seed_sets,
)
from benchmark.best_vs_best_runner import (
    aggregate_episode_rows,
    peak_orca_config,
    run_ap_orca_episode,
    run_forecast_stasac_episode,
)
from benchmark.train_human_forecaster import load_forecaster_checkpoint
from benchmark.train_stasac_cbf import load_stasac_checkpoint


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_final(
    selection_path: Path,
    actor_checkpoint: Path,
    forecaster_path: Path,
    out_dir: Path,
    *,
    max_steps: int = 600,
) -> dict:
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "selected_and_frozen":
        raise RuntimeError("selection artifact is not a frozen selected candidate")
    if selection.get("selection_protocol") != "validation_only_safety_first_lexicographic_v1":
        raise RuntimeError("unexpected candidate-selection protocol")
    if list(selection.get("holdout_seeds_used", [])):
        raise RuntimeError("holdout was already marked as used before the decisive run")

    actor_sha = sha256_file(actor_checkpoint)
    expected_actor_sha = str(selection["selected_checkpoint_sha256"])
    if actor_sha != expected_actor_sha:
        raise RuntimeError(f"selected actor SHA mismatch: {actor_sha} != {expected_actor_sha}")

    forecaster_sha = sha256_file(forecaster_path)
    expected_forecaster_sha = str(selection["forecaster_sha256"])
    if forecaster_sha != expected_forecaster_sha:
        raise RuntimeError(
            f"frozen forecaster SHA mismatch: {forecaster_sha} != {expected_forecaster_sha}"
        )

    actor, _, actor_meta = load_stasac_checkpoint(actor_checkpoint)
    forecaster, forecaster_meta = load_forecaster_checkpoint(forecaster_path)
    if str(actor_meta.get("forecaster_sha256", "")) != expected_forecaster_sha:
        raise RuntimeError("actor checkpoint was not trained with the frozen forecaster")
    if not bool(actor_meta.get("forecast_enabled", False)):
        raise RuntimeError("selected actor is not Forecast-ST-SAC")
    if int(actor_meta.get("final_stage", -1)) != 3:
        raise RuntimeError("selected actor metadata does not record stage 3")

    development_seeds, validation_seeds, holdout_seeds = split_seed_sets(frozen=True)
    if set(development_seeds) & set(validation_seeds):
        raise RuntimeError("development/validation seed overlap")
    if (set(development_seeds) | set(validation_seeds)) & set(holdout_seeds):
        raise RuntimeError("holdout seed leakage")

    scenarios = local_human_navigation_catalog()
    forecast_rows = []
    orca_rows = []
    fingerprints = {}
    for spec in scenarios:
        for seed in holdout_seeds:
            key = f"{spec.name}:{int(seed)}"
            fingerprints[key] = paired_world_fingerprint(spec, int(seed))
            forecast_rows.append(
                run_forecast_stasac_episode(
                    actor,
                    forecaster,
                    spec,
                    seed=int(seed),
                    max_steps=int(max_steps),
                )
            )
            orca_rows.append(
                run_ap_orca_episode(
                    spec,
                    seed=int(seed),
                    max_steps=int(max_steps),
                )
            )

    forecast_agg = aggregate_episode_rows(forecast_rows)
    orca_agg = aggregate_episode_rows(orca_rows)
    forward = compare_best_vs_best(forecast_agg, orca_agg, collision_tolerance=0.01)
    reverse = compare_best_vs_best(orca_agg, forecast_agg, collision_tolerance=0.01)
    if forward.rl_wins:
        verdict = "Forecast-ST-SAC+CBF"
    elif reverse.rl_wins:
        verdict = "Peak AP-ORCA+CBF"
    else:
        verdict = "no_superiority_claim"

    report = {
        "status": "decisive_holdout_complete",
        "protocol": "peak_ap_orca_cbf_vs_frozen_fully_trained_forecast_st_sac_cbf_v1",
        "verdict": verdict,
        "holdout_consumed_once": True,
        "holdout_seeds": list(holdout_seeds),
        "development_seeds_not_used_for_final": list(development_seeds),
        "validation_seeds_not_used_for_final": list(validation_seeds),
        "scenarios": [s.__dict__ for s in scenarios],
        "max_steps": int(max_steps),
        "pairing_fingerprints": fingerprints,
        "selected_candidate": selection["selected_candidate"],
        "selected_checkpoint_sha256": actor_sha,
        "forecaster_sha256": forecaster_sha,
        "actor_metadata": actor_meta,
        "forecaster_metadata": forecaster_meta,
        "peak_ap_orca_config": peak_orca_config().__dict__,
        "aggregates": {
            "Forecast-ST-SAC+CBF": forecast_agg.__dict__,
            "Peak AP-ORCA+CBF": orca_agg.__dict__,
        },
        "forecast_superiority_test": forward.__dict__,
        "orca_superiority_test": reverse.__dict__,
        "episodes": {
            "Forecast-ST-SAC+CBF": forecast_rows,
            "Peak AP-ORCA+CBF": orca_rows,
        },
        "interpretation_rule": (
            "Forecast-ST-SAC is declared superior only if its frozen comparison rule passes; "
            "Peak AP-ORCA is declared superior only if the same rule passes in the reverse direction. "
            "Otherwise report no superiority claim rather than forcing a winner."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "FINAL_HOLDOUT_USED").write_text("1\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--forecaster", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()
    report = run_final(
        Path(args.selection),
        Path(args.actor),
        Path(args.forecaster),
        Path(args.out),
        max_steps=int(args.max_steps),
    )
    print(json.dumps({
        "verdict": report["verdict"],
        "aggregates": report["aggregates"],
        "forecast_superiority_test": report["forecast_superiority_test"],
        "orca_superiority_test": report["orca_superiority_test"],
    }, indent=2))


if __name__ == "__main__":
    main()
