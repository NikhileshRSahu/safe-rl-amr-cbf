from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from benchmark.best_vs_best_evaluation import select_validation_candidate
from benchmark.best_vs_best_protocol import local_human_navigation_catalog, split_seed_sets
from benchmark.best_vs_best_runner import aggregate_episode_rows, run_forecast_stasac_episode
from benchmark.orca_teacher import should_promote_curriculum_stage
from benchmark.train_human_forecaster import load_forecaster_checkpoint
from benchmark.train_stasac_cbf import load_stasac_checkpoint


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def qualification_errors(summary: dict, *, expected_forecaster_sha: str, min_agent_steps: int) -> list[str]:
    errors: list[str] = []
    meta = dict(summary.get("metadata", {}))
    if int(meta.get("agent_steps", 0)) < int(min_agent_steps):
        errors.append(f"agent_steps<{min_agent_steps}")
    if not bool(meta.get("forecast_enabled", False)):
        errors.append("forecast_not_enabled")
    if str(meta.get("forecaster_sha256", "")) != str(expected_forecaster_sha):
        errors.append("wrong_forecaster_sha")
    if str(meta.get("training_seed_split", "")) != "development_gradient_only":
        errors.append("training_seed_split_not_development_only")
    if str(meta.get("architecture", "")) != "forecast_spatiotemporal_risk_attention_sac_cbf_v1":
        errors.append("wrong_architecture")
    if int(meta.get("final_stage", -1)) != 3:
        errors.append("stage3_not_reached")

    gates = list(summary.get("competence_gates", []))
    seen = {int(g.get("stage", -1)) for g in gates}
    missing = sorted({0, 1, 2, 3} - seen)
    if missing:
        errors.append(f"missing_stage_gates:{missing}")
    for stage in (0, 1, 2):
        if not any(int(g.get("stage", -1)) == stage and bool(g.get("promoted", False)) for g in gates):
            errors.append(f"stage{stage}_not_promoted")

    stage3 = [g for g in gates if int(g.get("stage", -1)) == 3]
    if not stage3:
        errors.append("stage3_not_competence_tested")
    else:
        policy = dict(stage3[-1].get("policy", {}))
        mastered = should_promote_curriculum_stage(
            episodes=int(policy.get("episodes", 0)),
            success_rate=float(policy.get("success_rate", 0.0)),
            collision_rate=float(policy.get("collision_rate", 1.0)),
        )
        if not mastered:
            errors.append("stage3_final_gate_not_mastered")
    return errors


def discover_candidates(root: Path) -> list[Path]:
    found: list[Path] = []
    for summary_path in sorted(root.rglob("summary.json")):
        parent = summary_path.parent
        if (parent / "stasac_cbf.pt").is_file():
            found.append(parent)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def candidate_name(path: Path, summary: dict) -> str:
    seed = summary.get("metadata", {}).get("seed")
    if seed is not None:
        return f"seed-{int(seed)}"
    return path.name


def select_candidate(
    candidate_root: Path,
    forecaster_path: Path,
    out_dir: Path,
    *,
    expected_forecaster_sha: str,
    min_agent_steps: int = 120000,
    max_steps: int = 600,
) -> dict:
    actual_forecaster_sha = sha256_file(forecaster_path)
    if actual_forecaster_sha != expected_forecaster_sha:
        raise RuntimeError(
            f"frozen forecaster SHA mismatch: {actual_forecaster_sha} != {expected_forecaster_sha}"
        )
    forecaster, forecaster_meta = load_forecaster_checkpoint(forecaster_path)

    eligible: dict[str, dict] = {}
    rejected: dict[str, dict] = {}
    for path in discover_candidates(candidate_root):
        summary = json.loads((path / "summary.json").read_text())
        name = candidate_name(path, summary)
        errors = qualification_errors(
            summary,
            expected_forecaster_sha=expected_forecaster_sha,
            min_agent_steps=min_agent_steps,
        )
        checkpoint = path / "stasac_cbf.pt"
        record = {
            "artifact_path": str(path),
            "checkpoint_sha256": sha256_file(checkpoint),
            "training_metadata": summary.get("metadata", {}),
            "final_stage3_gate": next(
                (g for g in reversed(summary.get("competence_gates", [])) if int(g.get("stage", -1)) == 3),
                None,
            ),
        }
        if errors:
            record["rejection_reasons"] = errors
            rejected[name] = record
        else:
            record["path"] = path
            eligible[name] = record

    if not eligible:
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "no_fully_trained_candidate",
            "expected_forecaster_sha256": expected_forecaster_sha,
            "eligible_candidates": [],
            "rejected_candidates": rejected,
        }
        (out_dir / "selection.json").write_text(json.dumps(failure, indent=2, default=str))
        raise RuntimeError("no candidate passed the frozen 0->3 curriculum mastery contract")

    _, validation_seeds, holdout_seeds = split_seed_sets(frozen=True)
    scenarios = local_human_navigation_catalog()
    aggregates = {}
    episode_rows = {}
    for name, record in eligible.items():
        actor, _, checkpoint_meta = load_stasac_checkpoint(record["path"] / "stasac_cbf.pt")
        if str(checkpoint_meta.get("forecaster_sha256", "")) != expected_forecaster_sha:
            raise RuntimeError(f"{name}: checkpoint metadata forecaster SHA mismatch")
        rows = []
        for spec in scenarios:
            for seed in validation_seeds:
                rows.append(
                    run_forecast_stasac_episode(
                        actor,
                        forecaster,
                        spec,
                        seed=int(seed),
                        max_steps=int(max_steps),
                    )
                )
        aggregate = aggregate_episode_rows(rows)
        aggregates[name] = aggregate
        episode_rows[name] = rows

    selected_name = select_validation_candidate(aggregates)
    selected = eligible[selected_name]
    selected_path = selected["path"] / "stasac_cbf.pt"

    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_checkpoint = out_dir / "selected_forecast_stasac.pt"
    shutil.copy2(selected_path, frozen_checkpoint)
    frozen_sha = sha256_file(frozen_checkpoint)
    if frozen_sha != selected["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint changed while freezing")

    report = {
        "status": "selected_and_frozen",
        "selection_protocol": "validation_only_safety_first_lexicographic_v1",
        "validation_seeds": list(validation_seeds),
        "holdout_seeds_used": [],
        "holdout_seed_count_reserved": len(holdout_seeds),
        "scenarios": [s.__dict__ for s in scenarios],
        "forecaster_sha256": expected_forecaster_sha,
        "forecaster_metadata": forecaster_meta,
        "eligible_candidates": {
            name: {
                "checkpoint_sha256": record["checkpoint_sha256"],
                "training_metadata": record["training_metadata"],
                "final_stage3_gate": record["final_stage3_gate"],
                "validation_aggregate": aggregates[name].__dict__,
            }
            for name, record in eligible.items()
        },
        "rejected_candidates": rejected,
        "selected_candidate": selected_name,
        "selected_checkpoint_sha256": frozen_sha,
        "selected_validation_aggregate": aggregates[selected_name].__dict__,
        "validation_episode_rows": episode_rows,
        "freeze_policy": (
            "The selected Forecast-ST-SAC checkpoint, frozen predictor, Peak AP-ORCA configuration, "
            "scenario catalog, and comparison rule must not change after this selection. The final "
            "holdout may be evaluated once for the decisive comparison."
        ),
    }
    (out_dir / "selection.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--forecaster", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-forecaster-sha", required=True)
    parser.add_argument("--min-agent-steps", type=int, default=120000)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()
    report = select_candidate(
        Path(args.candidate_root),
        Path(args.forecaster),
        Path(args.out),
        expected_forecaster_sha=str(args.expected_forecaster_sha),
        min_agent_steps=int(args.min_agent_steps),
        max_steps=int(args.max_steps),
    )
    print(json.dumps({
        "status": report["status"],
        "selected_candidate": report["selected_candidate"],
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "selected_validation_aggregate": report["selected_validation_aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
