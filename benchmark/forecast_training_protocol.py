from __future__ import annotations

FORECAST_GATE_SCENARIOS = (
    "human_crossing",
    "blind_shelf_corner",
    "human_hesitation",
    "human_reversal",
    "dense_human_flow",
    "mixed_local_traffic",
)

FORECAST_REQUIRED_WIN_SCENARIOS = (
    "human_crossing",
    "human_hesitation",
    "human_reversal",
    "dense_human_flow",
)


def forecaster_training_seeds() -> tuple[int, ...]:
    return tuple(range(10100, 10116))


def forecaster_eval_seeds() -> tuple[int, ...]:
    return tuple(range(10116, 10120))


def learned_forecaster_beats_cv(learned: dict, cv: dict) -> tuple[bool, dict]:
    failed = []
    if float(learned["ade"]) >= float(cv["ade"]) or float(learned["fde"]) >= float(cv["fde"]):
        failed.append("aggregate")
    learned_by = learned.get("per_scenario", {})
    cv_by = cv.get("per_scenario", {})
    for scenario in FORECAST_REQUIRED_WIN_SCENARIOS:
        if scenario not in learned_by or scenario not in cv_by:
            failed.append(scenario)
            continue
        lm = learned_by[scenario]
        cm = cv_by[scenario]
        if float(lm["ade"]) >= float(cm["ade"]) or float(lm["fde"]) >= float(cm["fde"]):
            failed.append(scenario)
    failed = list(dict.fromkeys(failed))
    return len(failed) == 0, {
        "passed": len(failed) == 0,
        "failed_scenarios": failed,
        "required_scenarios": list(FORECAST_REQUIRED_WIN_SCENARIOS),
    }
