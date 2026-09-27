from copy import deepcopy

from benchmark.select_fully_trained_forecast_stasac import qualification_errors


FORECASTER_SHA = "a" * 64


def mastered_summary():
    return {
        "metadata": {
            "agent_steps": 120000,
            "seed": 71,
            "architecture": "forecast_spatiotemporal_risk_attention_sac_cbf_v1",
            "forecast_enabled": True,
            "forecaster_sha256": FORECASTER_SHA,
            "training_seed_split": "development_gradient_only",
            "final_stage": 3,
        },
        "competence_gates": [
            {
                "stage": 0,
                "promoted": True,
                "policy": {"episodes": 12, "success_rate": 0.92, "collision_rate": 0.00},
            },
            {
                "stage": 1,
                "promoted": True,
                "policy": {"episodes": 12, "success_rate": 0.90, "collision_rate": 0.01},
            },
            {
                "stage": 2,
                "promoted": True,
                "policy": {"episodes": 12, "success_rate": 0.88, "collision_rate": 0.02},
            },
            {
                "stage": 3,
                "promoted": False,
                "policy": {"episodes": 12, "success_rate": 0.86, "collision_rate": 0.01},
            },
        ],
    }


def test_mastered_stage3_candidate_is_eligible_even_though_final_stage_cannot_promote_further():
    assert qualification_errors(
        mastered_summary(),
        expected_forecaster_sha=FORECASTER_SHA,
        min_agent_steps=120000,
    ) == []


def test_stage3_must_pass_its_own_competence_gate():
    summary = mastered_summary()
    summary["competence_gates"][-1]["policy"]["success_rate"] = 0.84
    errors = qualification_errors(
        summary,
        expected_forecaster_sha=FORECASTER_SHA,
        min_agent_steps=120000,
    )
    assert "stage3_final_gate_not_mastered" in errors


def test_wrong_predictor_lineage_is_rejected():
    summary = mastered_summary()
    summary["metadata"]["forecaster_sha256"] = "b" * 64
    errors = qualification_errors(
        summary,
        expected_forecaster_sha=FORECASTER_SHA,
        min_agent_steps=120000,
    )
    assert "wrong_forecaster_sha" in errors


def test_missing_prior_stage_promotion_is_rejected():
    summary = deepcopy(mastered_summary())
    summary["competence_gates"][1]["promoted"] = False
    errors = qualification_errors(
        summary,
        expected_forecaster_sha=FORECASTER_SHA,
        min_agent_steps=120000,
    )
    assert "stage1_not_promoted" in errors
