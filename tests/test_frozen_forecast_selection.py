import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from benchmark.best_vs_best_runner import peak_orca_config
from benchmark.run_frozen_forecast_vs_peak_orca import _load_frozen_orca_selection
from benchmark.select_fully_trained_forecast_stasac import qualification_errors
from benchmark.select_peak_ap_orca import ADAPTIVE_CANDIDATES


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


def frozen_orca_selection():
    return {
        "status": "selected_and_frozen",
        "controller": "Peak AP-ORCA+CBF",
        "selection_protocol": "validation_only_safety_first_lexicographic_v1",
        "perception_contract": "causal_observable_only_no_hidden_intent_no_future_state",
        "base_peak_orca_config": peak_orca_config().__dict__,
        "holdout_seeds_used": [],
        "selected_candidate": "default",
        "adaptive_config": asdict(ADAPTIVE_CANDIDATES["default"]),
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


def test_peak_ap_orca_candidate_set_is_predeclared_and_contains_current_default():
    assert "default" in ADAPTIVE_CANDIDATES
    assert len(ADAPTIVE_CANDIDATES) == 6
    for config in ADAPTIVE_CANDIDATES.values():
        assert config.horizon_min > 0.0
        assert config.horizon_max >= config.horizon_min
        assert config.max_uncertainty_extra >= 0.0
        assert 0.5 <= config.min_human_responsibility <= 1.0
        assert config.occlusion_memory_seconds > 0.0


def test_frozen_peak_ap_orca_selection_loads_exact_config(tmp_path):
    payload = frozen_orca_selection()
    path = tmp_path / "peak_ap_orca_selection.json"
    path.write_text(json.dumps(payload))
    loaded, config = _load_frozen_orca_selection(path)
    assert loaded["selected_candidate"] == "default"
    assert asdict(config) == payload["adaptive_config"]


def test_peak_ap_orca_selection_rejects_holdout_reuse(tmp_path):
    payload = frozen_orca_selection()
    payload["holdout_seeds_used"] = [14000]
    path = tmp_path / "peak_ap_orca_selection.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="already used"):
        _load_frozen_orca_selection(path)


def test_peak_ap_orca_selection_rejects_weakened_or_changed_base(tmp_path):
    payload = frozen_orca_selection()
    payload["base_peak_orca_config"]["human_margin"] = 0.0
    path = tmp_path / "peak_ap_orca_selection.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="base configuration"):
        _load_frozen_orca_selection(path)
