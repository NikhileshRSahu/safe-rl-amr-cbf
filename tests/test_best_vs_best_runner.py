import numpy as np
import torch

from benchmark.best_vs_best_protocol import ScenarioSpec
from benchmark.best_vs_best_runner import (
    aggregate_episode_rows,
    run_ap_orca_episode,
    run_stasac_episode,
)
from benchmark.spatiotemporal_policy import STASACActor
from benchmark.stasac_warehouse import EGO_DIM


def test_episode_runners_return_same_metric_contract_on_same_seed():
    spec = ScenarioSpec("density_06", "density", humans=6, n_amr=4, randomness_level="medium")
    torch.manual_seed(2)
    actor = STASACActor(EGO_DIM)
    rl = run_stasac_episode(actor, spec, seed=11100, max_steps=12)
    orca = run_ap_orca_episode(spec, seed=11100, max_steps=12)
    required = {
        "seed", "scenario", "agents", "success", "collision", "timeout",
        "fleet_success", "steps", "throughput_per_min", "cbf_interventions",
    }
    assert required <= rl.keys()
    assert required <= orca.keys()
    assert rl["agents"] == orca["agents"] == 4
    assert rl["success"] + rl["collision"] + rl["timeout"] == 4
    assert orca["success"] + orca["collision"] + orca["timeout"] == 4


def test_aggregate_episode_rows_computes_agent_and_fleet_rates():
    rows = [
        dict(agents=4, success=4, collision=0, timeout=0, fleet_success=True, throughput_per_min=4.0),
        dict(agents=4, success=2, collision=1, timeout=1, fleet_success=False, throughput_per_min=2.0),
    ]
    agg = aggregate_episode_rows(rows)
    assert agg.agents == 8
    assert agg.episodes == 2
    assert agg.success_rate == 0.75
    assert agg.collision_rate == 0.125
    assert agg.timeout_rate == 0.125
    assert agg.fleet_success_rate == 0.5
    assert agg.throughput_per_min == 3.0
