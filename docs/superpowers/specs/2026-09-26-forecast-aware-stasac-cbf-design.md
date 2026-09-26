# Forecast-Aware ST-SAC + CBF Design

## Goal

Extend the current spatio-temporal risk-attention SAC navigator with a causal human-motion forecasting subsystem so that the RL policy can anticipate short-horizon pedestrian motion in warehouse interactions while remaining the actual local navigator. The comparison target remains a strong AP-ORCA+CBF baseline under the same sensing and safety assumptions.

## Research Question

Can a forecast-aware recurrent SAC policy improve human-aware warehouse navigation efficiency and/or safety over both (a) the current temporal SAC+CBF policy and (b) AP-ORCA+CBF, particularly under hesitation, dense flow, mixed traffic, occlusion, and changing intent, without relying on future-state leakage or an unfair sensing advantage?

## Scope

Primary method:

A* global route -> causal observation/history -> human-motion forecaster -> forecast/risk representation -> ST-SAC actor -> normalized (v, omega) -> shared CBF safety filter.

The RL policy remains the runtime navigator and directly produces continuous differential-drive commands. ORCA is never used in the deployed RL control path. AP-ORCA may be used as a training teacher and remains the primary classical baseline.

## Non-Goals

- Do not use an LLM/VLM in the real-time low-level control loop.
- Do not let the forecaster directly command the robot.
- Do not claim superiority from easy scenarios where AP-ORCA already achieves near-perfect success.
- Do not use privileged simulator future state at deployment or evaluation.
- Do not replace the common CBF shield with a method-specific safety layer.

## Architecture

### 1. Causal Human History

Maintain a per-track history of only observations available to the robot at or before the current time:

- relative position
- relative velocity
- visibility / observation age
- optional heading estimate
- observation mask

History must respect the same range and shelf line-of-sight model already shared by ST-SAC and AP-ORCA.

Suggested window: 1.0-2.0 s of recent observations, sampled at the simulator control rate or downsampled deterministically.

No future simulator state, goal, RNG state, scripted waypoint, or hidden human intent label may enter this buffer.

### 2. Human Motion Forecaster

Initial implementation: compact GRU-based forecaster with shared weights across human tracks.

Input per human j:

H_j(t) = {x_j(t-k), ..., x_j(t)}

Output for forecast steps tau=1..H:

- predicted mean relative position mu_j,tau = (x, y)
- uncertainty sigma_j,tau = (sigma_x, sigma_y) or a bounded isotropic sigma

Recommended horizon: 1.5-2.0 s, using 5-10 forecast points.

The interface must be replaceable later by a transformer without changing the SAC actor API.

### 3. Forecast Risk Features

For each human, derive interpretable causal features from the predicted trajectory and uncertainty, for example:

- minimum predicted separation
- predicted time to minimum separation
- probability/score of entering the robot corridor
- prediction uncertainty
- stop/hesitation probability proxy
- turn/change-of-intent proxy
- visibility age

The exact feature set must remain bounded and normalized.

A useful risk form is:

R_j = max_tau exp(-tau*dt/T) * sigmoid((d_safe - d_hat_j,tau)/s) * (1 + lambda*sigma_j,tau)

where d_hat is the predicted robot-human separation under a short constant robot-motion hypothesis or route-aligned reference motion.

This feature is advisory to SAC; CBF remains the hard safety layer.

### 4. ST-SAC Integration

The existing recurrent actor keeps its temporal GRU/attention structure. Forecast-derived features are added to each entity embedding or fused as a separate forecast context vector.

Preferred first implementation:

entity_j = [current causal features, forecast summary features]

This preserves permutation invariance and variable crowd size.

The actor still outputs:

a_t = [v_norm, omega_norm]

which is mapped to physical differential-drive limits and passed through the shared CBF filter.

### 5. Training the Forecaster

Train the forecaster separately before joint RL integration.

Training data may use simulator trajectories, but samples must be constructed as:

past/current observation history -> future trajectory target

Targets may come from simulator truth because they are labels used only during offline training; those future states must never be included in the forecaster input.

Use development training seeds only. Keep evaluation seeds disjoint.

Metrics:

- ADE (average displacement error)
- FDE (final displacement error)
- uncertainty calibration / coverage if probabilistic uncertainty is predicted
- per-scenario metrics for crossing, hesitation, reversal, dense flow, mixed traffic, occlusion

A constant-velocity predictor must be retained as a baseline.

### 6. RL Training Protocol

After the forecaster passes its own evaluation gate:

1. freeze the initial forecaster for the first RL experiments;
2. train Forecast-ST-SAC+CBF using the existing performance-gated AP-ORCA teacher and safety/success-gated curriculum;
3. compare against the same architecture with forecast features removed;
4. only consider joint fine-tuning of the forecaster if the frozen-forecaster experiment shows a clear bottleneck.

This avoids simultaneously changing too many components.

## Fairness Rules

### Same raw information budget

AP-ORCA+CBF and Forecast-ST-SAC+CBF receive the same causal visible positions/velocities/history budget. The learned method may transform this history through a learned forecaster; AP-ORCA may use its own causal analytical prediction/adaptive horizon, but neither receives ground-truth future trajectories.

### Same route/task

Use the same A* route, start/goal, robot geometry, dynamic-obstacle realization, control limits, simulation horizon, and collision definition.

### Same safety layer

Both methods use the same CBF execution shield in the primary comparison. A separate no-CBF ablation may be reported, but must not be confused with the primary best-vs-best result.

### Frozen evaluation

Architecture, forecaster checkpoint, RL checkpoint-selection rule, AP-ORCA configuration, metrics, and win criteria are frozen before validation/holdout evaluation.

## Data Partitions

Maintain strict separation:

- gradient-training seeds: current 10100-10115 development subset
- competence-gate seeds: current 10116-10119 subset
- development screening: 10300-10319
- validation: 11100-11119
- final holdout: 13100-13129

Forecaster training/evaluation must respect the same split philosophy. No validation or holdout future trajectories may be used for forecaster training.

## Benchmark Scenarios

Keep easy scenarios as sanity checks:

- human crossing
- blind shelf corner
- human reversal
- forklift-like crossing

Primary discriminative scenarios:

- human hesitation / stop-restart
- dense human flow
- mixed local traffic
- interaction ambiguity at warehouse intersections
- partial shelf occlusion combined with later intent change

If AP-ORCA reaches near-perfect performance in a scenario, that scenario is not used as the headline proof of RL superiority.

## Metrics

Primary:

- success rate
- collision rate
- timeout rate
- mission/fleet success when applicable
- throughput

Secondary but important:

- traversal time
- path length
- CBF intervention count/rate
- stop/yield duration
- action/turn oscillation
- commitment reversals
- replans/backtracks if used
- minimum clearance (diagnostic, not headline)

Forecast metrics:

- ADE
- FDE
- uncertainty calibration/coverage

## Win Criterion

Forecast-aware RL may be called better only if safety is not materially degraded.

At minimum:

collision_rate_RL <= collision_rate_APORCA + epsilon

and one or more meaningful navigation improvements are demonstrated on frozen evaluation data, preferably:

success_rate_RL > success_rate_APORCA

and/or

throughput_RL > throughput_APORCA

and/or

traversal_delay_RL < traversal_delay_APORCA

with fewer or comparable pathological interaction behaviors such as repeated stop-go oscillation or excessive CBF rescue.

A lower timeout caused only by unsafe/aggressive motion is not a win.

## TDD / Verification Requirements

Before production integration, tests must prove:

1. history buffers never read future timestamps;
2. permuting human-track order does not change the aggregate scene representation;
3. 0-human and many-human cases are valid;
4. forecast tensors have deterministic shapes and finite bounded outputs;
5. masking/occlusion works and stale observations are marked correctly;
6. constant-velocity synthetic trajectories yield low forecast error;
7. scripted stop/turn trajectories produce non-zero uncertainty/error relative to constant velocity;
8. RL actor remains bounded and deterministic in evaluation mode;
9. disabling forecast features reproduces the non-forecast ST-SAC interface;
10. evaluation seed partitions remain disjoint.

## Implementation Files

New modules expected:

- benchmark/human_history.py
- benchmark/human_forecaster.py
- benchmark/forecast_risk_features.py
- benchmark/train_human_forecaster.py

Likely modified modules:

- benchmark/stasac_warehouse.py
- benchmark/spatiotemporal_policy.py
- benchmark/stasac_training_runner.py
- benchmark/best_vs_best_runner.py
- benchmark/best_vs_best_protocol.py

Tests should be added alongside existing STASAC development tests.

## Staged Research Plan

Phase A: causal history + constant-velocity baseline.

Phase B: GRU forecaster trained and evaluated independently.

Phase C: frozen forecaster integrated into ST-SAC.

Phase D: smoke training and development screening against non-forecast ST-SAC and AP-ORCA.

Phase E: choose/freeze the best development checkpoint and run formal validation.

Phase F: only after validation criteria are met, run untouched final holdout and render representative videos.

## Expected Scientific Contribution

The intended claim is not that RL universally dominates ORCA. The defensible contribution is that a forecast-aware temporal safe-RL local navigator can exploit learned short-horizon human-motion predictions to improve navigation decisions in uncertain warehouse interactions while retaining an explicit CBF safety shield, compared fairly against a strong adaptive predictive ORCA baseline under identical causal sensing and task conditions.
