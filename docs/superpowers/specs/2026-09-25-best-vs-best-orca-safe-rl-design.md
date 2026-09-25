# Best-vs-Best Adaptive ORCA vs Safe RL Benchmark Design

Date: 2026-09-25
Branch: `research/beast-baselines-v1`

## Goal

Build a scientifically defensible benchmark where the strongest classical controller and the strongest learning controller available in this project are compared under identical sensing, dynamics, maps, seeds, and safety definitions. The experiment must never handicap ORCA to make RL win, and must permit the conclusion that ORCA is better if that is what the data show.

The progression remains:

`VO -> RVO -> ORCA -> Safe RL`

but the decisive ORCA-vs-RL experiment targets interaction uncertainty rather than simple speed scaling or clearance tuning.

## Current findings motivating the redesign

1. Dense reciprocal 6-AMR testing already shows a genuine classical progression from VO to RVO to ORCA.
2. A 1.50x human-speed stress case shows SAC+CBF can preserve liveness where the frozen ORCA configuration times out, but speed alone is tunable and therefore is not a sufficient final claim.
3. The current RL policy is not yet a best-possible uncertainty-aware learner:
   - it is a feed-forward MLP;
   - it uses only one instantaneous observation;
   - it reserves only four human slots (`N_HUMAN_SLOTS=4`), even when 12+ humans exist;
   - therefore it cannot explicitly model longer temporal intent changes or all nearby crowd members.
4. The current ORCA controller is also not the strongest reasonable uncertainty-aware classical system. Fixed time horizon and fixed margins can be improved with uncertainty-aware prediction, adaptive response, and better liveness logic.

## Design principle

Both sides receive the same causal sensor information. Neither controller receives future human trajectories or hidden intent labels.

Both sides may use a short history of observed positions and velocities. The classical side uses this history analytically; the RL side learns from it.

Tuning/training uses development and validation seeds only. Final holdout seeds are generated and frozen before the final comparison and are never used for parameter selection.

## Scenario family

### S0: Structured reciprocal traffic

Purpose: confirm the classical progression.

- 4 and 6 AMRs
- predictable human motion or no humans
- reciprocal AMR interactions
- same map/global A* planner

Expected role: VO -> RVO -> ORCA capability demonstration, not the decisive RL benchmark.

### S1: Intent-switch pedestrians

Humans follow plausible warehouse waypoint motion but occasionally change intent causally:

- sudden stop for 0.5-2.0 s;
- restart after hesitation;
- 60-120 degree turn at aisle/intersection;
- occasional 180 degree reversal;
- variable walking speed within realistic bounds;
- cross-aisle entry;
- non-yielding or partially yielding behavior.

Transitions are stochastic but physically feasible. Humans do not teleport or change velocity infinitely fast; acceleration and turning-rate bounds are enforced.

### S2: Occluded-intersection emergence

Humans may emerge from behind shelf corners into a cross-intersection. Controllers receive the human only when line-of-sight/perception makes them observable. No controller gets future emergence knowledge.

### S3: Mixed-behavior crowd

A crowd contains multiple behavior classes in the same episode:

- cooperative/yielding;
- neutral waypoint-following;
- hesitant stop/restart;
- non-yielding crossers.

The class is not supplied to either controller. It must be inferred from observation history.

### S4: Density escalation under intent uncertainty

Use the same intent model while increasing pedestrian count, with all other parameters fixed. This tests the safety/liveness frontier rather than only a single stress point.

## Shared sensing model

Use the same perception range, update rate, noise model, and history for Adaptive ORCA and Safe RL.

Shared observable features per detected entity:

- relative position;
- relative velocity;
- entity type (AMR/human);
- short position/velocity history;
- visibility mask / age of last observation if occlusion is enabled.

No future positions, future waypoints, behavior labels, or simulator RNG state are exposed.

## Classical best: Adaptive Predictive ORCA (AP-ORCA)

The frozen Peak ORCA remains a baseline. The best classical contender adds only causal, interpretable extensions.

### 1. Motion-state estimator

Estimate velocity and acceleration from a short history using filtered finite differences / alpha-beta style filtering. Clamp estimates to physically plausible pedestrian limits.

### 2. Uncertainty-aware dynamic inflation

Predict a short reachable region rather than a single constant-velocity point. Inflate the effective obstacle radius by uncertainty growth over the prediction horizon.

Inflation depends on observed acceleration/heading variance, not the controller identity or final-test seed.

### 3. Adaptive time horizon

Increase horizon when:

- time-to-collision is short;
- uncertainty is low enough for longer prediction to be useful;
- density is moderate.

Reduce horizon when:

- pedestrian intent is highly nonstationary;
- long horizons over-constrain the feasible velocity set;
- repeated yielding indicates liveness loss.

Bounds are tuned only on validation seeds.

### 4. Non-reciprocal responsibility for humans

AMR-AMR interactions remain reciprocal. Human interactions are treated as non-reciprocal or responsibility-weighted; the robot can take more than 50% responsibility when the pedestrian appears non-yielding.

### 5. Liveness/deadlock manager

Retain the strong existing replanning/recovery logic and add deterministic hysteresis/right-of-way memory so the controller does not repeatedly switch pass side or enter yield-recover-yield loops.

### 6. Acceleration-aware DD realization

Keep identical robot speed/omega limits. Optionally constrain command change per tick to physically reasonable acceleration bounds so ORCA is not credited with instantaneous velocity changes unavailable to the RL controller.

## Learning best: Temporal Set-Encoder SAC + CBF (TS-SAC-CBF)

The exact Run-11 SAC+CBF remains a baseline and is never relabeled as the upgraded model.

### 1. Variable-neighbor encoder

Replace the fixed four-human slot bottleneck with a permutation-invariant entity encoder.

Recommended implementation:

- shared MLP embeds each human/AMR entity;
- attention or masked pooling aggregates a variable number of visible entities;
- type embedding distinguishes AMRs and humans;
- ego/goal/static-ray features are fused after aggregation.

This preserves variable crowd size without growing a fixed observation vector for every possible pedestrian.

### 2. Temporal memory

Use a compact GRU over the aggregated scene representation or over per-step scene embeddings. This gives the actor/critics access to recent intent changes without future information.

A feed-forward ablation is retained to measure whether memory is actually useful.

### 3. Training distribution

Train with domain randomization over:

- 1/2/4/6 AMRs;
- pedestrian count ranges covering the final benchmark without using holdout seeds;
- speed distributions;
- stop/restart events;
- bounded heading changes/reversals;
- cooperative/non-yielding behavior;
- observation noise/occlusion if enabled in the final protocol.

Use curriculum learning from structured to mixed-intent scenes, but sample all hard modes regularly in later stages to prevent forgetting.

### 4. Safety layer

Keep CBF as the execution-level safety filter. Use the same physical robot radius and collision definition as all classical controllers.

CBF must use only currently observable/estimated obstacle states. No privileged future human state is allowed.

### 5. Reward

Primary reward remains task/liveness based:

- progress;
- goal completion;
- collision penalty;
- prolonged stall/deadlock penalty;
- small control/smoothness cost.

Do not reward arbitrary large clearance; this would confound safety with conservatism.

Potential additional term: penalize repeated CBF overrides so the policy learns actions the safety filter does not constantly need to repair. This term must be validated because earlier shaping changes increased collisions in high-density tests.

## Approaches considered

### A. Only retune frozen ORCA and reuse Run-11 RL

Pros: cheapest, fastest.

Cons: not truly best-vs-best; current RL cannot represent all humans or temporal intent. Rejected for final thesis benchmark.

### B. Upgrade only ORCA, keep RL fixed

Pros: gives classical method a very fair challenge.

Cons: if RL loses, we would not know whether the loss is caused by its four-human/no-memory observation bottleneck. Useful as an intermediate baseline, not sufficient as the final benchmark.

### C. Upgrade both sides under identical sensing/history (recommended)

Pros: strongest scientific comparison; directly tests whether learned temporal interaction reasoning provides value beyond a strong analytical predictor plus adaptive ORCA.

Cons: requires retraining and careful validation; more compute.

Recommendation: Approach C.

## Tuning and freeze protocol

### Development

Use dedicated development seeds to debug correctness and broad parameter ranges.

### Validation

Use a separate fixed validation set to choose:

- AP-ORCA horizon bounds, uncertainty gains, responsibility weights, hysteresis/recovery parameters;
- TS-SAC-CBF architecture size, history length, curriculum/reward variants, checkpoint selection.

Each side gets a comparable tuning budget. Report the search budget and candidate count.

### Final holdout

Generate fresh holdout seeds after both sides are frozen. No parameter, architecture, checkpoint, or threshold may change after holdout results are observed.

If a bug is found after holdout, invalidate the affected holdout and generate a new untouched holdout after the bug fix is frozen.

## Metrics

Primary:

- individual success rate;
- fleet success rate;
- collision rate;
- timeout rate;
- throughput.

Secondary:

- traversal time among successful AMRs;
- path length among successful AMRs;
- minimum physical clearance;
- controller compute time;
- CBF intervention fraction;
- ORCA stop/yield fraction;
- recovery count;
- feasible-set failure / infeasible events;
- oscillation / steering-direction flip rate.

Report 95% confidence intervals for binary rates and paired seed-wise differences where appropriate.

## Decision rule

No single metric decides the winner.

A method is considered meaningfully better only if it improves mission completion/liveness or safety on the hard interaction scenarios without an unacceptable regression in collision rate or compute cost.

Examples:

- If AP-ORCA matches TS-SAC-CBF success with lower compute and equal safety, ORCA wins that scenario.
- If TS-SAC-CBF materially improves success/fleet completion while maintaining equal collision rate, RL demonstrates added value.
- If RL improves success but increases collisions, call it a liveness-safety trade-off, not a clean win.

## Verification outputs

For every final scenario family produce:

1. machine-readable JSON results;
2. per-seed episode records;
3. aggregate table with confidence intervals;
4. at least one representative same-seed video for each controller;
5. a proof workflow that asserts metadata and runs `ffprobe` on videos;
6. config/checkpoint hashes so results are reproducible.

## Acceptance criteria before thesis claim

- AP-ORCA and TS-SAC-CBF both pass unit/regression tests.
- Same causal sensing and physical limits verified by tests.
- Both tuned only on dev/validation seeds.
- Final configs/checkpoint frozen before holdout.
- At least 30 holdout episodes per major scenario family, preferably more for binary-rate precision.
- No claim of global superiority from a single seed or one stress parameter.
- Thesis wording distinguishes reciprocal structured traffic from uncertain/non-reciprocal human interaction.

## Expected thesis framing

The intended scientific question is:

> When a strong predictive, uncertainty-aware ORCA controller and a temporal Safe-RL controller receive the same causal observations, which method better preserves safety and mission liveness as human intent becomes uncertain, non-reciprocal, and partially observable?

The result is valid regardless of which method wins.
