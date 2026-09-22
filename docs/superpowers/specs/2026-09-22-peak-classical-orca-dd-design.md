# Peak Classical Baseline Design — A* + ORCA-DD

**Date:** 2026-09-22

## Purpose

Build the strongest defensible classical baseline for the Safe RL AMR thesis so the comparison against SAC+CBF remains fair even when the classical controller is tuned aggressively.

The benchmark must prefer a strong classical result over preserving any apparent RL advantage. If the improved baseline closes or reverses the gap, the thesis should report that result honestly.

## Current Baseline Limitation

The current `benchmark/beast_classical.py` controller is not full ORCA/RVO2. It samples candidate holonomic velocities, rejects candidates using closest-approach checks, then converts the selected 2-D velocity into a unicycle `(v, omega)` command. This creates a mismatch between the trajectory checked for safety and the trajectory actually executed.

Other limitations:
- finite velocity sampling can miss feasible high-quality commands;
- static obstacle checking assumes constant linear motion while execution may turn;
- reciprocal avoidance is heuristic rather than half-plane constrained;
- deadlock handling is heuristic and mixed into the core avoidance logic.

## Target Architecture

### Global planner

Use inflated-grid A* as the global path planner.

Requirements:
- same warehouse geometry as the RL environment;
- robot-radius-aware inflation;
- configurable additional planning margin;
- deterministic path generation;
- periodic replanning and replanning on persistent blockage;
- path smoothing only if every smoothed segment remains collision-free under the same inflation model.

### Local collision avoidance

Replace the sampled velocity-search core with ORCA-style reciprocal collision-avoidance constraints.

For controlled AMRs:
- construct reciprocal pairwise velocity-obstacle / ORCA half-plane constraints;
- split avoidance responsibility reciprocally;
- include right-of-way priority only as a bounded bias, never as permission to violate hard safety constraints.

For pedestrians:
- treat them as non-reciprocal moving obstacles;
- the AMR takes full avoidance responsibility;
- use current observed pedestrian position and velocity only;
- do not use future ground-truth trajectories.

### Differential-drive feasibility

The local planner must respect the robot's unicycle/differential-drive dynamics.

The selected command must be safety-checked in the same motion model that will be executed.

Preferred implementation:
1. build ORCA feasible velocity constraints in Cartesian velocity space;
2. generate dynamically reachable differential-drive candidate commands over the control interval;
3. project each candidate command to its realized translational velocity / short-horizon arc;
4. accept only candidates satisfying the ORCA constraints and static-clearance constraints;
5. choose the feasible command minimizing deviation from preferred path-following motion plus bounded smoothness cost.

Do not safety-check a holonomic velocity and then execute a materially different unicycle trajectory.

### Static obstacle safety

Static checks must follow the executed short-horizon unicycle arc, not only a straight-line extrapolation.

Check:
- shelves;
- walls;
- robot radius;
- configured static margin;
- sufficient temporal samples to avoid tunneling.

### Deadlock and liveness

ORCA is the primary avoidance mechanism.

Fallback liveness logic may include:
- priority-aware yielding;
- deterministic passing side;
- blocked-path replanning;
- short bounded recovery behavior.

Recovery must never bypass hard collision constraints.

## Fair Tuning Protocol

Do not tune the classical baseline on the final benchmark seeds.

Use three disjoint seed sets:

- **Development seeds:** implementation debugging only.
- **Validation seeds:** hyperparameter tuning and model selection.
- **Test seeds:** final thesis comparison only.

Freeze all classical parameters before running test seeds.

Tune at minimum:
- ORCA time horizon;
- neighbor distance;
- peer safety margin;
- pedestrian safety margin;
- static margin;
- A* planning inflation;
- preferred speed;
- replanning interval;
- path lookahead;
- differential-drive command lattice density;
- smoothness penalty;
- deadlock trigger threshold.

Use a validation objective that rewards completion and safety rather than route length alone.

Primary ordering:
1. minimize collisions;
2. minimize timeouts;
3. maximize fleet success;
4. maximize per-agent success;
5. then minimize traversal time/path length.

## Benchmark Fairness

Both controllers must use:
- identical world seeds;
- identical start and goal perturbations;
- identical pedestrian initial states and velocities;
- identical episode horizon;
- identical robot speed and angular-rate limits;
- identical collision geometry;
- identical termination rules.

The classical controller may use the same instantaneous environment state available to the RL policy, but no privileged future information.

SAC+CBF continues to use its exact frozen checkpoint.

## Evaluation

Run the final frozen comparison at:
- 2 AMRs;
- 4 AMRs;
- 6 AMRs.

Use at least the existing 30-seed test protocol.

Report:
- per-agent success rate + 95% CI;
- collision rate + 95% CI;
- timeout rate + 95% CI;
- fleet success rate + 95% CI;
- fleet collision rate + 95% CI;
- collision types;
- successful-path length;
- successful traversal time;
- throughput per minute;
- minimum physical clearance;
- controller compute time per agent-step.

Also record classical-specific diagnostics:
- number of ORCA constraints;
- infeasible-command events;
- stop/yield fraction;
- replan count;
- recovery/deadlock count.

## Acceptance Criteria

The replacement baseline is acceptable only when:

1. it is based on actual ORCA-style reciprocal constraints rather than only sampled closest-approach rejection;
2. differential-drive feasibility is incorporated before the final command is declared safe;
3. static safety follows the executed arc;
4. validation tuning is separated from test evaluation;
5. all test seeds are identical between SAC+CBF and the classical controller;
6. regression tests cover reciprocal head-on crossing, perpendicular crossing, pedestrian crossing, narrow aisles, static turns, multi-AMR congestion, and no-obstacle path following;
7. the benchmark exports enough diagnostics to explain failures rather than only aggregate rates.

## Files Expected to Change

- `benchmark/beast_classical.py` — replace heuristic sampled RVO core with ORCA-DD-strength controller.
- `benchmark/evaluate_beast_controllers.py` — add frozen-config loading and classical diagnostics.
- `tests/test_beast_classical.py` — focused local-controller and kinematic safety tests.
- `tests/test_beast_benchmark.py` — seed separation and fairness tests.
- `.github/workflows/beast-controller-eval.yml` — run validation-selected frozen config and final benchmark.
- Optional: `benchmark/tune_beast_classical.py` — validation-only hyperparameter search.

## Non-Goals

- No learned component in the classical baseline.
- No access to future pedestrian trajectories.
- No changing the RL checkpoint to improve the comparison.
- No tuning on the final test seeds.
- No deliberately weakening classical control to preserve thesis conclusions.
