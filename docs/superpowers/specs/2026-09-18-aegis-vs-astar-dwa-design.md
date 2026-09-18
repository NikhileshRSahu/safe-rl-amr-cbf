# Aegis-RL vs A* + DWA Comparison Design

## Goal
Build a reproducible benchmark and video pipeline that compares the current Aegis controller (SAC + CBF-QP + circulation/deadlock recovery) against a classical A* global planner plus Dynamic Window Approach (DWA) local controller in the same warehouse environment.

## Research integrity
- Both methods must start from the exact same robot pose, goal, dynamic-obstacle state, actuator state, observation history, and RNG state.
- Scenario selection may rank representative cases where Aegis wins, but selected clips must be labelled as selected examples.
- Aggregate benchmark CSVs must also be written so the presentation does not imply selected clips are the whole evaluation.
- The A*+DWA baseline must not receive the Aegis CBF safety filter.
- Velocity, angular-velocity, robot radius, obstacle geometry, timestep, episode limit, and collision definitions come from the existing environment/config.

## Architecture
### classical_baseline.py
Provides a self-contained classical controller:
1. Occupancy grid derived from warehouse bounds and inflated shelf geometry.
2. 8-connected A* global path planner.
3. Path simplification / waypoint selection.
4. DWA local velocity search over dynamically reachable commands.
5. Cost terms for waypoint progress, heading, clearance, speed, and collision rejection.
6. Conversion from physical (v, omega) to the normalized action space expected by AMRWarehouseEnv.

### compare_controllers.py
Orchestrates paired evaluation:
1. Reset an Aegis environment with a seed.
2. Snapshot state with get_state().
3. Run Aegis deterministically from that state.
4. Create a second environment with CBF disabled and restore the same snapshot.
5. Run A*+DWA from that identical state.
6. Save per-episode and per-step metrics for both controllers.
7. Rank seeds using an explicit win rule.
8. Optionally render selected paired runs to MP4.

## Paired-state fairness
The environment's get_state()/set_state() already captures robot state, goal, dynamic obstacles, step count, actuator state, observation buffer, LiDAR history, CBF diagnostics, and RNG state. The comparison runner will use this snapshot for both methods.

The classical environment is created with use_cbf_filter=False. Dynamic obstacles still evolve according to the same environment model and copied RNG state.

## A* planner
The planner operates on a configurable occupancy grid (default resolution 0.25 m). Static shelves and map boundaries are inflated by robot radius plus a planning margin.

A* uses Euclidean heuristic and 8-connected movement. If no path exists, the baseline records planner_failure rather than substituting a hidden recovery method.

## DWA controller
At every step DWA:
- derives the dynamic velocity window from current actual velocity, acceleration limits, angular acceleration limits, and DT;
- samples candidate (v, omega) commands;
- forward-simulates each candidate over a short horizon;
- rejects trajectories that collide with walls, inflated shelves, or predicted moving obstacle positions;
- scores remaining trajectories by local waypoint progress, final heading, minimum clearance, forward speed, and smoothness from the previous command.

DWA follows A* waypoints and advances the waypoint when within a configurable tolerance. It may replan if the current path becomes unusable or after a configurable interval.

## Metrics
Per episode:
- outcome: success / collision / timeout / planner_failure
- elapsed simulation time
- step count
- path length
- minimum physical clearance
- mean absolute angular velocity
- control smoothness (sum of absolute command deltas)
- collision type
- final distance to goal
- Aegis CBF intervention count

Aggregate:
- success rate
- collision rate
- timeout rate
- mean successful navigation time
- mean successful path length
- mean minimum clearance
- mean smoothness

## Aegis-win ranking
A seed is a strong representative Aegis win when, in order of priority:
1. Aegis succeeds and A*+DWA does not.
2. Both succeed, but Aegis has materially larger minimum clearance.
3. Both succeed with comparable safety, but Aegis is faster and/or shorter.
4. Ties are broken by smoother control.

The runner writes the reason and metric deltas for every selected seed. It never changes the environment after observing baseline performance.

## Video
The optional video shows the two real simulations side-by-side:
- left: A* + DWA
- right: Aegis-RL
- identical start/goal and dynamic obstacle realization
- live status: time, path length, min clearance, outcome
- Aegis panel additionally shows CBF interventions
- final card shows measured metrics and the text "Selected representative benchmark scenario"

Frames come from the existing rgb_array renderer.

## CLI
compare_controllers.py accepts:
- checkpoint path
- seed start / number of seeds
- output directory
- number of representative clips
- --render-top-k
- optional DWA/grid parameters for controlled experiments

The checkpoint is not committed because checkpoints are intentionally gitignored in this repository.

## Testing
Unit tests cover:
- occupancy inflation and A* path validity
- no-path handling
- normalized action conversion
- DWA collision rejection against a moving obstacle
- DWA command bounds
- deterministic Aegis-win ranking
- paired snapshot restoration

Integration tests may use a short environment episode with a stub policy so they do not require a large model checkpoint.

## Non-goals
- No retraining of Aegis.
- No tuning scenarios after seeing a single video.
- No ROS/Gazebo migration in this change.
- No attempt to make DWA intentionally weak.
