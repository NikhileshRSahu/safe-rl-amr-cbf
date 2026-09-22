# Peak A* + ORCA-DD Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current sampled “Beast” controller with a thesis-defensible A* + ORCA-DD-strength classical baseline that is tuned on validation seeds only and evaluated fairly against the frozen SAC+CBF checkpoint.

**Architecture:** Keep inflated-grid A* for global planning. Replace the heuristic sampled closest-approach local planner with ORCA-style reciprocal half-plane constraints, then select only dynamically reachable unicycle commands whose realized short-horizon arc satisfies ORCA and static-clearance constraints. Separate tuning from final evaluation with disjoint development, validation, and test seed sets.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyTorch (evaluation only), pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-22-peak-classical-orca-dd-design.md`

## Global Constraints

- Use actual ORCA-style reciprocal constraints rather than only sampled closest-approach rejection.
- Differential-drive feasibility must be incorporated before a command is declared safe.
- Static safety must be checked along the executed unicycle arc.
- Development, validation, and test seed sets must be disjoint.
- Classical hyperparameters must be frozen before final test evaluation.
- SAC+CBF must use the exact existing frozen checkpoint.
- Classical control must not use future pedestrian trajectories.
- Both controllers must use identical world seeds, dynamics, geometry, episode horizon, termination, and speed/angular-rate limits.
- Do not weaken the classical baseline to preserve an RL advantage.

## Review Focus

- Head-on AMR interaction where both agents must split responsibility without oscillation.
- Perpendicular crossing where one robot is already inside the conflict region and both remain dynamically feasible.
- Pedestrian crossing where the AMR must take full avoidance responsibility without future trajectory access.
- Narrow-aisle turns where straight-line holonomic safety would be misleading but the executed unicycle arc must remain collision-free.
- Dense 6-AMR congestion where the controller must prefer safe stopping/replanning over unsafe progress.

---

### Task 1: ORCA Geometry Primitives

**Files:**
- Create: `benchmark/orca_geometry.py`
- Create: `tests/test_orca_geometry.py`

**Interfaces:**
- Consumes: NumPy arrays representing 2-D positions and velocities.
- Produces:
  - `OrcaLine(point: np.ndarray, normal: np.ndarray)`
  - `build_orca_line(rel_position, rel_velocity, combined_radius, time_horizon, responsibility) -> OrcaLine`
  - `satisfies_orca_line(velocity, line, tol=1e-9) -> bool`
  - `project_velocity_to_orca_halfplanes(preferred_velocity, lines, max_speed) -> np.ndarray | None`

- [ ] **Step 1: Write failing tests for half-plane construction**

```python
import numpy as np
from benchmark.orca_geometry import build_orca_line, satisfies_orca_line

def test_head_on_constraint_rejects_collision_course():
    rel_p = np.array([2.0, 0.0])
    rel_v = np.array([-2.0, 0.0])
    line = build_orca_line(rel_p, rel_v, combined_radius=0.6, time_horizon=2.0, responsibility=0.5)
    assert not satisfies_orca_line(np.array([1.0, 0.0]), line)

def test_safe_lateral_velocity_satisfies_constraint():
    rel_p = np.array([2.0, 0.0])
    rel_v = np.array([-2.0, 0.0])
    line = build_orca_line(rel_p, rel_v, combined_radius=0.6, time_horizon=2.0, responsibility=0.5)
    assert satisfies_orca_line(np.array([0.0, 0.8]), line)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:
```bash
pytest tests/test_orca_geometry.py -v
```

Expected: import or symbol failures because `benchmark/orca_geometry.py` does not yet exist.

- [ ] **Step 3: Implement ORCA line representation and pairwise construction**

Implement:
- time-horizon velocity obstacle geometry;
- collision-region handling for overlapping/near-contact agents;
- configurable reciprocal responsibility fraction;
- normalized line normals;
- numerically stable epsilon handling.

- [ ] **Step 4: Add failing projection tests**

```python
from benchmark.orca_geometry import OrcaLine, project_velocity_to_orca_halfplanes

def test_projection_returns_closest_feasible_velocity():
    lines = [
        OrcaLine(point=np.array([0.2, 0.0]), normal=np.array([1.0, 0.0])),
    ]
    v = project_velocity_to_orca_halfplanes(np.array([1.0, 0.0]), lines, max_speed=1.0)
    assert v is not None
    assert v[0] <= 0.2 + 1e-6

def test_projection_respects_speed_limit():
    v = project_velocity_to_orca_halfplanes(np.array([3.0, 0.0]), [], max_speed=1.0)
    assert np.linalg.norm(v) <= 1.0 + 1e-9
```

- [ ] **Step 5: Implement half-plane projection**

Use deterministic incremental 2-D linear programming:
- first enforce speed-circle bound;
- then process each ORCA line;
- if a line is violated, project onto the feasible boundary while preserving all previously satisfied constraints;
- return `None` only if constraints are genuinely infeasible within the speed circle.

- [ ] **Step 6: Run geometry tests**

Run:
```bash
pytest tests/test_orca_geometry.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add benchmark/orca_geometry.py tests/test_orca_geometry.py
git commit -m "feat: add ORCA geometry primitives"
```

### Task 2: Differential-Drive Reachability and Arc Safety

**Files:**
- Create: `benchmark/dd_motion.py`
- Create: `tests/test_dd_motion.py`

**Interfaces:**
- Consumes: current pose, current speed/angular rate, `(v_cmd, omega_cmd)`, `DT`, horizon.
- Produces:
  - `simulate_unicycle_arc(...) -> np.ndarray` with shape `[T, 3]`
  - `reachable_commands(...) -> list[tuple[float, float]]`
  - `command_terminal_velocity(theta, v_cmd, omega_cmd, dt) -> np.ndarray`
  - `arc_static_clearance(arc, shelves, world_bound, robot_radius) -> float`

- [ ] **Step 1: Write failing kinematic tests**

```python
import numpy as np
from benchmark.dd_motion import simulate_unicycle_arc

def test_zero_omega_produces_straight_arc():
    arc = simulate_unicycle_arc(np.array([0.0,0.0,0.0]), 1.0, 0.0, dt=0.1, horizon=1.0)
    assert np.allclose(arc[-1,:2], [1.0,0.0], atol=1e-2)

def test_nonzero_omega_produces_curved_arc():
    arc = simulate_unicycle_arc(np.array([0.0,0.0,0.0]), 1.0, 1.0, dt=0.05, horizon=1.0)
    assert arc[-1,1] > 0.1
```

- [ ] **Step 2: Run tests and verify failure**

Run:
```bash
pytest tests/test_dd_motion.py -v
```

- [ ] **Step 3: Implement deterministic arc simulation**

Use the same heading-first integration convention as `World.step`:
- `theta_{t+1} = wrap(theta_t + omega * dt)`
- `p_{t+1} = p_t + [cos(theta_{t+1}), sin(theta_{t+1})] * v * dt`

- [ ] **Step 4: Add failing static-clearance tests**

Create a shelf-adjacent turning case where:
- straight-line extrapolation appears safe;
- the unicycle arc intersects the inflated shelf margin;
- `arc_static_clearance` must report the violation.

Also add wall-boundary coverage.

- [ ] **Step 5: Implement arc-based static clearance**

Reuse the same shelf geometry convention as `World.static_collision` and compute physical surface clearance at each arc sample.

- [ ] **Step 6: Implement dynamically reachable command lattice**

Generate commands around:
- current `v`, current `omega`;
- preferred `v`, preferred heading;
- explicit stop/rotate candidates.

Clamp exactly to `VMAX` and `WMAX`. Keep ordering deterministic.

- [ ] **Step 7: Run tests**

```bash
pytest tests/test_dd_motion.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add benchmark/dd_motion.py tests/test_dd_motion.py
git commit -m "feat: add differential drive arc safety"
```

### Task 3: Replace Beast Core with ORCA-DD-Strength Controller

**Files:**
- Modify: `benchmark/beast_classical.py`
- Create: `tests/test_beast_classical.py`

**Interfaces:**
- Consumes:
  - `build_orca_line`, `satisfies_orca_line`, `project_velocity_to_orca_halfplanes`
  - `simulate_unicycle_arc`, `reachable_commands`, `command_terminal_velocity`, `arc_static_clearance`
- Produces:
  - `BeastORCAConfig`
  - `AStarORCADD.action(w) -> np.ndarray`
  - `AStarORCADD.diagnostics() -> dict`

- [ ] **Step 1: Write failing no-obstacle path-following test**

Instantiate one robot in a clear aisle and verify:
- forward progress is positive;
- no unnecessary stopping;
- action stays within normalized bounds.

- [ ] **Step 2: Write failing head-on reciprocal avoidance test**

Create two robots on a collision course and verify after multiple controller steps:
- no AMR-AMR collision;
- both make eventual progress;
- neither controller violates `VMAX/WMAX`.

- [ ] **Step 3: Write failing perpendicular crossing test**

Create crossing trajectories with equal responsibility and verify:
- no collision;
- at least one controller yields;
- both eventually clear the conflict.

- [ ] **Step 4: Write failing pedestrian-crossing test**

Create one AMR and one pedestrian crossing its path. Verify:
- the AMR avoids collision;
- pedestrian velocity is read only from current state;
- no future pedestrian positions are consumed.

- [ ] **Step 5: Implement `BeastORCAConfig`**

Include exact configurable fields:
- `time_horizon`
- `neighbor_distance`
- `peer_margin`
- `human_margin`
- `static_margin`
- `planning_margin`
- `lookahead_distance`
- `waypoint_tolerance`
- `replan_interval`
- `preferred_speed`
- `command_speed_samples`
- `command_omega_samples`
- `smoothness_weight`
- `progress_weight`
- `stuck_progress_eps`
- `stuck_ticks`
- `recovery_ticks`

- [ ] **Step 6: Replace local velocity search with ORCA constraints**

For peers:
- only neighbors inside `neighbor_distance`;
- combined radius = `2 * ROBOT_R + peer_margin`;
- responsibility = 0.5, with bounded priority bias constrained to a narrow range such as `[0.35, 0.65]`.

For pedestrians:
- combined radius = `2 * ROBOT_R + human_margin`;
- responsibility = 1.0.

- [ ] **Step 7: Add unicycle-feasible command selection**

For each reachable command:
1. simulate the executed short-horizon arc;
2. reject if static clearance < `static_margin`;
3. derive realized near-term Cartesian velocity;
4. reject if any ORCA half-plane is violated;
5. score remaining command by path progress, preferred-velocity tracking, and smoothness;
6. choose the best deterministic command;
7. if none are feasible, stop/rotate only if that command itself passes hard safety checks.

- [ ] **Step 8: Keep recovery strictly safety-constrained**

Recovery may alter preferred direction and trigger A* replanning, but all recovered commands must pass the same ORCA and arc-static checks.

- [ ] **Step 9: Add diagnostics**

Track:
- `orca_constraints_total`
- `infeasible_command_events`
- `stop_yield_ticks`
- `replan_count`
- `recovery_count`
- `candidate_commands_evaluated`

- [ ] **Step 10: Run focused controller tests**

```bash
pytest tests/test_beast_classical.py -v
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add benchmark/beast_classical.py tests/test_beast_classical.py
git commit -m "feat: replace Beast with A* ORCA-DD controller"
```

### Task 4: Seed Separation and Frozen Configuration

**Files:**
- Create: `benchmark/beast_config.py`
- Create: `tests/test_beast_benchmark.py`

**Interfaces:**
- Produces:
  - `DEV_SEEDS_BY_N`
  - `VALIDATION_SEEDS_BY_N`
  - `TEST_SEEDS_BY_N`
  - `load_beast_config(path) -> BeastORCAConfig`
  - `save_beast_config(config, path)`

- [ ] **Step 1: Write failing disjoint-seed test**

```python
from benchmark.beast_config import DEV_SEEDS_BY_N, VALIDATION_SEEDS_BY_N, TEST_SEEDS_BY_N

def test_seed_sets_are_pairwise_disjoint():
    for n in (2,4,6):
        d=set(DEV_SEEDS_BY_N[n]); v=set(VALIDATION_SEEDS_BY_N[n]); t=set(TEST_SEEDS_BY_N[n])
        assert not (d & v)
        assert not (d & t)
        assert not (v & t)
```

- [ ] **Step 2: Define deterministic seed ranges**

Use explicit, version-controlled ranges, for example:
- dev: 2000-series;
- validation: 3000-series;
- final test: preserve the existing 5000-series protocol so prior SAC+CBF test comparability is retained.

For each `n`, keep exact seeds checked into code rather than deriving them from runtime randomness.

- [ ] **Step 3: Add config serialization tests**

Round-trip every `BeastORCAConfig` field through JSON and assert exact equality for ints/bools and numerical equality for floats.

- [ ] **Step 4: Implement config load/save**

Reject unknown fields and missing required fields so the final benchmark cannot silently run a partially tuned config.

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_beast_benchmark.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmark/beast_config.py tests/test_beast_benchmark.py
git commit -m "test: separate tuning and benchmark seeds"
```

### Task 5: Validation-Only Hyperparameter Tuner

**Files:**
- Create: `benchmark/tune_beast_classical.py`
- Modify: `tests/test_beast_benchmark.py`

**Interfaces:**
- Consumes: `VALIDATION_SEEDS_BY_N`, `BeastORCAConfig`, `AStarORCADD`
- Produces:
  - `results/beast_tuning/best_config.json`
  - `results/beast_tuning/validation_results.json`

- [ ] **Step 1: Write failing ranking test**

Create synthetic candidate summaries proving lexicographic preference:
1. lower collision rate wins;
2. then lower timeout rate;
3. then higher fleet success;
4. then higher per-agent success;
5. then lower traversal time/path length.

- [ ] **Step 2: Implement deterministic candidate generation**

Search a bounded grid over:
- time horizon;
- neighbor distance;
- peer/human/static/planning margins;
- preferred speed;
- replanning interval;
- path lookahead;
- command lattice density;
- smoothness weight;
- stuck/recovery thresholds.

Keep the search space finite enough for CI.

- [ ] **Step 3: Implement validation evaluation**

Run all candidate configs only on validation seeds for `n in (2,4,6)`.

Store per-candidate metrics and aggregate ranking keys.

- [ ] **Step 4: Implement lexicographic selection**

Select the best config using the acceptance ordering from the spec. Never inspect test-set metrics during tuning.

- [ ] **Step 5: Add a test that the tuner never imports or iterates `TEST_SEEDS_BY_N`**

Use monkeypatch/sentinel access so test execution fails if the tuner touches test seeds.

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_beast_benchmark.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add benchmark/tune_beast_classical.py tests/test_beast_benchmark.py
git commit -m "feat: add validation-only ORCA-DD tuning"
```

### Task 6: Upgrade Fair Evaluation and Diagnostics

**Files:**
- Modify: `benchmark/evaluate_beast_controllers.py`
- Modify: `tests/test_beast_benchmark.py`

**Interfaces:**
- Consumes: frozen `best_config.json`, exact Run 11 checkpoint, `TEST_SEEDS_BY_N`
- Produces:
  - `beast_eval_summary.json`
  - `beast_eval_rows.json`
  - `beast_eval_config.json`

- [ ] **Step 1: Write failing fairness test**

Verify both SAC+CBF and A*+ORCA-DD receive the same exact seed sequence for each fleet size.

- [ ] **Step 2: Replace `AStarReciprocalVO` with `AStarORCADD`**

Load config from a required JSON path. Fail fast if it is missing.

- [ ] **Step 3: Add classical diagnostics to row output**

Include:
- ORCA constraint count;
- infeasible-command events;
- stop/yield fraction;
- replan count;
- recovery count;
- candidates evaluated.

- [ ] **Step 4: Preserve all existing aggregate metrics**

Keep:
- success/collision/timeout + Wilson 95% CI;
- fleet success/collision + Wilson 95% CI;
- path length;
- traversal time;
- clearance;
- throughput;
- collision types;
- compute cost.

- [ ] **Step 5: Add config provenance**

Write the exact frozen config into `beast_eval_config.json` and include its SHA-256 digest in the summary.

- [ ] **Step 6: Run benchmark unit tests**

```bash
pytest tests/test_beast_benchmark.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add benchmark/evaluate_beast_controllers.py tests/test_beast_benchmark.py
git commit -m "feat: evaluate frozen A* ORCA-DD fairly"
```

### Task 7: End-to-End Regression Scenarios

**Files:**
- Create: `tests/test_beast_scenarios.py`

**Interfaces:**
- Consumes: production `AStarORCADD`.
- Produces: reproducible scenario-level safety/liveness regression coverage.

- [ ] **Step 1: Add head-on corridor regression**

Run two AMRs for a fixed number of ticks. Assert:
- zero AMR-AMR collisions;
- positive net progress for both;
- no command bound violations.

- [ ] **Step 2: Add perpendicular crossing regression**

Assert collision-free clearing and finite completion/progress.

- [ ] **Step 3: Add pedestrian crossing regression**

Assert no human collision and no future-state dependency.

- [ ] **Step 4: Add narrow-aisle static-turn regression**

Assert every executed pose remains at or above required shelf/wall clearance.

- [ ] **Step 5: Add six-AMR congestion regression**

Use a fixed dev seed and assert:
- zero catastrophic numerical failures;
- controller always emits finite bounded actions;
- if no feasible translation exists, safe stop/yield is preferred;
- progress resumes after conflict clears.

- [ ] **Step 6: Run scenario suite**

```bash
pytest tests/test_beast_scenarios.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_beast_scenarios.py
git commit -m "test: add ORCA-DD benchmark scenarios"
```

### Task 8: GitHub Actions Tuning and Final Benchmark

**Files:**
- Modify: `.github/workflows/beast-controller-eval.yml`

**Interfaces:**
- Produces:
  - tuning artifact;
  - frozen best config;
  - final 30-seed test artifact.

- [ ] **Step 1: Split workflow into tuning and evaluation jobs**

`tune` job:
- checkout;
- install dependencies;
- run unit/scenario tests;
- run `benchmark/tune_beast_classical.py`;
- upload `best_config.json` and validation results.

`eval` job:
- depend on `tune`;
- download exact Run 11 checkpoint;
- download frozen `best_config.json`;
- run final evaluation on test seeds only;
- upload final artifact.

- [ ] **Step 2: Ensure test seeds are inaccessible to tuning command**

The tuning script invocation must not accept a test-seed flag or test-seed path.

- [ ] **Step 3: Run YAML/static review locally**

Check workflow syntax and inspect the commands for:
- correct checkpoint artifact ID;
- correct branch;
- correct output paths;
- no accidental test-seed tuning.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/beast-controller-eval.yml
git commit -m "ci: tune and evaluate peak ORCA-DD baseline"
```

### Task 9: Full Verification Before Launch

**Files:**
- No production code changes unless failures are found.

**Interfaces:**
- Verifies the whole branch before starting the expensive GitHub run.

- [ ] **Step 1: Run all focused tests**

```bash
pytest tests/test_orca_geometry.py tests/test_dd_motion.py tests/test_beast_classical.py tests/test_beast_benchmark.py tests/test_beast_scenarios.py -v
```

Expected: all PASS.

- [ ] **Step 2: Run the repository regression suite**

```bash
pytest -q
```

Expected: all existing and new tests PASS.

- [ ] **Step 3: Run a small local validation smoke**

Run a tiny 2-AMR subset with 2–3 development seeds to ensure:
- config loading works;
- metrics serialize;
- diagnostics are finite;
- no runtime exceptions.

- [ ] **Step 4: Inspect git diff**

Verify:
- no SAC checkpoint/training code altered;
- no final test seeds used in tuner;
- no benchmark metric definitions changed asymmetrically.

- [ ] **Step 5: Trigger the GitHub workflow**

Push the final implementation commit to `research/beast-baselines-v1`.

- [ ] **Step 6: After completion, compare results without protecting the RL conclusion**

Report whether:
- classical closes the gap;
- classical wins some regimes;
- SAC+CBF still wins;
- tradeoffs change by fleet density.

Do not modify the classical baseline after seeing final test results unless a genuine implementation defect is found; any such correction requires a new frozen version and a fresh final evaluation.

- [ ] **Step 7: Commit verification notes if needed**

If any benchmark-protocol clarification is required, document it separately without changing measured results.
