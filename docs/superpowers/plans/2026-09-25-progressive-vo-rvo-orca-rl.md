# Progressive VO → RVO → ORCA → Safe RL Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a scientifically fair progression benchmark that isolates the capability added by VO, RVO, ORCA, and SAC+CBF in the same warehouse, with identical global planning, robot dynamics, geometry, seeds, and evaluation metrics.

**Architecture:** Keep the existing A* global planner and differential-drive command realization fixed for all classical controllers. Add two genuine velocity-space target generators: VO treats moving agents as passive obstacles and takes full avoidance responsibility; RVO uses reciprocal velocity-obstacle geometry for AMR peers while treating humans as non-reciprocal. Reuse the existing frozen Peak ORCA controller and exact Run-11 SAC+CBF checkpoint unchanged. Evaluate a staged scenario ladder whose difficulty targets the known limitation addressed by each successive method.

**Tech Stack:** Python 3.11, NumPy, existing benchmark World/RealisticHumanWorld, existing AStarORCADD, PyTorch SAC actor, pytest, GitHub Actions.

**Spec:** This plan is the benchmark specification.

## Global Constraints

- Do not retune Peak ORCA or SAC+CBF on final evaluation seeds.
- Use the same A* planner, map, robot radius, VMAX, WMAX, DT, goal tolerance, collision rules, episode horizon, and seed for methods being compared within a scenario.
- VO, RVO, and ORCA share the same differential-drive realization and static-arc feasibility checks.
- RVO reciprocity applies only to controlled AMR peers; humans remain non-reciprocal dynamic obstacles.
- Report raw per-method metrics; do not construct a weighted score designed to force monotonic ranking.
- Primary metrics: success rate, collision rate, timeout rate, fleet success, traversal time, path length, throughput.
- Progression metrics: steering/omega sign changes, stop-yield fraction, recovery count, and local-controller infeasibility.
- The final claim may be monotonic in capability without being monotonic in every metric.

## Review Focus

- Symmetric head-on AMRs: VO should not silently become reciprocal; RVO must use peer reciprocity.
- Non-reciprocal humans: RVO must not assume humans share collision responsibility.
- Near-zero relative velocity and overlapping velocity-space cases must not produce NaNs.
- Finite horizon collision prediction must clamp closest-approach time to [0, horizon].
- Common DD realization must not reintroduce ORCA half-plane constraints for VO/RVO.

---

### Task 1: Add velocity-space geometry primitives

**Files:**
- Create: `benchmark/progressive_velocity_baselines.py`
- Test: `tests/test_progressive_velocity_baselines.py`

**Interfaces:**
- Produces `finite_horizon_collision(...) -> bool`
- Produces `vo_candidate_safe(...) -> bool`
- Produces `rvo_peer_candidate_safe(...) -> bool`
- Produces `AStarVODD` and `AStarRVODD`

- [ ] **Step 1: Write failing geometry tests**

Test finite-horizon closing collision, separating motion, RVO transformation, and distinct controller classes.

- [ ] **Step 2: Run tests to verify red**

Run: `pytest -q tests/test_progressive_velocity_baselines.py`
Expected: import/module failure because the module does not yet exist.

- [ ] **Step 3: Implement minimal geometry and controller target selection**

VO evaluates candidate velocity directly against finite-horizon relative motion. RVO evaluates AMR peer candidates via the reciprocal transform `v_eff = 2*v_candidate - v_ego_current`; humans use direct VO geometry. Candidate targets include preferred/current/zero plus deterministic polar samples bounded by VMAX.

- [ ] **Step 4: Run targeted and full tests**

Run: `pytest -q tests/test_progressive_velocity_baselines.py`
Then: `pytest -q`
Expected: all tests pass.

### Task 2: Add progression-specific metrics

**Files:**
- Modify: `benchmark/progressive_velocity_baselines.py`
- Create: `benchmark/evaluate_progressive_ladder.py`
- Test: `tests/test_progressive_ladder_protocol.py`

**Interfaces:**
- Evaluator records success/collision/timeout/fleet-success/traversal/path/throughput.
- Classical diagnostics add target infeasible events, stop-yield ticks, recovery count, and omega-sign-flip count.

- [ ] **Step 1: Write protocol tests**

Assert scenario definitions keep map/dynamics fixed and alter only the intended interaction assumption.

- [ ] **Step 2: Verify tests fail before implementation**

Run: `pytest -q tests/test_progressive_ladder_protocol.py`

- [ ] **Step 3: Implement evaluator**

Scenarios:
1. `passive_dynamic`: 1 AMR + moving humans, constant motion. Demonstrates VO baseline competence.
2. `reciprocal_crossing`: 4 AMRs + 0 humans. Targets VO→RVO reciprocal interaction.
3. `dense_reciprocal`: 4 AMRs + controlled reciprocal traffic, symmetry pressure. Targets RVO→ORCA smoothness/liveness.
4. `warehouse_intent_changes`: 4 AMRs + 12 realistic humans with medium/high intent changes. Targets ORCA→SAC+CBF adaptability/liveness.

Use paired holdout seeds and preserve frozen ORCA/RL configs.

- [ ] **Step 4: Run targeted and full tests**

Run protocol tests and full pytest suite.

### Task 3: Run sharded benchmark and compile evidence

**Files:**
- Create: `.github/workflows/progressive-vo-rvo-orca-rl.yml`

**Interfaces:**
- One matrix job per scenario/controller to avoid runner-lifetime failures.
- Each job uploads JSON with aggregate and per-seed records.

- [ ] **Step 1: Add workflow after evaluator tests are green**

Matrix controllers: `vo`, `rvo`, `orca`, `sac_cbf` where applicable. Use exact Run-11 artifact for SAC+CBF and frozen Peak ORCA JSON.

- [ ] **Step 2: Run benchmark**

Use at least 10 paired seeds per scenario for exploratory ranking; expand to 30 for thesis-final statistics after the ladder is validated.

- [ ] **Step 3: Compile table without cherry-picking**

Report all methods in all applicable scenarios. Highlight the metric the next method was designed to improve, but preserve counter-results where the earlier method remains better.

- [ ] **Step 4: Thesis interpretation**

Expected structure, not guaranteed outcome:
- VO: strong with passive predictable movers; poor reciprocal coordination/oscillation.
- RVO: improved reciprocal cooperation vs VO; reciprocal-dance/symmetry can remain.
- ORCA: smoother, more scalable, formal half-plane selection and improved dense reciprocal liveness.
- SAC+CBF: potential higher mission completion under intent changes/non-stationarity, while ORCA may remain faster and sometimes safer.

Do not claim RL dominance if measured results do not support it.
