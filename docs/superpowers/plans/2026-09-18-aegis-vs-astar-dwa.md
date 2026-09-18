# Aegis vs A* + DWA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fair A*+DWA baseline, paired benchmark runner, representative-win selector, and real comparison-video export for the existing Aegis AMR environment.

**Architecture:** Keep classical planning/control isolated in `classical_baseline.py`. Keep paired evaluation, metrics, seed ranking, CSV output, and video composition in `compare_controllers.py`. Reuse `AMRWarehouseEnv.get_state()/set_state()` to guarantee identical initial conditions.

**Tech Stack:** Python 3, NumPy, Gymnasium environment already in the repo, Matplotlib/imageio for rendering/video, pytest for tests.

**Spec:** `docs/superpowers/specs/2026-09-18-aegis-vs-astar-dwa-design.md`

## Global Constraints
- A*+DWA runs with `use_cbf_filter=False`.
- Both controllers receive the exact same initial snapshot and RNG state.
- Existing robot/environment limits from `config.py` are authoritative.
- Scenario selection is post-benchmark ranking only; selected clips are labelled representative examples.
- Aggregate benchmark results are always saved alongside selected clips.
- The trained checkpoint is passed by CLI path and is not committed.

---

### Task 1: Classical planner/controller core

**Files:**
- Create: `classical_baseline.py`
- Create: `tests/test_classical_baseline.py`

**Interfaces:**
- Produces `AStarPlanner`, `DWAConfig`, `AStarDWAController`, `physical_to_normalized_action()`.
- `AStarDWAController.plan(start_xy, goal_xy) -> list[tuple[float,float]]`
- `AStarDWAController.command(robot_state, dynamic_obstacles) -> np.ndarray` normalized to [-1,1]^2.

- [ ] Write tests for grid path validity, blocked path, action normalization, DWA command limits, and moving-obstacle collision rejection.
- [ ] Run tests and verify RED because `classical_baseline.py` does not exist yet.
- [ ] Implement the minimum planner/controller needed for tests.
- [ ] Run the focused test file and verify GREEN.
- [ ] Commit planner/controller + tests.

### Task 2: Paired benchmark and metric calculation

**Files:**
- Create: `compare_controllers.py`
- Create: `tests/test_compare_controllers.py`

**Interfaces:**
- Produces `EpisodeResult`, `rank_aegis_wins()`, `run_paired_seed()`, and CSV writers.
- Consumes `AStarDWAController` and existing `AMRWarehouseEnv`.

- [ ] Write tests for deterministic win ranking, metric deltas, path-length accumulation, and paired state restoration using a short stub-policy episode.
- [ ] Run tests and verify RED.
- [ ] Implement benchmark dataclasses/helpers and paired runner.
- [ ] Run focused tests and verify GREEN.
- [ ] Commit benchmark code + tests.

### Task 3: Real side-by-side video export

**Files:**
- Modify: `compare_controllers.py`
- Modify: `tests/test_compare_controllers.py`

**Interfaces:**
- `render_comparison_video(aegis_trace, baseline_trace, output_path, fps) -> Path`.
- Uses environment `rgb_array` frames captured during the actual paired runs.

- [ ] Add a test that composes two synthetic RGB frame streams and verifies frame count/output creation via an injected writer.
- [ ] Run test and verify RED.
- [ ] Implement frame layout, live metric overlay, selected-example label, and final metric card.
- [ ] Run test and verify GREEN.
- [ ] Commit video support.

### Task 4: CLI, documentation, and verification

**Files:**
- Modify: `README.md`
- Create: `.github/workflows/comparison-tests.yml`

**Interfaces:**
- CLI example: `python compare_controllers.py CHECKPOINT.pt --seed-start 42 --num-seeds 200 --output-dir comparison_results --render-top-k 3`.

- [ ] Add a CI workflow that installs requirements + pytest and runs the two comparison test modules.
- [ ] Document benchmark invocation, output files, and research-integrity note.
- [ ] Run the full focused test suite in CI.
- [ ] Confirm generated CSV schema and representative-win reason fields.
- [ ] Create a PR only after CI reports zero failures.
