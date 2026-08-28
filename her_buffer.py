import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np

from config import GOAL_TOLERANCE, REWARD_CONFIG, MAP_MIN_X, MAP_MAX_X, MAP_MIN_Y, MAP_MAX_Y
from utils import distance, heading_to_goal, world_to_robot


def unnormalize_pose(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract and unnormalize the (x, y, theta) pose from the robot_state observation."""
    robot_obs = obs_dict["robot_state"]
    rx = (robot_obs[0] + 1.0) / 2.0 * (MAP_MAX_X - MAP_MIN_X) + MAP_MIN_X
    ry = (robot_obs[1] + 1.0) / 2.0 * (MAP_MAX_Y - MAP_MIN_Y) + MAP_MIN_Y
    theta = robot_obs[2] * math.pi
    return np.array([rx, ry, theta], dtype=np.float64)


@dataclass
class HERTransition:
    obs: Dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    next_obs: Dict[str, np.ndarray]
    done: float
    cost: float
    barrier_value: float
    is_goal: bool
    is_collision: bool
    max_diag: float
    info: Dict[str, Any]
    # Raw physical state the CBF filter was conditioned on at this
    # transition's `obs` (see AMRWarehouseEnv.step()'s
    # info["cbf_robot_state"] / info["cbf_dynamic_obstacles"]). Goal-
    # independent, so relabeling copies these through unchanged. Optional
    # (default None) so this dataclass stays constructible for callers not
    # yet passing them.
    robot_state_raw: Any = None
    dynamic_obstacles_raw: Any = None


@dataclass
class HERRelabelStats:
    pushed: int
    total_candidates: int          # non-terminal steps considered for relabeling
    total_relabeled: int           # actual relabeled transitions pushed
    reached_count: int
    skipped_no_valid_future: int   # steps where every candidate goal was < min_relabel_distance away
    # Ground-truth net displacement (start pose -> final pose) for this episode,
    # independent of HER strategy -- lets train.py separate "episode moved but
    # skip_ratio is high because of index-pool choice" from "episode never
    # moved at all, so no HER strategy could have produced a signal here."
    net_episode_displacement: float = 0.0
    # Count of explicitly-injected, guaranteed-reached transitions (see
    # `ensure_hindsight_success` on relabel_and_push). Tracked separately
    # from `reached_count` (which includes these plus any coincidental
    # reaches from the main sampling loop) so training curves can show
    # whether the guarantee is actually firing.
    hindsight_success_injected: int = 0

    @property
    def fraction_reached(self) -> float:
        return self.reached_count / self.total_relabeled if self.total_relabeled else 0.0

    @property
    def skip_ratio(self) -> float:
        return self.skipped_no_valid_future / self.total_candidates if self.total_candidates else 0.0


@dataclass
class HEREpisodeBuffer:
    transitions: List[HERTransition] = field(default_factory=list)

    def add(self, **kwargs) -> None:
        self.transitions.append(HERTransition(**kwargs))

    def clear(self) -> None:
        self.transitions.clear()

    def _recompute_goal_obs(self, robot_pose: np.ndarray, goal_pos: np.ndarray, max_diag: float) -> np.ndarray:
        """Must match AMRWarehouseEnv._get_observation()'s goal_obs block exactly."""
        gx, gy = goal_pos
        goal_rel_robot = world_to_robot((gx, gy), robot_pose)
        rx, ry = robot_pose[0], robot_pose[1]
        return np.array([
            np.clip(goal_rel_robot[0] / max_diag, -1.0, 1.0),
            np.clip(goal_rel_robot[1] / max_diag, -1.0, 1.0),
            np.clip(distance((rx, ry), (gx, gy)) / max_diag, 0.0, 1.0),
            heading_to_goal(robot_pose, goal_pos) / math.pi,
        ], dtype=np.float32)

    def _candidate_indices(self, t_idx: int, n: int, strategy: str) -> List[int]:
        """Returns the pool of transition indices eligible as relabel-goal
        sources for a step at t_idx, before the min_relabel_distance filter.

        - "future" (original, default): only steps strictly after t_idx.
          This is the standard HER "future" strategy and matches the
          project's established behavior -- unchanged default.
        - "episode": every other step in the episode, before or after
          t_idx. Only makes sense once the min_relabel_distance filter is
          also applied (below) -- otherwise near-duplicate/no-op goals
          would dominate.

        Added this session: previously `strategy` was accepted as a
        parameter but silently ignored; the loop always used the
        "future" pool regardless of what was passed in. This method makes
        the parameter actually do something, and validates it explicitly
        instead of failing silently on typos.
        """
        if strategy == "future":
            return list(range(t_idx + 1, n))
        if strategy == "episode":
            return [i for i in range(n) if i != t_idx]
        raise ValueError(f"Unknown HER strategy: {strategy!r}. Use 'future' or 'episode'.")

    def _push(self, replay_buffer, t: "HERTransition", obs, action, reward, next_obs, done, cost, barrier_value) -> None:
        """Single choke point for every replay_buffer.add() call in this
        class, so CBF raw-state passthrough (goal-independent -- it's the
        same physical world state regardless of which goal a transition
        gets relabeled toward) is written in exactly one place instead of
        duplicated across the original-push and relabel-push call sites.
        """
        replay_buffer.add(
            obs=obs, action=action, reward=reward, next_obs=next_obs,
            done=done, cost=cost, barrier_value=barrier_value,
            robot_state_raw=t.robot_state_raw, dynamic_obstacles_raw=t.dynamic_obstacles_raw,
        )

    def _relabeled_reward(self, t: "HERTransition", robot_pose_prev, robot_pose_next, new_goal_pos) -> Tuple[float, float, bool]:
        """Computes (new_reward, new_done, reached) for transition `t`
        relabeled toward `new_goal_pos`. Pulled out of the main loop so
        the hindsight-success injection path (which needs the identical
        computation for a *different* transition index) can't drift out
        of sync with the main per-candidate path.
        """
        from config import GOAL_TOLERANCE

        new_curr_dist = distance(robot_pose_next[:2], new_goal_pos)
        new_prev_dist = distance(robot_pose_prev[:2], new_goal_pos)
        reached = new_curr_dist <= GOAL_TOLERANCE

        if reached:
            return float(REWARD_CONFIG.SUCCESS_REWARD), 1.0, True

        r_progress = float(REWARD_CONFIG.PROGRESS_WEIGHT * (new_prev_dist - new_curr_dist))
        r_heading = float(REWARD_CONFIG.HEADING_WEIGHT * math.cos(heading_to_goal(robot_pose_next, new_goal_pos)))
        r_time = float(-REWARD_CONFIG.TIME_PENALTY)
        r_energy = t.info["reward_breakdown"].get("r_energy", 0.0)
        r_oscillation = t.info["reward_breakdown"].get("r_oscillation", 0.0)
        r_safety = t.info["reward_breakdown"].get("r_safety", 0.0)
        v_actual = t.info.get("v_actual", 0.0)
        r_deadlock = -0.5 if (abs(v_actual) < 0.05 and new_curr_dist > GOAL_TOLERANCE) else 0.0
        new_reward = float(sum([r_progress, r_heading, r_time, r_energy, r_oscillation, r_safety, r_deadlock]))
        return new_reward, t.done, False

    def _make_relabeled_pair(self, t: "HERTransition", robot_pose_prev, robot_pose_next, new_goal_pos):
        new_goal_obs_prev = self._recompute_goal_obs(robot_pose_prev, new_goal_pos, t.max_diag)
        new_obs = {key: np.copy(v) for key, v in t.obs.items()}
        new_obs["goal"] = new_goal_obs_prev

        new_next_obs = {key: np.copy(v) for key, v in t.next_obs.items()}
        new_next_obs["goal"] = self._recompute_goal_obs(robot_pose_next, new_goal_pos, t.max_diag)
        return new_obs, new_next_obs

    def relabel_and_push(
        self,
        replay_buffer,
        k: int = 4,
        strategy: str = "future",
        min_relabel_distance: float = None,
        ensure_hindsight_success: bool = True,
    ) -> "HERRelabelStats":
        """Relabels this episode's transitions with hindsight goals and
        pushes both the original and relabeled transitions into
        `replay_buffer`.

        On `ensure_hindsight_success` (why it exists)
        ----------------------------------------------
        Before this fix, `reached` for a relabeled transition ``t`` (at
        index ``t_idx``, relabeled toward the goal achieved at index
        ``f_idx``) was computed as
        ``distance(next_obs_of[t_idx], achieved_pos_of[f_idx]) <= GOAL_TOLERANCE``
        -- i.e. it only fired when transition ``t_idx``'s *own,
        unrelated* resulting position happened to coincidentally land
        within tolerance of a position visited at a *different* timestep.
        For continuous navigation that is a near-zero-probability event
        except in the single trivial case ``f_idx == t_idx + 1`` (where
        the "achieved position" literally *is* transition ``t_idx``'s own
        next_obs, giving `new_curr_dist == 0` by construction). Under the
        "future" strategy that trivial case has probability roughly
        ``k / (n - t_idx)`` per episode transition -- small but nonzero
        and larger for late-episode steps. Under "episode" strategy,
        goals are drawn near-uniformly from all ``n-1`` other transitions,
        diluting that already-small probability further by roughly
        ``(n - t_idx) / (n - 1)``. This is consistent with, and is the
        most likely explanation for, `fraction_reached` measuring exactly
        0.0% across a 440-episode, ~400k-relabeled-transition run: the
        replay buffer was receiving relabeled dense-reward signal but
        essentially never a genuine "this is what success looks like"
        terminal transition with `reward=SUCCESS_REWARD` -- which is the
        entire point of doing hindsight relabeling for a sparse-bonus task
        in the first place.

        The fix does not change how *dense* relabeled rewards are
        computed (that logic, in `_relabeled_reward`, is unchanged and
        was already goal-consistent). It adds one more, explicit
        transition per *distinct* achieved-goal index ``f_idx`` sampled
        anywhere in the episode: transition ``f_idx`` itself, relabeled
        toward its own achieved position. By construction
        ``new_curr_dist == 0 <= GOAL_TOLERANCE`` for that transition, so
        it is *guaranteed* `reached=True`, `reward=SUCCESS_REWARD`,
        `done=1.0` -- a real, well-formed "reaching this goal looks like
        this" example for the critic to bootstrap from, deduplicated (via
        `injected_success_idxs`) so a goal used by many `t_idx` candidates
        only contributes one such transition, not one per candidate.

        Args:
            ensure_hindsight_success: See above. Default True; set False
                to reproduce the previous (structurally success-starved)
                behavior for an A/B comparison.
        """
        from config import GOAL_TOLERANCE
        if min_relabel_distance is None:
            min_relabel_distance = 2.0 * GOAL_TOLERANCE

        n = len(self.transitions)
        pushed = 0
        total_candidates = 0
        total_relabeled = 0
        reached_count = 0
        skipped_no_valid_future = 0
        hindsight_success_injected = 0
        injected_success_idxs: set = set()

        net_episode_displacement = 0.0
        if n > 0:
            start_pose = unnormalize_pose(self.transitions[0].obs)[:2]
            end_pose = unnormalize_pose(self.transitions[-1].next_obs)[:2]
            net_episode_displacement = float(distance(start_pose, end_pose))

        for t_idx, t in enumerate(self.transitions):
            self._push(replay_buffer, t, t.obs, t.action, t.reward, t.next_obs, t.done, t.cost, t.barrier_value)
            pushed += 1

            if t.is_goal or t.is_collision:
                continue

            total_candidates += 1
            robot_pose_prev = unnormalize_pose(t.obs)

            candidate_idxs = self._candidate_indices(t_idx, n, strategy)
            valid_goal_idxs = [
                f_idx for f_idx in candidate_idxs
                if distance(
                    robot_pose_prev[:2],
                    unnormalize_pose(self.transitions[f_idx].next_obs)[:2]
                ) >= min_relabel_distance
            ]

            if not valid_goal_idxs:
                skipped_no_valid_future += 1
                continue

            sampled = np.random.choice(valid_goal_idxs, size=min(k, len(valid_goal_idxs)), replace=False)
            robot_pose_next = unnormalize_pose(t.next_obs)

            for f_idx in sampled:
                new_goal_pos = unnormalize_pose(self.transitions[f_idx].next_obs)[:2]
                new_obs, new_next_obs = self._make_relabeled_pair(t, robot_pose_prev, robot_pose_next, new_goal_pos)
                new_reward, new_done, reached = self._relabeled_reward(t, robot_pose_prev, robot_pose_next, new_goal_pos)
                total_relabeled += 1
                if reached:
                    reached_count += 1

                self._push(replay_buffer, t, new_obs, t.action, new_reward, new_next_obs, new_done, t.cost, t.barrier_value)
                pushed += 1

                # -- Guaranteed hindsight-success injection -------------- #
                if (
                    ensure_hindsight_success
                    and f_idx not in injected_success_idxs
                    and not self.transitions[f_idx].is_collision
                ):
                    injected_success_idxs.add(f_idx)
                    t_goal = self.transitions[f_idx]
                    goal_pose_prev = unnormalize_pose(t_goal.obs)
                    goal_pose_next = unnormalize_pose(t_goal.next_obs)
                    # achieved position == goal_pose_next by construction,
                    # so this is trivially reached; still route through
                    # _relabeled_reward for a single source of truth on
                    # the reward/done values rather than hardcoding them
                    # twice.
                    s_obs, s_next_obs = self._make_relabeled_pair(t_goal, goal_pose_prev, goal_pose_next, new_goal_pos)
                    s_reward, s_done, s_reached = self._relabeled_reward(t_goal, goal_pose_prev, goal_pose_next, new_goal_pos)
                    assert s_reached, (
                        "Hindsight-success injection invariant violated: relabeling "
                        "transition f_idx toward its own achieved position must be "
                        "trivially reached (distance 0 <= GOAL_TOLERANCE)."
                    )
                    self._push(replay_buffer, t_goal, s_obs, t_goal.action, s_reward, s_next_obs, s_done, t_goal.cost, t_goal.barrier_value)
                    pushed += 1
                    total_relabeled += 1
                    reached_count += 1
                    hindsight_success_injected += 1

        return HERRelabelStats(
            pushed=pushed,
            total_candidates=total_candidates,
            total_relabeled=total_relabeled,
            reached_count=reached_count,
            skipped_no_valid_future=skipped_no_valid_future,
            net_episode_displacement=net_episode_displacement,
            hindsight_success_injected=hindsight_success_injected,
        )