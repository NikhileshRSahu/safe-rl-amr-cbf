from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.orca_teacher import behavior_cloning_loss
from benchmark.spatiotemporal_policy import STASACActor, TwinRecurrentQ, reset_hidden
from benchmark.warehouse_interaction_features import FEATURE_DIM


class SequenceReplay:
    """Episode-preserving replay for recurrent SAC.

    Episodes are stored independently so sampled windows can never cross reset
    boundaries. Variable entity sets are padded only when a batch is sampled.
    Optional ORCA teacher actions are stored as training metadata; the actor
    itself has no runtime dependency on ORCA.
    """

    def __init__(self, capacity_episodes: int = 256, burn_in: int = 8, train_len: int = 16):
        if capacity_episodes < 1 or burn_in < 0 or train_len < 1:
            raise ValueError("invalid replay configuration")
        self.capacity_episodes = int(capacity_episodes)
        self.burn_in = int(burn_in)
        self.train_len = int(train_len)
        self.window_len = self.burn_in + self.train_len
        self.episodes = deque(maxlen=self.capacity_episodes)

    def __len__(self):
        return sum(len(ep) for ep in self.episodes)

    def add_episode(self, steps: list[dict[str, Any]]) -> None:
        if len(steps) < self.window_len:
            raise ValueError(f"episode needs at least {self.window_len} transitions")
        episode = []
        for step in steps:
            copied = dict(step)
            for key in ("ego", "entities", "entity_mask", "action", "next_ego", "next_entities", "next_entity_mask"):
                copied[key] = np.asarray(step[key]).copy()
            if "teacher_action" in step and step["teacher_action"] is not None:
                copied["teacher_action"] = np.asarray(step["teacher_action"], dtype=np.float32).copy()
            copied["reward"] = float(step["reward"])
            copied["done"] = bool(step["done"])
            episode.append(copied)
        self.episodes.append(episode)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        if not self.episodes:
            raise ValueError("replay is empty")
        rng = rng or np.random.default_rng()
        eligible = [ep for ep in self.episodes if len(ep) >= self.window_len]
        if not eligible:
            raise ValueError("no episode long enough for sequence sample")
        windows = []
        for _ in range(int(batch_size)):
            ep = eligible[int(rng.integers(0, len(eligible)))]
            max_start = len(ep) - self.window_len
            start = int(rng.integers(0, max_start + 1))
            windows.append(ep[start : start + self.window_len])
        return self._collate(windows)

    def _collate(self, windows):
        b = len(windows)
        l = self.window_len
        ego_dim = int(np.asarray(windows[0][0]["ego"]).shape[-1])
        max_n = 0
        has_teacher = False
        for window in windows:
            for step in window:
                max_n = max(max_n, len(step["entities"]), len(step["next_entities"]))
                has_teacher = has_teacher or ("teacher_action" in step and step["teacher_action"] is not None)

        ego = np.zeros((b, l, ego_dim), np.float32)
        next_ego = np.zeros_like(ego)
        entities = np.zeros((b, l, max_n, FEATURE_DIM), np.float32)
        next_entities = np.zeros_like(entities)
        entity_mask = np.zeros((b, l, max_n), np.bool_)
        next_entity_mask = np.zeros_like(entity_mask)
        action = np.zeros((b, l, 2), np.float32)
        teacher_action = np.zeros((b, l, 2), np.float32) if has_teacher else None
        teacher_mask = np.zeros((b, l), np.bool_) if has_teacher else None
        reward = np.zeros((b, l, 1), np.float32)
        done = np.zeros((b, l, 1), np.float32)

        for bi, window in enumerate(windows):
            for ti, step in enumerate(window):
                ego[bi, ti] = step["ego"]
                next_ego[bi, ti] = step["next_ego"]
                n = len(step["entities"])
                nn = len(step["next_entities"])
                if n:
                    entities[bi, ti, :n] = step["entities"]
                    entity_mask[bi, ti, :n] = step["entity_mask"]
                if nn:
                    next_entities[bi, ti, :nn] = step["next_entities"]
                    next_entity_mask[bi, ti, :nn] = step["next_entity_mask"]
                action[bi, ti] = step["action"]
                if has_teacher and "teacher_action" in step and step["teacher_action"] is not None:
                    teacher_action[bi, ti] = step["teacher_action"]
                    teacher_mask[bi, ti] = True
                reward[bi, ti, 0] = step["reward"]
                done[bi, ti, 0] = float(step["done"])

        train_mask = np.zeros((b, l), np.bool_)
        train_mask[:, self.burn_in :] = True
        out = {
            "ego": torch.from_numpy(ego),
            "entities": torch.from_numpy(entities),
            "entity_mask": torch.from_numpy(entity_mask),
            "action": torch.from_numpy(action),
            "reward": torch.from_numpy(reward),
            "next_ego": torch.from_numpy(next_ego),
            "next_entities": torch.from_numpy(next_entities),
            "next_entity_mask": torch.from_numpy(next_entity_mask),
            "done": torch.from_numpy(done),
            "train_mask": torch.from_numpy(train_mask),
        }
        if has_teacher:
            out["teacher_action"] = torch.from_numpy(teacher_action)
            out["teacher_mask"] = torch.from_numpy(teacher_mask)
        return out


def _sequence_scenes(actor, ego, entities, mask, done, *, grad: bool):
    b, l, _ = ego.shape
    hidden = ego.new_zeros((b, actor.hidden_dim))
    scenes = []
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for t in range(l):
            scene, hidden, _ = actor.encoder(ego[:, t], entities[:, t], mask[:, t], hidden)
            scenes.append(scene)
            if t < l - 1:
                hidden = reset_hidden(hidden, done[:, t, 0].bool())
    return torch.stack(scenes, dim=1)


def _polyak(source, target, tau: float):
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.mul_(1.0 - tau).add_(tau * p)


def recurrent_sac_update(
    batch,
    actor: STASACActor,
    q: TwinRecurrentQ,
    target_q: TwinRecurrentQ,
    actor_opt,
    q_opt,
    *,
    alpha: float = 0.08,
    gamma: float = 0.99,
    tau: float = 0.01,
    max_grad_norm: float = 5.0,
    bc_coeff: float = 0.0,
):
    ego = batch["ego"].float()
    entities = batch["entities"].float()
    mask = batch["entity_mask"].bool()
    action = batch["action"].float()
    reward = batch["reward"].float()
    next_ego = batch["next_ego"].float()
    next_entities = batch["next_entities"].float()
    next_mask = batch["next_entity_mask"].bool()
    done = batch["done"].float()
    train_mask = batch["train_mask"].bool()
    teacher_action = batch.get("teacher_action")
    teacher_mask = batch.get("teacher_mask")
    if teacher_action is not None:
        teacher_action = teacher_action.float()
        teacher_mask = teacher_mask.bool()
    b, l, _ = ego.shape

    current_scene = _sequence_scenes(actor, ego, entities, mask, done, grad=False)
    next_scene = _sequence_scenes(actor, next_ego, next_entities, next_mask, done, grad=False)

    with torch.no_grad():
        next_actions = []
        next_logps = []
        h = next_ego.new_zeros((b, actor.hidden_dim))
        for t in range(l):
            na, nlp, h, _ = actor.sample(
                next_ego[:, t], next_entities[:, t], next_mask[:, t], h, deterministic=False
            )
            next_actions.append(na)
            next_logps.append(nlp)
            if t < l - 1:
                h = reset_hidden(h, done[:, t, 0].bool())
        next_actions = torch.stack(next_actions, dim=1)
        next_logps = torch.stack(next_logps, dim=1)

    flat = train_mask.reshape(-1)
    cs = current_scene.reshape(b * l, -1)[flat]
    act = action.reshape(b * l, 2)[flat]
    ns = next_scene.reshape(b * l, -1)[flat]
    na = next_actions.reshape(b * l, 2)[flat]
    nlp = next_logps.reshape(b * l, 1)[flat]
    rew = reward.reshape(b * l, 1)[flat]
    dn = done.reshape(b * l, 1)[flat]

    with torch.no_grad():
        tq1, tq2 = target_q(ns, na)
        target = rew + float(gamma) * (1.0 - dn) * (torch.minimum(tq1, tq2) - float(alpha) * nlp)
    q1, q2 = q(cs, act)
    q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    q_opt.zero_grad(set_to_none=True)
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), max_grad_norm)
    q_opt.step()

    for p in q.parameters():
        p.requires_grad_(False)
    hidden = ego.new_zeros((b, actor.hidden_dim))
    actor_terms = []
    logp_values = []
    bc_terms = []
    for t in range(l):
        pa, logp, hidden, _ = actor.sample(ego[:, t], entities[:, t], mask[:, t], hidden, deterministic=False)
        if train_mask[:, t].any():
            tq1a, tq2a = q(hidden, pa)
            term = float(alpha) * logp - torch.minimum(tq1a, tq2a)
            actor_terms.append(term[train_mask[:, t]])
            logp_values.append(logp[train_mask[:, t]])
            if teacher_action is not None and float(bc_coeff) > 0.0:
                valid_teacher = train_mask[:, t] & teacher_mask[:, t]
                if valid_teacher.any():
                    # Deterministic actor action is the behavior target; ORCA is
                    # strictly a temporary training teacher and never executed
                    # by the deployed STASAC policy.
                    deterministic_action, _, _, _ = actor.sample(
                        ego[:, t], entities[:, t], mask[:, t], hidden.detach(), deterministic=True
                    )
                    bc_terms.append(
                        behavior_cloning_loss(
                            deterministic_action[valid_teacher],
                            teacher_action[:, t][valid_teacher],
                            coefficient=float(bc_coeff),
                        )
                    )
        if t < l - 1:
            hidden = reset_hidden(hidden, done[:, t, 0].bool())
    sac_actor_loss = torch.cat(actor_terms, dim=0).mean()
    bc_loss = torch.stack(bc_terms).mean() if bc_terms else sac_actor_loss * 0.0
    actor_loss = sac_actor_loss + bc_loss
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    actor_opt.step()
    for p in q.parameters():
        p.requires_grad_(True)

    _polyak(q, target_q, float(tau))
    mean_logp = torch.cat(logp_values, dim=0).mean()
    return {
        "q_loss": float(q_loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "sac_actor_loss": float(sac_actor_loss.detach().cpu()),
        "bc_loss": float(bc_loss.detach().cpu()),
        "mean_logp": float(mean_logp.detach().cpu()),
    }


def save_stasac_checkpoint(path, actor: STASACActor, q: TwinRecurrentQ, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "q": q.state_dict(),
            "ego_dim": actor.encoder.ego_dim,
            "hidden_dim": actor.hidden_dim,
            "observation_version": "warehouse_variable_entities_risk_history_v1",
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_stasac_checkpoint(path, ego_dim: int | None = None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ego_dim = int(ego_dim if ego_dim is not None else ck["ego_dim"])
    hidden_dim = int(ck.get("hidden_dim", 128))
    actor = STASACActor(ego_dim=ego_dim, hidden_dim=hidden_dim)
    q = TwinRecurrentQ(hidden_dim=hidden_dim)
    actor.load_state_dict(ck["actor"])
    q.load_state_dict(ck["q"])
    return actor, q, dict(ck.get("metadata", {}))
