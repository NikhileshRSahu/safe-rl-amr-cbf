from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from benchmark.history_sac import (
    HISTORY_OBS_DIM,
    HistoryActor,
    HistoryQ,
    augment_observation,
    transfer_run11_actor,
)
from benchmark.intent_shift_benchmark import IntentShiftWorld, TRAIN_SEEDS
from benchmark.train_multi_agent_research import Actor


class HistoryReplay:
    def __init__(self, cap=300000):
        self.cap = int(cap); self.n = 0; self.ptr = 0
        self.s = np.zeros((cap, HISTORY_OBS_DIM), np.float32)
        self.a = np.zeros((cap, 2), np.float32)
        self.r = np.zeros((cap, 1), np.float32)
        self.ns = np.zeros((cap, HISTORY_OBS_DIM), np.float32)
        self.d = np.zeros((cap, 1), np.float32)

    def add(self, s, a, r, ns, d):
        j = self.ptr
        self.s[j] = s; self.a[j] = a; self.r[j] = r; self.ns[j] = ns; self.d[j] = d
        self.ptr = (j + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, batch):
        ii = np.random.randint(0, self.n, batch)
        return [torch.tensor(x[ii]) for x in (self.s, self.a, self.r, self.ns, self.d)]


def _augment_batch(current, previous):
    return np.asarray([
        augment_observation(current[i], previous[i]) for i in range(len(current))
    ], dtype=np.float32)


def evaluate(actor, seeds):
    rows = []; actor.eval()
    for seed in seeds:
        w = IntentShiftWorld(4, 12, int(seed))
        obs = np.asarray(w.reset(), dtype=np.float32)
        prev = obs.copy()
        for _ in range(600):
            aug = _augment_batch(obs, prev)
            with torch.no_grad():
                action, _ = actor.sample(torch.tensor(aug), True)
            prev_before = obs.copy()
            next_obs, _, done = w.step(action.numpy(), True)
            prev = prev_before
            obs = np.asarray(next_obs, dtype=np.float32)
            if np.all(done): break
        success = w.done & ~w.hit
        rows.append({
            'seed': int(seed), 'success': int(success.sum()),
            'collision': int(w.hit.sum()), 'timeout': int((~w.done).sum()),
            'fleet_success': bool(success.all()), 'steps': int(w.steps),
            'interventions': int(w.interventions.sum()),
        })
    agents = 4 * len(rows)
    return {
        'rows': rows,
        'success_rate': sum(r['success'] for r in rows) / agents,
        'collision_rate': sum(r['collision'] for r in rows) / agents,
        'timeout_rate': sum(r['timeout'] for r in rows) / agents,
        'fleet_success_rate': sum(r['fleet_success'] for r in rows) / len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--agent-steps', type=int, default=80000)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(2)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    base = Actor(); base.load_state_dict(ck['actor']); base.eval()
    actor = transfer_run11_actor(base, HistoryActor())
    q1 = HistoryQ(); q2 = HistoryQ(); tq1 = HistoryQ(); tq2 = HistoryQ()
    tq1.load_state_dict(q1.state_dict()); tq2.load_state_dict(q2.state_dict())
    ao = Adam(actor.parameters(), 1.0e-4)
    qo = Adam(list(q1.parameters()) + list(q2.parameters()), 3e-4)
    replay = HistoryReplay()
    gamma = .99; alpha = .08; tau = .01
    rng = np.random.default_rng(args.seed + 11000)
    steps = 0; episodes = 0; w = None; obs = None; prev = None; history = []
    val_seeds = tuple(range(13005, 13010))

    while steps < args.agent_steps:
        if w is None or np.all(w.done) or w.steps >= 600:
            sd = int(TRAIN_SEEDS[int(rng.integers(0, len(TRAIN_SEEDS)))])
            w = IntentShiftWorld(4, 12, sd)
            obs = np.asarray(w.reset(), dtype=np.float32)
            prev = obs.copy(); episodes += 1

        aug = _augment_batch(obs, prev)
        with torch.no_grad():
            if steps < 1500:
                act = actor.sample(torch.tensor(aug), True)[0].numpy()
                act = np.clip(act + rng.normal(0, .08, size=(4, 2)), -1, 1).astype(np.float32)
            else:
                act = actor.sample(torch.tensor(aug), False)[0].numpy()

        current = obs.copy()
        next_obs, reward, done = w.step(act, False)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        next_aug = _augment_batch(next_obs, current)
        for i in range(4): replay.add(aug[i], act[i], reward[i], next_aug[i], float(done[i]))
        prev = current; obs = next_obs; steps += 4

        if replay.n >= 1800 and steps % 4 == 0:
            bs, ba, br, bns, bd = replay.sample(192)
            with torch.no_grad():
                na, nlp = actor.sample(bns)
                y = br + gamma * (1 - bd) * (torch.min(tq1(bns, na), tq2(bns, na)) - alpha * nlp)
            qloss = F.mse_loss(q1(bs, ba), y) + F.mse_loss(q2(bs, ba), y)
            qo.zero_grad(); qloss.backward(); torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5); qo.step()
            pa, lp = actor.sample(bs)
            aloss = (alpha * lp - torch.min(q1(bs, pa), q2(bs, pa))).mean()
            ao.zero_grad(); aloss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), 5); ao.step()
            with torch.no_grad():
                for p, tp in zip(q1.parameters(), tq1.parameters()): tp.mul_(1 - tau).add_(tau * p)
                for p, tp in zip(q2.parameters(), tq2.parameters()): tp.mul_(1 - tau).add_(tau * p)

        if steps % 20000 == 0 or steps >= args.agent_steps:
            ev = evaluate(actor, val_seeds)
            rec = {'agent_steps': steps, 'episodes': episodes, 'validation': ev}
            history.append(rec); print(json.dumps(rec), flush=True)
            torch.save({
                'actor': actor.state_dict(), 'architecture': 'history_delta_sac',
                'base_checkpoint': 'Run11', 'seed': args.seed, 'agent_steps': steps,
                'history_obs_dim': HISTORY_OBS_DIM,
            }, out / f'history_sac_{steps}.pt')

    def key(rec):
        v = rec['validation']
        return (v['collision_rate'], v['timeout_rate'], -v['fleet_success_rate'], -v['success_rate'])
    best = min(history, key=key)
    name = f"history_sac_{best['agent_steps']}.pt"
    (out / 'summary.json').write_text(json.dumps({
        'seed': args.seed, 'history': history, 'best': best, 'best_checkpoint': name,
    }, indent=2))
    print(json.dumps({'best': best, 'best_checkpoint': name}, indent=2))

if __name__ == '__main__': main()
