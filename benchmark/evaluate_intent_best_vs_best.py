from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmark.beast_classical import AStarORCADD
from benchmark.beast_config import load_beast_config
from benchmark.intent_shift_benchmark import AdaptiveORCADD, AdaptiveORCAExtras, IntentShiftWorld
from benchmark.train_multi_agent_research import DT


ADAPTIVE_CANDIDATES = [
    AdaptiveORCAExtras(0.45, 0.06, 0.5, 4.0, 0.08),
    AdaptiveORCAExtras(0.75, 0.10, 1.0, 5.0, 0.12),
    AdaptiveORCAExtras(1.00, 0.10, 1.5, 5.5, 0.12),
    AdaptiveORCAExtras(0.65, 0.14, 1.0, 4.5, 0.16),
    AdaptiveORCAExtras(1.15, 0.08, 2.0, 6.0, 0.10),
    AdaptiveORCAExtras(0.85, 0.12, 1.5, 5.0, 0.14),
]


def episode_metrics(w, diagnostics=None):
    success = w.done & ~w.hit
    timeout = ~w.done
    out = {
        'success': int(success.sum()),
        'collision': int(w.hit.sum()),
        'timeout': int(timeout.sum()),
        'fleet_success': bool(success.all()),
        'steps': int(w.steps),
        'path_length_success_mean': float(w.path_length[success].mean()) if success.any() else None,
        'traversal_time_success_mean': float((w.finish_step[success] * DT).mean()) if success.any() else None,
        'throughput_per_min': float(success.sum() / (max(1, w.steps) * DT) * 60.0),
        'intent_changes': int(w.intent_change_count),
    }
    if diagnostics is not None:
        out['stop_yield_ticks'] = int(sum(d.get('stop_yield_ticks', 0) for d in diagnostics))
        out['recovery_count'] = int(sum(d.get('recovery_count', 0) for d in diagnostics))
        out['orca_constraints_total'] = int(sum(d.get('orca_constraints_total', 0) for d in diagnostics))
    return out


def run_actor(actor, seed):
    import torch
    w = IntentShiftWorld(4, 12, seed)
    obs = w.reset()
    for _ in range(600):
        with torch.no_grad():
            action, _ = actor.sample(torch.tensor(np.asarray(obs), dtype=torch.float32), True)
        obs, _, done = w.step(action.numpy(), True)
        if np.all(done):
            break
    return episode_metrics(w)


def run_classical(controller_cls, config, seed, extras=None):
    w = IntentShiftWorld(4, 12, seed)
    w.reset()
    if extras is None:
        ctrls = [controller_cls(w, i, config) for i in range(4)]
    else:
        ctrls = [controller_cls(w, i, config, extras) for i in range(4)]
    for _ in range(600):
        acts = np.asarray([
            ctrls[i].action(w) if not w.done[i] else [-1.0, 0.0]
            for i in range(4)
        ], dtype=np.float32)
        _, _, done = w.step(acts, False)
        if np.all(done):
            break
    return episode_metrics(w, [c.diagnostics() for c in ctrls])


def aggregate(rows):
    agents = 4 * len(rows)
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    return {
        'episodes': len(rows),
        'success_rate': sum(r['success'] for r in rows) / agents,
        'collision_rate': sum(r['collision'] for r in rows) / agents,
        'timeout_rate': sum(r['timeout'] for r in rows) / agents,
        'fleet_success_rate': sum(r['fleet_success'] for r in rows) / len(rows),
        'mean_path_length_success': mean('path_length_success_mean'),
        'mean_traversal_time_success': mean('traversal_time_success_mean'),
        'mean_throughput_per_min': mean('throughput_per_min'),
        'mean_stop_yield_ticks': mean('stop_yield_ticks'),
        'mean_recovery_count': mean('recovery_count'),
        'mean_intent_changes': mean('intent_changes'),
    }


def main():
    import torch
    from benchmark.train_multi_agent_research import Actor

    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['sac', 'peak_orca', 'adaptive'], required=True)
    ap.add_argument('--candidate', type=int, default=0)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--checkpoint')
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    config = load_beast_config(args.config)

    actor = None
    extras = None
    if args.mode == 'sac':
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        actor = Actor(); actor.load_state_dict(ck['actor']); actor.eval()
    elif args.mode == 'adaptive':
        extras = ADAPTIVE_CANDIDATES[args.candidate]

    rows = []
    for seed in seeds:
        if args.mode == 'sac': row = run_actor(actor, seed)
        elif args.mode == 'peak_orca': row = run_classical(AStarORCADD, config, seed)
        else: row = run_classical(AdaptiveORCADD, config, seed, extras)
        rows.append({'seed': seed, **row})
        print(json.dumps(rows[-1]), flush=True)

    payload = {
        'mode': args.mode,
        'candidate': args.candidate if args.mode == 'adaptive' else None,
        'extras': asdict(extras) if extras else None,
        'seeds': seeds,
        'aggregate': aggregate(rows),
        'episodes': rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

if __name__ == '__main__':
    main()
