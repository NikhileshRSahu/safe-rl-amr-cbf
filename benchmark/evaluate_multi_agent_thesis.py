import argparse, json, math
from pathlib import Path
import numpy as np
import torch

from classical_baseline import AStarPlanner, physical_to_normalized_action
from benchmark.train_multi_agent_research import (
    Actor, World, DT, WORLD, ROBOT_R, VMAX, WMAX, GOAL_TOL, SHELVES
)

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def wilson(k, n, z=1.96):
    if n <= 0: return [0.0, 0.0]
    p = k / n
    den = 1 + z*z/n
    cen = (p + z*z/(2*n))/den
    half = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)/den
    return [max(0.0, cen-half), min(1.0, cen+half)]

class AStarVO:
    def __init__(self, world, i):
        self.i=i
        self.planner=AStarPlanner(
            map_bounds=(-WORLD, WORLD, -WORLD, WORLD),
            shelves=SHELVES,
            robot_radius=ROBOT_R,
            margin=0.12,
            resolution=0.25,
        )
        self.path=self.planner.plan(tuple(world.p[i]), tuple(world.g[i]))
        self.idx=0

    def _target(self, p):
        if not self.path: return None
        while self.idx < len(self.path)-1 and np.linalg.norm(np.asarray(self.path[self.idx])-p) < 0.6:
            self.idx += 1
        j=self.idx
        acc=0.0
        while j+1 < len(self.path) and acc < 1.4:
            a=np.asarray(self.path[j]); b=np.asarray(self.path[j+1])
            acc += float(np.linalg.norm(b-a)); j += 1
        return np.asarray(self.path[j], dtype=float)

    def _safe(self, w, i, v, om, horizon=1.5):
        x,y,th=map(float,[w.p[i,0],w.p[i,1],w.th[i]])
        steps=max(1,int(round(horizon/DT)))
        minc=999.0
        for k in range(1,steps+1):
            th=wrap(th+om*DT); x += v*math.cos(th)*DT; y += v*math.sin(th)*DT
            q=np.array([x,y],dtype=float)
            if w.static_collision(q): return False,-1.0
            t=k*DT
            for j in range(w.n):
                if j==i or w.done[j]: continue
                pj=w.p[j] + np.array([math.cos(float(w.th[j]))*float(w.v[j]), math.sin(float(w.th[j]))*float(w.v[j])])*t
                c=float(np.linalg.norm(q-pj)-2*ROBOT_R); minc=min(minc,c)
                if c < 0.10: return False,minc
            for j in range(w.nppl):
                pj=w.hp[j] + w.hv[j]*t
                c=float(np.linalg.norm(q-pj)-2*ROBOT_R); minc=min(minc,c)
                if c < 0.10: return False,minc
        return True,minc

    def action(self, w, i):
        p=np.asarray(w.p[i],dtype=float)
        target=self._target(p)
        if target is None:
            return np.array([-1.0,0.0],np.float32)
        d0=float(np.linalg.norm(target-p))
        best=None
        nearest_peer=min([np.linalg.norm(w.p[j]-w.p[i]) for j in range(w.n) if j!=i and not w.done[j]]+[99.0])
        yield_scale=0.45+0.55*float(w.priority[i]) if nearest_peer<1.6 else 1.0
        for v in np.linspace(0.0, VMAX, 9):
            for om in np.linspace(-WMAX, WMAX, 25):
                safe,clear=self._safe(w,i,float(v),float(om))
                if not safe: continue
                th=wrap(float(w.th[i])+float(om)*0.8)
                pred=p+np.array([math.cos(th),math.sin(th)])*float(v)*0.8
                prog=d0-float(np.linalg.norm(target-pred))
                hd=math.cos(wrap(math.atan2(target[1]-pred[1],target[0]-pred[0])-th))
                score=4.0*prog+1.2*hd+0.5*min(clear,1.5)+0.35*(float(v)/VMAX)*yield_scale-0.05*abs(float(om))
                if best is None or score>best[0]: best=(score,float(v),float(om))
        if best is None:
            return np.array([-1.0,0.0],np.float32)
        return physical_to_normalized_action(best[1],best[2],v_min=0.0,v_max=VMAX,omega_max=WMAX)

def run_actor(actor,n,seed,use_cbf):
    w=World(n,max(3,n+2),seed); obs=w.reset()
    for _ in range(600):
        with torch.no_grad():
            a,_=actor.sample(torch.tensor(np.asarray(obs),dtype=torch.float32),True)
        obs,_,done=w.step(a.numpy(),use_cbf)
        if np.all(done): break
    return summarize_episode(w,n,seed,"sac_cbf" if use_cbf else "sac_actor_no_cbf")

def run_astar_vo(n,seed):
    w=World(n,max(3,n+2),seed); w.reset()
    ctrls=[AStarVO(w,i) for i in range(n)]
    for _ in range(600):
        acts=np.asarray([ctrls[i].action(w,i) if not w.done[i] else [-1.0,0.0] for i in range(n)],np.float32)
        _,_,done=w.step(acts,False)
        if np.all(done): break
    return summarize_episode(w,n,seed,"astar_vo")

def summarize_episode(w,n,seed,controller):
    success=(w.done & ~w.hit)
    collision=w.hit.copy()
    timeout=~w.done
    effective=np.where(w.finish_step>0,w.finish_step,600)
    types={"amr_amr":0,"human":0,"static":0}
    for t in w.collision_type:
        if t in types: types[t]+=1
    return {
        "seed":seed,"controller":controller,"n":n,"steps":int(w.steps),
        "success":int(success.sum()),"collision":int(collision.sum()),"timeout":int(timeout.sum()),
        "fleet_success":bool(success.all()),"fleet_collision":bool(collision.any()),
        "path_length_success_mean":float(w.path_length[success].mean()) if success.any() else None,
        "traversal_time_success_mean":float((w.finish_step[success]*DT).mean()) if success.any() else None,
        "min_clearance":float(np.min(w.min_clearance[np.isfinite(w.min_clearance)])),
        "interventions":int(w.interventions.sum()),"deadlock":int(w.deadlock.sum()),
        "intervention_fraction":float(w.interventions.sum()/max(1,effective.sum())),
        "throughput_per_min":float(success.sum()/(max(1,w.steps)*DT)*60.0),
        "collision_types":types,
    }

def aggregate(rows,n):
    agents=n*len(rows)
    succ=sum(r["success"] for r in rows); coll=sum(r["collision"] for r in rows); tout=sum(r["timeout"] for r in rows)
    fs=sum(1 for r in rows if r["fleet_success"])
    fc=sum(1 for r in rows if r["fleet_collision"])
    def mean_nonnull(key):
        vals=[r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None
    ct={k:sum(r["collision_types"][k] for r in rows) for k in ("amr_amr","human","static")}
    return {
        "episodes":len(rows),"agents":agents,
        "success_rate":succ/agents,"success_ci95":wilson(succ,agents),
        "collision_rate":coll/agents,"collision_ci95":wilson(coll,agents),
        "timeout_rate":tout/agents,"timeout_ci95":wilson(tout,agents),
        "fleet_success_rate":fs/len(rows),"fleet_success_ci95":wilson(fs,len(rows)),
        "fleet_collision_rate":fc/len(rows),"fleet_collision_ci95":wilson(fc,len(rows)),
        "path_length_success_mean":mean_nonnull("path_length_success_mean"),
        "traversal_time_success_mean":mean_nonnull("traversal_time_success_mean"),
        "episode_min_clearance_mean":float(np.mean([r["min_clearance"] for r in rows])),
        "worst_min_clearance":float(np.min([r["min_clearance"] for r in rows])),
        "mean_intervention_fraction":float(np.mean([r["intervention_fraction"] for r in rows])),
        "mean_deadlock_events":float(np.mean([r["deadlock"] for r in rows])),
        "mean_throughput_per_min":float(np.mean([r["throughput_per_min"] for r in rows])),
        "collision_type_counts":ct,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--seeds",type=int,default=30)
    ap.add_argument("--seed-base",type=int,default=5000)
    ap.add_argument("--out",default="results/thesis_eval")
    ap.add_argument("--only-n",type=int,choices=[1,2,4,6],default=None)
    ap.add_argument("--only-controller",choices=["sac_cbf","sac_actor_no_cbf","astar_vo"],default=None)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    actor=Actor(); actor.load_state_dict(ck["actor"]); actor.eval()
    configs=[args.only_n] if args.only_n is not None else [1,2,4,6]
    all_rows=[]; summary={}
    for n in configs:
        seeds=range(args.seed_base+n*100,args.seed_base+n*100+args.seeds)
        summary[str(n)]={}
        runners=[
            ("sac_cbf",lambda sd:run_actor(actor,n,sd,True)),
            ("sac_actor_no_cbf",lambda sd:run_actor(actor,n,sd,False)),
            ("astar_vo",lambda sd:run_astar_vo(n,sd)),
        ]
        if args.only_controller is not None:
            runners=[x for x in runners if x[0]==args.only_controller]
        for name,runner in runners:
            rows=[runner(sd) for sd in seeds]
            all_rows.extend(rows)
            summary[str(n)][name]=aggregate(rows,n)
            print(n,name,summary[str(n)][name],flush=True)
    (out/"thesis_eval_summary.json").write_text(json.dumps(summary,indent=2))
    (out/"thesis_eval_rows.json").write_text(json.dumps(all_rows,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
