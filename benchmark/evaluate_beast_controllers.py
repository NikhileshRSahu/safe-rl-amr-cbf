import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch

from benchmark.train_multi_agent_research import Actor, World, DT
from benchmark.beast_classical import AStarReciprocalVO

def wilson(k,n,z=1.96):
    if n <= 0: return [None,None]
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return [max(0.0,c-h), min(1.0,c+h)]

def summarize_episode(w,n,seed,controller,compute_s):
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
        "controller_compute_seconds":float(compute_s),
        "controller_ms_per_agent_step":float(1000.0*compute_s/max(1,w.steps*n)),
    }

def aggregate(rows,n):
    agents=n*len(rows)
    succ=sum(r["success"] for r in rows); coll=sum(r["collision"] for r in rows); tout=sum(r["timeout"] for r in rows)
    fs=sum(1 for r in rows if r["fleet_success"]); fc=sum(1 for r in rows if r["fleet_collision"])
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
        "mean_throughput_per_min":float(np.mean([r["throughput_per_min"] for r in rows])),
        "collision_type_counts":ct,
        "mean_controller_ms_per_agent_step":float(np.mean([r["controller_ms_per_agent_step"] for r in rows])),
    }

def run_actor(actor,n,seed):
    w=World(n,max(3,n+2),seed); obs=w.reset(); compute=0.0
    for _ in range(600):
        t0=time.perf_counter()
        with torch.no_grad():
            a,_=actor.sample(torch.tensor(np.asarray(obs),dtype=torch.float32),True)
        compute += time.perf_counter()-t0
        obs,_,done=w.step(a.numpy(),True)
        if np.all(done): break
    return summarize_episode(w,n,seed,"sac_cbf",compute)

def run_beast(n,seed):
    w=World(n,max(3,n+2),seed); w.reset()
    ctrls=[AStarReciprocalVO(w,i) for i in range(n)]
    compute=0.0
    for _ in range(600):
        t0=time.perf_counter()
        acts=np.asarray([ctrls[i].action(w) if not w.done[i] else [-1.0,0.0] for i in range(n)],np.float32)
        compute += time.perf_counter()-t0
        _,_,done=w.step(acts,False)
        if np.all(done): break
    return summarize_episode(w,n,seed,"astar_rvo_beast",compute)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--seeds",type=int,default=30)
    ap.add_argument("--seed-base",type=int,default=5000)
    ap.add_argument("--out",default="results/beast_eval")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    actor=Actor(); actor.load_state_dict(ck["actor"]); actor.eval()

    summary={}; all_rows=[]
    for n in (2,4,6):
        seeds=range(args.seed_base+n*100,args.seed_base+n*100+args.seeds)
        summary[str(n)]={}
        for name,runner in (
            ("sac_cbf",lambda sd: run_actor(actor,n,sd)),
            ("astar_rvo_beast",lambda sd: run_beast(n,sd)),
        ):
            rows=[runner(sd) for sd in seeds]
            all_rows.extend(rows)
            summary[str(n)][name]=aggregate(rows,n)
            print(n,name,json.dumps(summary[str(n)][name]),flush=True)

    (out/"beast_eval_summary.json").write_text(json.dumps(summary,indent=2))
    (out/"beast_eval_rows.json").write_text(json.dumps(all_rows,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
