from __future__ import annotations

import math
from statistics import mean

import numpy as np
import torch

from benchmark.adaptive_predictive_orca import AStarAdaptivePredictiveORCADD
from benchmark.beast_classical import BeastORCAConfig
from benchmark.best_vs_best_evaluation import ControllerAggregate
from benchmark.best_vs_best_protocol import ScenarioSpec, make_paired_worlds
from benchmark.spatiotemporal_policy import reset_hidden
from benchmark.stasac_warehouse import WarehouseObservationBuilder
from benchmark.train_multi_agent_research import DT


def peak_orca_config() -> BeastORCAConfig:
    return BeastORCAConfig(time_horizon=3.0, neighbor_distance=4.0, peer_margin=0.08, human_margin=0.12, static_margin=0.08, planning_margin=0.10, lookahead_distance=1.8, waypoint_tolerance=0.45, replan_interval=12, preferred_speed=1.0, command_speed_samples=7, command_omega_samples=15, smoothness_weight=0.12, progress_weight=4.0, stuck_progress_eps=0.01, stuck_ticks=22, recovery_ticks=20, arc_horizon=1.0)


def _interaction_metrics(action_history):
    if not action_history:
        return {"stop_yield_fraction":0.0,"angular_oscillation":0.0,"commitment_reversals":0.0}
    a=np.asarray(action_history,dtype=np.float32)
    # normalized forward command -1 maps to zero physical speed
    speed01=(a[...,0]+1.0)*0.5
    stop=float(np.mean(speed01 < 0.12))
    omega=a[...,1]
    if omega.shape[0] < 2:
        return {"stop_yield_fraction":stop,"angular_oscillation":0.0,"commitment_reversals":0.0}
    delta=np.abs(np.diff(omega,axis=0))
    osc=float(np.mean(delta))
    signs=np.sign(omega); signs[np.abs(omega)<0.12]=0
    reversals=0
    for agent in range(signs.shape[1]):
        nz=signs[:,agent][signs[:,agent]!=0]
        if len(nz)>1:
            reversals += int(np.sum(nz[1:] != nz[:-1]))
    return {"stop_yield_fraction":stop,"angular_oscillation":osc,"commitment_reversals":float(reversals)}


def _episode_metrics(world, *, seed:int, scenario:str, steps:int, diagnostics=None, action_history=None):
    success_mask=world.done & ~world.hit; collision_mask=world.hit; timeout_mask=~world.done
    success=int(success_mask.sum()); collision=int(collision_mask.sum()); timeout=int(timeout_mask.sum()); duration=max(int(steps),1)*DT
    row={"seed":int(seed),"scenario":str(scenario),"agents":int(world.n),"success":success,"collision":collision,"timeout":timeout,"fleet_success":bool(success==world.n),"steps":int(steps),"throughput_per_min":float(success/(duration/60.0)),"cbf_interventions":int(world.interventions.sum()),"cbf_intervention_rate":float(world.interventions.sum()/max(1,steps*world.n)),"path_length_success_mean":float(np.mean(world.path_length[success_mask])) if success else None,"traversal_time_success_mean":float(np.mean(world.finish_step[success_mask])*DT) if success else None,"min_clearance":float(np.min(world.min_clearance[np.isfinite(world.min_clearance)])) if np.any(np.isfinite(world.min_clearance)) else None}
    row.update(_interaction_metrics(action_history))
    if diagnostics is not None:
        keys=set().union(*(d.keys() for d in diagnostics)) if diagnostics else set()
        for key in keys:
            vals=[d.get(key,0) for d in diagnostics]
            if all(isinstance(v,(int,float,np.integer,np.floating)) for v in vals): row[f"ap_orca_{key}"]=float(sum(float(v) for v in vals))
    return row


def _run_actor_episode(actor, spec:ScenarioSpec, *, seed:int, max_steps:int=600, use_forecast:bool=False, forecaster=None):
    world,_=make_paired_worlds(spec,int(seed)); builder=WarehouseObservationBuilder(world,perception_range=6.0,use_forecast=use_forecast,forecaster=forecaster)
    hidden=[torch.zeros(1,actor.hidden_dim) for _ in range(world.n)]; steps=0; action_history=[]
    for _ in range(int(max_steps)):
        actions=[]
        for i in range(world.n):
            if world.done[i]:
                actions.append(np.array([-1.0,0.0],dtype=np.float32)); hidden[i].zero_(); continue
            ego,batch=builder.observe(world,i); ego_t=torch.as_tensor(ego,dtype=torch.float32).unsqueeze(0); ent_t=torch.as_tensor(batch.features,dtype=torch.float32).unsqueeze(0); mask_t=torch.as_tensor(batch.mask,dtype=torch.bool).unsqueeze(0)
            with torch.no_grad(): action,_,next_hidden,_=actor.sample(ego_t,ent_t,mask_t,hidden[i],deterministic=True)
            actions.append(action[0].cpu().numpy().astype(np.float32)); hidden[i]=next_hidden
        actions=np.asarray(actions,dtype=np.float32); action_history.append(actions.copy()); _,_,done=world.step(actions,use_cbf=True); steps+=1
        for i in range(world.n):
            if done[i]: hidden[i]=reset_hidden(hidden[i],torch.tensor([True]))
        if np.all(done): break
    return _episode_metrics(world,seed=seed,scenario=spec.name,steps=steps,action_history=action_history)


def run_stasac_episode(actor, spec:ScenarioSpec, *, seed:int, max_steps:int=600):
    return _run_actor_episode(actor,spec,seed=seed,max_steps=max_steps,use_forecast=False,forecaster=None)


def run_forecast_stasac_episode(actor, forecaster, spec:ScenarioSpec, *, seed:int, max_steps:int=600):
    return _run_actor_episode(actor,spec,seed=seed,max_steps=max_steps,use_forecast=True,forecaster=forecaster)


def run_ap_orca_episode(spec:ScenarioSpec, *, seed:int, max_steps:int=600):
    _,world=make_paired_worlds(spec,int(seed)); cfg=peak_orca_config(); controllers=[AStarAdaptivePredictiveORCADD(world,i,cfg) for i in range(world.n)]; steps=0; action_history=[]
    for _ in range(int(max_steps)):
        actions=np.asarray([c.action(world) for c in controllers],dtype=np.float32); action_history.append(actions.copy()); _,_,done=world.step(actions,use_cbf=True); steps+=1
        if np.all(done): break
    return _episode_metrics(world,seed=seed,scenario=spec.name,steps=steps,diagnostics=[c.diagnostics() for c in controllers],action_history=action_history)


def aggregate_episode_rows(rows) -> ControllerAggregate:
    rows=list(rows)
    if not rows: raise ValueError("rows must not be empty")
    agents=int(sum(int(r["agents"]) for r in rows)); episodes=len(rows); successes=sum(int(r["success"]) for r in rows); collisions=sum(int(r["collision"]) for r in rows); timeouts=sum(int(r["timeout"]) for r in rows)
    return ControllerAggregate(agents=agents,episodes=episodes,success_rate=float(successes/agents),collision_rate=float(collisions/agents),timeout_rate=float(timeouts/agents),fleet_success_rate=float(sum(bool(r["fleet_success"]) for r in rows)/episodes),throughput_per_min=float(mean(float(r["throughput_per_min"]) for r in rows)))


def run_three_way_development_screen(base_actor, forecast_actor, forecaster, scenarios, seeds, *, max_steps:int=600):
    result={"AP-ORCA+CBF":[],"ST-SAC+CBF":[],"Forecast-ST-SAC+CBF":[]}
    for spec in scenarios:
        for seed in seeds:
            result["AP-ORCA+CBF"].append(run_ap_orca_episode(spec,seed=int(seed),max_steps=max_steps))
            result["ST-SAC+CBF"].append(run_stasac_episode(base_actor,spec,seed=int(seed),max_steps=max_steps))
            result["Forecast-ST-SAC+CBF"].append(run_forecast_stasac_episode(forecast_actor,forecaster,spec,seed=int(seed),max_steps=max_steps))
    return {name:{"aggregate":aggregate_episode_rows(rows).__dict__,"episodes":rows} for name,rows in result.items()}
