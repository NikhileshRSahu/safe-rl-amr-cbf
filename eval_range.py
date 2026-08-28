import argparse, csv, time
from pathlib import Path
import numpy as np
import torch
from environment import AMRWarehouseEnv
from safe_sac import SafeSACAgent, SafeSACConfig

p=argparse.ArgumentParser()
p.add_argument('--start',type=int,required=True)
p.add_argument('--count',type=int,required=True)
p.add_argument('--out',type=str,required=True)
p.add_argument('--max-steps',type=int,default=500)
a=p.parse_args()

torch.set_num_threads(1)
ckpt=torch.load('/mnt/data/rl_work/final_model_v2.pt',map_location='cpu',weights_only=False)
env=AMRWarehouseEnv(max_episode_steps=a.max_steps)
env.cbf_filter.config.enable_circulation=True
env.cbf_filter.config.circulation_threshold=0.5
env.cbf_filter.config.circulation_gain=0.6
agent=SafeSACAgent(policy_config=ckpt['policy_config'],observation_space=env.observation_space,action_dim=2,sac_config=SafeSACConfig(device='cpu'),replay_buffer_size=10)
agent.policy.load_state_dict(ckpt['policy_state_dict']); agent.eval()
rows=[]
t0=time.time()
for j,seed in enumerate(range(a.start,a.start+a.count)):
    obs,_=env.reset(seed=seed)
    start_dist=float(np.linalg.norm(env.robot_state[:2]-env.goal_pos))
    ep_ret=0.; hard=pred=deadpred=strict=solver_fail=interventions=0
    outcome='timeout'; ctype='none'
    for step in range(1,a.max_steps+1):
        action=agent.select_action(obs,deterministic=True)
        obs,reward,term,trunc,info=env.step(action); ep_ret+=float(reward)
        d=env.last_cbf_diagnostics
        tier=d.get('tier','strict')
        if tier=='hard_escape': hard+=1
        elif tier=='predictive_recovery': pred+=1
        elif tier=='deadlock_predictive_recovery': deadpred+=1
        else: strict+=1
        if not d.get('solver_success',True): solver_fail+=1
        if d.get('intervened',False): interventions+=1
        if term or trunc:
            if info.get('goal_reached',False): outcome='success'
            elif info.get('collision',False): outcome='collision'; ctype=str(info.get('collision_type','unknown'))
            else: outcome='timeout'
            break
    final_dist=float(np.linalg.norm(env.robot_state[:2]-env.goal_pos))
    rows.append(dict(seed=seed,outcome=outcome,steps=step,start_distance=start_dist,final_distance=final_dist,progress=start_dist-final_dist,return_=ep_ret,collision_type=ctype,hard_escape_steps=hard,predictive_recovery_steps=pred,deadlock_predictive_recovery_steps=deadpred,strict_steps=strict,solver_fail_steps=solver_fail,cbf_intervention_steps=interventions))
    print(f'{seed}: {outcome} steps={step} final={final_dist:.3f} hard={hard} pred={pred} deadpred={deadpred}', flush=True)

out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('elapsed',time.time()-t0,'sec')
