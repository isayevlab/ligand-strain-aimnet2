import os, sys, time, json, numpy as np, torch
sys.argv=['x']; exec(open(os.path.expanduser('../../data/steps_timing.py')).read().split("res = {'n_conformers'")[0])
torch.set_num_threads(1)
ens = torch.jit.load(MODEL, map_location='cpu').eval(); mem = getattr(ens.models, '0')
rows=[]
sub = confs[::4]
opt(mem,'cpu',confs[0])
for c in sub:
    opt(mem,'cpu',c); t,n,ok = opt(mem,'cpu',c); rows.append(dict(name=c[3], natoms=int(len(c[0])), cpu_member_s=t, steps=n, converged=ok)); print(rows[-1], flush=True)
s=dict(cpu_member_median_s=float(np.median([r['cpu_member_s'] for r in rows])), cpu_member_median_s_per_step=float(np.median([r['cpu_member_s']/max(r['steps'],1) for r in rows])), median_steps=float(np.median([r['steps'] for r in rows])))
print(s); json.dump(dict(rows=rows, summary=s), open(os.path.expanduser('../../data/cpu_member_results.json'),'w'), indent=1)
