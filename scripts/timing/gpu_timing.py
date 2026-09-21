"""GPU vs CPU timing of AIMNet2-CPCM geometry optimization on production-like workload.
Workload: stratified subset of ligands from global_conformers.sdf; K RDKit ETKDG conformers per ligand
(stand-in for Omega starting geometries); batched FIRE optimization to fmax < 0.05 eV/A (max 1000 steps)
with the frozen B97-3c/CPCM ensemble (4 members, as used in production).
Modes: batched GPU (several batch sizes), unbatched GPU (batch 1), unbatched CPU 1 thread (batch 1).
"""
import os, sys, time, json, random, argparse, numpy as np, torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
ap = argparse.ArgumentParser()
ap.add_argument('--nlig', type=int, default=200); ap.add_argument('--nconf', type=int, default=32)
ap.add_argument('--gpu', default='cuda:1'); ap.add_argument('--model', default=os.path.expanduser('../../data/b973c_cpcm_ens_f.jpt'))
ap.add_argument('--batches', default='1024'); ap.add_argument('--repeat', type=int, default=2); ap.add_argument('--cpu_n', type=int, default=20); ap.add_argument('--gpu1_n', type=int, default=100)
ap.add_argument('--skip_steps', action='store_true'); ap.add_argument('--out', default=os.path.expanduser('../../data/gpu_timing_results_fire.json'))
a = ap.parse_args()
random.seed(11); torch.manual_seed(11)
FMAX = 0.05; MAXSTEP = 500

# ---- workload
mols = [m for m in Chem.SDMolSupplier(os.path.expanduser('../../data/global_conformers.sdf'), removeHs=False) if m is not None]
bins = [(0, 20), (20, 30), (30, 40), (40, 200)]
per = a.nlig // len(bins); sel = []
for lo, hi in bins:
    pool = [m for m in mols if lo <= m.GetNumHeavyAtoms() < hi]; random.shuffle(pool); sel += pool[:per]
confs = []   # (numbers, coords, charge, ligname)
for m in sel:
    q = float(sum(x.GetFormalCharge() for x in m.GetAtoms())); z = np.array([x.GetAtomicNum() for x in m.GetAtoms()])
    mh = Chem.Mol(m); cids = AllChem.EmbedMultipleConfs(mh, numConfs=a.nconf, randomSeed=7, useRandomCoords=False)
    for cid in cids:
        confs.append((z, mh.GetConformer(cid).GetPositions().astype(np.float32), q, m.GetProp('_Name')))
print(f'ligands {len(sel)}  conformers {len(confs)}  atoms/conformer mean {np.mean([len(c[0]) for c in confs]):.1f}', flush=True)
confs.sort(key=lambda c: len(c[0]))

def pad(batch, dev):
    N = max(len(c[0]) for c in batch); B = len(batch)
    Z = torch.zeros(B, N, dtype=torch.int64); X = torch.zeros(B, N, 3); Q = torch.zeros(B)
    for i, (z, x, q, _) in enumerate(batch):
        Z[i, :len(z)] = torch.from_numpy(z); X[i, :len(z)] = torch.from_numpy(x); Q[i] = q
    return Z.to(dev), X.to(dev), Q.to(dev)

def fire_batch(model, batch, dev):
    """batched FIRE; returns (n_converged, n_steps_total_conformer_steps, wall)"""
    Z, X, Q = pad(batch, dev); mask = (Z > 0).unsqueeze(-1).float(); B = Z.shape[0]
    dt = torch.full((B,), 0.1, device=dev); alpha = torch.full((B,), 0.1, device=dev); Np = torch.zeros(B, device=dev)
    V = torch.zeros_like(X); done = torch.zeros(B, dtype=torch.bool, device=dev); steps_done = torch.zeros(B, device=dev)
    dtmax, Nmin, finc, fdec, astart, fa = 1.0, 5, 1.1, 0.5, 0.1, 0.99
    t0 = time.perf_counter()
    for it in range(MAXSTEP):
        with torch.no_grad():
            out = model({'coord': X, 'numbers': Z, 'charge': Q})   # ensemble returns energy and forces (eV, eV/A)
        F = out['forces'] * mask
        fmax = F.norm(dim=-1).amax(dim=1)
        newly = (fmax < FMAX) & ~done; done |= newly
        steps_done += (~done).float()
        if done.all(): break
        with torch.no_grad():
            V = V + dt[:, None, None] * F
            P = (F * V).sum(dim=(1, 2))
            vn = V.norm(dim=(1, 2), keepdim=True) + 1e-12; fn = F.norm(dim=(1, 2), keepdim=True) + 1e-12
            V = (1 - alpha)[:, None, None] * V + alpha[:, None, None] * vn / fn * F
            pos = P > 0; Np = torch.where(pos, Np + 1, torch.zeros_like(Np))
            grow = pos & (Np > Nmin); dt = torch.where(grow, torch.clamp(dt * finc, max=dtmax), dt); alpha = torch.where(grow, alpha * fa, alpha)
            dt = torch.where(pos, dt, dt * fdec); alpha = torch.where(pos, alpha, torch.full_like(alpha, astart)); V = torch.where(pos[:, None, None], V, torch.zeros_like(V))
            dr = dt[:, None, None] * V; dr = dr * torch.clamp(0.2 / (dr.norm(dim=-1, keepdim=True) + 1e-12), max=1.0)
            X = X + dr * mask * (~done)[:, None, None].float()
    if dev.startswith('cuda'): torch.cuda.synchronize()
    return int(done.sum()), float(steps_done.sum()), time.perf_counter() - t0, it + 1

res = {'n_ligands': len(sel), 'n_conformers': len(confs), 'mean_atoms': float(np.mean([len(c[0]) for c in confs])), 'model': os.path.basename(a.model), 'gpu': torch.cuda.get_device_name(a.gpu), 'fmax': FMAX, 'maxstep': MAXSTEP}

# ---- batched GPU
dev = a.gpu; model = torch.jit.load(a.model, map_location=dev).eval()
member0 = getattr(model.models, '0')
def step_throughput(mod, batch, nrep=10):
    Z, X, Q = pad(batch, dev)
    with torch.no_grad():
        for _ in range(3): mod({'coord': X, 'numbers': Z, 'charge': Q})
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(nrep): mod({'coord': X, 'numbers': Z, 'charge': Q})
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / nrep
    return dict(batch=len(batch), max_atoms=int(Z.shape[1]), s_per_step=dt, conformer_steps_per_s=len(batch) / dt, mem_GiB=torch.cuda.max_memory_allocated(dev) / 2**30)
res['step_throughput'] = {}
for bs in ([] if a.skip_steps else [256, 512, 1024, 2048]):
    b = confs[len(confs)//2 - bs//2: len(confs)//2 + bs//2] if bs <= len(confs) else confs
    for name, mod in [('ensemble4', model), ('member1', member0)]:
        torch.cuda.reset_peak_memory_stats(dev)
        r = step_throughput(mod, b); r['model'] = name; res['step_throughput'][f'{name}_{bs}'] = r; print('step', r, flush=True)
fire_batch(model, confs[:64], dev)  # warm-up
for bs in [int(x) for x in a.batches.split(',')]:
    runs = []
    for rep in range(a.repeat):   # repeat; keep the last (warm) run
        conv = steps = wall = 0; nb = 0
        for i in range(0, len(confs), bs):
            c, s, w, _ = fire_batch(model, confs[i:i + bs], dev); conv += c; steps += s; wall += w; nb += 1
        runs.append(dict(batch=bs, conformers=len(confs), converged=conv, wall_s=wall, conf_per_s=len(confs) / wall, conformer_steps=steps, conf_steps_per_s=steps / wall, mem_GiB=torch.cuda.max_memory_allocated(dev) / 2**30))
        print('rep', rep, runs[-1], flush=True)
    res[f'gpu_batched_{bs}'] = runs[-1]; res[f'gpu_batched_{bs}_allruns'] = runs

# ---- unbatched GPU (batch 1), subset spread over sizes
sub = confs[::max(1, len(confs) // a.gpu1_n)][:a.gpu1_n]
fire_batch(model, sub[:1], dev)
ts = []; st = []
for c in sub:
    fire_batch(model, [c], dev)
    _, s, w, _ = fire_batch(model, [c], dev); ts.append(w); st.append(s)
res['gpu_unbatched'] = dict(n=len(sub), median_s=float(np.median(ts)), mean_s=float(np.mean(ts)), median_steps=float(np.median(st))); print(res['gpu_unbatched'], flush=True)

# ---- CPU single thread (batch 1)
torch.set_num_threads(1); os.environ['OMP_NUM_THREADS'] = '1'
mc = torch.jit.load(a.model, map_location='cpu').eval()
subc = sub[::max(1, len(sub) // a.cpu_n)][:a.cpu_n]
fire_batch(mc, subc[:1], 'cpu')
ts = []; st = []
for c in subc:
    fire_batch(mc, [c], 'cpu')
    _, s, w, _ = fire_batch(mc, [c], 'cpu'); ts.append(w); st.append(s)
res['cpu_1thread'] = dict(n=len(subc), median_s=float(np.median(ts)), mean_s=float(np.mean(ts)), median_steps=float(np.median(st)), cpu=open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].split(':')[1].strip()); print(res['cpu_1thread'], flush=True)
json.dump(res, open(a.out, 'w'), indent=1); print('saved', a.out)
