"""Batched GPU optimization throughput with SaddleForge (production-grade batched optimizer, progressive
filtering of converged molecules) driving the single-member B97-3c/CPCM model (as in production).
Workload: 64 ligands x 16 RDKit ETKDG conformers (1024) from global_conformers.sdf; batches of 256/512/1024;
each configuration run twice, second (warm) run reported.  Convergence: SaddleForge NORMAL / force-only
(fmax and frms, Gaussian-calibrated thresholds), max 500 steps."""
import os, sys, time, json, random, argparse, numpy as np, torch
sys.path.insert(0, os.path.expanduser('~/saddleforge'))
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
from saddleforge import optimize
from saddleforge.core.molecule import Molecule
ap = argparse.ArgumentParser(); ap.add_argument('--gpu', default='cuda:1'); ap.add_argument('--nlig', type=int, default=64); ap.add_argument('--nconf', type=int, default=16)
ap.add_argument('--batches', default='256,512,1024'); ap.add_argument('--methods', default='lbfgs,fire'); ap.add_argument('--conv', default='normal'); ap.add_argument('--max_steps', type=int, default=500)
a = ap.parse_args(); dev = a.gpu; random.seed(11)
MODEL = os.path.expanduser('../../data/b973c_cpcm_ens_f.jpt')
ens = torch.jit.load(MODEL, map_location=dev).eval(); member = getattr(ens.models, '0')
class Calc:
    def __init__(self, m): self.m = m
    def __call__(self, coords, atomic_nums, charges=None, spin_multiplicity=None):
        q = charges if charges is not None else torch.zeros(coords.shape[0], device=coords.device)
        with torch.no_grad():
            o = self.m({'coord': coords.float(), 'numbers': atomic_nums.long(), 'charge': q.float()})
        return o['energy'].flatten().to(coords.dtype), o['forces'].to(coords.dtype)
calc = Calc(member)
mols = [m for m in Chem.SDMolSupplier(os.path.expanduser('../../data/global_conformers.sdf'), removeHs=False) if m is not None]
sel = []
for lo, hi in [(0, 20), (20, 30), (30, 40), (40, 200)]:
    pool = [m for m in mols if lo <= m.GetNumHeavyAtoms() < hi]; random.shuffle(pool); sel += pool[:a.nlig // 4]
work = []
for m in sel:
    q = int(sum(x.GetFormalCharge() for x in m.GetAtoms())); z = [x.GetAtomicNum() for x in m.GetAtoms()]
    mh = Chem.Mol(m); cids = AllChem.EmbedMultipleConfs(mh, numConfs=a.nconf, randomSeed=7)
    for cid in cids: work.append(Molecule.from_arrays(z, mh.GetConformer(cid).GetPositions().tolist(), charge=q, device=dev, name=m.GetProp('_Name')))
work.sort(key=lambda mm: len(mm.atomic_numbers))
natoms = [len(mm.atomic_numbers) for mm in work]
print(f'ligands {len(sel)} conformers {len(work)} atoms mean {np.mean(natoms):.1f} max {max(natoms)}', flush=True)
res = dict(gpu=torch.cuda.get_device_name(dev), n_conformers=len(work), mean_atoms=float(np.mean(natoms)), model='b973c_cpcm member 0', convergence=a.conv, max_steps=a.max_steps, runs={})
optimize(work[:64], calc, method='lbfgs', convergence=a.conv, convergence_criteria='force', max_steps=50, device=dev)  # warm-up
for method in a.methods.split(','):
    for bs in [int(x) for x in a.batches.split(',')]:
        for rep in range(2):
            torch.cuda.synchronize(); t0 = time.perf_counter(); conv = 0; steps = []; nsteps_tot = 0
            for i in range(0, len(work), bs):
                r = optimize(work[i:i + bs], calc, method=method, convergence=a.conv, convergence_criteria='force', max_steps=a.max_steps, device=dev)
                conv += int(r.converged.sum()); nsteps_tot += r.n_steps
                if r.n_steps_per_molecule is not None: steps += r.n_steps_per_molecule.tolist()
            torch.cuda.synchronize(); wall = time.perf_counter() - t0
            out = dict(method=method, batch=bs, rep=rep, wall_s=wall, conformers=len(work), converged=conv, conf_per_s=len(work) / wall, median_steps=float(np.median(steps)) if steps else None, mean_steps=float(np.mean(steps)) if steps else None, peak_mem_GiB=torch.cuda.max_memory_allocated(dev) / 2**30)
            print(out, flush=True); res['runs'][f'{method}_{bs}_rep{rep}'] = out
        json.dump(res, open(os.path.expanduser('../../data/sf_timing_results.json'), 'w'), indent=1)
print('done')
