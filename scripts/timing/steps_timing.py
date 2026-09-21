"""Steps-to-convergence and unbatched timings with the standard ASE FIRE optimizer (fmax 0.05 eV/A),
using the frozen B97-3c/CPCM ensemble through a minimal ASE calculator. GPU (batch 1) and CPU (1 thread)."""
import os, sys, time, json, random, numpy as np, torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE
random.seed(11)
MODEL = os.path.expanduser('../../data/b973c_cpcm_ens_f.jpt')
class Jit(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(self, model, dev, charge): super().__init__(); self.m, self.dev, self.q = model, dev, charge
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        d = {'coord': torch.tensor(atoms.get_positions()[None], dtype=torch.float32, device=self.dev),
             'numbers': torch.tensor(atoms.get_atomic_numbers()[None], dtype=torch.int64, device=self.dev),
             'charge': torch.tensor([self.q], dtype=torch.float32, device=self.dev)}
        with torch.no_grad(): o = self.m(d)
        self.results = {'energy': float(o['energy'].flatten()[0]), 'forces': o['forces'][0].cpu().numpy().astype(float)}
mols = [m for m in Chem.SDMolSupplier(os.path.expanduser('../../data/global_conformers.sdf'), removeHs=False) if m is not None]
sel = []
for lo, hi in [(0, 20), (20, 30), (30, 40), (40, 200)]:
    pool = [m for m in mols if lo <= m.GetNumHeavyAtoms() < hi]; random.shuffle(pool); sel += pool[:8]
confs = []
for m in sel:
    q = float(sum(x.GetFormalCharge() for x in m.GetAtoms())); mh = Chem.Mol(m)
    cid = AllChem.EmbedMolecule(mh, randomSeed=7)
    if cid < 0: continue
    confs.append((np.array([x.GetAtomicNum() for x in m.GetAtoms()]), mh.GetConformer(cid).GetPositions(), q, m.GetProp('_Name')))
def opt(model, dev, c):
    z, x, q, _ = c; at = Atoms(numbers=z, positions=x); at.calc = Jit(model, dev, q)
    t = time.perf_counter(); o = FIRE(at, logfile=None, maxstep=0.2); ok = o.run(fmax=0.05, steps=2000); return time.perf_counter() - t, o.nsteps, bool(ok)
res = {'n_conformers': len(confs), 'atoms': [int(len(c[0])) for c in confs]}
dev = 'cuda:1'; mg = torch.jit.load(MODEL, map_location=dev).eval()
opt(mg, dev, confs[0]); opt(mg, dev, confs[0])
rows = []
for c in confs:
    opt(mg, dev, c)                       # warm run discarded
    t, n, ok = opt(mg, dev, c); rows.append(dict(name=c[3], natoms=int(len(c[0])), gpu_s=t, steps=n, converged=ok)); print(rows[-1], flush=True)
torch.set_num_threads(1); mc = torch.jit.load(MODEL, map_location='cpu').eval()
sub = rows[::4]
opt(mc, 'cpu', confs[0])
for r in sub:
    c = [x for x in confs if x[3] == r['name']][0]
    opt(mc, 'cpu', c); t, n, ok = opt(mc, 'cpu', c); r['cpu_s'] = t; r['cpu_steps'] = n; print('cpu', r, flush=True)
res['rows'] = rows
res['summary'] = dict(median_steps=float(np.median([r['steps'] for r in rows])), mean_steps=float(np.mean([r['steps'] for r in rows])),
                      frac_converged=float(np.mean([r['converged'] for r in rows])),
                      gpu_unbatched_median_s=float(np.median([r['gpu_s'] for r in rows])), gpu_unbatched_median_s_per_step=float(np.median([r['gpu_s'] / max(r['steps'], 1) for r in rows])),
                      cpu_1thread_median_s=float(np.median([r['cpu_s'] for r in sub])), cpu_1thread_median_s_per_step=float(np.median([r['cpu_s'] / max(r['cpu_steps'], 1) for r in sub])))
print(res['summary']); json.dump(res, open(os.path.expanduser('../../data/steps_timing_results.json'), 'w'), indent=1)
