"""Re-time AIMNet2 on CPU: warm-up run excluded, 3 repeats per ligand, report per-ligand median."""
import os, gzip
from pathlib import Path
DATA = Path(os.environ.get('LCSE_DATA_DIR', Path(__file__).resolve().parent.parent / 'data'))
OUT = DATA
FIG = DATA.parent / 'figures'; FIG.mkdir(exist_ok=True)

import json, time, random, os, sys, numpy as np, pandas as pd
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import torch; torch.set_num_threads(1)
from aimnet.calculators import AIMNet2ASE
from ase import Atoms
from ase.optimize import FIRE
prev = pd.read_csv(str(OUT / 'timing' / 'cpu_timing_container.csv'))
subset = list(prev.key)
meta = json.load(gzip.open(DATA / 'LCSE_details.json.gz', 'rt'))
mols = {}
for m in Chem.ForwardSDMolSupplier(gzip.open(DATA / 'global_conformers.sdf.gz', 'rb'), removeHs=False):
    if m is not None and m.GetProp('_Name') in subset: mols[m.GetProp('_Name')] = m
calc = AIMNet2ASE('aimnet2')
def run(m, key):
    at = Atoms(numbers=[a.GetAtomicNum() for a in m.GetAtoms()], positions=m.GetConformer().GetPositions())
    at.calc = calc; calc.set_charge(meta[key]['charge'])
    t = time.perf_counter(); FIRE(at, logfile=None).run(fmax=0.05, steps=1000); return time.perf_counter() - t
# warm-up: two full runs on the first ligand, discarded
for _ in range(2): run(mols[subset[0]], subset[0])
rows = []
for key in subset:
    ts = [run(mols[key], key) for _ in range(3)]
    rows.append(dict(key=key, aimnet_cpu_s_r1=ts[0], aimnet_cpu_s_r2=ts[1], aimnet_cpu_s_r3=ts[2], aimnet_cpu_s_median=float(np.median(ts))))
    print(key, [round(x, 2) for x in ts], file=sys.stderr)
df = pd.DataFrame(rows)
out = prev.drop(columns=['aimnet_cpu_s']).merge(df, on='key')
out.to_csv(str(OUT / 'timing' / 'cpu_timing_container.csv'), index=False)
print('AIMNet2 CPU per-ligand median of 3 repeats: median %.2f s, mean %.2f s' % (out.aimnet_cpu_s_median.median(), out.aimnet_cpu_s_median.mean()))
print('xtb median %.2f  mmff median %.3f' % (out.xtb_s.median(), out.mmff_s.median()))
