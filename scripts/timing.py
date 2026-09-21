"""CPU reference timings on one stratified subset: MMFF94 (RDKit), GFN2-xTB/ALPB (xtb), AIMNet2 gas (same architecture as CPCM), all geometry optimizations from the AIMNet2-CPCM global conformer. Single core where controllable."""
import os, gzip
from pathlib import Path
DATA = Path(os.environ.get('LCSE_DATA_DIR', Path(__file__).resolve().parent.parent / 'data'))
OUT = DATA
FIG = DATA.parent / 'figures'; FIG.mkdir(exist_ok=True)

import json, time, random, os, subprocess, tempfile, sys, numpy as np, pandas as pd
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
from rdkit import Chem, RDLogger
from rdkit.Chem import rdForceFieldHelpers as FF
RDLogger.DisableLog('rdApp.*')
meta = json.load(gzip.open(DATA / 'LCSE_details.json.gz', 'rt'))
random.seed(3)
subset = []
for lo, hi in [(10, 20), (20, 30), (30, 40), (40, 60)]:
    pool = [k for k, v in meta.items() if lo <= v['Nheavy'] < hi]; random.shuffle(pool); subset += pool[:5]
mols = {}
for m in Chem.ForwardSDMolSupplier(gzip.open(DATA / 'global_conformers.sdf.gz', 'rb'), removeHs=False):
    if m is not None and m.GetProp('_Name') in subset: mols[m.GetProp('_Name')] = m

XTB = '/tmp/claude-0/-home-claude/e65721a8-caa8-5333-887f-9ad2cf244866/scratchpad/mm/envs/omm/bin/xtb'
def t_mmff(m):
    m = Chem.Mol(m); props = FF.MMFFGetMoleculeProperties(m); ff = FF.MMFFGetMoleculeForceField(m, props)
    t = time.perf_counter(); ff.Minimize(maxIts=2000); return time.perf_counter() - t
def t_xtb(m, key):
    q = meta[key]['charge']
    with tempfile.TemporaryDirectory() as d:
        Chem.MolToXYZFile(m, f'{d}/in.xyz')
        t = time.perf_counter()
        r = subprocess.run([XTB, 'in.xyz', '--opt', '--gfn', '2', '--alpb', 'water', '--chrg', str(q), '-P', '1'], cwd=d, capture_output=True, text=True, env={**os.environ, 'OMP_NUM_THREADS': '1'})
        dt = time.perf_counter() - t
        ok = 'GEOMETRY OPTIMIZATION CONVERGED' in r.stdout
    return dt, ok

import torch; torch.set_num_threads(1)
from aimnet.calculators import AIMNet2ASE
from ase import Atoms
from ase.optimize import FIRE
calc = AIMNet2ASE('aimnet2')
def t_aimnet(m, key):
    at = Atoms(numbers=[a.GetAtomicNum() for a in m.GetAtoms()], positions=m.GetConformer().GetPositions())
    at.calc = calc; calc.set_charge(meta[key]['charge'])
    t = time.perf_counter(); n = FIRE(at, logfile=None).run(fmax=0.05, steps=1000); return time.perf_counter() - t, at.calc.results.get('nsteps', None)

rows = []
for key in subset:
    m = mols[key]; n = m.GetNumAtoms()
    r = dict(key=key, natoms=n, nheavy=meta[key]['Nheavy'], charge=meta[key]['charge'])
    r['mmff_s'] = t_mmff(m)
    r['xtb_s'], r['xtb_ok'] = t_xtb(m, key)
    r['aimnet_cpu_s'], _ = t_aimnet(m, key)
    rows.append(r); print(r, file=sys.stderr)
df = pd.DataFrame(rows); df.to_csv(str(OUT / 'timing' / 'cpu_timing_container.csv'), index=False)
print(df[['mmff_s', 'xtb_s', 'aimnet_cpu_s']].describe().round(2))
