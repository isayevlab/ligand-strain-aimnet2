"""Same 32 ligands/conformers as steps_timing.py: MMFF94 (RDKit) and GFN2-xTB/ALPB (xtb) single-core optimization times, warm repeats."""
import os, sys, time, json, subprocess, tempfile, numpy as np
os.environ['OMP_NUM_THREADS']='1'
sys.argv=['x']; src=open(os.path.expanduser('../../data/steps_timing.py')).read().split("res = {'n_conformers'")[0]
src=src.replace("import torch","import torch\ntorch.set_num_threads(1)")
exec(src)
from rdkit.Chem import rdForceFieldHelpers as FF
XTB=os.path.expanduser('~/miniforge3/bin/xtb')
def t_mmff(c):
    z,x,q,name=c; m=[mm for mm in sel if mm.GetProp('_Name')==name][0]; m=Chem.Mol(m); conf=m.GetConformer()
    for i in range(m.GetNumAtoms()): conf.SetAtomPosition(i, x[i].tolist())
    props=FF.MMFFGetMoleculeProperties(m); ff=FF.MMFFGetMoleculeForceField(m,props); t=time.perf_counter(); ff.Minimize(maxIts=2000); return time.perf_counter()-t
def t_xtb(c):
    z,x,q,name=c
    with tempfile.TemporaryDirectory() as d:
        with open(f'{d}/in.xyz','w') as f:
            f.write(f'{len(z)}\n\n'); [f.write(f'{Chem.GetPeriodicTable().GetElementSymbol(int(zz))} {xx[0]:.5f} {xx[1]:.5f} {xx[2]:.5f}\n') for zz,xx in zip(z,x)]
        t=time.perf_counter(); r=subprocess.run([XTB,'in.xyz','--opt','--gfn','2','--alpb','water','--chrg',str(int(q)),'-P','1'],cwd=d,capture_output=True,text=True); dt=time.perf_counter()-t
        return dt, 'CONVERGED' in r.stdout
rows=[]
for c in confs:
    t_mmff(c); m1=min(t_mmff(c),t_mmff(c)); t_xtb(c); x1,ok=t_xtb(c)
    rows.append(dict(name=c[3],natoms=int(len(c[0])),mmff_s=m1,xtb_s=x1,xtb_ok=ok)); print(rows[-1],flush=True)
s=dict(mmff_median_s=float(np.median([r['mmff_s'] for r in rows])), xtb_median_s=float(np.median([r['xtb_s'] for r in rows])), xtb_all_ok=all(r['xtb_ok'] for r in rows), n=len(rows))
print(s); json.dump(dict(rows=rows,summary=s),open(os.path.expanduser('../../data/cpu_refs_results.json'),'w'),indent=1)
