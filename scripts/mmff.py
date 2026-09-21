"""MMFF94 strain, three flavours, for every ligand in bound/global SDFs.
  sp     : single point on AIMNet2-CPCM geometries (what v4 reports)
  reopt  : bound re-optimized with all rotatable-bond dihedrals restrained to
           deposited values, global re-optimized freely (R1 #1 request)
Charges: MMFF assigns formal charges from the SDF; net charge checked vs JSON.
"""
import os, gzip
from pathlib import Path
DATA = Path(os.environ.get('LCSE_DATA_DIR', Path(__file__).resolve().parent.parent / 'data'))
OUT = DATA
FIG = DATA.parent / 'figures'; FIG.mkdir(exist_ok=True)

import json, sys, math
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem import rdForceFieldHelpers as FF
RDLogger.DisableLog('rdApp.*')

meta = json.load(gzip.open(DATA / 'LCSE_details.json.gz', 'rt'))
ROT = Chem.MolFromSmarts('[!$([NH])&!$([NH2])&!$(C(F)(F)F)&!D1&!$(*#*)]-&!@[!$([NH])&!$([NH2])&!$(C(F)(F)F)&!D1&!$(*#*)]')

def load(path):
    out = {}
    for m in Chem.SDMolSupplier(path, removeHs=False, sanitize=True):
        if m is None: continue
        out[m.GetProp('_Name')] = m
    return out

def torsions(m):
    tors = []
    for b, c in m.GetSubstructMatches(ROT):
        bond = m.GetBondBetweenAtoms(b, c)
        if bond.IsInRing(): continue
        a = [n.GetIdx() for n in m.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != c]
        d = [n.GetIdx() for n in m.GetAtomWithIdx(c).GetNeighbors() if n.GetIdx() != b]
        if a and d: tors.append((a[0], b, c, d[0]))
    return tors

def mmff_energy(m, opt=False, restrain=None, maxit=2000):
    props = FF.MMFFGetMoleculeProperties(m, mmffVariant='MMFF94')
    if props is None: return None
    ff = FF.MMFFGetMoleculeForceField(m, props, nonBondedThresh=100.0)
    if opt:
        if restrain:
            conf = m.GetConformer()
            for t in restrain:
                ang = rdMolTransforms.GetDihedralDeg(conf, *t)
                ff.MMFFAddTorsionConstraint(*t, False, ang - 0.5, ang + 0.5, 1000.0)
        ff.Initialize()
        for _ in range(5):
            if ff.Minimize(maxIts=maxit) == 0: break
        if restrain:   # report energy without the restraint term
            ff2 = FF.MMFFGetMoleculeForceField(m, props, nonBondedThresh=100.0)
            return ff2.CalcEnergy()
    return ff.CalcEnergy()

bound = load(gzip.open(DATA / 'bound_conformers.sdf.gz', 'rb'))
glob_ = load(gzip.open(DATA / 'global_conformers.sdf.gz', 'rb'))
print('loaded', len(bound), len(glob_), file=sys.stderr)

rows = []
for i, (key, mb) in enumerate(bound.items()):
    mg = glob_.get(key)
    if mg is None: continue
    mb = Chem.Mol(mb); mg = Chem.Mol(mg)
    q = sum(a.GetFormalCharge() for a in mb.GetAtoms())
    try:
        eb_sp = mmff_energy(mb); eg_sp = mmff_energy(mg)
        if eb_sp is None or eg_sp is None:
            rows.append(dict(key=key, q_sdf=q, ok=False)); continue
        tors = torsions(mb)
        eb_ro = mmff_energy(mb, opt=True, restrain=tors)
        eg_ro = mmff_energy(mg, opt=True)
        rows.append(dict(key=key, q_sdf=q, ok=True, ntors=len(tors),
                         E_bound_sp=eb_sp, E_glob_sp=eg_sp, strain_sp=eb_sp - eg_sp,
                         E_bound_reopt=eb_ro, E_glob_reopt=eg_ro, strain_reopt=eb_ro - eg_ro))
    except Exception as e:
        rows.append(dict(key=key, q_sdf=q, ok=False, err=str(e)[:80]))
    if i % 500 == 0: print(i, file=sys.stderr)

df = pd.DataFrame(rows).set_index('key')
mt = pd.DataFrame.from_dict(meta, orient='index')[['pdbid','ligand_code','charge','Estrain_aimnet_kcal','Nrotatable','Nheavy']]
df = df.join(mt, how='left')
df.to_csv(str(OUT / 'mmff94_strain_all.csv'))
print(df.ok.value_counts(), file=sys.stderr)
print((df.q_sdf != df.charge).sum(), 'charge mismatches vs JSON', file=sys.stderr)
