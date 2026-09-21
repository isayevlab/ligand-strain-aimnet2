"""GAFF2 (AM1-BCC) strain in vacuum and in OBC2 implicit solvent for a stratified subset.
Per ligand: single point on AIMNet2-CPCM geometries; re-optimized (bound: rotatable torsions restrained
to deposited values; global: free). Both in vacuum and OBC2."""
import os, gzip
from pathlib import Path
DATA = Path(os.environ.get('LCSE_DATA_DIR', Path(__file__).resolve().parent.parent / 'data'))
OUT = DATA
FIG = DATA.parent / 'figures'; FIG.mkdir(exist_ok=True)

import json, sys, random, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolTransforms
RDLogger.DisableLog('rdApp.*')
import openmm, openmm.app as app, openmm.unit as u
from openff.toolkit import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator

meta = json.load(gzip.open(DATA / 'LCSE_details.json.gz', 'rt'))
mm = pd.read_csv(str(OUT / 'mmff94_strain_all.csv'), index_col=0)
random.seed(7)
subset = []
for q in [0, 1, -1, -2, -3]:
    pool = [k for k, v in meta.items() if v['charge'] == q and 12 <= v['Nheavy'] <= 40 and k in mm.index and mm.loc[k, 'ok'] == True]
    random.shuffle(pool); subset += pool[:10]
print(len(subset), file=sys.stderr)

def load(path, keys):
    out = {}
    for m in Chem.SDMolSupplier(path, removeHs=False):
        if m is not None and m.GetProp('_Name') in keys: out[m.GetProp('_Name')] = m
    return out
bound = load(gzip.open(DATA / 'bound_conformers.sdf.gz', 'rb'), set(subset))
glob_ = load(gzip.open(DATA / 'global_conformers.sdf.gz', 'rb'), set(subset))

ROT = Chem.MolFromSmarts('[!$([NH])&!$([NH2])&!$(C(F)(F)F)&!D1&!$(*#*)]-&!@[!$([NH])&!$([NH2])&!$(C(F)(F)F)&!D1&!$(*#*)]')
def torsions(m):
    tors = []
    for b, c in m.GetSubstructMatches(ROT):
        if m.GetBondBetweenAtoms(b, c).IsInRing(): continue
        a = [n.GetIdx() for n in m.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != c]
        d = [n.GetIdx() for n in m.GetAtomWithIdx(c).GetNeighbors() if n.GetIdx() != b]
        if a and d: tors.append((a[0], b, c, d[0]))
    return tors

KCAL = u.kilocalories_per_mole
def make_systems(rdmol):
    offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    offmol.assign_partial_charges('am1bcc')
    gen = GAFFTemplateGenerator(molecules=[offmol], forcefield='gaff-2.11')
    top = offmol.to_topology().to_openmm()
    systems = {}
    for tag, xmls in [('vac', []), ('obc2', ['implicit/obc2.xml'])]:
        ff = app.ForceField(*xmls); ff.registerTemplateGenerator(gen.generator)
        systems[tag] = ff.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    return systems

def energy(system, pos, restrain=None, opt=False):
    sysc = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    if restrain:
        f = openmm.CustomTorsionForce('0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.141592653589793')
        f.addPerTorsionParameter('k'); f.addPerTorsionParameter('theta0')
        for (a, b, c, d), th in restrain: f.addTorsion(a, b, c, d, [1000.0, th])
        sysc.addForce(f)
    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(sysc, integ, openmm.Platform.getPlatformByName('CPU'))
    ctx.setPositions(pos)
    if opt:
        openmm.LocalEnergyMinimizer.minimize(ctx, tolerance=0.01, maxIterations=5000)
        if restrain:   # re-evaluate without restraint term
            p = ctx.getState(getPositions=True).getPositions()
            return energy(system, p)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KCAL)

rows = []
for i, key in enumerate(subset):
    mb, mg = bound[key], glob_[key]
    try:
        systems = make_systems(mb)
        match = mb.GetSubstructMatch(mg, useChirality=False)
        if len(match) != mb.GetNumAtoms(): raise RuntimeError('atom mapping failed')
        pg_raw = mg.GetConformer().GetPositions(); pg_arr = np.zeros_like(pg_raw)
        for gi, bi in enumerate(match): pg_arr[bi] = pg_raw[gi]
        pb = mb.GetConformer().GetPositions() * u.angstrom; pg = pg_arr * u.angstrom
        tors = torsions(mb); conf = mb.GetConformer()
        rest = [(t, np.deg2rad(rdMolTransforms.GetDihedralDeg(conf, *t))) for t in tors]
        r = dict(key=key, charge=meta[key]['charge'], Nheavy=meta[key]['Nheavy'], Nrot=meta[key]['Nrotatable'],
                 lcse_aimnet=meta[key]['Estrain_aimnet_kcal'], mmff_sp=mm.loc[key, 'strain_sp'], mmff_reopt=mm.loc[key, 'strain_reopt'])
        for tag in ['vac', 'obc2']:
            s = systems[tag]
            r[f'gaff_{tag}_sp'] = energy(s, pb) - energy(s, pg)
            r[f'gaff_{tag}_reopt'] = energy(s, pb, restrain=rest, opt=True) - energy(s, pg, opt=True)
        rows.append(r); print(i, key, {k: round(v, 1) for k, v in r.items() if isinstance(v, float)}, file=sys.stderr)
    except Exception as e:
        print(i, key, 'FAILED', str(e)[:200], file=sys.stderr)
    pd.DataFrame(rows).to_csv(str(OUT / 'gaff2_obc2_subset.csv'), index=False)
