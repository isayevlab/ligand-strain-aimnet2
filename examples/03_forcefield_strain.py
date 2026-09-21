"""Force-field strain for the released conformers (no model needed): MMFF94 single point on the
AIMNet2-CPCM geometries and re-optimized on the MMFF94 surface (bound: rotatable dihedrals restrained
to deposited values; global: free). Reproduces data/mmff94_strain_all.csv for the first N ligands.

    python examples/03_forcefield_strain.py 20
"""
import sys
from itertools import islice
from rdkit.Chem import rdForceFieldHelpers as FF, rdMolTransforms
from rdkit import Chem
from lcse import load_conformers, load_master_table
from lcse.strain import rotatable_dihedrals

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
df = load_master_table()
glob = {m.GetProp("_Name"): m for m in islice(load_conformers("global"), 5 * N)}


def mmff(m, opt=False, restrain=None):
    m = Chem.Mol(m); p = FF.MMFFGetMoleculeProperties(m); ff = FF.MMFFGetMoleculeForceField(m, p)
    if opt:
        if restrain:
            c = m.GetConformer()
            for t in restrain:
                a = rdMolTransforms.GetDihedralDeg(c, *t); ff.MMFFAddTorsionConstraint(*t, False, a - 0.5, a + 0.5, 1000.0)
        ff.Initialize(); ff.Minimize(maxIts=2000)
        if restrain:
            return FF.MMFFGetMoleculeForceField(m, p).CalcEnergy()
    return ff.CalcEnergy()


print(f"{'ligand':22s} {'AIMNet2-CPCM':>13s} {'MMFF sp':>9s} {'MMFF reopt':>11s}")
for mb in islice(load_conformers("bound"), N):
    k = mb.GetProp("_Name"); mg = glob.get(k)
    if mg is None or FF.MMFFGetMoleculeProperties(mb) is None:
        continue
    sp = mmff(mb) - mmff(mg)
    ro = mmff(mb, True, rotatable_dihedrals(mb)) - mmff(mg, True)
    print(f"{k:22s} {df.loc[k, 'lcse_kcal']:13.2f} {sp:9.2f} {ro:11.2f}")
