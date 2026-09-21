"""Compute solvated LCSE for one ligand from its bound pose with the released AIMNet2-CPCM model.

    python examples/02_compute_lcse.py                      # first ligand of the released set
    python examples/02_compute_lcse.py my_pose.sdf          # any SDF with explicit hydrogens

Requires models/b973c_cpcm_ens_f.jpt (see models/README.md). GPU recommended; on CPU a
300-conformer search takes ~15 min for a 50-atom ligand.
"""
import sys
from rdkit import Chem
from lcse import AIMNet2CPCM, ligand_strain, load_conformers, EV_TO_KCAL

if len(sys.argv) > 1:
    mol = next(m for m in Chem.SDMolSupplier(sys.argv[1], removeHs=False) if m is not None)
else:
    mol = next(load_conformers("bound"))
name = mol.GetProp("_Name") if mol.HasProp("_Name") else "ligand"
print(f"{name}: {mol.GetNumAtoms()} atoms, net charge {sum(a.GetFormalCharge() for a in mol.GetAtoms())}")

calc = AIMNet2CPCM(member=0)          # one ensemble member, as in the production search
res = ligand_strain(mol, calc, n_conformers=300)
print(f"E_bound  = {res['E_bound_eV']:.4f} eV   E_global = {res['E_global_eV']:.4f} eV   "
      f"({res['n_conformers']} conformers optimized)")
print(f"LCSE = {res['lcse_kcal']:.2f} kcal/mol")
if mol.HasProp("energy_hartree"):
    print(f"(deposited bound-state energy, ensemble mean: {float(mol.GetProp('energy_hartree')):.6f} hartree)")
res["bound"].write(f"{name}_bound_relaxed.xyz"); res["global"].write(f"{name}_global_min.xyz")
