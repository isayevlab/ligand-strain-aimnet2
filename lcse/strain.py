"""Ligand conformational strain energy (LCSE) with AIMNet2-CPCM.

    LCSE = E(bound, torsion-constrained relaxation) - E(global minimum in CPCM water)

This module reproduces the paper's protocol for one ligand at a time with ASE. The
production search used an in-house batched GPU optimizer over up to 5000 Omega conformers;
here conformers come from RDKit ETKDG by default. For screening-scale use, batch the
optimizations (e.g. with SaddleForge) rather than looping over ASE.
"""
from __future__ import annotations
import numpy as np
from ase import Atoms
from ase.constraints import FixInternals
from ase.optimize import FIRE, LBFGS
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from .calculator import AIMNet2CPCM

EV_TO_KCAL = 23.060547830619026
ROTATABLE = Chem.MolFromSmarts("[!$([NH])&!$([NH2])&!$(C(F)(F)F)&!D1&!$(*#*)]-&!@[!$([NH])&!$([NH2])&!$(C(F)(F)F)&!D1&!$(*#*)]")


def net_charge(mol: Chem.Mol) -> int:
    return int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))


def rotatable_dihedrals(mol: Chem.Mol) -> list[tuple[int, int, int, int]]:
    """One (a,b,c,d) quadruple per non-ring rotatable bond."""
    out = []
    for b, c in mol.GetSubstructMatches(ROTATABLE):
        if mol.GetBondBetweenAtoms(b, c).IsInRing():
            continue
        a = [n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != c]
        d = [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetIdx() != b]
        if a and d:
            out.append((a[0], b, c, d[0]))
    return out


def _atoms(mol: Chem.Mol, conf_id: int = -1) -> Atoms:
    return Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()], positions=mol.GetConformer(conf_id).GetPositions())


def bound_state_energy(mol: Chem.Mol, calc: AIMNet2CPCM, fmax: float = 0.05, steps: int = 1000) -> tuple[float, Atoms]:
    """Relax bond lengths and angles of the deposited pose with all rotatable-bond dihedrals
    fixed at their deposited values. Returns (energy in eV, relaxed Atoms)."""
    at = _atoms(mol)
    calc.set_charge(net_charge(mol)); at.calc = calc
    conf = mol.GetConformer()
    dih = [[rdMolTransforms.GetDihedralDeg(conf, *t), list(t)] for t in rotatable_dihedrals(mol)]
    if dih:
        at.set_constraint(FixInternals(dihedrals_deg=dih))
    FIRE(at, logfile=None).run(fmax=fmax, steps=steps)
    return float(at.get_potential_energy()), at


def global_minimum(mol: Chem.Mol, calc: AIMNet2CPCM, n_conformers: int = 300, fmax: float = 0.05,
                   steps: int = 1000, seed: int = 7, optimizer=LBFGS) -> tuple[float, Atoms, np.ndarray]:
    """Generate ETKDG conformers, optimize each with AIMNet2-CPCM, return the lowest.
    Returns (energy in eV, Atoms, array of all optimized energies in eV)."""
    mh = Chem.Mol(mol)
    cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_conformers, randomSeed=seed, pruneRmsThresh=0.3))
    if not cids:
        raise RuntimeError("ETKDG produced no conformers")
    calc.set_charge(net_charge(mol))
    energies, best = [], None
    for cid in cids:
        at = _atoms(mh, cid); at.calc = calc
        optimizer(at, logfile=None).run(fmax=fmax, steps=steps)
        e = float(at.get_potential_energy()); energies.append(e)
        if best is None or e < best[0]:
            best = (e, at.copy())
    return best[0], best[1], np.array(energies)


def ligand_strain(bound_mol: Chem.Mol, calc: AIMNet2CPCM, **kw) -> dict:
    """LCSE in kcal/mol for one ligand given its bound pose (RDKit Mol with explicit H)."""
    e_b, at_b = bound_state_energy(bound_mol, calc)
    e_g, at_g, all_e = global_minimum(bound_mol, calc, **kw)
    return {"lcse_kcal": (e_b - e_g) * EV_TO_KCAL, "E_bound_eV": e_b, "E_global_eV": e_g,
            "n_conformers": len(all_e), "bound": at_b, "global": at_g}
