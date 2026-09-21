"""lcse: solvated ligand conformational strain energy with AIMNet2-CPCM.

Companion package to
Gokcan, H.; Isayev, O. Efficient and Accurate Ligand Strain Calculations in Solution
with the AIMNet2 Neural Network Potential. J. Chem. Inf. Model. (2026).
"""
from .data import DATA_DIR, load_master_table, load_conformers, load_details
from .calculator import AIMNet2CPCM
from .strain import bound_state_energy, global_minimum, ligand_strain

__all__ = ["DATA_DIR", "load_master_table", "load_conformers", "load_details",
           "AIMNet2CPCM", "bound_state_energy", "global_minimum", "ligand_strain"]
__version__ = "1.0.0"
HARTREE_TO_KCAL = 627.5094740631
EV_TO_KCAL = 23.060547830619026
