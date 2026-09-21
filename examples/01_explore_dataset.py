"""Reproduce the headline numbers of the paper from the released master table. No model needed.

    python examples/01_explore_dataset.py
"""
import numpy as np
from lcse import load_master_table

df = load_master_table()
print(f"ligands: {len(df)}   cofactors: {(df.ligand_function == 'cofactor').sum()}   "
      f"multi-entry ligands: {(df.groupby('global_pair').size() > 1).sum()}")
print(f"LCSE median {df.lcse_kcal.median():.1f}  mean {df.lcse_kcal.mean():.1f} kcal/mol")
print("\nmedian LCSE by net charge")
print(df.groupby('net_charge').lcse_kcal.agg(['size', 'median']).round(1).to_string())
print("\nmedian LCSE by enzyme class (non-cofactors)")
ec = df[df.ligand_function.isna() & df.enzyme_class.notna()]
print(ec.groupby('enzyme_class').lcse_kcal.agg(['size', 'median']).round(1).sort_values('size', ascending=False).to_string())
neu = df[df.net_charge == 0]
print(f"\nMMFF94 single-point vs AIMNet2-CPCM, neutral ligands: median |diff| = "
      f"{(neu.mmff94_strain_sp_kcal - neu.lcse_kcal).abs().median():.1f} kcal/mol")
chg = df[df.net_charge == -3]
print(f"net charge -3: MMFF94 re-optimized minus AIMNet2-CPCM, median = "
      f"{(chg.mmff94_strain_reopt_kcal - chg.lcse_kcal).median():.1f} kcal/mol")
fmn = df[df.ligand_code == 'FMN']
print(f"\nFMN: {len(fmn)} entries, LCSE range {fmn.lcse_kcal.min():.1f} to {fmn.lcse_kcal.max():.1f} kcal/mol")
