"""Data-integrity tests; run with `pytest`. They need only the data/ directory (no model)."""
import gzip
import numpy as np
import pytest
from lcse import load_master_table, load_conformers, DATA_DIR

HARTREE_TO_KCAL = 627.5094740631


@pytest.fixture(scope="module")
def df():
    return load_master_table()


def test_shape_and_columns(df):
    assert len(df) == 7887
    for c in ["pdbid", "ligand_code", "smiles", "net_charge", "n_heavy", "n_rotatable", "lcse_kcal",
              "E_bound_hartree", "E_global_hartree", "enzyme_class", "mmff94_strain_sp_kcal"]:
        assert c in df.columns


def test_lcse_consistent_with_energies(df):
    calc = (df.E_bound_hartree - df.E_global_hartree) * HARTREE_TO_KCAL
    assert np.allclose(calc, df.lcse_kcal, atol=1e-3)


def test_headline_numbers(df):
    assert abs(df.lcse_kcal.median() - 4.0) < 0.1
    assert abs(df[df.net_charge == 0].lcse_kcal.median() - 2.7) < 0.1
    assert (df.ligand_function == "cofactor").sum() == 304


def test_sdf_files_match_table(df):
    for which in ["bound", "global"]:
        names = []
        for i, m in enumerate(load_conformers(which)):
            names.append(m.GetProp("_Name"))
            if i == 0:
                assert m.HasProp("energy_hartree") and m.HasProp("model_name")
            if i >= 200:
                break
        assert all(n in df.index for n in names)


def test_bound_energy_tag_matches_table(df):
    m = next(load_conformers("bound"))
    assert abs(float(m.GetProp("energy_hartree")) - df.loc[m.GetProp("_Name"), "E_bound_hartree"]) < 1e-6


def test_files_present():
    for f in ["bound_conformers.sdf.gz", "global_conformers.sdf.gz", "LCSE_details.json.gz",
              "mmff94_strain_all.csv", "gaff2_obc2_subset.csv"]:
        assert (DATA_DIR / f).exists(), f
