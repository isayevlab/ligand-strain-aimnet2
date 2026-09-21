"""Loaders for the released dataset (data/ directory)."""
from __future__ import annotations
import gzip, json, os
from pathlib import Path

DATA_DIR = Path(os.environ.get("LCSE_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


def load_master_table(path: str | Path | None = None):
    """Per-ligand table (7887 rows): identity, descriptors, annotations, AIMNet2-CPCM energies,
    LCSE and MMFF94 strain. Returns a pandas DataFrame indexed by ligand_id."""
    import pandas as pd
    p = Path(path) if path else DATA_DIR / "lcse_master_table.csv"
    return pd.read_csv(p, index_col="ligand_id")


def load_details(path: str | Path | None = None) -> dict:
    """The original per-ligand JSON (same content as the master table, nested)."""
    p = Path(path) if path else DATA_DIR / "LCSE_details.json.gz"
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as f:
        return json.load(f)


def load_conformers(which: str = "bound", path: str | Path | None = None, remove_hs: bool = False):
    """Iterate RDKit molecules from bound_conformers.sdf.gz or global_conformers.sdf.gz.

    Each molecule carries the SD tags ``energy_hartree`` (AIMNet2-CPCM, ensemble mean),
    ``final_fmax_eV/A`` and ``model_name``; the title is the ligand_id
    (``<ligand code>_<PDB id>_<chain>_<residue number>``).

    Note: atom order differs between the bound and global files for the same ligand.
    Map atoms with a substructure match before comparing coordinates.
    """
    from rdkit import Chem
    p = Path(path) if path else DATA_DIR / f"{which}_conformers.sdf.gz"
    fh = gzip.open(p, "rb") if str(p).endswith(".gz") else open(p, "rb")
    supp = Chem.ForwardSDMolSupplier(fh, removeHs=remove_hs)
    for m in supp:
        if m is not None:
            yield m


def conformer_dict(which: str = "bound", **kw) -> dict:
    """ligand_id -> RDKit Mol."""
    return {m.GetProp("_Name"): m for m in load_conformers(which, **kw)}
