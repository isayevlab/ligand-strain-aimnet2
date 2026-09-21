"""ASE calculator wrapping the released AIMNet2-CPCM TorchScript model.

The model file ``models/b973c_cpcm_ens_f.jpt`` is a 4-member ensemble trained to
B97-3c/CPCM(water) energies and forces. It takes a dict with keys ``coord`` [B,N,3] (Å),
``numbers`` [B,N] and ``charge`` [B] and returns ``energy`` (eV), ``forces`` (eV/Å) and
their ensemble standard deviations. Long-range Coulomb and D3 dispersion are inside the graph.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

MODEL_DIR = Path(os.environ.get("LCSE_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
DEFAULT_MODEL = MODEL_DIR / "b973c_cpcm_ens_f.jpt"
SUPPORTED_Z = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53}


class AIMNet2CPCM(Calculator):
    """ASE interface. ``member=None`` uses the 4-member ensemble mean (as deposited energies);
    ``member=0..3`` uses one member (as in the production conformer search; ~4x faster)."""
    implemented_properties = ["energy", "forces", "energy_std"]

    def __init__(self, model: str | Path = DEFAULT_MODEL, device: str | None = None,
                 charge: float = 0.0, member: int | None = None, **kw):
        super().__init__(**kw)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(model), map_location=self.device).eval()
        if member is not None:
            self.model = getattr(self.model.models, str(member))
        self.charge = float(charge)

    def set_charge(self, q: float):
        self.charge = float(q); self.reset()

    def evaluate(self, coords: np.ndarray, numbers: np.ndarray, charge: float) -> dict:
        """Batched evaluation: coords [B,N,3], numbers [B,N] (0 = padding), charge scalar or [B]."""
        q = np.broadcast_to(np.asarray(charge, dtype=np.float32), (coords.shape[0],))
        d = {"coord": torch.as_tensor(coords, dtype=torch.float32, device=self.device),
             "numbers": torch.as_tensor(numbers, dtype=torch.int64, device=self.device),
             "charge": torch.as_tensor(np.ascontiguousarray(q), device=self.device)}
        with torch.no_grad():
            o = self.model(d)
        return {k: v.detach().cpu().numpy() for k, v in o.items() if k in ("energy", "forces", "energy_std", "forces_std")}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        z = atoms.get_atomic_numbers()
        bad = set(z.tolist()) - SUPPORTED_Z
        if bad:
            raise ValueError(f"unsupported elements Z={sorted(bad)}; AIMNet2 supports {sorted(SUPPORTED_Z)}")
        o = self.evaluate(atoms.get_positions()[None], z[None], self.charge)
        self.results = {"energy": float(o["energy"][0]), "forces": o["forces"][0].astype(float)}
        if "energy_std" in o:
            self.results["energy_std"] = float(o["energy_std"][0])
