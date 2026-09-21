# AIMNet2-CPCM model

`b973c_cpcm_ens_f.jpt` (35 MB): TorchScript ensemble of four AIMNet2 networks trained on ~20 M
B97-3c/CPCM(water) energies and forces (the same reference geometries as the gas-phase AIMNet2 models).
Elements: H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I; neutral and charged closed-shell molecules.

The weights are not yet public. They will be released with the published paper as a GitHub release of this repository. Until then, requests for the model file go to the corresponding
author (olexandr@olexandrisayev.com). Everything else in this repository (dataset, analysis, force-field
comparisons, examples 01 and 03) works without the model; only `examples/02_compute_lcse` and the
GPU timing scripts need it. Place the file at `models/b973c_cpcm_ens_f.jpt` or set `LCSE_MODEL_DIR`.

## Interface

```python
import torch
m = torch.jit.load("models/b973c_cpcm_ens_f.jpt", map_location="cuda").eval()
out = m({"coord": coord,      # float32 [B, N, 3], Å (zero-padded)
         "numbers": numbers,  # int64   [B, N]     (0 = padding)
         "charge": charge})   # float32 [B]
out["energy"], out["forces"]          # eV, eV/Å (ensemble mean)
out["energy_std"], out["forces_std"]  # ensemble spread
member0 = getattr(m.models, "0")      # one member (4x faster; used for the conformer search)
```

Long-range Coulomb and D3 dispersion are inside the graph; do not add them again. `lcse.AIMNet2CPCM`
wraps this as an ASE calculator. The gas-phase counterpart `b973c_gas_ens_f.jpt` (same architecture,
gas-phase B97-3c reference) is available on request and was used for the FreeSolv solvation-energy
benchmark in Figure 2a.

Model weights will be released under CC BY 4.0.
