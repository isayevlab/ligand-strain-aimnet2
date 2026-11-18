# AIMNet2-CPCM model

`b973c_cpcm_ens_f.jpt` (35 MB): TorchScript ensemble of four AIMNet2 networks trained on ~20 M
B97-3c/CPCM(water) energies and forces (the same reference geometries as the gas-phase AIMNet2 models).
Elements: H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I; neutral and charged closed-shell molecules.

The [CPCM ensemble weights](b973c_cpcm_ens_f.jpt) are included in this repository and accompany the
[published paper](https://doi.org/10.1021/acs.jcim.6c02468). Example `02_compute_lcse` and the GPU
timing scripts look for `models/b973c_cpcm_ens_f.jpt`. Set `LCSE_MODEL_DIR` to use another directory.

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

The CPCM ensemble weights are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
