---
layout: default
title: Reproduce
---

# Reproduce

```bash
git clone https://github.com/isayevlab/ligand-strain-aimnet2.git && cd ligand-strain-aimnet2
conda env create -f environment.yml && conda activate lcse      # or: pip install -e .[analysis]
pytest                                                          # data-integrity tests, ~10 s
```

| Result | Command | Needs |
|---|---|---|
| Headline numbers | `python examples/01_explore_dataset.py` | data only |
| Figures 3 to 6, S2, S7 to S13 | `jupyter lab notebooks/LCSE_analysis.ipynb` | data, matplotlib, seaborn |
| Figure S14 (conformer statistics) | `python scripts/confstats.py` | data |
| Figure S15, Table S4 (MMFF94) | `python scripts/mmff.py && python scripts/mmff_fig.py` | data, RDKit; ~20 min |
| Figure S16, Table S5 (GAFF2/OBC2) | `python scripts/gaff.py && python scripts/gaff_fig.py` | OpenMM, openmmforcefields, openff-toolkit, ambertools; ~1 h |
| Table S3 timings | `scripts/timing/*.py` | model; GPU rows also need [SaddleForge](https://github.com/isayevlab/saddleforge) and xtb |
| LCSE for a new ligand | `python examples/02_compute_lcse.py pose.sdf` | model, GPU recommended |

Scripts read from `data/` (override with `LCSE_DATA_DIR`) and the model from `models/` (`LCSE_MODEL_DIR`). The example notebooks in `examples/` are executed copies of the scripts with output.

## Getting the model

`models/b973c_cpcm_ens_f.jpt` (35 MB) will be released with the published paper; see `models/README.md`. All results except example 02 and the GPU timing rows can be reproduced without it.
