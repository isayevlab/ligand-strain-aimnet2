import os, gzip
from pathlib import Path
DATA = Path(os.environ.get('LCSE_DATA_DIR', Path(__file__).resolve().parent.parent / 'data'))
OUT = DATA
FIG = DATA.parent / 'figures'; FIG.mkdir(exist_ok=True)

import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
df = pd.read_csv(str(OUT / 'gaff2_obc2_subset.csv'))
order = [-3, -2, -1, 0, 1]; labels = ['-3', '-2', '-1', '0', '+1']
series = [('mmff_reopt', '#9db8d0', 'MMFF94, vacuum'), ('gaff_vac_reopt', '#4a7ba6', 'GAFF2, vacuum'), ('gaff_obc2_reopt', '#c8553d', 'GAFF2, OBC2 solvent')]
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8), dpi=300)
w = 0.25
for j, (col, color, name) in enumerate(series):
    data = [(df[df.charge == q][col] - df[df.charge == q].lcse_aimnet).values for q in order]
    pos = np.arange(len(order)) + (j - 1) * (w + 0.03)
    ax[0].boxplot(data, positions=pos, widths=w, showfliers=True, patch_artist=True, flierprops=dict(marker='.', markersize=3, markeredgecolor=color),
                  medianprops=dict(color='white', lw=1.1), boxprops=dict(facecolor=color, edgecolor=color), whiskerprops=dict(color=color), capprops=dict(color=color))
    ax[0].plot([], [], color=color, lw=6, label=name)
ax[0].axhline(0, color='0.4', lw=0.8, ls='--'); ax[0].set_xticks(range(len(order)), labels)
ax[0].set_xlabel('Net charge'); ax[0].set_ylabel('ΔStrain, FF − AIMNet2-CPCM (kcal/mol)')
ax[0].legend(frameon=False, fontsize=7, loc='upper right'); ax[0].set_title('(a) re-optimized endpoints', loc='left', fontweight='bold', fontsize=8)
for col, color, name in series[1:]:
    ax[1].scatter(df.lcse_aimnet, df[col], s=14, color=color, alpha=0.8, lw=0, label=name)
lim = (-5, 50); ax[1].plot(lim, lim, color='0.4', lw=0.8, ls='--'); ax[1].set_xlim(-2, 30); ax[1].set_ylim(lim)
ax[1].set_xlabel('AIMNet2-CPCM strain (kcal/mol)'); ax[1].set_ylabel('GAFF2 re-optimized strain (kcal/mol)')
ax[1].legend(frameon=False, fontsize=7, loc='upper left'); ax[1].set_title('(b)', loc='left', fontweight='bold')
fig.tight_layout(); fig.savefig(str(FIG / 'FigS_gaff2_obc2.png')); fig.savefig(str(FIG / 'FigS_gaff2_obc2.pdf'))
