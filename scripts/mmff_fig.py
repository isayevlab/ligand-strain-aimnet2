import os, gzip
from pathlib import Path
DATA = Path(os.environ.get('LCSE_DATA_DIR', Path(__file__).resolve().parent.parent / 'data'))
OUT = DATA
FIG = DATA.parent / 'figures'; FIG.mkdir(exist_ok=True)

import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
df = pd.read_csv(str(OUT / 'mmff94_strain_all.csv'), index_col=0); df = df[df.ok == True]
df['qcls'] = df.charge.clip(-3, 2)
order = [-3, -2, -1, 0, 1, 2]; labels = ['<= -3', '-2', '-1', '0', '+1', '>= +2']
c_sp, c_ro = '#7fa7c9', '#1f5f8b'
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8), dpi=300)
w = 0.36
for j, (col, color, name) in enumerate([('strain_sp', c_sp, 'MMFF94 single point'), ('strain_reopt', c_ro, 'MMFF94 re-optimized')]):
    data = [(df[df.qcls == q][col] - df[df.qcls == q].Estrain_aimnet_kcal).values for q in order]
    pos = np.arange(len(order)) + (j - 0.5) * (w + 0.04)
    bp = ax[0].boxplot(data, positions=pos, widths=w, showfliers=False, patch_artist=True,
                       medianprops=dict(color='white', lw=1.1), boxprops=dict(facecolor=color, edgecolor=color),
                       whiskerprops=dict(color=color), capprops=dict(color=color))
    ax[0].plot([], [], color=color, lw=6, label=name)
ax[0].axhline(0, color='0.4', lw=0.8, ls='--'); ax[0].set_xticks(range(len(order)), labels)
ax[0].set_xlabel('Net charge'); ax[0].set_ylabel('ΔStrain, FF − AIMNet2-CPCM (kcal/mol)')
ax[0].legend(frameon=False, fontsize=7, loc='upper right'); ax[0].set_title('(a)', loc='left', fontweight='bold')
# (b) scatter re-opt vs AIMNet, neutral vs charged
neu = df[df.charge == 0]; chg = df[df.charge.abs() >= 2]
ax[1].scatter(neu.Estrain_aimnet_kcal, neu.strain_reopt, s=3, alpha=0.35, color=c_ro, lw=0, label=f'neutral (n={len(neu)})')
ax[1].scatter(chg.Estrain_aimnet_kcal, chg.strain_reopt, s=3, alpha=0.35, color='#c8553d', lw=0, label=f'|q| >= 2 (n={len(chg)})')
lim = (-5, 60); ax[1].plot(lim, lim, color='0.4', lw=0.8, ls='--'); ax[1].set_xlim(-2, 40); ax[1].set_ylim(lim)
ax[1].set_xlabel('AIMNet2-CPCM strain (kcal/mol)'); ax[1].set_ylabel('MMFF94 re-optimized strain (kcal/mol)')
ax[1].legend(frameon=False, fontsize=7, loc='upper left', markerscale=4); ax[1].set_title('(b)', loc='left', fontweight='bold')
fig.tight_layout(); fig.savefig(str(FIG / 'FigS_mmff94_reopt.png')); fig.savefig(str(FIG / 'FigS_mmff94_reopt.pdf'))
