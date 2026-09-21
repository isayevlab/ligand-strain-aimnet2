import os, gzip
from pathlib import Path
DATA = Path(os.environ.get('LCSE_DATA_DIR', Path(__file__).resolve().parent.parent / 'data'))
OUT = DATA
FIG = DATA.parent / 'figures'; FIG.mkdir(exist_ok=True)

import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
d = pd.DataFrame.from_dict(json.load(gzip.open(DATA / 'LCSE_details.json.gz', 'rt')), orient='index')
d['rot_bin'] = pd.cut(d.Nrotatable, [-1, 2, 5, 8, 11, 100], labels=['0-2', '3-5', '6-8', '9-11', '>=12'])
g = d.groupby('rot_bin', observed=True).N_conformers
tab = pd.DataFrame({'n_ligands': g.size(), 'median_conformers': g.median(), 'p90_conformers': g.quantile(0.9),
                    'n_at_cap': g.apply(lambda s: (s >= 5000).sum())})
tab['frac_at_cap_%'] = 100 * tab.n_at_cap / tab.n_ligands
tab.to_csv(str(OUT / 'conformer_count_stats.csv'))
print(tab.round(1))
print('total', d.N_conformers.sum(), 'median', d.N_conformers.median(), 'mean', d.N_conformers.mean().round(), 'cap', (d.N_conformers >= 5000).sum())

blue = '#1f5f8b'
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.7), dpi=300)
ax[0].hist(d.N_conformers, bins=np.logspace(0, np.log10(5000), 40), color=blue, edgecolor='white', linewidth=0.3)
ax[0].set_xscale('log'); ax[0].set_xlabel('Conformers generated per ligand'); ax[0].set_ylabel('Ligands')
ax[0].axvline(5000, color='0.4', lw=0.8, ls='--'); ax[0].text(4600, ax[0].get_ylim()[1] * 0.99, 'cap = 5000', ha='right', va='top', rotation=90, fontsize=7, color='0.3')
ax[0].set_title('(a)', loc='left', fontweight='bold')
bins = list(tab.index)
data = [d.N_conformers[d.rot_bin == b].values for b in bins]
bp = ax[1].boxplot(data, tick_labels=bins, widths=0.6, showfliers=False, patch_artist=True,
                   medianprops=dict(color='white', lw=1.2), boxprops=dict(facecolor=blue, edgecolor=blue), whiskerprops=dict(color=blue), capprops=dict(color=blue))
ax[1].set_yscale('log'); ax[1].set_xlabel('Rotatable bonds'); ax[1].set_ylabel('Conformers generated per ligand')
for i, b in enumerate(bins):
    ax[1].text(i + 1, 7000, f'{tab.loc[b, "frac_at_cap_%"]:.1f}%\nat cap', ha='center', fontsize=6.5, color='0.3')
ax[1].set_ylim(1, 20000); ax[1].axhline(5000, color='0.4', lw=0.8, ls='--')
ax[1].set_title('(b)', loc='left', fontweight='bold')
fig.tight_layout(); fig.savefig(str(FIG / 'FigS_conformer_counts.png')); fig.savefig(str(FIG / 'FigS_conformer_counts.pdf'))
