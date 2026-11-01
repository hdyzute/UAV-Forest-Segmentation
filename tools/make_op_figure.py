"""Precision-recall curves of the operating-point study (OP1) for the paper.

Usage:  python tools/make_op_figure.py <dir with sweep_*.npz from OP1_results.zip> [--out out]
Each curve shifts the logit of one class over the bias grid; the dot marks the arg-max rule (no shift),
the dashed line the recall of weighted CE without a shift."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
MARKERS = ['o', 's', '^', 'D']
INK, INK_MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9,
    'axes.edgecolor': INK_MUTED, 'axes.labelcolor': INK, 'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'legend.frameon': False,
})

RUNS = [('Exp1', 'Exp 1: CE'), ('Exp2-CED', 'Exp 2-CED: CE + Dice'),
        ('Exp2-CEFD', 'Exp 2-CEFD: 1:1:1'), ('Exp3', 'Exp 3: Uncertainty')]
CLASS_NAMES = ['Sky', 'Deciduous trees', 'Coniferous trees', 'Fallen trees', 'Dirt ground',
               'Ground vegetation', 'Rocks', 'Building', 'Fence', 'Car', 'Empty']


def rec_prec(cm, c):
    cm = cm.reshape(11, 11).astype(float)
    tp = cm[c, c]
    return 100 * tp / max(cm[c].sum(), 1), 100 * tp / max(cm[:, c].sum(), 1)


ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('--out', default='out')
args = ap.parse_args()

data = {k: np.load(Path(args.src) / f'sweep_{k}.npz') for k, _ in RUNS}
grid = data['Exp1']['grid']
classes = [int(c) for c in data['Exp1']['classes']]
zero = int(np.argmin(np.abs(grid)))

fig, axes = plt.subplots(1, len(classes), figsize=(7.2, 2.35))
for ax, (i, c) in zip(axes, enumerate(classes)):
    ref_rec = rec_prec(data['Exp1']['test'][i, zero], c)[0]
    ax.axvline(ref_rec, color=INK_MUTED, lw=0.8, ls='--', zorder=1)
    lo = 100
    for j, (k, label) in enumerate(RUNS):
        pts = np.array([rec_prec(data[k]['test'][i, g], c) for g in range(len(grid))])
        lo = min(lo, pts[:, 1].min())
        ax.plot(pts[:, 0], pts[:, 1], color=SERIES[j], lw=1.5, label=label, zorder=2)
        ax.plot(*pts[zero], MARKERS[j], color=SERIES[j], ms=5, mec='white', mew=0.8, zorder=3)
    ax.set_title(CLASS_NAMES[c], fontsize=9, color=INK, loc='left')
    ax.set_xlabel('Recall (%)')
    ax.set_xlim(right=100)
axes[0].set_ylabel('Precision (%)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(w_pad=0.8)
out = Path(args.out); out.mkdir(exist_ok=True)
fig.savefig(out / 'fig_operating_point.png', dpi=500, bbox_inches='tight', facecolor='white')
print('wrote', out / 'fig_operating_point.png')
