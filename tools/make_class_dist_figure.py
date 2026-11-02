"""Fig. 1: pixel frequency of each class on the Forest Inspection training split (log scale).

Frequencies are those reported in Section 4.1 of the manuscript. Same style as make_tables_figures.py.

Usage:  python tools/make_class_dist_figure.py [--out out]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from make_tables_figures import INK, SERIES, save

FREQ = [('Ground veg.', 35.41), ('Coniferous', 28.33), ('Deciduous', 16.36), ('Sky', 15.21),
        ('Dirt ground', 1.99), ('Rocks', 1.88), ('Fallen trees', 0.57), ('Building', 0.19),
        ('Car', 0.045), ('Fence', 0.0131)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent.parent / 'out'))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, (name, p) in enumerate(FREQ):
        rare = p < 1
        ax.barh(i, p, height=0.62, color=SERIES[1] if rare else SERIES[0], hatch='///' if rare else None,
                edgecolor='white', linewidth=0.6)
        ax.annotate(f'{p:g}%', xy=(p, i), xytext=(3, 0), textcoords='offset points', va='center',
                    color=INK, fontsize=8)
    ax.set_yticks(range(len(FREQ))); ax.set_yticklabels([n for n, _ in FREQ]); ax.invert_yaxis()
    ax.set_xscale('log'); ax.set_xlim(0.005, 150)
    ticks = [0.01, 0.1, 1, 10, 100]
    ax.set_xticks(ticks); ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.set_xlabel('Share of training pixels (%, log scale)'); ax.grid(axis='y', visible=False)
    ax.legend(handles=[Patch(facecolor=SERIES[0], label='≥ 1% of pixels'),
                       Patch(facecolor=SERIES[1], hatch='///', edgecolor='white', label='< 1% of pixels')],
              loc='lower right', fontsize=8)
    save(fig, out / 'fig_class_distribution'); plt.close(fig)
    print('wrote', out / 'fig_class_distribution.pdf')


if __name__ == '__main__':
    main()
