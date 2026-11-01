"""Build the LaTeX tables and Figures 3-5 of the manuscript from the run outputs.

Inputs
  results/              results_<RUN>.json files downloaded from Kaggle (any sub-folder)
  results/logs/         training_log.csv of the run used for Fig. 3 (named <RUN>_training_log.csv)
  tools/legacy_results.json   seed-123 Exp1 / CED / CEF results that are not re-trained
Outputs (written to out/)
  table_ablation.tex, table_baselines.tex, table_per_class.tex, table_small_objects.tex
  fig_loss_weights_convergence.png, fig_ablation_dotplot.png, fig_small_object_tradeoff.png

Usage:  python make_tables_figures.py [--results ../results] [--out ../out]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BACKBONES = ['deeplabv3plus', 'segformer', 'upernet', 'unet', 'manet', 'unetpp']
BB_LABEL = {'deeplabv3plus': 'DeepLabV3+', 'segformer': 'SegFormer', 'upernet': 'UPerNet',
            'unet': 'UNet', 'manet': 'MA-Net', 'unetpp': 'UNet++'}
ABLATION = ['ce', 'ced', 'cef', 'cefd', 'adaptive']
LOSS_LABEL = {'ce': 'Exp 1: CE (baseline)', 'ced': 'Exp 2-CED: CE + Dice', 'cef': 'Exp 2-CEF: CE + Focal',
              'cefd': 'Exp 2-CEFD: CE + Focal + Dice (1:1:1)', 'adaptive': 'Exp 3: Uncertainty-weighted',
              'lovasz': 'CE + Lov\\\'{a}sz-Softmax', 'focal_tversky': 'CE + Focal-Tversky',
              'unified_focal': 'Unified Focal loss', 'dwa': 'DWA (CE, Focal, Dice)',
              'famo': 'FAMO (CE, Focal, Dice)'}
BASELINE_ORDER = ['ce', 'ced', 'lovasz', 'focal_tversky', 'unified_focal', 'dwa', 'famo', 'cefd', 'adaptive']
CLASSES = ['Sky', 'Deciduous_trees', 'Coniferous_trees', 'Fallen_trees', 'Dirt_ground',
           'Ground_vegetation', 'Rocks', 'Building', 'Fence', 'Car']
CLASS_LABEL = {'Sky': 'Sky', 'Deciduous_trees': 'Deciduous', 'Coniferous_trees': 'Coniferous',
               'Fallen_trees': 'Fallen trees', 'Dirt_ground': 'Dirt ground', 'Ground_vegetation': 'Ground veg.',
               'Rocks': 'Rocks', 'Building': 'Building', 'Fence': 'Fence', 'Car': 'Car'}
SMALL = ['Fallen_trees', 'Fence', 'Car', 'Building']

# Categorical palette (fixed order; validated adjacent-pair palette, light mode).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
MARKERS = ['o', 's', '^', 'D', '*']           # secondary encoding: identity is never colour alone
INK, INK_MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9,
    'axes.edgecolor': INK_MUTED, 'axes.labelcolor': INK, 'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'legend.frameon': False, 'savefig.dpi': 500, 'savefig.bbox': 'tight',
})


# ------------------------------------------------------------------ loading
def load(results_dir, seeds=None):
    runs = json.loads((HERE / 'legacy_results.json').read_text())
    retrained = {(r['model'], r['loss'], r['seed']) for r in runs}
    for p in sorted(Path(results_dir).rglob('results_*.json')):
        r = json.loads(p.read_text())
        key = (r['model'], r['loss'], r['seed'])
        runs = [x for x in runs if (x['model'], x['loss'], x['seed']) != key]   # new run replaces legacy
        runs.append(r)
        retrained.discard(key)
    by = defaultdict(list)
    for r in runs:
        if seeds is None or r['seed'] in seeds:
            by[(r['model'], r['loss'])].append(r)
    return by


def stat(rs, f):
    v = np.array([f(r) for r in rs], dtype=float) * 100
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else None), len(v)


def fmt(m, s, n, bold=False):
    txt = f'{m:.2f}' if s is None else f'{m:.2f} $\\pm$ {s:.2f}'
    return f'\\textbf{{{txt}}}' if bold else txt


def seed123(by, model, loss):
    rs = [r for r in by.get((model, loss), []) if r['seed'] == 123]
    return rs[0] if rs else None


# ------------------------------------------------------------------ tables
def table_ablation(by):
    lines = ['\\begin{tabular}{l' + 'c' * len(BACKBONES) + '}', '\\toprule',
             'Configuration & ' + ' & '.join(BB_LABEL[b] for b in BACKBONES) + ' \\\\', '\\midrule']
    best = {b: max((stat(by[(b, l)], lambda r: r['test']['miou'])[0] for l in ABLATION if by.get((b, l))), default=None)
            for b in BACKBONES}
    for l in ABLATION:
        cells = []
        for b in BACKBONES:
            rs = by.get((b, l))
            if not rs:
                cells.append('--'); continue
            m, s, n = stat(rs, lambda r: r['test']['miou'])
            cells.append(fmt(m, s, n, bold=abs(m - best[b]) < 1e-9))
        lines.append(f'{LOSS_LABEL[l]} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines)


def table_baselines(by, model='unetpp'):
    cols = ['miou'] + SMALL
    lines = ['\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule',
             'Loss & mIoU & Fallen trees & Fence & Car & Building \\\\', '\\midrule']
    for l in BASELINE_ORDER:
        rs = by.get((model, l))
        if not rs:
            continue
        cells = []
        for c in cols:
            if c == 'miou':
                have, f = rs, (lambda r: r['test']['miou'])
            else:
                have, f = [r for r in rs if 'iou' in r['test']], (lambda r, c=c: r['test']['iou'][c])
            cells.append(fmt(*stat(have, f)) if have else '--')
        lines.append(f'{LOSS_LABEL[l]} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines)


def table_per_class(by):
    cols = []
    for b in BACKBONES:
        for l in (['ce', 'cefd', 'adaptive'] if b in ('manet', 'unetpp') else ['ce', 'adaptive']):
            r = seed123(by, b, l)
            if r is not None and 'iou' in r['test']:
                cols.append((b, l, r))
    head1 = ' & ' + ' & '.join(f'\\multicolumn{{{sum(1 for c in cols if c[0] == b)}}}{{c}}{{{BB_LABEL[b]}}}'
                                 for b in BACKBONES if any(c[0] == b for c in cols)) + ' \\\\'
    short = {'ce': 'Exp 1', 'cefd': '1:1:1', 'adaptive': 'Exp 3'}
    head2 = 'Class & ' + ' & '.join(short[l] for _, l, _ in cols) + ' \\\\'
    lines = ['\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule', head1, head2, '\\midrule']
    for c in CLASSES + ['mIoU']:
        vals = [100 * (r['test']['miou'] if c == 'mIoU' else r['test']['iou'][c]) for _, _, r in cols]
        out = []
        for i, (b, _, _) in enumerate(cols):
            group = [vals[j] for j, cc in enumerate(cols) if cc[0] == b]
            out.append(f'\\textbf{{{vals[i]:.2f}}}' if abs(vals[i] - max(group)) < 1e-9 else f'{vals[i]:.2f}')
        if c == 'mIoU':
            lines.append('\\midrule')
        lines.append(f'{CLASS_LABEL.get(c, c)} & ' + ' & '.join(out) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines)


def table_small_objects(by):
    pairs = [(b, seed123(by, b, 'ce'), seed123(by, b, 'adaptive')) for b in BACKBONES]
    pairs = [p for p in pairs if p[1] is not None and p[2] is not None and 'recall' in p[2]['test']]
    lines = ['\\begin{tabular}{ll' + 'cc' * len(pairs) + '}', '\\toprule',
             ' & & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{BB_LABEL[b]}}}' for b, _, _ in pairs) + ' \\\\',
             'Class & Metric & ' + ' & '.join('Exp 1 & Exp 3' for _ in pairs) + ' \\\\', '\\midrule']
    for k, c in enumerate(SMALL):
        rec = [100 * r['test']['recall'][c] for _, a, e in pairs for r in (a, e)]
        prec = [100 * r['test']['precision'][c] for _, a, e in pairs for r in (a, e)]
        lines.append(f'\\multirow{{3}}{{*}}{{{CLASS_LABEL[c]}}} & Recall & ' + ' & '.join(f'{v:.2f}' for v in rec) + ' \\\\')
        lines.append(' & Miss rate & ' + ' & '.join(f'{100 - v:.2f}' for v in rec) + ' \\\\')
        lines.append(' & Precision & ' + ' & '.join(f'{v:.2f}' for v in prec) + ' \\\\')
        if k < len(SMALL) - 1:
            lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines)


# ------------------------------------------------------------------ figures
def fig_loss_weights(log_csv, out):
    rows = list(csv.DictReader(open(log_csv)))
    ep = [int(r['epoch']) for r in rows]
    series = [('w_focal', 'Focal'), ('w_dice', 'Dice'), ('w_ce', 'Weighted CE')]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, (k, name) in enumerate(series):
        w = [0.5 * float(r[k]) for r in rows]           # logged exp(-s); effective weight is 0.5*exp(-s)
        ax.plot(ep, w, color=SERIES[i], lw=1.5, marker=MARKERS[i], ms=3.5, markevery=5)
        ax.annotate(f'{name}  {w[-1]:.2f}', xy=(ep[-1], w[-1]), xytext=(6, 0), textcoords='offset points',
                    va='center', color=INK, fontsize=8)
    ax.set_xlabel('Epoch'); ax.set_ylabel(r'Effective weight $w_k = \frac{1}{2}e^{-s_k}$')
    ax.set_xlim(1, ep[-1] + 9); ax.set_ylim(0, None)
    fig.savefig(out / 'fig_loss_weights_convergence.png'); plt.close(fig)


def fig_ablation(by, out):
    """Dot plot: one row per backbone, one marker per configuration (mean, with s.d. when available)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for j, l in enumerate(ABLATION):
        xs, ys, es = [], [], []
        for i, b in enumerate(BACKBONES):
            rs = by.get((b, l))
            if not rs:
                continue
            m, s, _ = stat(rs, lambda r: r['test']['miou'])
            xs.append(m); ys.append(len(BACKBONES) - 1 - i + (j - 2) * 0.12); es.append(s or 0)
        ax.errorbar(xs, ys, xerr=es, fmt=MARKERS[j], color=SERIES[j], ms=9 if MARKERS[j] == '*' else 6,
                    mec='white', mew=0.8,
                    elinewidth=1, capsize=0, label=LOSS_LABEL[l])
    ax.set_yticks(range(len(BACKBONES)))
    ax.set_yticklabels([BB_LABEL[b] for b in reversed(BACKBONES)])
    ax.set_xlabel('Test mIoU (%)'); ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    fig.savefig(out / 'fig_ablation_dotplot.png'); plt.close(fig)


def fig_small_objects(by, out, model='unetpp'):
    a, e = seed123(by, model, 'ce'), seed123(by, model, 'adaptive')
    if a is None or e is None or 'recall' not in e['test']:
        print('  [skip] Fig. 5 needs seed-123 Exp1 and Exp3 results for', model); return
    x = np.arange(len(SMALL)); w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    panels = [('iou', 'IoU (%)', '(a) Class IoU'), ('miss', 'Miss rate (%)', '(b) Ground-truth miss rate')]
    for ax, (key, ylab, title) in zip(axes, panels):
        for k, (r, name, hatch) in enumerate([(a, 'Exp 1 (CE)', None), (e, 'Exp 3', '///')]):
            v = [100 * r['test']['iou'][c] if key == 'iou' else 100 * (1 - r['test']['recall'][c]) for c in SMALL]
            bars = ax.bar(x + (k - 0.5) * w, v, w - 0.03, color=SERIES[k], hatch=hatch, edgecolor='white',
                          linewidth=0.6, label=name)
            ax.bar_label(bars, fmt='%.1f', fontsize=7, color=INK, padding=1.5)
        ax.set_xticks(x); ax.set_xticklabels([CLASS_LABEL[c] for c in SMALL])
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=9, color=INK, loc='left'); ax.grid(axis='x', visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(); fig.savefig(out / 'fig_small_object_tradeoff.png'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE.parent / 'results'))
    ap.add_argument('--out', default=str(HERE.parent / 'out'))
    ap.add_argument('--weights-log', default=None, help='training_log.csv for Fig. 3 (UNet++ Exp3 seed 123)')
    ap.add_argument('--seeds', type=int, nargs='*', default=None,
                    help='only use these seeds (e.g. --seeds 123 for an interim single-seed version)')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    by = load(args.results, args.seeds)

    for name, fn in [('table_ablation', table_ablation), ('table_baselines', table_baselines),
                     ('table_per_class', table_per_class), ('table_small_objects', table_small_objects)]:
        (out / f'{name}.tex').write_text(fn(by), encoding='utf-8'); print('wrote', name)
    fig_ablation(by, out); print('wrote fig_ablation_dotplot')
    fig_small_objects(by, out)
    log = args.weights_log or next(Path(args.results).rglob('R01_*training_log*.csv'), None)
    if log:
        fig_loss_weights(log, out); print('wrote fig_loss_weights_convergence')
    else:
        print('  [skip] Fig. 3 needs the training_log.csv of R01 (UNet++ Exp3 seed 123)')

    # runs per configuration, so missing seeds are visible at a glance
    print('\nRuns per (model, loss):')
    for (m, l), rs in sorted(by.items()):
        print(f'  {m:14s} {l:14s} seeds={sorted(r["seed"] for r in rs)}')


if __name__ == '__main__':
    main()
