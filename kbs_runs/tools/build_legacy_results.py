"""Write legacy_results.json for the seed-123 runs that are NOT re-trained (Exp1, CED, CEF).

mIoU values come from Table 2 of the manuscript, per-class IoU of Exp1 from Table 3, and the
Exp1 recall/precision of the four small classes from the notebooks' confusion-matrix outputs."""
import json
import re
from pathlib import Path

BS = chr(92)
root = Path(__file__).resolve().parents[2]
tex = (root / 'submission_kbs_final' / 'main.tex').read_text(encoding='utf-8')

def table_block(label):
    s = tex.index(BS + 'label{' + label + '}')
    return tex[s:tex.index(BS + 'end{table*}', s)]

def numbers(line):
    return [float(re.sub(r'[^0-9.]', '', c)) for c in line.replace(BS + BS, '').split('&')[1:]]

BACKBONES = ['deeplabv3plus', 'segformer', 'upernet', 'unet', 'manet', 'unetpp']
NAMES = ['Sky', 'Deciduous_trees', 'Coniferous_trees', 'Fallen_trees', 'Dirt_ground',
         'Ground_vegetation', 'Rocks', 'Building', 'Fence', 'Car']

abl = table_block('tab:ablation_study').splitlines()
row = lambda prefix: numbers(next(l for l in abl if l.strip().startswith(prefix)))
ce, ced, cef = row('Exp 1:'), row('Exp 2-CED'), row('Exp 2-CEF')

pc = [l for l in table_block('tab:per_class_iou').splitlines()
      if '&' in l and l.split('&')[0].strip() and not l.strip().startswith(('Class', BS))]
per_class = [numbers(l) for l in pc[:10]]           # 10 class rows, in NAMES order
EXP1_COL = [0, 2, 4, 6, 8, 11]

SMALL = {  # (recall, precision) from the Exp1 notebooks
    'deeplabv3plus': {'Fallen_trees': (.9201, .5330), 'Fence': (.9309, .5577), 'Car': (.9601, .7628), 'Building': (.9878, .8580)},
    'segformer':     {'Fallen_trees': (.9356, .5307), 'Fence': (.9441, .5753), 'Car': (.9588, .7833), 'Building': (.9881, .8698)},
    'upernet':       {'Fallen_trees': (.9432, .5512), 'Fence': (.9439, .6104), 'Car': (.9570, .8184), 'Building': (.9878, .8815)},
    'unet':          {'Fallen_trees': (.8993, .5742), 'Fence': (.9256, .5684), 'Car': (.9464, .8113), 'Building': (.9852, .8769)},
    'manet':         {'Fallen_trees': (.9276, .6081), 'Fence': (.9392, .5880), 'Car': (.9596, .8033), 'Building': (.9879, .8785)},
    'unetpp':        {'Fallen_trees': (.8991, .6259), 'Fence': (.9381, .5795), 'Car': (.9533, .8260), 'Building': (.9857, .8891)},
}

out = []
for j, b in enumerate(BACKBONES):
    iou = {n: per_class[i][EXP1_COL[j]] / 100 for i, n in enumerate(NAMES)}
    assert abs(sum(iou.values()) / 10 - ce[j] / 100) < 6e-5, (b, sum(iou.values()) / 10, ce[j])
    out.append({'run_name': f'legacy_{b}_ce_s123', 'model': b, 'loss': 'ce', 'seed': 123, 'source': 'legacy',
                'test': {'miou': ce[j] / 100, 'iou': iou,
                         'recall': {k: v[0] for k, v in SMALL[b].items()},
                         'precision': {k: v[1] for k, v in SMALL[b].items()}}})
    for loss, vals in (('ced', ced), ('cef', cef)):
        out.append({'run_name': f'legacy_{b}_{loss}_s123', 'model': b, 'loss': loss, 'seed': 123,
                    'source': 'legacy', 'test': {'miou': vals[j] / 100}})

dst = Path(__file__).resolve().parent / 'legacy_results.json'
dst.write_text(json.dumps(out, indent=1), encoding='utf-8')
print(f'wrote {len(out)} legacy entries -> {dst}')
