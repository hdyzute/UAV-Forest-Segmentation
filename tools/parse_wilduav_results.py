"""Build results_WU_<model>_<loss>_s<seed>.json from the printed TEST tables of finished WU notebooks
(WildUAV, 9 classes including Empty), for notebooks whose results zip was not downloaded.
Usage: python parse_wilduav_results.py <notebook.ipynb>... --out results_wilduav"""
import argparse
import json
import re
from pathlib import Path

CLASSES = ['Ground_vegetation', 'Trees', 'Dirt_ground', 'Road', 'Water', 'Empty',
           'Static_car', 'Moving_car', 'People']

ap = argparse.ArgumentParser()
ap.add_argument('notebooks', nargs='+')
ap.add_argument('--out', default='results_wilduav')
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
for nb_path in args.notebooks:
    cells = json.loads(Path(nb_path).read_text(encoding='utf-8'))['cells']
    text = '\n'.join(''.join(o.get('text', '')) for c in cells if c['cell_type'] == 'code'
                     for o in c.get('outputs', []) if o.get('output_type') == 'stream').replace('\r', '\n')
    for block in re.split(r'(?m)^===== ', text)[1:]:
        run = block.split()[0]
        test = re.search(r'TEST mIoU \(9 classes\) ([\d.]+)', block)
        if not test or run.endswith('_SMOKE'):
            continue
        rows = {m[0]: tuple(map(float, m[1:])) for m in
                re.findall(r'(?m)^(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$', block) if m[0] in CLASSES}
        model, loss, seed = re.match(r'WU_(\w+?)_(\w+)_s(\d+)$', run).groups()
        r = {'run_name': run, 'dataset': 'wilduav', 'model': model, 'loss': loss, 'seed': int(seed),
             'source': f'parsed from {Path(nb_path).name}', 'epochs': 50, 'evaluated': 'final model on val.txt',
             'test': {'miou': float(test.group(1)),
                      'iou': {c: rows[c][0] for c in CLASSES},
                      'recall': {c: rows[c][1] for c in CLASSES},
                      'precision': {c: rows[c][2] for c in CLASSES}}}
        (out / f'results_{run}.json').write_text(json.dumps(r, indent=2))
        print(f'{run}: {r["test"]["miou"]:.4f}')
