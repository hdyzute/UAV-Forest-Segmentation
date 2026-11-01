"""Build results_<RUN>.json from the printed TEST table of a finished notebook,
for runs whose JSON was not downloaded. No confusion matrix (not printed).
Usage: python parse_notebook_results.py <notebook.ipynb>... --out results/parsed"""
import argparse
import json
import re
from pathlib import Path

CLASSES = ['Sky', 'Deciduous_trees', 'Coniferous_trees', 'Fallen_trees', 'Dirt_ground',
           'Ground_vegetation', 'Rocks', 'Building', 'Fence', 'Car']

ap = argparse.ArgumentParser()
ap.add_argument('notebooks', nargs='+')
ap.add_argument('--out', default='results/parsed')
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
for nb_path in args.notebooks:
    cells = json.loads(Path(nb_path).read_text(encoding='utf-8'))['cells']
    text = '\n'.join(''.join(o.get('text', '')) for c in cells if c['cell_type'] == 'code'
                     for o in c.get('outputs', []) if o.get('output_type') == 'stream')
    cfg = re.search(r'\[CONFIG\] (\S+) \| (\w+)/(\w+) \| loss=(\w+) \| seed=(\d+) \| amp=(\w+) \| wd=([\d.e-]+)', text)
    test = re.search(r'TEST mIoU \(10 classes\): ([\d.]+)', text)
    if not (cfg and test):
        print('[skip] no CONFIG/TEST output in', nb_path); continue
    run, model, enc, loss, seed, amp, wd = cfg.groups()
    res = {'run_name': run, 'model': model, 'encoder': enc, 'loss': loss, 'seed': int(seed),
           'use_amp': amp == 'True', 'weight_decay': float(wd), 'source': 'parsed from notebook output',
           'test': {'miou': float(test.group(1)), 'iou': {}, 'recall': {}, 'precision': {}}}
    for c in CLASSES:
        m = re.search(rf'^{c}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', text[test.end():], re.M)
        res['test']['iou'][c], res['test']['recall'][c], res['test']['precision'][c] = map(float, m.groups())
    best = re.findall(r'\[BEST\] val mIoU ([\d.]+) @ epoch (\d+)', text)
    if best:
        res['best_val_miou'], res['best_epoch'] = float(best[-1][0]), int(best[-1][1])
    (out / f'results_{run}.json').write_text(json.dumps(res, indent=2))
    print(f'{run}: test mIoU {res["test"]["miou"]:.4f}')
