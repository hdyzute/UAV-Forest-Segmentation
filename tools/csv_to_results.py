"""Build results_<RUN>.json from a kbs_done *_test_metrics.csv (runs whose JSON was not downloaded).
Per-class IoU for all classes; recall/precision only for the four small-object classes in the CSV.
Usage: python csv_to_results.py ../kbs_done/r07-r16_test_metrics.csv --out results/from_csv"""
import argparse
import csv
import json
from pathlib import Path

MODEL = {'UNet++': 'unetpp', 'MAnet': 'manet', 'MA-Net': 'manet'}
SMALL = ['Fallen_trees', 'Building', 'Fence', 'Car']

ap = argparse.ArgumentParser()
ap.add_argument('csv')
ap.add_argument('--out', default='results/from_csv')
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
for row in csv.DictReader(open(args.csv, encoding='utf-8')):
    model, seed = MODEL[row['model']], int(row['seed'])
    run = f"R{row['run'][1:]}_{model}_{row['loss']}_s{seed}"
    iou = {k[4:]: float(v) for k, v in row.items() if k.startswith('iou_')}
    r = {'run_name': run, 'model': model, 'loss': row['loss'], 'seed': seed,
         'source': 'kbs_done/' + Path(args.csv).name,
         'best_epoch': int(row['best_epoch']), 'best_val_miou': float(row['best_val_miou']),
         'test': {'miou': float(row['test_miou']), 'iou': iou,
                  'recall': {c: float(row[f'recall_{c}']) for c in SMALL},
                  'precision': {c: float(row[f'precision_{c}']) for c in SMALL}}}
    (out / f'results_{run}.json').write_text(json.dumps(r, indent=2))
    print(run, r['test']['miou'])
