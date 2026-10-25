"""End-to-end CPU smoke test of a generated notebook on tiny synthetic data.

Session 1 trains with a tiny MAX_HOURS so it stops after epoch 1 (time limit),
session 2 resumes from last.pth, finishes 3 epochs, and runs the test cell.
Usage: python smoke_test_pipeline.py <notebook.ipynb> <work_dir>"""
import json
import matplotlib
matplotlib.use("Agg")
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

nb_path, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)
os.chdir(work)
cells = [''.join(c['source']) for c in json.loads(nb_path.read_text(encoding='utf-8'))['cells'] if c['cell_type'] == 'code']

# --- synthetic dataset: 3 splits of tiny RGB images and colour-coded masks
COLORS = [(0, 255, 255), (0, 127, 0), (19, 132, 69), (0, 53, 65), (130, 76, 0), (152, 251, 152),
          (151, 126, 171), (250, 150, 0), (115, 176, 195), (123, 123, 123)]
rng = np.random.default_rng(0)
files = {}
for split, n in [('train', 8), ('val', 4), ('test', 4)]:
    d = work / 'data' / split
    d.mkdir(parents=True, exist_ok=True)
    files[split] = []
    for i in range(n):
        img = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
        lab = rng.integers(0, 10, (16, 24))
        mask = np.array(COLORS, dtype=np.uint8)[np.kron(lab, np.ones((4, 4), dtype=int))]
        Image.fromarray(img).save(d / f'{i}.png'); Image.fromarray(mask).save(d / f'{i}_m.png')
        files[split].append((str(d / f'{i}.png'), str(d / f'{i}_m.png')))

def patched(i, src, resume_dir, max_hours):
    if src.startswith('!pip'):
        return ''
    if 'RUN_NAME' in src and 'SESSION_START' in src:   # config
        src = src.replace('EPOCHS       = 50', 'EPOCHS       = 3').replace('IMG_SIZE     = (768, 1280)', 'IMG_SIZE     = (64, 96)')
        src = src.replace('MAX_HOURS    = 11.0', f'MAX_HOURS    = {max_hours}').replace("RESUME_DIR   = ''", f"RESUME_DIR   = {resume_dir!r}")
        src = src.replace('USE_AMP      = True', 'USE_AMP      = False').replace('BATCH_SIZE   = 4', 'BATCH_SIZE   = 2')
        src = src.replace('NUM_WORKERS  = 32', 'NUM_WORKERS  = 0')
    if 'collect_pairs_from_dirs' in src:              # data paths -> synthetic files
        src = "import os\nfrom pathlib import Path\n" + f"train_files, val_files, test_files = {files['train']!r}, {files['val']!r}, {files['test']!r}"
    if 'MODELS = {' in src:                           # small model, no weight download
        src = "import torch.nn as nn, shutil, glob\nimport segmentation_models_pytorch as smp\nmodel = smp.Unet('resnet18', encoder_weights=None, classes=NUM_CLASSES).to(DEVICE)"
    src = src.replace('persistent_workers=True, prefetch_factor=2,', '').replace(',\n                        persistent_workers=True, prefetch_factor=2)', ')')
    src = src.replace(', prefetch_factor=2', '')
    src = src.replace('num_workers=16', 'num_workers=0')
    src = src.replace("torch.cuda.get_rng_state_all()", "[]").replace("torch.cuda.set_rng_state_all([s.cpu() for s in rs['cuda']])", "pass")
    return src

def session(resume_dir, max_hours):
    g = {}
    for i, src in enumerate(cells):
        exec(patched(i, src, resume_dir, max_hours), g)
    return g

out = work / 'outputs'
g1 = session('', 1e-9)                                  # stops after epoch 1 (time limit)
assert not g1['TRAINING_FINISHED'] and (out / 'checkpoints' / 'last.pth').exists()
print('\n--- session 1 stopped as expected; resuming ---\n')
prev = work / 'prev_session'
if prev.exists():
    import shutil; shutil.rmtree(prev)
os.rename(out, prev)                                     # like adding the old output as an input dataset
g2 = session(str(prev), 11.0)
assert g2['TRAINING_FINISHED']
res = json.loads(next(out.glob('results_*.json')).read_text())
log = (out / 'training_log.csv').read_text().strip().splitlines()
print('\nepochs logged:', [l.split(',')[0] for l in log[1:]])
assert [l.split(',')[0] for l in log[1:]] == ['1', '2', '3'], 'log not continued across sessions'
assert len(res['test']['iou']) == 10 and len(res['test']['confusion_matrix']) == 11
print('PIPELINE OK:', res['run_name'], 'test mIoU', round(res['test']['miou'], 4), 'best epoch', res['best_epoch'])
