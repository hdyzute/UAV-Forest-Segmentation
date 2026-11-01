"""Add per-class test IoU of the legacy CED / CEF runs (read from the notebooks' test output)
to legacy_results.json, checking that their mean matches the mIoU already stored."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / 'sunny+overcast'
NOTEBOOKS = {
    ('deeplabv3plus', 'ced'): 'deeplabv3plus/exp2-CED/deeplabv3-ced.ipynb',
    ('deeplabv3plus', 'cef'): 'deeplabv3plus/exp2-CEF/deeplabv3-cef.ipynb',
    ('segformer', 'ced'): 'segformer/exp2-CED/segformer-ced-s123.ipynb',
    ('segformer', 'cef'): 'segformer/exp2-CEF/segformer-cef-s123.ipynb',
    ('upernet', 'ced'): 'upernet/exp2-CED/upernet-ced.ipynb',
    ('upernet', 'cef'): 'upernet/exp2-CEF/upernet-cef.ipynb',
    ('unet', 'ced'): 'unet/exp2-CED/unet-ced.ipynb',
    ('unet', 'cef'): 'unet/exp2-CEF/unet-cef.ipynb',
    ('manet', 'ced'): 'ma-net/exp2-CED/ma-net-CED.ipynb',
    ('manet', 'cef'): 'ma-net/exp2-CEF/manet-cef.ipynb',
    ('unetpp', 'ced'): 'unet++/exp2-CED/unetpp-ced-s123.ipynb',
    ('unetpp', 'cef'): 'unet++/exp2-CEF/unetpp-cef-s123.ipynb',
}
NAMES = ['Sky', 'Deciduous_trees', 'Coniferous_trees', 'Fallen_trees', 'Dirt_ground',
         'Ground_vegetation', 'Rocks', 'Building', 'Fence', 'Car']

legacy = json.loads((HERE / 'legacy_results.json').read_text())
for (model, loss), rel in NOTEBOOKS.items():
    nb = json.loads((SRC / rel).read_text(encoding='utf-8'))
    out = '\n'.join(''.join(o.get('text', [])) for c in nb['cells'] if c['cell_type'] == 'code'
                    for o in c.get('outputs', []) if 'text' in o)
    blocks = re.findall(r'KẾT QUẢ TEST SET(.*?)mIoU \(10 classes\)', out, re.S)
    entry = next(r for r in legacy if r['model'] == model and r['loss'] == loss)
    if not blocks:
        print(f'  [no test block] {model} {loss}'); continue
    iou = {n: float(re.search(rf'{n}\s+([0-9.]+)', blocks[-1]).group(1)) for n in NAMES}
    mean = sum(iou.values()) / 10
    ok = abs(mean - entry['test']['miou']) < 6e-5
    print(f'  {model:14s} {loss}: mean per-class {mean:.4f} vs Table 2 {entry["test"]["miou"]:.4f} -> {"OK" if ok else "MISMATCH (not added)"}')
    if ok:
        entry['test']['iou'] = iou
(HERE / 'legacy_results.json').write_text(json.dumps(legacy, indent=1), encoding='utf-8')
