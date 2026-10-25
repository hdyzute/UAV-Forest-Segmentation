"""Generate tools/../figures/make_qualitative.ipynb (runs on Kaggle, needs the test data and two checkpoints).

Output figure: rows = selected test frames, columns = Input | Ground truth | Exp 1 (CE) | Exp 3.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import CELLS, RESNET_DIR, ROOT, nb_json

CONFIG = r'''# ============================================================
# CONFIG
# ============================================================
MODEL_NAME, ENCODER, SEED, USE_AMP = 'unetpp', 'resnet50', 123, True
ENCODER_WEIGHT_DIR = '%s'

# best.pth of the two UNet++ seed-123 runs (add their outputs as input datasets)
CKPT_EXP1 = '/kaggle/input/<dataset-of-UNet++-Exp1-s123>/best.pth'      # old checkpoints also work
CKPT_EXP3 = '/kaggle/input/<dataset-of-R01_unetpp_adaptive_s123>/checkpoints/best.pth'

# (sequence, frame number in the file name, (top, left) of the crop)
FRAMES = [('seq6', 290, (192, 320)), ('seq11', 3701, (192, 320)), ('seq6', 654, (192, 320))]
CROP = (384, 640)            # crop height, width (full frame is 768 x 1280)
GAMMA = 0.7                  # display-only brightening of the input image
''' % RESNET_DIR

QUAL = r'''import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import albumentations as A
from albumentations.pytorch import ToTensorV2

val_tf = A.Compose([A.Resize(height=768, width=1280),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

def load_model(path):
    m = MODELS[MODEL_NAME](encoder_name=ENCODER, encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck['model_state_dict'])
    print(f'loaded {path} (epoch {ck.get("epoch")}, val mIoU {ck.get("miou")})')
    return m.to(DEVICE).eval()

def find_frame(seq, frame):
    cands = [p for p in test_files if Path(p[0]).parent.parent.name == seq]
    hit = [p for p in cands if re.findall(r'\d+', Path(p[0]).stem) and int(re.findall(r'\d+', Path(p[0]).stem)[-1]) == frame]
    if not hit:
        raise ValueError(f'{seq} frame {frame} not found; example names: {[Path(p[0]).name for p in cands[:3]]}')
    return hit[0]

@torch.no_grad()
def predict(m, img):
    x = val_tf(image=img)['image'].unsqueeze(0).to(DEVICE)
    with torch.amp.autocast('cuda', enabled=USE_AMP):
        return m(x).argmax(1)[0].cpu().numpy()

models = {'Exp 1 (CE)': load_model(CKPT_EXP1), 'Exp 3': load_model(CKPT_EXP3)}
ch, cw = CROP
fig, axes = plt.subplots(len(FRAMES), 4, figsize=(12, 2.0 * len(FRAMES) + 0.6))
axes = np.atleast_2d(axes)
for r, (seq, frame, (top, left)) in enumerate(FRAMES):
    img_p, mask_p = find_frame(seq, frame)
    img = np.array(Image.open(img_p).convert('RGB'))
    gt = fast_rgb_to_mask(np.array(Image.open(mask_p).convert('RGB')))
    panels = [np.clip(255 * (img / 255.0) ** GAMMA, 0, 255).astype(np.uint8), PALETTE[gt]]
    panels += [PALETTE[predict(m, img)] for m in models.values()]
    titles = ['Input', 'Ground truth'] + list(models)
    for c, (p, t) in enumerate(zip(panels, titles)):
        ax = axes[r, c]
        ax.imshow(p[top:top + ch, left:left + cw]); ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(t, fontsize=11)
        if c == 0:
            ax.set_ylabel(f'{seq}, frame {frame}', fontsize=9)
legend = [Patch(facecolor=PALETTE[i] / 255.0, edgecolor='#999999', label=n.replace('_', ' '))
          for i, n in enumerate(CLASS_NAMES[:-1])]
fig.legend(handles=legend, loc='lower center', ncol=10, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUTPUT_DIR / 'fig_qualitative.png', dpi=300, bbox_inches='tight')
print('saved', OUTPUT_DIR / 'fig_qualitative.png')
'''

cells = [CELLS[0], ('code', CONFIG), CELLS[2], CELLS[3], CELLS[4], CELLS[6], ('code', QUAL)]
title = ('# Qualitative figure (UNet++, Exp 1 vs Exp 3)\n\n'
         'Set the two checkpoint paths and the frames/crops in the CONFIG cell, then run all cells. '
         'Output: `outputs/fig_qualitative.png`.\n')
d = ROOT / 'figures'
d.mkdir(exist_ok=True)
(d / 'make_qualitative.ipynb').write_text(json.dumps(nb_json([('markdown', title)] + cells), indent=1, ensure_ascii=False),
                                          encoding='utf-8')
print('wrote', d / 'make_qualitative.ipynb')
