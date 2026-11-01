"""Generate the week-4 Kaggle notebooks (see WEEK4_PLAN.md).

Usage:  python build_week4_notebooks.py
Writes week4/ next to this tools/ folder:
  WU1..WU6  WildUAV loss ablation (UNet++, real data); one notebook trains several losses for one seed.
  OP1       operating-point study on Forest Inspection (inference only, existing best.pth files).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import CELLS as FOREST_CELLS, RESNET_DIR, nb_json

ROOT = Path(__file__).resolve().parent.parent

WILDUAV_NOTEBOOKS = [
    ('WU1_wilduav_s123', ['ce', 'ced', 'cefd', 'adaptive'], 123),
    ('WU2_wilduav_s42', ['ce', 'ced', 'cefd', 'adaptive'], 42),
    ('WU3_wilduav_s2026', ['ce', 'ced', 'cefd', 'adaptive'], 2026),
    ('WU4_wilduav_unified_focal_s123', ['unified_focal'], 123),
    ('WU5_wilduav_unified_focal_s42', ['unified_focal'], 42),
    ('WU6_wilduav_unified_focal_s2026', ['unified_focal'], 2026),
]

# ================================================================ WildUAV template
WU = []

WU.append(('code', FOREST_CELLS[0][1]))   # offline pip install

WU.append(('code', r'''# ============================================================
# CONFIG — the only cell that differs between WildUAV notebooks
# ============================================================
import time
SESSION_START = time.time()

NOTEBOOK     = '{nb_name}'
LOSSES       = {losses}      # trained one after another in this session
SEED         = {seed}
MODEL_NAME   = 'unetpp'
ENCODER      = 'resnet50'
USE_AMP      = True
AMP_DTYPE    = 'float16'

LR           = 1e-4
WEIGHT_DECAY = 1e-4            # same as every Forest Inspection run
EPOCHS       = 50              # fixed length, no early stopping: the FINAL model is evaluated,
                               # so no checkpoint is selected on the images that are reported
BATCH_SIZE   = 4
IMG_SIZE     = (768, 1024)     # (H, W); WildUAV Mapping frames are 5280x3956 (4:3)
MAX_HOURS    = 11.0            # do not start another loss if it would not finish before this
EVAL_EVERY   = 10              # print test mIoU every N epochs - monitoring only, never used to pick a model

SMOKE        = False           # True: 2 epochs on 16 training images per loss, to check the notebook (NOT for the paper)
MULTI_GPU    = True            # T4 x2: each batch of 4 is split over both GPUs (nn.DataParallel).
                               # Use the same GPU type for WU1-WU4 so that all runs share one setting.

# Output of an earlier session of THIS notebook (add it as an input dataset): finished losses are skipped.
DONE_DIR     = ''

WILD_UAV_OLD = '/kaggle/input/datasets/dinhh04/wilduav/WildUAV'
WILD_UAV_NEW = '/kaggle/input/datasets/phmnguynkimquyn/wuav-data'
TRAIN_TXT    = '/kaggle/input/datasets/itmeagain77/wilduav-splits/train.txt'
VAL_TXT      = '/kaggle/input/datasets/itmeagain77/wilduav-splits/val.txt'
ENCODER_WEIGHT_DIR = '{resnet_dir}'
print(f'[CONFIG] {{NOTEBOOK}} | losses={{LOSSES}} | seed={{SEED}} | {{IMG_SIZE}} | wd={{WEIGHT_DECAY}} | SMOKE={{SMOKE}}')'''))

WU.append(('code', r'''import os
import numpy as np

# Official partition: train.txt (998 frames) for training, val.txt (246 frames) for reporting.
# Paths follow the earlier WildUAV notebook (images: seq00 in the old dataset, others in the new one).
def read_split_file(txt_path):
    pairs = []
    for line in open(txt_path):
        line = line.strip()
        if not line:
            continue
        seq, fname = line.split('/')
        base = fname.split('.')[0]
        if seq == 'seq00':
            img_dir = os.path.join(WILD_UAV_OLD, 'mapping_set', f'{seq}_img+metadata', 'img')
        else:
            img_dir = os.path.join(WILD_UAV_NEW, f'{seq}_imgmetadata.7z', f'{seq}_img+metadata.7z', 'img')
        img = next((p for p in (os.path.join(img_dir, base + e) for e in ('.png', '.jpg')) if os.path.exists(p)), None)
        lbl = next((p for p in (os.path.join(r, 'mapping_set', 'semantic_extension', f'{seq}_semantic', 'semantic', base + '.png')
                                for r in (WILD_UAV_OLD, WILD_UAV_NEW)) if os.path.exists(p)), None)
        if img and lbl:
            pairs.append((img, lbl))
    return pairs

train_pairs, test_pairs = read_split_file(TRAIN_TXT), read_split_file(VAL_TXT)
print(f'TRAIN {len(train_pairs)} | TEST (val.txt) {len(test_pairs)}')
assert (len(train_pairs), len(test_pairs)) == (998, 246), 'Unexpected split sizes - check the input dataset paths'
if SMOKE:
    train_pairs, test_pairs = train_pairs[:16], test_pairs[:8]

# 16 raw label colours -> the 9 benchmark classes (same mapping as the earlier notebook;
# unknown colours and rocks/building/fence/sky -> Empty). Empty is an active class, as in the official protocol.
WILDUAV_COLORS = [
    (0, 255, 255), (0, 127, 0), (19, 132, 69), (0, 53, 65), (130, 76, 0),
    (152, 251, 152), (151, 126, 171), (0, 0, 255), (250, 150, 0), (115, 176, 195),
    (128, 64, 128), (255, 77, 228), (123, 123, 123), (255, 255, 255), (200, 0, 0), (0, 0, 0),
]
CLASS_NAMES = ['Ground_vegetation', 'Trees', 'Dirt_ground', 'Road', 'Water', 'Empty',
               'Static_car', 'Moving_car', 'People']
NUM_CLASSES = len(CLASS_NAMES)
EMPTY_CLASS = 5
IGNORE = 255                   # only for border pixels created by the affine augmentation
RAW_TO_CLASS = np.full(16, EMPTY_CLASS, dtype=np.uint8)
for raw, cls in {0: 0, 5: 0, 1: 1, 2: 1, 3: 1, 4: 2, 10: 3, 11: 3, 7: 4, 15: 5, 12: 6, 13: 7, 14: 8}.items():
    RAW_TO_CLASS[raw] = cls
RGB_TO_CLASS = np.full((256, 256, 256), EMPTY_CLASS, dtype=np.uint8)
for raw, color in enumerate(WILDUAV_COLORS):
    RGB_TO_CLASS[color] = RAW_TO_CLASS[raw]
'''))

WU.append(('code', r'''# Decode and resize every frame once and keep it in RAM (about 4 GB): the 20-Mpixel PNGs are too slow
# to decode in every epoch. Image: area interpolation; label: nearest neighbour on the colour image.
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

H, W = IMG_SIZE

def load_pair(pair):
    img = cv2.imread(pair[0], cv2.IMREAD_COLOR)
    lbl = cv2.imread(pair[1], cv2.IMREAD_COLOR)
    assert img is not None and lbl is not None, pair
    assert img.shape[:2] == lbl.shape[:2], f'image/label size mismatch: {pair}'
    shape = img.shape[:2]
    img = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    lbl = cv2.cvtColor(cv2.resize(lbl, (W, H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
    return img, RGB_TO_CLASS[lbl[..., 0], lbl[..., 1], lbl[..., 2]], shape

def load_all(pairs, desc):
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        out = list(tqdm(ex.map(load_pair, pairs), total=len(pairs), desc=desc))
    return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]

train_imgs, train_masks, shapes = load_all(train_pairs, 'cache train')
test_imgs, test_masks, shapes_t = load_all(test_pairs, 'cache test')
aspects = {round(w / h, 3) for h, w in shapes + shapes_t}
print('native sizes:', sorted({f'{w}x{h}' for h, w in shapes + shapes_t}), '| aspect', aspects, '| resized to', f'{W}x{H}')
if any(abs(a - W / H) > 0.03 for a in aspects):
    print('[WARN] IMG_SIZE does not match the native aspect ratio - images are stretched')

counts = sum(np.bincount(m.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES] for m in train_masks)
freq = counts / counts.sum()
for n, f in zip(CLASS_NAMES, freq):
    print(f'  {n:<18}{100 * f:8.3f}%')
'''))

WU.append(('code', r'''import random
from pathlib import Path
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

AMP_DTYPE_T = torch.bfloat16 if AMP_DTYPE == 'bfloat16' else torch.float16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('[INFO] Device:', torch.cuda.get_device_name() if DEVICE == 'cuda' else 'cpu')

OUTPUT_DIR = Path('/kaggle/working/outputs') if os.path.exists('/kaggle') else Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ENet weights w = 1/ln(1.02 + f_c) on the WildUAV training split, normalised to sum to NUM_CLASSES
w = 1.0 / np.log(1.02 + freq)
class_weights = torch.tensor(w / w.sum() * NUM_CLASSES, dtype=torch.float32).to(DEVICE)
print('[ENet weights]', dict(zip(CLASS_NAMES, np.round(class_weights.cpu().numpy(), 4))))
'''))

WU.append(('code', r'''from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CachedDataset(Dataset):
    def __init__(self, imgs, masks, transform):
        self.imgs, self.masks, self.transform = imgs, masks, transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        aug = self.transform(image=self.imgs[idx], mask=self.masks[idx])
        return aug['image'], aug['mask'].long()

# Same augmentation as Forest Inspection (images are already resized). Empty is an active class here,
# so the border pixels created by the affine transform are ignored instead of being labelled Empty.
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), rotate=(-15, 15),
             translate_percent={'x': (-0.0625, 0.0625), 'y': (-0.0625, 0.0625)},
             p=0.3, border_mode=0, fill=0, fill_mask=IGNORE),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
test_tf = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

NUM_WORKERS = min(8, os.cpu_count() or 2)
train_ds = CachedDataset(train_imgs, train_masks, train_tf)
test_loader = DataLoader(CachedDataset(test_imgs, test_masks, test_tf), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)
'''))

WU.append(('code', r'''import shutil, glob
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

CACHE_DIR = '/root/.cache/torch/hub/checkpoints/'
os.makedirs(CACHE_DIR, exist_ok=True)
pth_files = glob.glob(os.path.join(ENCODER_WEIGHT_DIR, '*.pth'))
if pth_files and not os.path.exists(os.path.join(CACHE_DIR, 'resnet50-19c8e357.pth')):
    shutil.copy(pth_files[0], os.path.join(CACHE_DIR, 'resnet50-19c8e357.pth'))

N_GPUS = torch.cuda.device_count() if MULTI_GPU else 1

def build_model():
    m = smp.UnetPlusPlus(encoder_name=ENCODER, encoder_weights='imagenet',
                         in_channels=3, classes=NUM_CLASSES).to(DEVICE)
    return nn.DataParallel(m) if N_GPUS > 1 else m

print(f'[GPU] {N_GPUS} x', torch.cuda.get_device_name() if DEVICE == 'cuda' else 'cpu')

# ---- losses: same definitions as on Forest Inspection, but all 9 classes are active and only
# augmentation borders (IGNORE) are excluded from every term.
def make_ce():
    return nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE)

def make_focal():
    return smp.losses.FocalLoss(mode=smp.losses.MULTICLASS_MODE, gamma=2.0, ignore_index=IGNORE)

def make_dice():
    return smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True, ignore_index=IGNORE)

class FixedWeightLoss(nn.Module):
    """Exp1 (ce), Exp2-CED, Exp2-CEFD (1:1:1): static weights."""
    def __init__(self, w_ce=1.0, w_focal=0.0, w_dice=0.0):
        super().__init__()
        self.w = {'ce': w_ce, 'focal': w_focal, 'dice': w_dice}
        self.ce = make_ce()
        self.focal = make_focal() if w_focal else None
        self.dice = make_dice() if w_dice else None

    def forward(self, logits, targets):
        l = {'ce': self.ce(logits, targets)}
        l['focal'] = self.focal(logits, targets) if self.focal else torch.zeros((), device=logits.device)
        l['dice'] = self.dice(logits, targets) if self.dice else torch.zeros((), device=logits.device)
        loss = sum(self.w[k] * l[k] for k in l)
        return loss, {**{k: v.item() for k, v in l.items()}, **{f'w_{k}': self.w[k] for k in l}}

    def get_weights(self):
        return {f'w_{k}': v for k, v in self.w.items()}

class AdaptiveFocalDiceLoss(nn.Module):
    """Exp3: homoscedastic uncertainty weighting. Logged w_* = exp(-s_k); effective weight = 0.5*exp(-s_k)."""
    def __init__(self):
        super().__init__()
        self.focal, self.dice, self.ce = make_focal(), make_dice(), make_ce()
        self.log_sigma_focal = nn.Parameter(torch.zeros(1))
        self.log_sigma_dice = nn.Parameter(torch.zeros(1))
        self.log_sigma_ce = nn.Parameter(torch.zeros(1))

    def forward(self, logits, targets):
        l_f, l_d, l_c = self.focal(logits, targets), self.dice(logits, targets), self.ce(logits, targets)
        w_f, w_d, w_c = (torch.exp(-s) for s in (self.log_sigma_focal, self.log_sigma_dice, self.log_sigma_ce))
        loss = (0.5 * w_f * l_f + 0.5 * self.log_sigma_focal
                + 0.5 * w_d * l_d + 0.5 * self.log_sigma_dice
                + 0.5 * w_c * l_c + 0.5 * self.log_sigma_ce)
        return loss.squeeze(), {'focal': l_f.item(), 'dice': l_d.item(), 'ce': l_c.item(),
                                'w_focal': w_f.item(), 'w_dice': w_d.item(), 'w_ce': w_c.item()}

    def get_weights(self):
        return {'w_focal': torch.exp(-self.log_sigma_focal).item(),
                'w_dice': torch.exp(-self.log_sigma_dice).item(),
                'w_ce': torch.exp(-self.log_sigma_ce).item()}

class UnifiedFocalLoss(nn.Module):
    """Symmetric Unified Focal loss (Yeung et al., 2022), lam=0.5, delta=0.6, gamma=0.5, as on Forest Inspection."""
    def __init__(self, lam=0.5, delta=0.6, gamma=0.5, eps=1e-7):
        super().__init__()
        self.lam, self.delta, self.gamma, self.eps = lam, delta, gamma, eps

    def forward(self, logits, targets):
        logits = logits.float()
        valid = targets != IGNORE
        prob = F.softmax(logits, dim=1).clamp(self.eps, 1 - self.eps)
        tgt = targets.clone()
        tgt[~valid] = 0
        p_t = prob.gather(1, tgt.unsqueeze(1)).squeeze(1)[valid]
        l_focal = ((1 - p_t) ** self.gamma * -torch.log(p_t)).mean()

        onehot = F.one_hot(tgt, NUM_CLASSES).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
        p = prob * valid.unsqueeze(1)
        dims = (0, 2, 3)
        tp, fn, fp = (p * onehot).sum(dims), ((1 - p) * onehot).sum(dims), (p * (1 - onehot)).sum(dims)
        ti = (tp + self.eps) / (tp + self.delta * fn + (1 - self.delta) * fp + self.eps)
        present = onehot.sum(dims) > 0
        l_ft = ((1 - ti[present]) ** (1 - self.gamma)).mean()

        loss = self.lam * l_ft + (1 - self.lam) * l_focal
        return loss, {'focal': l_focal.item(), 'dice': l_ft.item(), 'ce': 0.0,
                      'w_focal': 1 - self.lam, 'w_dice': self.lam, 'w_ce': 0.0}

    def get_weights(self):
        return {'w_focal': 1 - self.lam, 'w_dice': self.lam, 'w_ce': 0.0}

def build_criterion(name):
    if name == 'ce':            return FixedWeightLoss(1.0, 0.0, 0.0)
    if name == 'ced':           return FixedWeightLoss(1.0, 0.0, 1.0)
    if name == 'cefd':          return FixedWeightLoss(1.0, 1.0, 1.0)
    if name == 'adaptive':      return AdaptiveFocalDiceLoss()
    if name == 'unified_focal': return UnifiedFocalLoss()
    raise ValueError(name)

def confusion(model_, loader):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for imgs, masks in loader:
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE_T, enabled=USE_AMP):
                out = model_(imgs.to(DEVICE))
            p = out.argmax(1).cpu().numpy().ravel()
            t = masks.numpy().ravel()
            keep = t != IGNORE
            cm += np.bincount(t[keep] * NUM_CLASSES + p[keep], minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return cm

def metrics_from_cm(cm):
    """Per-class IoU/recall/precision; mIoU over all 9 classes including Empty (benchmark protocol)."""
    tp = np.diag(cm).astype(float)
    gt, pred = cm.sum(1).astype(float), cm.sum(0).astype(float)
    iou = np.divide(tp, gt + pred - tp, out=np.zeros_like(tp), where=(gt + pred - tp) > 0)
    rec = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    return float(iou.mean()), iou, rec, prec
'''))

WU.append(('code', r'''# ============================================================
# TRAIN + TEST, one loss after another. Each run re-seeds, so its result does not depend on the order.
# ============================================================
import csv, json, gc

def train_one(loss_name):
    run = f'WU_{MODEL_NAME}_{loss_name}_s{SEED}' + ('_SMOKE' if SMOKE else '')
    seed_everything(SEED)
    model = build_model()
    criterion = build_criterion(loss_name).to(DEVICE)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    epochs = 2 if SMOKE else EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP and AMP_DTYPE_T == torch.float16)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn,
                              generator=torch.Generator().manual_seed(SEED))
    log_file = OUTPUT_DIR / f'{run}_training_log.csv'
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'w_focal', 'w_dice', 'w_ce', 'l_focal', 'l_dice', 'l_ce', 'seconds'])
    print(f'\n===== {run} | {type(criterion).__name__} =====')
    t_run = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train(); criterion.train()
        sums = {k: 0.0 for k in ('focal', 'dice', 'ce', 'w_focal', 'w_dice', 'w_ce')}
        total, n, skipped = 0.0, 0, 0
        for imgs, masks in tqdm(train_loader, desc=f'E{epoch}', leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE_T, enabled=USE_AMP):
                loss, comp = criterion(model(imgs), masks)
            if not torch.isfinite(loss):
                skipped += 1
                if skipped > 20:
                    raise RuntimeError(f'{skipped} non-finite losses in epoch {epoch} of {run}')
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            total += loss.item(); n += 1
            for k in sums:
                sums[k] += comp[k]
        scheduler.step()
        avg = {k: v / max(n, 1) for k, v in sums.items()}
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, total / max(n, 1), avg['w_focal'], avg['w_dice'], avg['w_ce'],
                                    avg['focal'], avg['dice'], avg['ce'], round(time.time() - t0)])
        msg = f'E{epoch:02d} | loss {total / max(n, 1):.4f} | w F/D/CE {avg["w_focal"]:.2f}/{avg["w_dice"]:.2f}/{avg["w_ce"]:.2f} | {time.time() - t0:.0f}s'
        if skipped:
            msg += f' | skipped {skipped}'
        if epoch == 1 and DEVICE == 'cuda':
            msg += ' | peak GB ' + '/'.join(f'{torch.cuda.max_memory_allocated(d) / 1e9:.1f}' for d in range(max(N_GPUS, 1)))
        if epoch % EVAL_EVERY == 0 and epoch < epochs:
            model.eval()
            msg += f' | monitor mIoU {metrics_from_cm(confusion(model, test_loader))[0]:.4f}'
        print(msg)

    model.eval(); criterion.eval()
    cm = confusion(model, test_loader)
    miou, iou, rec, prec = metrics_from_cm(cm)
    results = {
        'run_name': run, 'dataset': 'WildUAV (train.txt -> val.txt)', 'model': MODEL_NAME, 'encoder': ENCODER,
        'loss': loss_name, 'seed': SEED, 'img_size': list(IMG_SIZE), 'epochs': epochs, 'selection': 'final epoch',
        'weight_decay': WEIGHT_DECAY, 'lr': LR, 'amp_dtype': str(AMP_DTYPE_T), 'smoke': SMOKE,
        'gpu': torch.cuda.get_device_name() if DEVICE == 'cuda' else 'cpu', 'n_gpus': N_GPUS,
        'train_minutes': round((time.time() - t_run) / 60, 1),
        'class_weights': dict(zip(CLASS_NAMES, class_weights.cpu().tolist())),
        'loss_weights_final': criterion.get_weights(),
        'test': {'miou': miou,
                 'iou': dict(zip(CLASS_NAMES, iou.tolist())),
                 'recall': dict(zip(CLASS_NAMES, rec.tolist())),
                 'precision': dict(zip(CLASS_NAMES, prec.tolist())),
                 'confusion_matrix': cm.tolist()},
    }
    json.dump(results, open(OUTPUT_DIR / f'results_{run}.json', 'w'), indent=2)
    torch.save({'run_name': run, 'model_state_dict': getattr(model, 'module', model).state_dict(),
                'criterion_state_dict': criterion.state_dict()}, OUTPUT_DIR / f'{run}_final.pth')
    print(f'TEST mIoU (9 classes) {miou:.4f} | {results["train_minutes"]} min')
    print(f"{'Class':<20}{'IoU':>8}{'Recall':>8}{'Prec':>8}")
    for c, a, b, d in zip(CLASS_NAMES, iou, rec, prec):
        print(f'{c:<20}{a:8.4f}{b:8.4f}{d:8.4f}')
    del model, criterion, optimizer, train_loader
    gc.collect(); torch.cuda.empty_cache()
    return time.time() - t_run

last_duration = 0.0
for loss_name in LOSSES:
    run = f'WU_{MODEL_NAME}_{loss_name}_s{SEED}' + ('_SMOKE' if SMOKE else '')
    done = sorted(Path(DONE_DIR).rglob(f'results_{run}.json')) if DONE_DIR else []
    if done:
        for p in Path(done[0]).parent.glob(f'*{run}*'):
            if p.suffix in ('.json', '.csv'):
                shutil.copy(p, OUTPUT_DIR / p.name)
        print(f'[SKIP] {run}: already finished in {done[0].parent}')
        continue
    if (time.time() - SESSION_START) + 1.1 * last_duration > MAX_HOURS * 3600:
        print(f'[TIME LIMIT] not starting {run}. Save this version, add its output as a dataset, set DONE_DIR and run again.')
        break
    last_duration = train_one(loss_name)
print('ALL DONE:', sorted(p.name for p in OUTPUT_DIR.glob('results_*.json')))
'''))

WU.append(('code', r'''# ============================================================
# DOWNLOAD — one zip with every results JSON and training log of this notebook
# ============================================================
import zipfile
WORK_DIR = OUTPUT_DIR.parent
zip_path = WORK_DIR / f'{NOTEBOOK}_results.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in OUTPUT_DIR.glob('*'):
        if p.suffix in ('.json', '.csv'):
            z.write(p, arcname=p.name)
print(f'{zip_path.name}: {zip_path.stat().st_size / 1e3:.0f} kB (the *_final.pth files stay in outputs/)')
os.chdir(WORK_DIR)
try:
    from IPython.display import FileLink, display
    display(FileLink(zip_path.name))
except ImportError:
    print(zip_path)
'''))

# ================================================================ operating-point template
OP = [('code', FOREST_CELLS[0][1])]

OP.append(('code', r'''# ============================================================
# CONFIG — operating-point study (inference only, no training)
# ============================================================
import numpy as np

# best.pth of finished UNet++ seed-123 runs (add their outputs as input datasets). Leave '' to skip a model.
# Old-format checkpoints (plain state_dict, with or without a 'module.' prefix) are accepted too.
CHECKPOINTS = {
    'Exp 1: CE':          '',   # old UNet++ Exp1 seed-123 best.pth (the one used for the qualitative figure)
    'Exp 2-CED: CE+Dice': '',   # old UNet++ CE+Dice seed-123 best.pth, if it still exists
    'Exp 3: Uncertainty': '',   # R01_unetpp_adaptive_s123 .../best.pth
    'Unified Focal':      '',   # R11_unetpp_unified_focal_s123 .../best.pth
    'Exp 2-CEFD: 1:1:1':  '',   # R07_unetpp_cefd_s123 .../best.pth (when finished)
}
REFERENCE = 'Exp 1: CE'      # its validation recall per class is the recall target
FALLBACK_TARGET_RECALL = 0.90 # used when REFERENCE is not given

TARGET_CLASSES = [3, 8, 9, 7]                             # Fallen_trees, Fence, Car, Building
BIAS_GRID = np.round(np.arange(-2.0, 6.0001, 0.25), 2)    # additive bias on the logit of one class
VAL_STRIDE = 1               # use every k-th validation frame (2 halves the validation time)
BATCH_SIZE = 4
NUM_WORKERS = 4
SMOKE = False                # True: 8 validation and 8 test frames, to check the notebook

MODEL_NAME, ENCODER, IMG_SIZE = 'unetpp', 'resnet50', (768, 1280)
print('[CONFIG] models:', [k for k, v in CHECKPOINTS.items() if v], '| grid', BIAS_GRID[0], '...', BIAS_GRID[-1])'''))

OP.append(('code', FOREST_CELLS[2][1]))   # Forest Inspection split (same as training)
OP.append(('code', FOREST_CELLS[3][1]))   # label colours -> class ids

OP.append(('code', r'''import torch, json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = Path('/kaggle/working/outputs') if os.path.exists('/kaggle') else Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ForestDataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list, self.transform = file_list, transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_list[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = fast_rgb_to_mask(np.array(Image.open(mask_path).convert('RGB')))
        aug = self.transform(image=img, mask=mask)
        return aug['image'], aug['mask'].long()

val_tf = A.Compose([A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
val_list, test_list = val_files[::VAL_STRIDE], test_files
if SMOKE:
    val_list, test_list = val_list[:8], test_list[:8]
make_loader = lambda files: DataLoader(ForestDataset(files, val_tf), batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=NUM_WORKERS, pin_memory=True)

def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck.get('state_dict', ck)) if isinstance(ck, dict) else ck
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    m = smp.UnetPlusPlus(encoder_name=ENCODER, encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()

def logits_of(model, imgs):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEVICE == 'cuda'):
        z = model(imgs.to(DEVICE))
    return z.float().permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)

def sweep_and_eval(model, loader, bias_vectors=()):
    """(a) per-class sweep: confusion matrix for every (target class, bias) with only that class biased;
       (b) confusion matrix for each full bias vector in bias_vectors. One forward pass per frame."""
    K, G = NUM_CLASSES, len(BIAS_GRID)
    grid = torch.tensor(BIAS_GRID, dtype=torch.float32, device=DEVICE)
    cms = torch.zeros(len(TARGET_CLASSES), G, K * K, dtype=torch.int64, device=DEVICE)
    bvs = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in bias_vectors]
    cms_bv = torch.zeros(len(bvs), K * K, dtype=torch.int64, device=DEVICE)
    for imgs, masks in tqdm(loader, leave=False):
        z = logits_of(model, imgs)
        t = masks.to(DEVICE).reshape(-1) * K
        for i, c in enumerate(TARGET_CLASSES):
            other = z.clone(); other[:, c] = -float('inf')
            m_other, a_other = other.max(1)
            margin = z[:, c] - m_other                     # class c wins iff margin + bias > 0
            for j in range(G):
                p = torch.where(margin + grid[j] > 0, c, a_other)
                cms[i, j] += torch.bincount(t + p, minlength=K * K)
        for i, b in enumerate(bvs):
            cms_bv[i] += torch.bincount(t + (z + b).argmax(1), minlength=K * K)
    return (cms.view(len(TARGET_CLASSES), G, K, K).cpu().numpy(),
            cms_bv.view(len(bvs), K, K).cpu().numpy())

def metrics_from_cm(cm):
    """Per-class IoU/recall/precision for the 10 foreground classes; mIoU = their mean (as in the paper)."""
    tp = np.diag(cm).astype(float)
    gt, pred = cm.sum(1).astype(float), cm.sum(0).astype(float)
    iou = np.divide(tp, gt + pred - tp, out=np.zeros_like(tp), where=(gt + pred - tp) > 0)
    rec = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    fg = slice(0, NUM_CLASSES - 1)
    return float(iou[fg].mean()), iou, rec, prec
'''))

OP.append(('code', r'''# ============================================================
# RUN — biases are chosen on the VALIDATION split only, then applied once to the test split
#   rule 'recall': smallest bias whose validation recall reaches the target (REFERENCE recall)
#   rule 'iou'   : bias with the highest validation IoU of that class
# ============================================================
models = {k: v for k, v in CHECKPOINTS.items() if v}
assert models, 'Set at least one path in CHECKPOINTS'
order = ([REFERENCE] if REFERENCE in models else []) + [k for k in models if k != REFERENCE]
target_recall, results = {}, {}

for name in order:
    print(f'\n===== {name} =====')
    model = load_model(models[name])
    val_cms, _ = sweep_and_eval(model, make_loader(val_list))
    zero = int(np.argmin(np.abs(BIAS_GRID)))
    if name == REFERENCE:
        target_recall = {c: metrics_from_cm(val_cms[i, zero])[2][c] for i, c in enumerate(TARGET_CLASSES)}
        print('target recall (validation, reference):', {CLASS_NAMES[c]: round(r, 4) for c, r in target_recall.items()})
    chosen = {'none': np.zeros(NUM_CLASSES), 'recall': np.zeros(NUM_CLASSES), 'iou': np.zeros(NUM_CLASSES)}
    for i, c in enumerate(TARGET_CLASSES):
        rec_v = np.array([metrics_from_cm(val_cms[i, j])[2][c] for j in range(len(BIAS_GRID))])
        iou_v = np.array([metrics_from_cm(val_cms[i, j])[1][c] for j in range(len(BIAS_GRID))])
        goal = target_recall.get(c, FALLBACK_TARGET_RECALL)
        ok = np.where(rec_v >= goal)[0]
        chosen['recall'][c] = BIAS_GRID[ok[0]] if len(ok) else BIAS_GRID[-1]
        chosen['iou'][c] = BIAS_GRID[int(np.argmax(iou_v))]
    print('biases (recall rule):', {CLASS_NAMES[c]: chosen['recall'][c] for c in TARGET_CLASSES})
    print('biases (IoU rule):   ', {CLASS_NAMES[c]: chosen['iou'][c] for c in TARGET_CLASSES})

    test_cms, test_bv = sweep_and_eval(model, make_loader(test_list), [chosen[r] for r in chosen])
    np.savez_compressed(OUTPUT_DIR / f'sweep_{name.split(":")[0].replace(" ", "")}.npz',
                        val=val_cms, test=test_cms, grid=BIAS_GRID, classes=TARGET_CLASSES)
    results[name] = {}
    for (rule, b), cm in zip(chosen.items(), test_bv):
        miou, iou, rec, prec = metrics_from_cm(cm)
        results[name][rule] = {'bias': {CLASS_NAMES[c]: float(b[c]) for c in TARGET_CLASSES}, 'miou': miou,
                               **{CLASS_NAMES[c]: {'iou': iou[c], 'recall': rec[c], 'precision': prec[c]}
                                  for c in TARGET_CLASSES}}
    del model; torch.cuda.empty_cache()

json.dump({'target_recall_val': {CLASS_NAMES[c]: r for c, r in target_recall.items()}, 'results': results},
          open(OUTPUT_DIR / 'operating_point_results.json', 'w'), indent=2, default=float)

print(f"\n{'Model':<22}{'Rule':<8}{'mIoU':>7}" + ''.join(f'{CLASS_NAMES[c][:7]:>22}' for c in TARGET_CLASSES))
print(' ' * 37 + '      IoU / Rec / Prec' * len(TARGET_CLASSES))
for name, rules in results.items():
    for rule, r in rules.items():
        cells = ''.join(f"{100 * r[CLASS_NAMES[c]]['iou']:8.2f}{100 * r[CLASS_NAMES[c]]['recall']:7.2f}"
                        f"{100 * r[CLASS_NAMES[c]]['precision']:7.2f}" for c in TARGET_CLASSES)
        print(f'{name:<22}{rule:<8}{100 * r["miou"]:7.2f}{cells}')
print("\nCheck: the 'none' rows must match the test results already in the paper (differences <0.05 pp).")
'''))

OP.append(('code', r'''# Precision-recall curves on the test split (bias sweep of one class at a time), for the figure.
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(TARGET_CLASSES), figsize=(5 * len(TARGET_CLASSES), 4.5))
zero = int(np.argmin(np.abs(BIAS_GRID)))
for name in results:
    d = np.load(OUTPUT_DIR / f'sweep_{name.split(":")[0].replace(" ", "")}.npz')
    for ax, (i, c) in zip(axes, enumerate(TARGET_CLASSES)):
        m = [metrics_from_cm(d['test'][i, j]) for j in range(len(BIAS_GRID))]
        rec, prec = [100 * x[2][c] for x in m], [100 * x[3][c] for x in m]
        line, = ax.plot(rec, prec, label=name)
        ax.plot(rec[zero], prec[zero], 'o', color=line.get_color())
for ax, c in zip(axes, TARGET_CLASSES):
    ax.set_title(CLASS_NAMES[c].replace('_', ' ')); ax.set_xlabel('Recall (%)'); ax.set_ylabel('Precision (%)')
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle('Test precision vs. recall when the logit of one class is shifted (dot: no shift)')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_operating_point.png', dpi=200, bbox_inches='tight')
plt.show()

import zipfile
zip_path = OUTPUT_DIR.parent / 'OP1_results.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in OUTPUT_DIR.glob('*'):
        z.write(p, arcname=p.name)
os.chdir(OUTPUT_DIR.parent)
try:
    from IPython.display import FileLink, display
    display(FileLink(zip_path.name))
except ImportError:
    print(zip_path)
'''))


# ================================================================ build
def write(nb_name, cells, title):
    d = ROOT / 'week4'
    d.mkdir(parents=True, exist_ok=True)
    nb = nb_json([('markdown', title)] + cells)
    (d / f'{nb_name}.ipynb').write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('wrote week4', nb_name)


def main():
    for nb_name, losses, seed in WILDUAV_NOTEBOOKS:
        cfg = dict(nb_name=nb_name, losses=repr(losses), seed=seed, resnet_dir=RESNET_DIR)
        cells = [(k, s.format(**cfg) if i == 1 else s) for i, (k, s) in enumerate(WU)]
        title = (f'# {nb_name}\n\nWildUAV loss ablation, UNet++/ResNet50, seed **{seed}**, losses **{", ".join(losses)}** '
                 '(trained one after another, 50 epochs each, final model evaluated on `val.txt`).\n\n'
                 '1. First run with `SMOKE = True` (a few minutes) and check that every loss prints a TEST table.\n'
                 '2. Set `SMOKE = False`, then Save Version -> Save & Run All.\n'
                 '3. If it stops with `[TIME LIMIT]`: add the output as a dataset, set `DONE_DIR`, run again '
                 '(finished losses are skipped).\n'
                 f'4. Download `{nb_name}_results.zip`.\n')
        write(nb_name, cells, title)
    write('OP1_operating_point', OP,
          '# OP1_operating_point\n\nInference only: shifts the logit of each rare class, chooses the shift on the '
          'validation split, reports it once on the test split.\n\n'
          '1. Add the outputs that contain the `best.pth` files as input datasets; put the paths in `CHECKPOINTS`.\n'
          '2. Run once with `SMOKE = True`, then with `SMOKE = False`.\n'
          '3. Download `OP1_results.zip`.\n')


if __name__ == '__main__':
    main()
