"""CPU smoke test for the loss cell of the generated notebooks.

Runs every criterion on random data, checks that losses are finite, that gradients flow,
that the static losses reproduce the original notebook formulas, and that state_dict
round-trips (needed for resume)."""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

nb = json.loads(sorted(Path(__file__).resolve().parent.parent.glob('week*/R*_*.ipynb'))[-1].read_text(encoding='utf-8'))
code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
loss_cell = next(c for c in code if 'def build_criterion' in c)
loss_cell = loss_cell.split('criterion = build_criterion')[0]   # definitions only

g = {'torch': torch, 'nn': nn, 'smp': smp, 'NUM_CLASSES': 11, 'EMPTY_CLASS': 10, 'DEVICE': 'cpu',
     'USE_AMP': False, 'AMP_DTYPE_T': torch.float16,
     'class_weights': torch.tensor([0.2260, 0.2128, 0.1355, 1.4142, 0.9162, 0.1129,
                                    0.9421, 1.6558, 1.8006, 1.7720, 1.8119])}
exec(loss_cell, g)

torch.manual_seed(0)
logits = torch.randn(2, 11, 64, 96, requires_grad=True)
targets = torch.randint(0, 10, (2, 64, 96))
targets[:, :4, :] = 10            # some Empty pixels at the border, as produced by Affine fill
targets[:, 10:12, 10:12] = 8      # tiny Fence region

ce = nn.CrossEntropyLoss(weight=g['class_weights'], ignore_index=10)
focal = smp.losses.FocalLoss(mode='multiclass', gamma=2.0)
dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True, classes=list(range(10)))
ref = {'ce': ce(logits, targets), 'ced': ce(logits, targets) + dice(logits, targets),
       'cef': ce(logits, targets) + focal(logits, targets),
       'cefd': ce(logits, targets) + focal(logits, targets) + dice(logits, targets)}

ok = True
for name in ['ce', 'ced', 'cef', 'cefd', 'adaptive', 'lovasz', 'focal_tversky', 'unified_focal', 'dwa', 'famo']:
    crit = g['build_criterion'](name)
    loss, comp = crit(logits, targets)
    logits.grad = None
    loss.backward()
    finite = torch.isfinite(loss).item() and torch.isfinite(logits.grad).all().item()
    msg = f'{name:14s} loss={loss.item():.4f} finite={finite} keys_ok={set(comp) >= {"focal", "dice", "ce", "w_focal", "w_dice", "w_ce"}}'
    if name in ref:
        same = torch.allclose(loss, ref[name])
        msg += f' matches_original={same}'
        ok &= same
    if name == 'adaptive':
        # at s_k = 0 the adaptive loss equals 0.5 * (focal + dice + ce)
        same = torch.allclose(loss, 0.5 * ref['cefd'])
        msg += f' equals_0.5*cefd_at_init={same}'
        ok &= same
    if name == 'dwa':
        for avg in [{'ce': 1.0, 'focal': 0.2, 'dice': 0.5}, {'ce': 0.8, 'focal': 0.19, 'dice': 0.3}]:
            crit.end_epoch(avg)
        lam = crit.lam.tolist()
        sd = crit.state_dict()
        crit2 = g['build_criterion']('dwa'); crit2.load_state_dict(sd)
        same = torch.allclose(crit2.lam, crit.lam) and int(crit2.n_epochs) == 2
        msg += f' lam_after_2_epochs={[round(x, 3) for x in lam]} sum={sum(lam):.3f} resume_ok={same}'
        ok &= same and abs(sum(lam) - 3) < 1e-5
    if name == 'famo':
        # at xi = 0 the gradient equals that of mean(L_k / 3 / ...) -> effective weights sum to 1
        wsum = comp['w_ce'] + comp['w_focal'] + comp['w_dice']
        crit.train(); crit(logits, targets)
        crit.after_step(lambda x: x, logits.detach(), targets)          # identity "model": no decrease -> only decay
        lin = nn.Conv2d(11, 11, 1); lin.weight.data = torch.eye(11).view(11, 11, 1, 1) * 1.5; lin.bias.data.zero_()
        crit(logits, targets); crit.after_step(lambda x: lin(x), logits.detach(), targets)
        sd = crit.state_dict(); crit2 = g['build_criterion']('famo'); crit2.load_state_dict(sd)
        same = torch.equal(crit2.xi, crit.xi) and int(crit2.adam_t) == 2 and torch.equal(crit2.adam_v, crit.adam_v)
        msg += f' w_sum={wsum:.4f} xi_after_2_steps={[round(x, 4) for x in crit.xi.tolist()]} resume_ok={same}'
        ok &= same and abs(wsum - 1) < 1e-4 and torch.isfinite(crit.xi).all().item()
    if name == 'adaptive':
        sd = crit.state_dict(); crit2 = g['build_criterion']('adaptive'); crit2.load_state_dict(sd)
        ok &= {'log_sigma_focal', 'log_sigma_dice', 'log_sigma_ce'} <= set(sd) and torch.equal(crit2.log_sigma_dice, crit.log_sigma_dice)
    ok &= finite
    print(msg)

print('ALL OK' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
