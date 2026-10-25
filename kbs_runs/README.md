# KBS re-runs — checklist

All notebooks come from one template (`tools/build_notebooks.py`). They differ **only in the CONFIG cell**.
Shared settings: AdamW, lr 1e-4, **weight decay 1e-4**, cosine schedule to 1e-6, 50 epochs, batch 4,
768×1280, ENet weights fixed (no 2 h `compute_class_weights`), `NUM_WORKERS = 32`.
Per-backbone settings are copied from the original seed-123 notebooks:
MA-Net runs **without AMP**; DeepLabV3+ and UNet++ CE+Dice use patience 15, all others 10.

## How to run one notebook on Kaggle
1. Upload the `.ipynb`, attach the same input datasets as before (data, offline packages, encoder weights), GPU on.
2. **Save Version → Save & Run All**. Each run needs about 20 h = 2 sessions.
3. Session 1 stops by itself after ~11 h with `[TIME LIMIT]` (`last.pth` is saved).
   Add that version's output as an input dataset, put its folder in `RESUME_DIR` (CONFIG cell), Save & Run All again.
   In an interactive session, the last cell (DOWNLOAD) shows download links for `<RUN>_logs.zip`,
   `<RUN>_best.pth` and `<RUN>_last.pth`; upload these as a dataset to resume (the RUN_NAME prefix is accepted).
4. When `TRAINING_FINISHED = True`, the test cell writes `outputs/results_<RUN>.json`.
   Download into `results/` here: `results_<RUN>.json`, `training_log.csv` (rename to `<RUN>_training_log.csv`),
   and keep `checkpoints/best.pth` on Kaggle (needed for the qualitative figure).
   The last cell also saves `small_object_error_analysis_<RUN>.png` (row-normalized confusion matrix with the
   small classes outlined, IoU / precision / miss rate of the small classes, and the top confusions printed)
   and `sample_predictions_<RUN>.png` (the same 3 test frames in every run, so runs can be compared side by side).

## Schedule (8 runs ≈ 160 GPU-h per week; 6 accounts × 30 h = 180 h)

| Week | Run | Model | Loss | Seed | Done |
|---|---|---|---|---|---|
| 1 | R01 | UNet++ | Exp3 adaptive | 123 | [ ] |
| 1 | R02 | MA-Net | Exp3 adaptive | 123 | [ ] |
| 1 | R03 | UPerNet | Exp3 adaptive | 123 | [ ] |
| 1 | R04 | SegFormer | Exp3 adaptive | 123 | [ ] |
| 1 | R05 | DeepLabV3+ | Exp3 adaptive | 123 | [ ] |
| 1 | R06 | UNet | Exp3 adaptive | 123 | [ ] |
| 1 | R07 | UNet++ | 1:1:1 (cefd) | 123 | [ ] |
| 1 | R08 | MA-Net | 1:1:1 (cefd) | 123 | [ ] |
| 2 | R09 | UNet++ | CE + Lovász | 123 | [ ] |
| 2 | R10 | UNet++ | CE + Focal-Tversky | 123 | [ ] |
| 2 | R11 | UNet++ | Unified Focal | 123 | [ ] |
| 2 | R12 | UNet++ | DWA (CE, Focal, Dice) | 123 | [ ] |
| 2 | R13 | UNet++ | CE | 42 | [ ] |
| 2 | R14 | UNet++ | CE + Dice | 42 | [ ] |
| 2 | R15 | UNet++ | 1:1:1 | 42 | [ ] |
| 2 | R16 | UNet++ | Exp3 adaptive | 42 | [ ] |
| 3 | R17 | UNet++ | CE | 2026 | [ ] |
| 3 | R18 | UNet++ | CE + Dice | 2026 | [ ] |
| 3 | R19 | UNet++ | 1:1:1 | 2026 | [ ] |
| 3 | R20 | UNet++ | Exp3 adaptive | 2026 | [ ] |
| 3 | R21 | MA-Net | 1:1:1 | 42 | [ ] |
| 3 | R22 | MA-Net | Exp3 adaptive | 42 | [ ] |
| 3 | R23 | MA-Net | 1:1:1 | 2026 | [ ] |
| 3 | R24 | MA-Net | Exp3 adaptive | 2026 | [ ] |

**Decision point after week 1:** compare R01 vs R07 and R02 vs R08 (adaptive vs 1:1:1) → choose the title/framing.
Not re-trained (already weight decay 1e-4): Exp1, CE+Dice, CE+Focal for all six backbones, seed 123
(stored in `tools/legacy_results.json`). Exp2-CED (CE + Dice, 1:1) is Combo loss with α = 0.5, so no separate Combo run.

## After the runs
```
python tools/make_tables_figures.py            # reads results/, writes out/
```
Produces `out/table_ablation.tex` (mean ± s.d. when several seeds exist), `table_baselines.tex`,
`table_per_class.tex`, `table_small_objects.tex`, and Figures 3–5 (`fig_loss_weights_convergence.png`,
`fig_ablation_dotplot.png`, `fig_small_object_tradeoff.png`). It also prints which seeds exist per configuration.

Qualitative figure: run `figures/make_qualitative.ipynb` on Kaggle with the `best.pth` of UNet++ Exp1 (old, seed 123)
and R01; set frames and crop positions in its CONFIG cell.

## Tools
| File | Purpose |
|---|---|
| `tools/build_notebooks.py` | regenerates all run notebooks (edit `RUNS` to add runs) |
| `tools/smoke_test_losses.py` | CPU check of every loss (original formulas reproduced, gradients finite, resume state) |
| `tools/smoke_test_pipeline.py` | CPU end-to-end test: train → time-limit stop → resume → test → JSON |
| `tools/make_tables_figures.py` | tables + Figures 3–5 from `results/` |
| `tools/build_legacy_results.py`, `tools/add_legacy_per_class.py` | build `legacy_results.json` from the manuscript and the old notebooks |
| `tools/build_qualitative_notebook.py` | regenerates `figures/make_qualitative.ipynb` |
