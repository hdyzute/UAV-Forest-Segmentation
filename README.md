# Revisiting Compound Losses for Class-Imbalanced UAV Forest Segmentation

Code, run notebooks, and results for the manuscript
*Revisiting Compound Losses for Class-Imbalanced UAV Forest Segmentation: A Multi-Backbone Study of Dice, Focal, and Learned Weighting*
(H.-D. Pham, H.-H. Pham Le, T.-S. Huynh; submitted to *Image and Vision Computing*).

The study separates ENet-weighted cross-entropy, Dice loss, focal loss, and their combination with static or
uncertainty-learned weights on six segmentation architectures (Forest Inspection benchmark), compares five
alternative losses and loss-balancing methods on UNet++, repeats the core ablation on the real WildUAV benchmark,
and studies the operating point of the trained models with a post-hoc logit bias.

## Repository layout

| Path | Content |
|---|---|
| `tools/build_notebooks.py` | Template of every Forest Inspection run (data split, augmentation, all losses, training loop, test, error analysis). Regenerates `week1/`–`week3/`, `week5/`. |
| `tools/build_week4_notebooks.py` | Template of the WildUAV runs and of the operating-point study (`week4/`). |
| `tools/build_qualitative_notebook.py` | Regenerates `figures/make_qualitative.ipynb` (qualitative figure). |
| `tools/smoke_test_losses.py`, `tools/smoke_test_pipeline.py` | CPU tests of the losses and of train → resume → test. |
| `tools/make_tables_figures.py` | Tables and Figs. 3-5 of the paper from `results/` (PNG and vector PDF; the paper uses `--seeds 123`). |
| `tools/make_class_dist_figure.py` | Class-frequency figure (Fig. 1). |
| `tools/make_op_figure.py` | Precision–recall figure of the operating-point study from `results_operating_point/`. |
| `tools/parse_*.py`, `tools/csv_to_results.py` | Rebuild `results_<RUN>.json` from notebook outputs or logs. |
| `week1/` … `week5/` | Run notebooks R01–R25 (Forest Inspection), WU1–WU6 (WildUAV), OP1 (operating point). They differ only in the CONFIG cell. |
| `executed_runs/` | The same notebooks as executed on Kaggle, with outputs (for runs whose notebook was kept). |
| `legacy_seed123/` | Executed notebooks of the seed-123 runs that were not re-trained: Exp 1 (CE), Exp 2-CED, Exp 2-CEF on the six architectures. |
| `results/`, `results_wilduav/`, `results_operating_point/` | Test metrics (mIoU, per-class IoU, recall, precision) of every run, training logs, and the bias sweeps of OP1. |

## Runs

Forest Inspection, all with weight decay 1e-4 and the protocol of the paper (Section 4.3):

| Runs | Model | Loss | Seed |
|---|---|---|---|
| R01–R06 | UNet++, MA-Net, UPerNet, SegFormer, DeepLabV3+, UNet | Exp 3, uncertainty-weighted | 123 |
| R07, R08 | UNet++, MA-Net | Exp 2-CEFD, static 1:1:1 | 123 |
| R09–R12, R25 | UNet++ | CE + Lovász-Softmax, CE + Focal-Tversky, Unified Focal, DWA, FAMO | 123 |
| R13–R16 | UNet++ | CE, CE + Dice, 1:1:1, Exp 3 | 42 |
| R17–R20 | UNet++ | CE, CE + Dice, 1:1:1, Exp 3 | 2026 |
| R21–R24 | MA-Net | 1:1:1, Exp 3 | 42, 2026 |
| `legacy_seed123/` | all six | CE, CE + Dice, CE + Focal | 123 |

WildUAV (UNet++, 50 epochs, final model on `val.txt`): WU1–WU3 run CE, CE + Dice, 1:1:1, and Exp 3 with seeds 123, 42,
2026; WU4–WU6 run the Unified Focal loss with the same seeds. OP1 sweeps a per-class logit bias on the seed-2026
UNet++ models R17–R20.

## Reproducing

The runs were made on Kaggle (Python 3.13, PyTorch 2.11 + CUDA 12.8, segmentation-models-pytorch 0.5.0,
timm 1.0.29, albumentations 2.0.8). Each run takes about 20 GPU hours and resumes from `last.pth` after the
Kaggle time limit (see the RESUME cell).

1. Download the Forest Inspection dataset (sunny: https://doi.org/10.5281/zenodo.15511426,
   overcast: https://doi.org/10.5281/zenodo.17912570) and WildUAV (https://github.com/hrflr/wuav).
2. Open a notebook of `week1/`–`week5/`, set the dataset paths in the data cell, and run all cells.
   The test cell writes `outputs/results_<RUN>.json`.
3. Copy the JSON files into `results/` and run `python tools/make_tables_figures.py`.

Check the losses without a GPU: `python tools/smoke_test_losses.py`.

## Citation

See `CITATION.cff`. Please cite the Forest Inspection dataset (Blaga and Nedevschi, *Scientific Data*, 2026) and
WildUAV (Florea et al., ICCP 2021) when using the data.

## License

MIT (see `LICENSE`). The datasets are distributed by their authors under their own licenses.
