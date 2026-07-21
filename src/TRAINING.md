# RGC Training Guide

This document describes how to run **BYOL self-supervised pretraining** and **downstream fine-tuning** for the RGC radio AGN classifier.

For project background, see the main [README.md](../README.md).

---

## Overview

Training is a two-stage pipeline:

1. **Pretrain** — BYOL self-supervised learning on unlabeled radio images (`scripts/byol_pretrain.py`)
2. **Fine-tune** — Supervised classification with a head on top of the pretrained encoder (`scripts/downstream_trainer.py`)

Both stages share a single config file: `config.yaml` (in this directory).

```
Unlabeled images  →  BYOL pretrain  →  best.pt (encoder weights)
                                              ↓
Labeled ImageFolder  →  downstream finetune  →  checkpoints + metrics
```

---

## 1. Environment setup

### Dependencies

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key packages: PyTorch, PyTorch Lightning, `e2cnn`, `albumentations`, `wandb`, `pyyaml`.

### GPU

Both scripts default to GPU training (`cuda` / `gpu`). A CUDA-capable GPU is strongly recommended.

### Weights & Biases

- **Pretrain:** optional — set `logging_params.wandb.enabled: false` in `config.yaml` to disable.
- **Fine-tune:** always calls `wandb.login()`. Log in first:

```bash
wandb login
```

### Project root (`RGC_PROJECT_ROOT`)

Relative paths in `config.yaml` resolve from the current working directory, or from `RGC_PROJECT_ROOT` if set:

```bash
export RGC_PROJECT_ROOT=/path/to/your/project/root
```

Torch and W&B caches are written under `$RGC_PROJECT_ROOT/.torch` and `$RGC_PROJECT_ROOT/wandb_runs`.

---

## 2. Data preparation

### Pretraining data (unlabeled)

- **Layout:** any directory tree of image files (searched recursively).
- **Formats:** `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`
- **Content:** single-channel grayscale radio cutouts; images are resized to 151×151 during loading.

Example layout:

```
20k_images/rgz20k_png_raw/
├── image_001.png
├── image_002.png
└── subfolder/
    └── image_003.png
```

Set the path in `config.yaml`:

```yaml
data_params:
  data_path: ./20k_images/rgz20k_png_raw/
```

### Fine-tuning data (labeled)

- **Layout:** [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) — one subfolder per class:

```
datasets/mian_dataset/no_spurious_masked_png/
├── nat/
│   ├── nat_L0001_....png
│   └── ...
├── wat/
├── bent/
└── ...
```

Class folder names become label names in metrics and W&B reports.

To build PNGs from FITS sources, use the preprocessing notebooks:

- `preprocessing/process_single_file.ipynb` — one source file
- `preprocessing/process_bulk.ipynb` — batch processing

Catalog references live in `catalogs/` (e.g. `first-2060.csv` with columns `idx`, `radioname`, `ra`, `dec`, `Label`).

Set the path in `config.yaml`:

```yaml
finetune:
  data_dir: "./datasets/mian_dataset/no_spurious_masked_png"
```

---

## 3. Configuration (`config.yaml`)

One file controls both stages.

### Pretrain sections

| Section | Purpose |
|---------|---------|
| `data_params` | Unlabeled image directory |
| `exp_params` | LR, batch size, epochs, device, BYOL projection sizes |
| `model_params` | `image_size` (151), `kernel_size` (5) |
| `logging_params` | Output dir and W&B settings |

### Fine-tune section

| Key | Purpose |
|-----|---------|
| `data_dir` | Labeled ImageFolder root |
| `byol_checkpoint` | Path to pretrain `best.pt` |
| `experiment_root` | Where finetune outputs are saved |
| `learning_rate`, `weight_decay`, `batch_size`, `max_epochs` | Training hyperparameters |
| `freeze_encoder` | If `true`, only the classifier head is trained |
| `k_fold` | Cross-validation folds (`5` default); use `1` for a single train/test split |
| `test_size`, `split_seed` | Used when `k_fold: 1` |
| `accelerator`, `devices`, `precision` | PyTorch Lightning settings |
| `wandb_entity`, `wandb_project`, `model_slug` | W&B run naming |

**Important:** After pretraining, point `finetune.byol_checkpoint` at the checkpoint that was actually written. Pretrain saves under:

```
{results_dir}/models/byol/run_{N}/best.pt
```

(default `results_dir` is `./results/`). Update the path in config before fine-tuning.

---

## 4. Stage 1 — BYOL pretraining

### Run

Run from `src/` (the script puts `src/` on `sys.path`, so `models`/`utils` import correctly):

```bash
cd src
python scripts/byol_pretrain.py --config config.yaml
```

With a custom project root:

```bash
cd src
RGC_PROJECT_ROOT=/path/to/project python scripts/byol_pretrain.py --config config.yaml
```

Custom config path:

```bash
python scripts/byol_pretrain.py --config /path/to/my_pretrain_config.yaml
```

### What it does

- Builds a `DSteerableLeNet` encoder wrapped in BYOL (`models/byol_trainer.py`).
- Trains with two augmented views per image (color jitter, flips, rotation, crop, normalize).
- Saves checkpoints each epoch; keeps the best by lowest epoch loss.

### Outputs

```
results/
└── models/
    └── byol/
        └── run_0/          # auto-increments: run_1, run_2, ...
            ├── best.pt     # use this for fine-tuning
            └── last.pt
```

Each checkpoint contains:

- `model_state_dict` — encoder weights (this is what fine-tuning loads)
- `optimizer_state_dict`, `epoch`, `loss`

### Disable W&B for pretrain

In `config.yaml`:

```yaml
logging_params:
  wandb:
    enabled: false
```

---

## 5. Stage 2 — Downstream fine-tuning

### Run

From `src/`:

```bash
cd src
python scripts/downstream_trainer.py --config config.yaml
```

Ensure `finetune.byol_checkpoint` in config points to your pretrain `best.pt`. If the file is missing, training continues from a randomly initialized encoder (a warning is printed).

### What it does

- Loads the BYOL encoder from `best.pt`.
- Adds a 3-layer MLP classifier head.
- Splits data with **stratified k-fold** (`k_fold: 5` by default) or a single stratified train/test split when `k_fold: 1`.
- Trains with PyTorch Lightning; logs metrics, confusion matrix, and classification report to W&B.
- Runs test evaluation after each fold.

### Outputs

Under `finetune.experiment_root` (default `./finetune_runs/no_mask_byol+mask_data`):

```
finetune_runs/<experiment_name>/
├── checkpoints_byol/           # Lightning checkpoints per fold
├── saved_test_outputs/         # Per-fold test predictions (.pt)
└── kfold_logs/
    └── kfold_<dataset>_<slug>_<timestamp>.jsonl   # fold + averaged metrics
```

Each `*_test_outputs.pt` file contains `logits`, `probs`, `y_true`, `y_pred`, `class_names`, and `test_paths`.

At the end, a k-fold summary is printed with average accuracy and the JSONL log path.

---

## 6. End-to-end example

```bash
# 1. Setup
cd /path/to/rgc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login

# 2. Edit src/config.yaml:
#    - data_params.data_path  → your unlabeled images
#    - finetune.data_dir      → your labeled ImageFolder
#    - finetune.byol_checkpoint → will set after step 3

# 3. Pretrain
cd src
python scripts/byol_pretrain.py --config config.yaml
# Note the run directory, e.g. ./results/models/byol/run_0/best.pt

# 4. Update config.yaml:
#    finetune:
#      byol_checkpoint: "./results/models/byol/run_0/best.pt"

# 5. Fine-tune
python scripts/downstream_trainer.py --config config.yaml
```

---

## 7. Inference after training

- **Single image:** `notebooks/inference_single_image.ipynb`
- **Grad-CAM / attention:** `notebooks/gradcam_maps_byol.ipynb`, `notebooks/gradcam_maps_supervised.ipynb`, `notebooks/fig6_figD2_gradcam_metrics.ipynb`

Load the encoder from a BYOL checkpoint or a full Lightning checkpoint from the finetune run.

---

## 8. Troubleshooting

| Issue | Fix |
|-------|-----|
| `No images found under data_path=...` | Check `data_params.data_path`; images must match supported extensions |
| `finetune.data_dir does not exist` | Create the ImageFolder layout or fix the path in config |
| `WARNING: BYOL checkpoint not found` | Set `finetune.byol_checkpoint` to the correct `best.pt` path |
| Import errors (`models` not found) | Run the entry points from `src/` (`python scripts/...`); they add `src/` to `sys.path`, but the config's relative paths resolve from there |
| W&B 404 / project errors | Set `wandb_entity` to your W&B username/team, or `null` for default; `/` in project names is converted to `-` |
| CUDA OOM | Lower `batch_size` in `exp_params` or `finetune` |
| Slow dataloading | Adjust `num_workers`; ensure images are on local/fast storage |

---

## 9. Default hyperparameters (from `config.yaml`)

**Pretrain:** 500 epochs, batch 16, LR `3e-4`, image 151×151, projection 256 / hidden 4096, moving average decay 0.99.

**Fine-tune:** 500 epochs, batch 8, LR ≈ `8.5e-4`, weight decay ≈ `2.1e-5`, 5-fold CV, encoder not frozen (`freeze_encoder: false`).

Tune these in `config.yaml` for your hardware and dataset size.

---

## Related files

| File | Role |
|------|------|
| `scripts/byol_pretrain.py` | Pretrain entry point |
| `scripts/downstream_trainer.py` | Fine-tune entry point (runner) |
| `models/byol_trainer.py` | BYOL training loop |
| `models/downstream_trainer.py` | `BYOLDownstreamClassifier` (LightningModule) |
| `models/dsteerablelenet.py` | Steerable CNN encoder |
| `utils/unlabeled_dataset.py` | Unlabeled image dataset |
| `utils/dataloader.py` | Labeled `GalaxyDataset` data module |
| `config.yaml` | Shared configuration |
