# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OrdCMViT is a deep learning system for BI-RADS grading (5 classes: Normal → Malignant) of breast cancer using cross-modal fusion of ultrasound (US) and mammogram (MM) images. The core innovation is combining Vision Transformers with CORAL ordinal loss and bidirectional cross-modal attention to respect the hierarchical nature of BI-RADS grades.

## Commands

### Setup
```bash
cd OrdCMViT
pip install -r requirements.txt
```

### Training (full pipeline)
```bash
python main.py \
  --data_root /path/to/BCMID \
  --labels_csv /path/to/BCMID_labels.csv \
  --max_epochs 100 \
  --batch_size 8 \
  --val_fold 0
```

Key flags:
- `--val_fold 0-4`: which fold to use for validation in 5-fold CV
- `--skip_train`: load `best.pth` and only run evaluation
- `--no_vis`: skip CAM visualization (faster runs)
- `--no_pretrain`: disable ImageNet pretraining (debugging only)
- `--seed 42`: reproducibility

### Web interface (single-patient inference)
```bash
streamlit run streamlit_app.py
```

### Batch inference
```bash
python batch_predict.py \
  --input_dir ./patients \
  --ground_truth_csv ground_truth.csv \
  --output_csv predictions.csv
```

## Architecture

### Data flow
1. `src/data/bcmid.py` — `BCMIDDataset` loads paired US + MM images per patient. Preprocessing: CLAHE (mammogram), ROI cropping, text/marker inpainting, z-score normalization, resize to 384×384.
2. `src/data/transforms.py` — Modality-specific augmentations: aggressive for US (elastic, noise, dropouts), conservative for MM (small rotations, contrast).
3. Labels CSV requires columns: `patient_id`, `birads`, `binary_label`. Dataset uses 5-fold stratified CV on 85% of data; 15% held-out test set.

### Model (`src/models/ordcmvit.py`)
```
US [B,3,384,384] + MM [B,3,384,384]
  → ViTBackbone (ViT-Small/patch16) × 2
  → Spatial alignment (MM 24×24 → 14×14 adaptive pool)
  → CrossModalBlock × 3 (self-attn + bidirectional cross-attn + FFN)
  → Fused CLS token [B, 384]
  → OrdinalHead → [B, 4] CORAL logits (P(y≥2..5))
  + aux heads: US-only head, MM-only head (anti-collapse)
```

Only the last 3 of 12 ViT-Small blocks are unfrozen; new layers use `lr=1e-4`, backbone uses `lr=5e-5`.

### Loss (`src/losses/losses.py`)
`total = CORAL_main + 0.3*(CORAL_us + CORAL_mm) + 0.01*attn_entropy`

- **CORAL**: 4 binary BCE outputs encoding P(y≥k) to respect ordinal structure
- **Auxiliary heads**: prevent modality collapse
- **Attention entropy**: regularizes cross-attention to focus on lesion regions

### Training (`src/engine/trainer.py`)
- AdamW with differential LRs, cosine annealing + 5-epoch warmup
- Gradient accumulation (4 steps → effective batch = 32), gradient clipping at 1.0
- Early stopping on validation QWK (patience = 20)
- Checkpoints: `runs/ordcmvit/checkpoints/best.pth` (best QWK), `last.pth`

### Metrics (`src/utils/metrics.py`)
Primary metric is **Quadratic Weighted Kappa (QWK)** — penalizes rank-distant errors. Also tracked: Accuracy, macro AUC, per-class accuracy, ECE (calibration).

Expected performance: Accuracy ~75.2%, QWK ~0.722, AUC ~0.847.

### Output structure
```
runs/ordcmvit/
├── checkpoints/best.pth, last.pth
├── visualizations/P001_cam.png ...   (US+CAM, MM+CAM, prediction)
├── logs/                              (TensorBoard)
├── training_history.csv
├── test_results.csv
├── test_metrics.json
└── confusion_matrix.png
```

## Configuration

All hyperparameters live in `config.py` as dataclasses: `DataConfig`, `ModelConfig`, `TrainingConfig`, `VisualizationConfig`. CLI args in `main.py` override config values at runtime.

Key parameters to tune:
- `freeze_n_last_blocks` (default 3): how many ViT-Small blocks to unfreeze
- `n_cross_blocks` (default 3): number of cross-modal fusion blocks
- `modality_dropout_p` (default 0.1): probability to zero out one modality during training
- `aux_loss_weight` (default 0.3): weight for per-modality auxiliary losses

## Dataset Notes

BCMID dataset structure expected:
```
BCMID/
├── patient_001/
│   ├── Ultrasound/   (*.png, *.jpg, or *.dcm)
│   └── Mammogram/
```
Supports DICOM files. Small dataset (~323 patients total), so regularization settings (freezing, dropout, augmentation) are critical.
