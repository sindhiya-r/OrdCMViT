"""
config.py — All hyperparameters for OrdCMViT.
Every number that matters is here. Change ONLY this file.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os


@dataclass
class DataConfig:
    # ── BCMID paths ────────────────────────────────────────────────────────────
    data_root: str = "./BCMID"                     # root folder of BCMID
    labels_csv: str = "./BCMID/BCMID_labels.csv"  # CSV: patient_id, birads
    output_dir: str = "./runs/ordcmvit"            # all outputs saved here

    # ── Column names in CSV (update if BCMID CSV uses different names) ─────────
    patient_id_col: str = "patient_id"
    label_col: str = "birads"                      # BI-RADS integer 1-5

    # ── Image subfolder names inside each patient folder ──────────────────────
    us_subdir: str = "Ultrasound"
    mm_subdir: str = "Mammogram"

    # ── Supported image extensions ─────────────────────────────────────────────
    img_extensions: Tuple[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".dcm")

    # ── Resolution (CRITICAL — justified below) ────────────────────────────────
    # Mammogram: 384×384 with CLAHE preprocessing
    #   Original BCMID mammograms are typically >2000×2000.
    #   384 is a compromise: patch 16×16 → 24×24=576 tokens.
    #   Each patch = 16/384 = 4.2% of image width — adequate for mass detection.
    #   ViT-Small/16-384 is pretrained at this resolution → no positional interp needed.
    mm_size: int = 384

    # Ultrasound: 224×224 — native ViT-Small/16 size
    #   US lesions are generally larger relative to image area than mammo calcifications.
    #   224 is appropriate; lesions typically span 10-30% of US image.
    us_size: int = 224

    # ── Patch size ─────────────────────────────────────────────────────────────
    # Both modalities use patch_size=16:
    #   Mammo: 384/16=24 → 24×24=576 tokens. Patch = 16px = 4.2% width.
    #   US:    224/16=14 → 14×14=196 tokens. Patch = 16px = 7.1% width.
    # Chosen over patch=8 to reduce sequence length for 323-sample dataset.
    patch_size: int = 16

    # ── Dataset split ──────────────────────────────────────────────────────────
    n_folds: int = 5           # 5-fold stratified CV
    val_fold: int = 0          # which fold is validation (0-4)
    test_size: float = 0.15    # held-out test set (never seen during CV)
    random_seed: int = 42

    # ── DataLoader ─────────────────────────────────────────────────────────────
    batch_size: int = 8        # BCMID is small; accumulate gradient to effective=32
    num_workers: int = 4
    pin_memory: bool = True

    # ── Augmentation probabilities ─────────────────────────────────────────────
    # Strong augmentation is ESSENTIAL given only 226 training samples
    aug_hflip_p: float = 0.5
    aug_vflip_p: float = 0.2
    aug_rotate_p: float = 0.5
    aug_rotate_limit: int = 15        # degrees — clinical images rarely rotated >15°
    aug_brightness_p: float = 0.4
    aug_brightness_limit: float = 0.2
    aug_contrast_p: float = 0.4
    aug_contrast_limit: float = 0.2
    aug_noise_p: float = 0.3
    aug_blur_p: float = 0.2
    aug_elastic_p: float = 0.3       # simulates tissue deformation (clinically valid)
    aug_cutout_p: float = 0.3        # regularization — masks random patches
    aug_cutout_max_h_size: int = 30  # px
    aug_cutout_max_w_size: int = 30

    # ── CLAHE for mammogram ────────────────────────────────────────────────────
    # Contrast Limited Adaptive Histogram Equalization:
    # Enhances local contrast → microcalcifications become more visible at 384×384
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # ImageNet normalization (ViT-Small pretrained on ImageNet)
    normalize_mean: Tuple[float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    # ── Backbone ───────────────────────────────────────────────────────────────
    # WHY ViT-Small, not ViT-Base:
    #   ViT-Base = 86M params. With 226 training samples → catastrophic overfitting.
    #   ViT-Small = 22M params, 12 layers, 384-dim, 6 heads → much safer.
    #   vit_small_patch16_384: pretrained on ImageNet-21k at 384×384 → used for mammo
    #   vit_small_patch16_224: pretrained on ImageNet-21k at 224×224 → used for US
    mm_backbone_name: str = "vit_small_patch16_384"   # timm model name
    us_backbone_name: str = "vit_small_patch16_224"   # timm model name
    embed_dim: int = 384              # ViT-Small feature dimension
    pretrained: bool = True

    # ── Backbone freezing strategy ─────────────────────────────────────────────
    # CRITICAL: With 226 samples, training all 22M params from pretrained → overfits.
    # We freeze early layers (generic features) and only train last N blocks.
    # Each ViT-Small has 12 transformer blocks.
    # Last 3 blocks = ~5M params per modality → reasonable for 226 samples.
    freeze_n_last_blocks: int = 3     # number of transformer blocks to UNFREEZE

    # ── Mammo token spatial pooling ───────────────────────────────────────────
    # Mammo produces 576 tokens (24×24), US produces 196 tokens (14×14).
    # For cross-attention efficiency, pool mammo to 196 tokens (14×14 grid).
    # Using 2D spatial adaptive average pooling: 24×24 → 14×14
    # This is NOT a learned compression — it's deterministic spatial pooling.
    # Rationale: spatial structure preserved; same token count → aligned grids.
    mm_pool_size: int = 14            # pool mammo to 14×14 = 196 tokens (matches US)

    # ── Cross-Modal Attention ─────────────────────────────────────────────────
    # 3 bidirectional cross-attention blocks:
    # Block 1: US queries Mammo → US tokens enriched with mammo context
    # Block 2: Mammo queries US → Mammo tokens enriched with US context
    # Block 3: Both + self-attention for refinement
    n_cross_blocks: int = 3
    cross_attn_heads: int = 8         # 384 / 8 = 48 dims per head
    cross_attn_dropout: float = 0.1
    cross_ffn_ratio: float = 4.0      # FFN hidden dim = embed_dim × ratio

    # ── Modality Dropout (anti-collapse mechanism 1) ──────────────────────────
    # Randomly zero-out entire modality during training.
    # Forces each branch to maintain independent discriminative power.
    # Without this: model learns to use only the easier modality → collapse.
    modality_dropout_p: float = 0.1   # p=0.1 for each modality independently

    # ── Number of BI-RADS classes ─────────────────────────────────────────────
    num_classes: int = 5              # BI-RADS 1, 2, 3, 4, 5

    # ── Ordinal encoding ──────────────────────────────────────────────────────
    # CORAL ordinal regression: num_classes-1 = 4 binary sigmoid outputs
    # f_k = w^T z + b_k, k=1,2,3,4
    # P(y≥k) = sigmoid(f_k)
    # Prediction: ŷ = 1 + Σ_k 1[P(y≥k)>0.5]  → integer in {1,2,3,4,5}
    num_ordinal: int = 4              # = num_classes - 1

    # ── Auxiliary heads (anti-collapse mechanism 2) ───────────────────────────
    # Each modality has its OWN ordinal head on its CLS token.
    # This enforces that both branches must independently solve the task.
    use_aux_heads: bool = True
    aux_loss_weight: float = 0.3      # total_loss = main + 0.3*(aux_us + aux_mm)

    # ── Dropout in classification heads ──────────────────────────────────────
    head_dropout: float = 0.3         # stronger dropout given small dataset

    # ── Label smoothing for ordinal loss ─────────────────────────────────────
    label_smoothing: float = 0.05


@dataclass
class TrainingConfig:
    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Differential learning rates:
    #   Backbone (unfrozen blocks): very small LR (pretrained features)
    #   Cross-attention + heads: larger LR (new parameters)
    backbone_lr: float = 5e-6         # for unfrozen backbone blocks
    new_layers_lr: float = 1e-4       # for cross-attention, heads
    weight_decay: float = 0.05        # AdamW standard weight decay
    betas: Tuple[float, float] = (0.9, 0.999)

    # ── Gradient accumulation ─────────────────────────────────────────────────
    # Effective batch size = batch_size × grad_accum_steps = 8 × 4 = 32
    # Needed because BCMID is small → small physical batch is best
    grad_accum_steps: int = 4

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler: str = "cosine"         # cosine annealing with warmup
    warmup_epochs: int = 5
    max_epochs: int = 100

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp: bool = True              # FP16 training (faster + lower memory)

    # ── Gradient clipping ────────────────────────────────────────────────────
    grad_clip: float = 1.0

    # ── Early stopping ───────────────────────────────────────────────────────
    patience: int = 20                # stop if val QWK doesn't improve for 20 epochs
    min_delta: float = 0.001

    # ── Class weighting ──────────────────────────────────────────────────────
    # BCMID likely has BI-RADS class imbalance (fewer BI-RADS 1/5).
    # Computed dynamically from training set in dataset.py.
    use_class_weights: bool = True

    # ── Model checkpoint ─────────────────────────────────────────────────────
    save_best_metric: str = "val_qwk"  # metric to monitor for best checkpoint
    save_last: bool = True


@dataclass
class VisualizationConfig:
    save_cams: bool = True             # Save CAM overlays for test set
    n_vis_samples: int = 20            # number of test patients to visualize
    cam_alpha: float = 0.5             # blend weight for heatmap overlay
    upsample_cam: bool = True          # upsample CAM to original image size
    save_format: str = "png"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vis: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.data.output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.data.output_dir, "logs"), exist_ok=True)


# ── Default config instance ────────────────────────────────────────────────────
CFG = Config()
