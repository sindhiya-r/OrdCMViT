"""
src/data/transforms.py
───────────────────────
Modality-specific augmentation pipelines using Albumentations.

Key design decisions:
  1. US and Mammogram get DIFFERENT augmentation pipelines
     - US: aggressive spatial + noise augmentation (US artifacts are clinically diverse)
     - Mammogram: conservative spatial + contrast augmentation (preserve microcalcifications)
  2. CLAHE is applied in bcmid.py BEFORE transform (at native resolution)
     - Here we only apply post-resize augmentations
  3. Normalize last using ImageNet stats (ViT pretrained on ImageNet)
  4. Returns albumentations transforms (take numpy HWC uint8, return tensor CHW float)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple


def build_train_transform_us(cfg) -> A.Compose:
    """
    Ultrasound training augmentation.
    US is robust to spatial transforms; speckle noise is clinically present.
    """
    dc = cfg.data
    return A.Compose([
        # ── Spatial ─────────────────────────────────────────────────────────
        A.HorizontalFlip(p=dc.aug_hflip_p),
        A.VerticalFlip(p=dc.aug_vflip_p),
        A.Rotate(limit=dc.aug_rotate_limit, p=dc.aug_rotate_p,
                 border_mode=0),          # fill with black (not reflect) — avoid artifacts
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.3,
            border_mode=0,
        ),
        A.ElasticTransform(
            alpha=50, sigma=10, p=dc.aug_elastic_p
        ),  # simulates tissue deformation during probe pressure
        # ── Intensity ────────────────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=dc.aug_brightness_limit,
            contrast_limit=dc.aug_contrast_limit,
            p=dc.aug_brightness_p,
        ),
        A.GaussNoise(
            var_limit=(10.0, 50.0),       # simulates US speckle noise
            p=dc.aug_noise_p,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=dc.aug_blur_p),
        # ── Regularization ──────────────────────────────────────────────────
        A.CoarseDropout(
            max_holes=4,
            max_height=dc.aug_cutout_max_h_size,
            max_width=dc.aug_cutout_max_w_size,
            fill_value=0,
            p=dc.aug_cutout_p,
        ),
        # ── Final: convert to tensor ─────────────────────────────────────────
        A.Normalize(mean=dc.normalize_mean, std=dc.normalize_std),
        ToTensorV2(),                     # HWC uint8 → CHW float32
    ])


def build_train_transform_mm(cfg) -> A.Compose:
    """
    Mammogram training augmentation.
    Conservative spatial (microcalcifications are tiny; aggressive spatial = false patterns).
    Moderate contrast (clinical mammo protocols vary → contrast robustness is valid).
    NO vertical flip (clinical mammo is always CC or MLO — anatomically consistent).
    """
    dc = cfg.data
    return A.Compose([
        # ── Spatial (conservative) ───────────────────────────────────────────
        A.HorizontalFlip(p=dc.aug_hflip_p),   # bilateral symmetry is valid
        A.Rotate(limit=5, p=0.3,              # very limited rotation (mammography is standardized)
                 border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.03,                 # small shifts
            scale_limit=0.05,                 # small scale variation
            rotate_limit=5,
            p=0.2,
            border_mode=0,
        ),
        # ── Intensity (moderate) ─────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.1,             # conservative for mammo
            contrast_limit=0.15,
            p=0.4,
        ),
        # Note: NO GaussNoise for mammo (mammographic noise is structured, not Gaussian)
        # Note: NO ElasticTransform (breast tissue deformation in mammo is minimal)
        # ── Regularization ──────────────────────────────────────────────────
        A.CoarseDropout(
            max_holes=2,                      # fewer holes than US
            max_height=20,
            max_width=20,
            fill_value=0,
            p=0.2,
        ),
        # ── Final ────────────────────────────────────────────────────────────
        A.Normalize(mean=dc.normalize_mean, std=dc.normalize_std),
        ToTensorV2(),
    ])


def build_val_transform(cfg, size: int) -> A.Compose:
    """
    Validation/test transform: no augmentation, just normalize.
    Note: CLAHE is applied in bcmid.py before this.
    """
    dc = cfg.data
    return A.Compose([
        A.Normalize(mean=dc.normalize_mean, std=dc.normalize_std),
        ToTensorV2(),
    ])


def build_val_transform_us(cfg)  -> A.Compose:
    return build_val_transform(cfg, cfg.data.us_size)


def build_val_transform_mm(cfg)  -> A.Compose:
    return build_val_transform(cfg, cfg.data.mm_size)
