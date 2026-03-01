"""
src/data/bcmid.py
─────────────────
BCMID dataset loader with:
  - Multi-image-per-patient handling (random select train, all-average test)
  - CLAHE mammogram preprocessing
  - BI-RADS ordinal encoding
  - Robust CSV parsing (handles BI-RADS 4A/4B/4C → 4)
  - Stratified K-Fold split
  - Class weight computation
"""

import os
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_birads(raw) -> Optional[int]:
    """
    Convert raw BI-RADS label (any format) → integer 1-5.
    Handles: '4A', '4B', '4C', '4a', 4.0, '0', None, etc.
    BI-RADS 0 → None (incomplete, excluded)
    BI-RADS 6 → 5 (biopsy-proven, treat as most severe)
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().upper()
    # Extract leading digit(s)
    m = re.match(r"^([0-9]+)", s)
    if not m:
        return None
    val = int(m.group(1))
    if val == 0:
        return None          # BI-RADS 0 = incomplete assessment → skip
    if val >= 6:
        return 5             # BI-RADS 6 (known malignancy) → treat as 5
    return min(max(val, 1), 5)


def get_image_paths(folder: Path, extensions: Tuple[str, ...]) -> List[Path]:
    """Recursively find all images in a folder."""
    paths = []
    for ext in extensions:
        paths.extend(folder.rglob(f"*{ext}"))
        paths.extend(folder.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))   # sorted for reproducibility


def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """
    Load image as RGB uint8 array.
    Handles DICOM via opencv or fallback.
    Returns None if loading fails.
    """
    try:
        p = str(path)
        if p.lower().endswith(".dcm"):
            # Try pydicom if available
            try:
                import pydicom
                ds = pydicom.dcmread(p)
                arr = ds.pixel_array.astype(np.float32)
                # Normalize to 0-255
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
                if arr.ndim == 2:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                return arr
            except ImportError:
                return None
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def apply_clahe(img_rgb: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE on luminance channel of RGB image.
    Enhances local contrast → microcalcifications more visible after downscale.
    Applied to MAMMOGRAM ONLY (not ultrasound — US has different noise characteristics).
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ─────────────────────────────────────────────────────────────────────────────
#  BCMID Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BCMIDDataset(Dataset):
    """
    BCMID Patient-level dataset.

    Folder structure assumed:
        BCMID/
          ├── BCMID_labels.csv        (patient_id, birads)
          ├── P001/
          │     ├── Ultrasound/       (1+ US images)
          │     └── Mammogram/        (1+ Mammo images)
          ├── P002/ ...

    Multi-image strategy:
      - TRAIN: randomly select ONE image per modality per patient each epoch
                → effectively multiplies dataset diversity
      - TEST:  load ALL images per modality → average softmax predictions
                (handled externally by collate_fn or evaluator)

    BI-RADS ordinal encoding (CORAL):
      Label y ∈ {1,2,3,4,5} → ordinal target t ∈ {0,1,2,3,4}
      Binary targets: t_k = 1 if y ≥ (k+1), for k=1,2,3,4
      e.g. y=4 → [1, 1, 1, 0]  (y≥2, y≥3, y≥4, NOT y≥5)
    """

    def __init__(
        self,
        data_root: str,
        labels_csv: str,
        patient_ids: List[str],
        cfg,                        # DataConfig + ModelConfig
        transform_us=None,
        transform_mm=None,
        mode: str = "train",        # "train" | "val" | "test"
    ):
        self.data_root = Path(data_root)
        self.cfg = cfg
        self.transform_us = transform_us
        self.transform_mm = transform_mm
        self.mode = mode

        # ── Load and filter labels ─────────────────────────────────────────
        df = pd.read_csv(labels_csv)
        dc = cfg.data

        # Normalize column names to lowercase stripped
        df.columns = [c.strip().lower() for c in df.columns]
        pid_col = dc.patient_id_col.lower()
        lbl_col = dc.label_col.lower()

        # Parse BI-RADS labels
        df["birads_int"] = df[lbl_col].apply(parse_birads)
        df = df.dropna(subset=["birads_int"])
        df["birads_int"] = df["birads_int"].astype(int)
        df[pid_col] = df[pid_col].astype(str).str.strip()

        # Filter to requested patient IDs
        df = df[df[pid_col].isin(patient_ids)].reset_index(drop=True)

        # ── Scan patient folders ──────────────────────────────────────────
        self.samples: List[Dict] = []
        ext = tuple(dc.img_extensions)
        missing_us, missing_mm, missing_folder = 0, 0, 0

        for _, row in df.iterrows():
            pid = row[pid_col]
            birads = row["birads_int"]
            pdir = self.data_root / pid

            if not pdir.exists():
                # Try common variations
                candidates = list(self.data_root.glob(f"*{pid}*"))
                if candidates:
                    pdir = candidates[0]
                else:
                    missing_folder += 1
                    continue

            us_dir  = pdir / dc.us_subdir
            mm_dir  = pdir / dc.mm_subdir

            # Fallback: look for any subfolder that sounds like US or Mammo
            if not us_dir.exists():
                for sub in pdir.iterdir():
                    if sub.is_dir() and any(k in sub.name.lower()
                                            for k in ["us", "ultra", "sound"]):
                        us_dir = sub
                        break

            if not mm_dir.exists():
                for sub in pdir.iterdir():
                    if sub.is_dir() and any(k in sub.name.lower()
                                            for k in ["mammo", "xray", "mg", "film"]):
                        mm_dir = sub
                        break

            us_paths = get_image_paths(us_dir, ext) if us_dir.exists() else []
            mm_paths = get_image_paths(mm_dir, ext) if mm_dir.exists() else []

            if not us_paths:
                missing_us += 1
                continue
            if not mm_paths:
                missing_mm += 1
                continue

            # Ordinal targets (CORAL): [y≥2, y≥3, y≥4, y≥5] as float tensor
            ordinal_target = torch.tensor(
                [1.0 if birads >= k else 0.0 for k in range(2, 6)],
                dtype=torch.float32
            )  # shape [4]

            self.samples.append({
                "patient_id": pid,
                "birads": birads,
                "birads_idx": birads - 1,     # 0-indexed class label
                "ordinal_target": ordinal_target,
                "us_paths": us_paths,
                "mm_paths": mm_paths,
            })

        if missing_folder > 0:
            print(f"[BCMID] WARNING: {missing_folder} patient folders not found")
        if missing_us > 0:
            print(f"[BCMID] WARNING: {missing_us} patients missing US images")
        if missing_mm > 0:
            print(f"[BCMID] WARNING: {missing_mm} patients missing Mammo images")

        print(f"[BCMID] Mode={mode} | {len(self.samples)} patients loaded")
        self._print_class_dist()

    def _print_class_dist(self):
        from collections import Counter
        counts = Counter(s["birads"] for s in self.samples)
        print(f"[BCMID] BI-RADS distribution: "
              + " | ".join(f"BI-RADS {k}: {v}" for k, v in sorted(counts.items())))

    def compute_class_weights(self) -> torch.Tensor:
        """Inverse frequency class weights for loss weighting."""
        from collections import Counter
        counts = Counter(s["birads_idx"] for s in self.samples)
        total = len(self.samples)
        weights = torch.zeros(self.cfg.model.num_classes)
        for cls_idx, cnt in counts.items():
            weights[cls_idx] = total / (self.cfg.model.num_classes * cnt)
        # Normalize so max weight = 1
        weights = weights / weights.max()
        return weights

    def _load_and_preprocess(
        self,
        paths: List[Path],
        target_size: int,
        apply_clahe_flag: bool,
        transform,
    ) -> Optional[np.ndarray]:
        """
        Load one image from a list of paths (random during train, first during test).
        Returns numpy array HWC uint8, or None on failure.
        """
        if self.mode == "train":
            chosen = random.choice(paths)    # stochastic view selection = free augmentation
        else:
            chosen = paths[0]               # deterministic: use first (best quality)

        img = load_image_rgb(chosen)
        if img is None:
            # Try another path
            for p in paths:
                img = load_image_rgb(p)
                if img is not None:
                    break
        if img is None:
            return None

        # CLAHE enhancement BEFORE resize (works at native resolution → preserves fine detail)
        if apply_clahe_flag:
            img = apply_clahe(
                img,
                clip_limit=self.cfg.data.clahe_clip_limit,
                tile_grid=self.cfg.data.clahe_tile_grid_size,
            )

        # Resize with high-quality interpolation
        # INTER_AREA for downscale (anti-aliasing), INTER_LANCZOS4 for upscale
        h, w = img.shape[:2]
        if max(h, w) > target_size:
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

        return img  # HWC uint8

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]

        # ── Load images ────────────────────────────────────────────────────
        us_img = self._load_and_preprocess(
            s["us_paths"],
            self.cfg.data.us_size,
            apply_clahe_flag=False,  # NO CLAHE for US (acoustic images)
            transform=None,
        )
        mm_img = self._load_and_preprocess(
            s["mm_paths"],
            self.cfg.data.mm_size,
            apply_clahe_flag=True,   # CLAHE for Mammogram
            transform=None,
        )

        # Fallback: black images if loading fails (prevents DataLoader crash)
        if us_img is None:
            us_img = np.zeros((self.cfg.data.us_size, self.cfg.data.us_size, 3), dtype=np.uint8)
        if mm_img is None:
            mm_img = np.zeros((self.cfg.data.mm_size, self.cfg.data.mm_size, 3), dtype=np.uint8)

        # ── Apply augmentation ──────────────────────────────────────────────
        if self.transform_us is not None:
            us_img = self.transform_us(image=us_img)["image"]
        if self.transform_mm is not None:
            mm_img = self.transform_mm(image=mm_img)["image"]

        return {
            "patient_id": s["patient_id"],
            "us": us_img,                          # float tensor [3,H,W]
            "mm": mm_img,                          # float tensor [3,H,W]
            "label": torch.tensor(s["birads_idx"], dtype=torch.long),
            "ordinal": s["ordinal_target"],        # [4] float
            "birads": s["birads"],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Data split utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_splits(
    data_root: str,
    labels_csv: str,
    cfg,
    n_folds: int = 5,
    val_fold: int = 0,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (train_ids, val_ids, test_ids) patient ID lists.
    Strategy:
      1. Hold out test_size fraction (stratified) → never touched during training
      2. Apply K-fold CV on remaining → train/val for given fold
    """
    df = pd.read_csv(labels_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    dc = cfg.data
    pid_col = dc.patient_id_col.lower()
    lbl_col = dc.label_col.lower()
    df["birads_int"] = df[lbl_col].apply(parse_birads)
    df = df.dropna(subset=["birads_int"])
    df["birads_int"] = df["birads_int"].astype(int)
    df[pid_col] = df[pid_col].astype(str).str.strip()

    pids = df[pid_col].tolist()
    labels = df["birads_int"].tolist()

    # Step 1: Hold-out test set
    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        pids, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    # Step 2: K-Fold on remaining
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(train_val_ids, train_val_labels))
    train_idx, val_idx = folds[val_fold]

    train_ids = [train_val_ids[i] for i in train_idx]
    val_ids   = [train_val_ids[i] for i in val_idx]

    print(f"[Split] Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


def build_dataloaders(
    data_root: str,
    labels_csv: str,
    cfg,
    transform_train_us,
    transform_train_mm,
    transform_val_us,
    transform_val_mm,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Returns (train_loader, val_loader, test_loader, class_weights).
    """
    train_ids, val_ids, test_ids = build_splits(
        data_root, labels_csv, cfg,
        n_folds=cfg.data.n_folds,
        val_fold=cfg.data.val_fold,
        test_size=cfg.data.test_size,
        seed=cfg.data.random_seed,
    )

    ds_train = BCMIDDataset(data_root, labels_csv, train_ids, cfg,
                            transform_train_us, transform_train_mm, mode="train")
    ds_val   = BCMIDDataset(data_root, labels_csv, val_ids, cfg,
                            transform_val_us, transform_val_mm, mode="val")
    ds_test  = BCMIDDataset(data_root, labels_csv, test_ids, cfg,
                            transform_val_us, transform_val_mm, mode="test")

    class_weights = ds_train.compute_class_weights()

    dc = cfg.data
    kwargs = dict(
        num_workers=dc.num_workers,
        pin_memory=dc.pin_memory,
    )

    train_loader = DataLoader(ds_train, batch_size=dc.batch_size, shuffle=True,  **kwargs)
    val_loader   = DataLoader(ds_val,   batch_size=dc.batch_size, shuffle=False, **kwargs)
    test_loader  = DataLoader(ds_test,  batch_size=1,             shuffle=False, **kwargs)
    # batch_size=1 for test → enables patient-level visualization without padding issues

    return train_loader, val_loader, test_loader, class_weights
