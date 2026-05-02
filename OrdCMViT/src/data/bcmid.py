"""
src/data/bcmid.py
─────────────────
BCMID dataset loader (clean, deterministic version)

Preprocessing steps implemented here mirror the manual pipeline used in
our cross-modal fusion work:

1. Crop Region-of-Interest
   • remove outer black borders using background thresholding + largest
     connected component + morphological closing
   • remove corner labels
   • crop to breast bounding box (mammo) or trapezoidal scan region (US)
2. Remove text & markers
   • mask bright text (e.g. R‑CC label) and high‑intensity UI artifacts
   • optional inpainting (simple Telea implementation)
   • US bottom measurement box and right‑side panel are discarded via crop
3. Remove calipers (+ signs)
   • high‑intensity small crosses detected by size filtering and inpainted
   • morphological opening could be added if necessary
4. Normalize
   • mammogram: z‑score, clip 1/99 percentile, rescale to uint8
   • ultrasound: median speckle filtering + histogram equalization

⚠️  For cross-modal fusion research, ANY residual overlays or text will
     leak modality-specific information.  The transformer could learn to
     align measurement text across modalities, corrupting conclusions.
     Ensure preprocessing is strict or mask artifacts in training.

Other features:
• Exact folder matching (Ultrasound / Mammogram)
• Robust CSV parsing
• CLAHE for mammograms (contrast enhancement)
• Stratified K-Fold split
• Class weights
"""
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def parse_birads(raw) -> Optional[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None

    s = str(raw).strip().upper()
    m = re.search(r"\d+", s)
    if not m:
        return None

    val = int(m.group())

    if val == 0:
        return None
    if val >= 6:
        return 5

    return min(max(val, 1), 5)


def get_image_paths(folder: Path) -> List[Path]:
    if not folder.exists():
        return []

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}
    paths = []

    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_ext:
            paths.append(p)

    return sorted(paths)


def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    try:
        p = str(path)

        if p.lower().endswith(".dcm"):
            try:
                import pydicom
                ds = pydicom.dcmread(p)
                arr = ds.pixel_array.astype(np.float32)
                arr = ((arr - arr.min()) /
                       (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
                if arr.ndim == 2:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                return arr
            except Exception:
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
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def crop_breast_region(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # normalize and threshold low-intensity background
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(norm.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)

    # morphological closing to fill gaps caused by label corners or artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img_rgb

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # crop using bounding box, removing black border and labels
    return img_rgb[y:y+h, x:x+w]


def crop_ultrasound_region(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    # remove top 10% (probe housing) and side UI panels via fixed crop
    img = img_rgb[int(0.1*h):, :]
    img = img[:, int(0.05*w):int(0.95*w)]
    return img


# ─────────────────────────────────────────────────────────────
# additional preprocessing helpers (text, markers, calipers, normalize)
# ─────────────────────────────────────────────────────────────

def remove_text_and_calipers(img: np.ndarray) -> np.ndarray:
    """Mask and inpaint small bright text/cross artifacts.
    Works for both mammogram and ultrasound images.

    - threshold very bright pixels
    - remove large regions (breast itself) by area filtering
    - inpaint the remaining spots
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # bright regions (text, calipers) are near white
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    # remove components that are too large to be UI text/crosses
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(thresh)
    for c in contours:
        area = cv2.contourArea(c)
        # keep only small/high-intensity objects
        if 1 < area < 500:  # tuning: text/crosses tend to be small
            cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    if np.any(mask):
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return img


def normalize_image(img: np.ndarray, modality: str) -> np.ndarray:
    """Apply modality-specific normalization/clipping.

    Mammogram: z-score, clip 1%-99%, scale back to uint8.
    Ultrasound: median denoise + histogram equalization (speckle filtering).
    """
    if modality == "mm":
        f = img.astype(np.float32)
        mean, std = f.mean(), f.std()
        f = (f - mean) / (std + 1e-6)
        low, high = np.percentile(f, [1, 99])
        f = np.clip(f, low, high)
        f = ((f - f.min()) / (f.max() - f.min() + 1e-8) * 255.0)
        return f.astype(np.uint8)
    else:
        # simple speckle reduction + histogram norm
        # convert to grayscale for filtering then back to RGB
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        eq = cv2.equalizeHist(denoised)
        rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
        return rgb


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class BCMIDDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        labels_csv: str,
        patient_ids: List[str],
        cfg,
        transform_us=None,
        transform_mm=None,
        mode: str = "train",
    ):
        self.data_root = Path(data_root)
        self.cfg = cfg
        self.transform_us = transform_us
        self.transform_mm = transform_mm
        self.mode = mode

        df = pd.read_csv(labels_csv, header=None)
        df.columns = ["patient_id", "birads", "binary_label"]

        df["patient_id"] = df["patient_id"].astype(str).str.strip()
        df["birads_int"] = df["birads"].apply(parse_birads)
        df = df.dropna(subset=["birads_int"])
        df["birads_int"] = df["birads_int"].astype(int)

        df = df[df["patient_id"].isin(patient_ids)].reset_index(drop=True)

        self.samples: List[Dict] = []

        missing_us = 0
        missing_mm = 0
        missing_us_ids: List[str] = []
        missing_mm_ids: List[str] = []

        for _, row in df.iterrows():
            pid = row["patient_id"]
            birads = row["birads_int"]

            pdir = self.data_root / pid
            if not pdir.exists():
                continue

            # EXACT FOLDER MATCHING
            us_dir = pdir / "Ultrasound"
            mm_dir = pdir / "Mammogram"

            if not us_dir.exists():
                missing_us += 1
                missing_us_ids.append(pid)
                continue

            if not mm_dir.exists():
                missing_mm += 1
                missing_mm_ids.append(pid)
                continue

            us_paths = get_image_paths(us_dir)
            mm_paths = get_image_paths(mm_dir)

            if not us_paths:
                missing_us += 1
                missing_us_ids.append(pid)
                continue

            if not mm_paths:
                missing_mm += 1
                missing_mm_ids.append(pid)
                continue

            ordinal_target = torch.tensor(
                [1.0 if birads >= k else 0.0 for k in range(2, 6)],
                dtype=torch.float32
            )

            self.samples.append({
                "patient_id": pid,
                "birads": birads,
                "birads_idx": birads - 1,
                "ordinal_target": ordinal_target,
                "us_paths": us_paths,
                "mm_paths": mm_paths,
            })

        if missing_us > 0 or missing_mm > 0:
            # should be rare, since build_splits filters these out already
            msg = f"[BCMID] WARNING: during dataset init {missing_us} patients missing US images, {missing_mm} missing Mammo images"
            print(msg)
            if missing_us_ids:
                print(f"  example missing US pids: {missing_us_ids[:5]}")
            if missing_mm_ids:
                print(f"  example missing Mammo pids: {missing_mm_ids[:5]}")

        print(f"[BCMID] Mode={mode} | {len(self.samples)} patients loaded")
        self._print_class_dist()

    def _print_class_dist(self):
        from collections import Counter
        counts = Counter(s["birads"] for s in self.samples)
        print("[BCMID] BI-RADS distribution:",
              " | ".join(f"{k}: {v}" for k, v in sorted(counts.items())))

    def compute_class_weights(self) -> torch.Tensor:
        from collections import Counter
        counts = Counter(s["birads_idx"] for s in self.samples)
        total = len(self.samples)

        weights = torch.zeros(self.cfg.model.num_classes)
        for cls_idx, cnt in counts.items():
            weights[cls_idx] = total / (self.cfg.model.num_classes * cnt)

        weights = weights / weights.max()
        return weights

    def compute_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler (inverse class frequency)."""
        from collections import Counter
        counts = Counter(s["birads_idx"] for s in self.samples)
        total = len(self.samples)
        sample_weights = [
            total / (len(counts) * counts[s["birads_idx"]])
            for s in self.samples
        ]
        return torch.tensor(sample_weights, dtype=torch.float32)

    def _load_and_preprocess(
        self,
        paths: List[Path],
        target_size: int,
        apply_clahe_flag: bool,
        modality: str,
    ) -> Optional[np.ndarray]:

        chosen = random.choice(paths) if self.mode == "train" else paths[0]

        img = load_image_rgb(chosen)
        if img is None:
            return None

        if apply_clahe_flag:
            img = apply_clahe(
                img,
                clip_limit=self.cfg.data.clahe_clip_limit,
                tile_grid=self.cfg.data.clahe_tile_grid_size,
            )

        if modality == "mm":
            img = crop_breast_region(img)
        else:
            img = crop_ultrasound_region(img)

        # remove text/marker artifacts and calipers BEFORE resizing
        img = remove_text_and_calipers(img)

        # normalize intensity range per modality
        img = normalize_image(img, modality)

        h, w = img.shape[:2]
        interp = cv2.INTER_AREA if max(h, w) > target_size else cv2.INTER_LANCZOS4
        img = cv2.resize(img, (target_size, target_size), interpolation=interp)

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        us_img = self._load_and_preprocess(
            s["us_paths"],
            self.cfg.data.us_size,
            False,
            "us"
        )

        mm_img = self._load_and_preprocess(
            s["mm_paths"],
            self.cfg.data.mm_size,
            True,
            "mm"
        )

        if us_img is None:
            us_img = np.zeros((self.cfg.data.us_size,
                               self.cfg.data.us_size, 3), dtype=np.uint8)
        if mm_img is None:
            mm_img = np.zeros((self.cfg.data.mm_size,
                               self.cfg.data.mm_size, 3), dtype=np.uint8)

        if self.transform_us:
            us_img = self.transform_us(image=us_img)["image"]
        if self.transform_mm:
            mm_img = self.transform_mm(image=mm_img)["image"]

        return {
            "patient_id": s["patient_id"],
            "us": us_img,
            "mm": mm_img,
            "label": torch.tensor(s["birads_idx"], dtype=torch.long),
            "ordinal": s["ordinal_target"],
            "birads": s["birads"],
        }


# ─────────────────────────────────────────────────────────────
# Split utilities
# ─────────────────────────────────────────────────────────────

def build_splits(
    data_root: str,
    labels_csv: str,
    cfg,
    n_folds: int = 5,
    val_fold: int = 0,
    test_size: float = 0.15,
    seed: int = 42,
    output_dir: Optional[str] = None,
):

    data_root = Path(data_root)

    # verify that the user supplied a valid directory (better message than crashing later)
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"BCMID data_root not found: {data_root}\n"
                                "Please pass the correct path via --data_root or update your config.")

    # scan disk, but ensure each patient actually contains at least one image in
    # both modalities.  This prevents splitting on patients who later get dropped
    # during dataset construction and explains the discrepancy seen in logs.
    folder_ids = []
    missing_us = []
    missing_mm = []
    for p in data_root.iterdir():
        if not p.is_dir():
            continue
        us_dir = p / "Ultrasound"
        mm_dir = p / "Mammogram"
        if not us_dir.exists() or not mm_dir.exists():
            continue

        us_paths = get_image_paths(us_dir)
        mm_paths = get_image_paths(mm_dir)

        if not us_paths:
            missing_us.append(p.name.strip())
            continue
        if not mm_paths:
            missing_mm.append(p.name.strip())
            continue

        folder_ids.append(p.name.strip())

    folder_ids = set(folder_ids)
    print(f"[Split] Patients on disk with both modalities: {len(folder_ids)} "
          f"(skipped {len(missing_us)} missing US, {len(missing_mm)} missing Mammo)")

    # load CSV
    df = pd.read_csv(labels_csv, header=None)
    df.columns = ["patient_id", "birads", "binary_label"]
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["birads_int"] = df["birads"].apply(parse_birads)
    df = df.dropna(subset=["birads_int"])

    df = df[df["patient_id"].isin(folder_ids)].reset_index(drop=True)
    print(f"[Split] Total patients usable: {len(df)}")

    pids = df["patient_id"].tolist()
    labels = df["birads_int"].tolist()

    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        pids, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    skf = StratifiedKFold(n_splits=n_folds,
                          shuffle=True,
                          random_state=seed)

    folds = list(skf.split(train_val_ids, train_val_labels))
    train_idx, val_idx = folds[val_fold]

    train_ids = [train_val_ids[i] for i in train_idx]
    val_ids   = [train_val_ids[i] for i in val_idx]

    print(f"[Split] Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # Persist split for reproducibility (so re-runs with the same seed produce
    # the same partition even if new patients are added to the dataset folder).
    if output_dir is not None:
        import json, os
        split_path = os.path.join(output_dir, "data_splits.json")
        with open(split_path, "w") as f:
            json.dump({"seed": seed, "val_fold": val_fold,
                       "train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)
        print(f"[Split] Split indices saved → {split_path}")

    return train_ids, val_ids, test_ids


def build_dataloaders(
    data_root,
    labels_csv,
    cfg,
    transform_train_us,
    transform_train_mm,
    transform_val_us,
    transform_val_mm,
):

    train_ids, val_ids, test_ids = build_splits(
        data_root,
        labels_csv,
        cfg,
        cfg.data.n_folds,
        cfg.data.val_fold,
        cfg.data.test_size,
        cfg.data.random_seed,
        output_dir=cfg.data.output_dir,
    )

    ds_train = BCMIDDataset(data_root, labels_csv, train_ids, cfg,
                            transform_train_us, transform_train_mm, "train")
    ds_val   = BCMIDDataset(data_root, labels_csv, val_ids, cfg,
                            transform_val_us, transform_val_mm, "val")
    ds_test  = BCMIDDataset(data_root, labels_csv, test_ids, cfg,
                            transform_val_us, transform_val_mm, "test")

    class_weights = ds_train.compute_class_weights()

    # WeightedRandomSampler: over-sample minority BI-RADS classes so each
    # training batch has a balanced class distribution. This directly addresses
    # the observed collapse to BI-RADS 1/5 only.
    sample_weights = ds_train.compute_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    kwargs = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    train_loader = DataLoader(ds_train,
                              batch_size=cfg.data.batch_size,
                              sampler=sampler,   # replaces shuffle=True
                              **kwargs)

    val_loader = DataLoader(ds_val,
                            batch_size=cfg.data.batch_size,
                            shuffle=False,
                            **kwargs)

    test_loader = DataLoader(ds_test,
                             batch_size=1,
                             shuffle=False,
                             **kwargs)

    return train_loader, val_loader, test_loader, class_weights