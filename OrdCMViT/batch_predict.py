"""
batch_predict.py
────────────────
Batch inference script for processing multiple patients with OrdCMViT.

Usage:
    python batch_predict.py --input_dir ./patients --output_csv results.csv --model_run 4

Structure expected:
    patients/
    ├── patient_001/
    │   ├── Ultrasound/
    │   │   └── image.png
    │   └── Mammogram/
    │       └── image.jpg
    ├── patient_002/
    │   ├── Ultrasound/
    │   │   └── image.png
    │   └── Mammogram/
    │       └── image.jpg
    └── ...
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from src.models.ordcmvit import OrdCMViT, OrdinalHead
from src.utils.seed import get_device
from src.data.transforms import build_val_transform_us, build_val_transform_mm
import cv2
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """Load image from file."""
    try:
        if str(path).lower().endswith(".dcm"):
            try:
                import pydicom
                ds = pydicom.dcmread(str(path))
                arr = ds.pixel_array.astype(np.float32)
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
                if arr.ndim == 2:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                return arr
            except ImportError:
                print("⚠️ pydicom not installed. Cannot load DICOM files.")
                return None
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None


def get_latest_image(folder: Path) -> Optional[Tuple[np.ndarray, str]]:
    """Get latest/first image from folder."""
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}
    images = sorted([
        p for p in folder.rglob("*") 
        if p.is_file() and p.suffix.lower() in valid_ext
    ])
    
    if not images:
        return None
    
    img = load_image_rgb(images[0])
    return (img, str(images[0])) if img is not None else None


def find_modality_folders(patient_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find Ultrasound and Mammogram folders in patient directory."""
    us_folder = None
    mm_folder = None
    
    for folder in patient_dir.iterdir():
        if folder.is_dir():
            name_lower = folder.name.lower()
            if "ultrasound" in name_lower or "us" in name_lower:
                us_folder = folder
            elif "mammogram" in name_lower or "mm" in name_lower:
                mm_folder = folder
    
    return us_folder, mm_folder


# ─────────────────────────────────────────────────────────────────────────────
#  Model & Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> Optional[OrdCMViT]:
    """Load OrdCMViT model from checkpoint."""
    cfg = Config()
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        model = OrdCMViT(cfg.model)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None


@torch.no_grad()
def predict_patient(
    us_image: np.ndarray,
    mm_image: np.ndarray,
    model: OrdCMViT,
    device: torch.device,
    cfg: Config,
) -> Dict:
    """Predict BI-RADS for a single patient."""
    transform_us = build_val_transform_us(cfg)
    transform_mm = build_val_transform_mm(cfg)
    
    # Transform images
    us_tensor = transform_us(image=us_image)["image"].unsqueeze(0).to(device)
    mm_tensor = transform_mm(image=mm_image)["image"].unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(us_tensor, mm_tensor)
        logits = output["main_logits"]  # [1, 4] ordinal logits
    
    # Predict class
    pred_class = OrdinalHead.predict(logits).item()
    pred_birads = pred_class + 1
    
    # Get probabilities
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Convert ordinal probabilities to class probabilities
    class_probs = np.zeros(5)
    class_probs[0] = 1 - probs[0]
    for k in range(1, 4):
        class_probs[k] = probs[k-1] - probs[k]
    class_probs[4] = probs[3]
    class_probs = np.clip(class_probs, 0, 1)
    
    confidence = float(np.max(class_probs))
    
    return {
        "pred_class": int(pred_class),
        "pred_birads": int(pred_birads),
        "confidence": confidence,
        "class_probabilities": class_probs.tolist(),
        "ordinal_logits": logits.squeeze().cpu().numpy().tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Batch Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_patient_directory(
    patient_dir: Path,
    model: OrdCMViT,
    device: torch.device,
    cfg: Config,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Process a single patient directory.
    
    Returns:
        {
            "patient_id": str,
            "us_found": bool,
            "mm_found": bool,
            "success": bool,
            "pred_birads": int or None,
            "confidence": float or None,
            ...
        }
    """
    patient_id = patient_dir.name
    result = {"patient_id": patient_id}
    
    # Find modality folders
    us_folder, mm_folder = find_modality_folders(patient_dir)
    
    result["us_found"] = us_folder is not None
    result["mm_found"] = mm_folder is not None
    
    if not result["us_found"] or not result["mm_found"]:
        if verbose:
            print(f"⚠️  {patient_id}: Missing modality folder(s)")
        result["success"] = False
        return result
    
    # Load images
    us_result = get_latest_image(us_folder)
    mm_result = get_latest_image(mm_folder)
    
    if us_result is None or mm_result is None:
        if verbose:
            print(f"⚠️  {patient_id}: Could not load images")
        result["success"] = False
        return result
    
    us_image, _ = us_result
    mm_image, _ = mm_result
    
    # Predict
    try:
        pred = predict_patient(us_image, mm_image, model, device, cfg)
        result.update(pred)
        result["success"] = True
        if verbose:
            print(f"✅ {patient_id}: BI-RADS {pred['pred_birads']} (conf: {pred['confidence']:.2%})")
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        if verbose:
            print(f"❌ {patient_id}: Error during prediction - {e}")
    
    return result


def batch_process(
    input_dir: Path,
    model: OrdCMViT,
    device: torch.device,
    cfg: Config,
) -> List[Dict]:
    """Process all patient directories in input folder."""
    results = []
    
    # Find all patient directories
    patient_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    
    print(f"\n🔍 Found {len(patient_dirs)} patient directories")
    
    # Process each patient
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        result = process_patient_directory(patient_dir, model, device, cfg, verbose=False)
        results.append(result)
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Results Processing
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    results: List[Dict],
    output_csv: Path,
    output_json: Optional[Path] = None,
    ground_truth_csv: Optional[Path] = None,
) -> None:
    """Save batch prediction results."""
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Rename columns for clarity
    output_cols = ["patient_id", "pred_birads", "confidence"]
    if "class_probabilities" in df.columns:
        # Expand class probabilities
        for i, col in enumerate([f"prob_birads_{j}" for j in range(1, 6)]):
            df[col] = df["class_probabilities"].apply(lambda x: x[i] if isinstance(x, list) else None)
    
    if "us_found" in df.columns:
        output_cols.extend(["us_found", "mm_found", "success"])
    
    df_output = df[output_cols + [c for c in df.columns if c not in output_cols]]
    
    # Save CSV
    df_output.to_csv(output_csv, index=False)
    print(f"\n💾 Results saved to {output_csv}")
    
    # Save JSON for detailed info
    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"📋 Detailed results saved to {output_json}")
    
    # Print summary
    successful = sum(1 for r in results if r.get("success", False))
    print(f"\n📊 Summary:")
    print(f"  Total patients: {len(results)}")
    print(f"  Successful: {successful} ({100*successful/len(results):.1f}%)")
    print(f"  Failed: {len(results) - successful}")
    
    # Class distribution
    if successful > 0:
        preds = [r["pred_birads"] for r in results if r.get("success", False)]
        print(f"\n  BI-RADS Distribution:")
        for birads in range(1, 6):
            count = sum(1 for p in preds if p == birads)
            pct = 100 * count / successful
            print(f"    BI-RADS {birads}: {count:3d} ({pct:5.1f}%)")


def evaluate_with_ground_truth(
    results_df: pd.DataFrame,
    ground_truth_csv: Path,
) -> Dict:
    """Evaluate results against ground truth labels."""
    
    gt_df = pd.read_csv(ground_truth_csv)
    
    # Merge on patient_id
    merged = results_df.merge(
        gt_df,
        on="patient_id",
        how="inner",
        suffixes=("_pred", "_true")
    )
    
    if len(merged) == 0:
        print("⚠️  No matching patients in ground truth")
        return {}
    
    # Calculate accuracy
    correct = (merged["pred_birads"] == merged["true_birads"]).sum()
    accuracy = correct / len(merged)
    
    # Calculate ordinal metrics
    from sklearn.metrics import cohen_kappa_score
    qwk = cohen_kappa_score(
        merged["true_birads"],
        merged["pred_birads"],
        weights="quadratic"
    )
    
    metrics = {
        "n_samples": len(merged),
        "accuracy": float(accuracy),
        "qwk": float(qwk),
    }
    
    print(f"\n📈 Evaluation against ground truth:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Quadratic Weighted Kappa: {qwk:.4f}")
    
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch process patients with OrdCMViT")
    
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing patient folders"
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results.csv"),
        help="Output CSV file with predictions"
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Output JSON file with detailed results"
    )
    parser.add_argument(
        "--model_run",
        type=int,
        default=4,
        help="Model run number (from ./ordcmvit_runs/)"
    )
    parser.add_argument(
        "--ground_truth_csv",
        type=Path,
        default=None,
        help="CSV with ground truth labels for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    print("="*60)
    print("OrdCMViT Batch Inference")
    print("="*60)
    print(f"📁 Input directory: {args.input_dir}")
    print(f"💾 Output CSV: {args.output_csv}")
    print(f"🔧 Model run: {args.model_run}")
    print(f"📱 Device: {args.device}")
    
    # Find model checkpoint
    checkpoint_path = Path(f"./ordcmvit_runs/{args.model_run}/ordcmvit/checkpoints/best.pth")
    if not checkpoint_path.exists():
        checkpoint_path = Path(f"./ordcmvit_runs/{args.model_run}/checkpoints/best.pth")
    
    if not checkpoint_path.exists():
        print(f"❌ Model checkpoint not found for run {args.model_run}")
        return
    
    # Setup
    device = torch.device(args.device)
    cfg = Config()
    
    # Load model
    print("\n🔄 Loading model...")
    model = load_model(str(checkpoint_path), device)
    if model is None:
        return
    
    # Process patients
    print("\n🚀 Processing patients...")
    results = batch_process(args.input_dir, model, device, cfg)
    
    # Save results
    print("\n💾 Saving results...")
    save_results(
        results,
        args.output_csv,
        args.output_json,
        args.ground_truth_csv,
    )
    
    # Evaluate if ground truth provided
    if args.ground_truth_csv and args.ground_truth_csv.exists():
        print("\n📊 Evaluating against ground truth...")
        results_df = pd.read_csv(args.output_csv)
        evaluate_with_ground_truth(results_df, args.ground_truth_csv)
    
    print("\n✅ Batch processing complete!")


if __name__ == "__main__":
    main()
