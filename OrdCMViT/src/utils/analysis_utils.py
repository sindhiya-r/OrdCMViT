"""
src/utils/analysis_utils.py
───────────────────────────
Advanced analysis utilities for model evaluation:
  - Hard example mining
  - Per-modality contribution
  - Attention stability analysis
  - Calibration metrics
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, List, Optional


def mine_hard_examples(
    patient_ids: List[str],
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
) -> pd.DataFrame:
    """
    Identify and save hard examples (where model is uncertain or wrong).

    Args:
        patient_ids: [N] list of patient IDs
        labels: [N] true class indices (0-4)
        preds: [N] predicted class indices (0-4)
        probs: [N, K] predicted probabilities
        output_dir: where to save the CSV

    Returns:
        df: DataFrame with hard examples, sorted by difficulty
    """
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    # Compute metrics per sample
    confidences = probs.max(axis=1)
    is_correct = (preds == labels).astype(float)
    error_margin = np.abs(preds - labels)
    predicted_class = preds + 1
    true_class = labels + 1

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "true_birads": true_class,
        "pred_birads": predicted_class,
        "correct": is_correct.astype(bool),
        "confidence": confidences,
        "error_margin": error_margin,  # ±1 OK, ±4 bad
    })

    # Sort by: first by correctness (wrong first), then by confidence (low first)
    df = df.sort_values(["correct", "confidence"], ascending=[True, True])

    # Save all examples
    csv_path = Path(output_dir) / "all_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Analysis] All predictions saved → {csv_path}")

    # Save hard examples (wrong or uncertain)
    hard_examples = df[
        (df["correct"] == False) | (df["confidence"] < 0.6)
    ].reset_index(drop=True)
    
    if len(hard_examples) > 0:
        hard_csv_path = Path(output_dir) / "hard_examples.csv"
        hard_examples.to_csv(hard_csv_path, index=False)
        
        n_wrong = (~hard_examples["correct"]).sum()
        n_uncertain = (hard_examples["confidence"] < 0.6).sum()
        print(f"[Analysis] Found {len(hard_examples)} hard examples:")
        print(f"  - {n_wrong} incorrect predictions")
        print(f"  - {n_uncertain} uncertain predictions (confidence < 0.6)")
        print(f"  Saved → {hard_csv_path}")

    return df


def analyze_modality_contribution(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    cfg,
) -> Dict:
    """
    Measure which modality contributes more to the final prediction.
    Runs inference with:
      - Full model (US + Mammo)
      - US only (Mammo zeroed)
      - Mammo only (US zeroed)

    Args:
        model: trained OrdCMViT
        test_loader: test data
        device: torch device
        cfg: config object

    Returns:
        analysis: Dict with AUC/accuracy for each modality combo
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    model.eval()
    results = {
        "full": {"preds": [], "labels": []},
        "us_only": {"preds": [], "labels": []},
        "mm_only": {"preds": [], "labels": []},
    }

    with torch.no_grad():
        for batch in test_loader:
            us = batch["us"].to(device)
            mm = batch["mm"].to(device)
            labels = batch["label"].to(device)

            # Full model
            out_full = model(us, mm)
            preds_full = torch.sigmoid(out_full["main_logits"]).cpu().numpy()
            results["full"]["preds"].extend(preds_full.tolist())
            results["full"]["labels"].extend(labels.cpu().numpy().tolist())

            # US only
            out_us = model(us, torch.zeros_like(mm))
            preds_us = torch.sigmoid(out_us["main_logits"]).cpu().numpy()
            results["us_only"]["preds"].extend(preds_us.tolist())
            results["us_only"]["labels"].extend(labels.cpu().numpy().tolist())

            # Mammo only
            out_mm = model(torch.zeros_like(us), mm)
            preds_mm = torch.sigmoid(out_mm["main_logits"]).cpu().numpy()
            results["mm_only"]["preds"].extend(preds_mm.tolist())
            results["mm_only"]["labels"].extend(labels.cpu().numpy().tolist())

    # Compute metrics
    analysis = {}
    for mode_name, mode_data in results.items():
        labels = np.array(mode_data["labels"])
        probs = np.array(mode_data["preds"])
        preds = probs.argmax(axis=1)

        acc = accuracy_score(labels, preds)
        analysis[mode_name] = {
            "accuracy": float(acc),
            "n_samples": len(labels),
        }

        # AUC if multi-class
        try:
            unique_l = np.unique(labels)
            if len(unique_l) > 1:
                auc = roc_auc_score(
                    labels, probs,
                    multi_class="ovr", average="macro",
                    labels=list(unique_l),
                )
                analysis[mode_name]["auc_macro"] = float(auc)
        except Exception:
            pass

    # Print summary
    print("\n[Analysis] Modality Contribution:")
    print(f"  Full (US+Mammo)  | Acc: {analysis['full'].get('accuracy', 0):.4f} | "
          f"AUC: {analysis['full'].get('auc_macro', 0):.4f}")
    print(f"  US only          | Acc: {analysis['us_only'].get('accuracy', 0):.4f}")
    print(f"  Mammo only       | Acc: {analysis['mm_only'].get('accuracy', 0):.4f}")

    fusion_gain = (
        analysis["full"].get("accuracy", 0) -
        max(analysis["us_only"].get("accuracy", 0), analysis["mm_only"].get("accuracy", 0))
    )
    print(f"  Fusion gain: {fusion_gain:+.4f}")

    if fusion_gain < 0.02:
        print("  ⚠️  WARNING: minimal fusion benefit; check cross-attention")

    return analysis


def check_prediction_calibration(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 5,
) -> Dict:
    """
    Check if prediction confidence matches accuracy (calibration).
    In medical imaging, well-calibrated models are essential.

    Args:
        labels: [N] true class indices
        probs: [N, K] predicted probabilities
        n_bins: number of confidence bins

    Returns:
        calib: Dict with ECE and bin-wise analysis
    """
    labels = np.array(labels)
    probs = np.array(probs)
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    calib = {
        "ece": 0.0,
        "mce": 0.0,
        "bins": [],
    }

    max_error = 0.0
    for bin_idx in range(n_bins):
        bin_lower = bin_idx / n_bins
        bin_upper = (bin_idx + 1) / n_bins

        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        if in_bin.sum() == 0:
            continue

        bin_accuracy = (preds[in_bin] == labels[in_bin]).mean()
        bin_confidence = confidences[in_bin].mean()
        bin_error = abs(bin_accuracy - bin_confidence)

        bin_weight = in_bin.sum() / len(labels)
        calib["ece"] += bin_weight * bin_error
        max_error = max(max_error, bin_error)

        calib["bins"].append({
            "confidence_range": f"[{bin_lower:.2f}, {bin_upper:.2f})",
            "accuracy": float(bin_accuracy),
            "avg_confidence": float(bin_confidence),
            "calibration_error": float(bin_error),
            "n_samples": int(in_bin.sum()),
        })

    calib["mce"] = float(max_error)

    print("\n[Analysis] Prediction Calibration (Confidence vs Accuracy):")
    print(f"  Expected Calibration Error (ECE): {calib['ece']:.4f}")
    print(f"  Maximum Calibration Error (MCE):  {calib['mce']:.4f}")
    if calib["ece"] < 0.05:
        print("  ✓ Excellent calibration (ECE < 0.05)")
    elif calib["ece"] < 0.15:
        print("  ⚠️  Fair calibration (0.05 < ECE < 0.15)")
    else:
        print("  ✗ Poor calibration (ECE > 0.15) — model is overconfident or uncertain")

    for b in calib["bins"]:
        print(f"    {b['confidence_range']} | Acc: {b['accuracy']:.3f} | "
              f"Conf: {b['avg_confidence']:.3f} | "
              f"Error: {b['calibration_error']:+.3f} | N={b['n_samples']}")

    return calib


def analyze_attention_stability(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    n_runs: int = 5,
    noise_std: float = 0.01,
) -> Dict:
    """
    Check if CAM is stable (robust) by running inference with slight noise.
    Stability > 0.85 means CAM is reliable; < 0.6 means gradients are noisy.

    Args:
        model: trained model
        input_tensor: [B, 3, H, W] input image
        device: torch device
        n_runs: number of noisy forward passes
        noise_std: std of Gaussian noise to add

    Returns:
        stability: Dict with correlation scores
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    # Forward pass clean
    with torch.no_grad():
        out_clean = model(input_tensor[:, :3], input_tensor[:, 3:])
        logits_clean = out_clean["main_logits"][0].cpu().numpy()

    stability_scores = []

    with torch.no_grad():
        for run in range(n_runs):
            # Add noise
            noise = torch.randn_like(input_tensor) * noise_std
            input_noisy = (input_tensor + noise).clamp(0, 1)

            out_noisy = model(input_noisy[:, :3], input_noisy[:, 3:])
            logits_noisy = out_noisy["main_logits"][0].cpu().numpy()

            # Correlation
            corr = np.corrcoef(logits_clean, logits_noisy)[0, 1]
            stability_scores.append(corr)

    mean_stability = np.mean(stability_scores)
    std_stability = np.std(stability_scores)

    result = {
        "mean_stability": float(mean_stability),
        "std_stability": float(std_stability),
        "per_run": stability_scores,
    }

    print(f"\n[Analysis] Attention Stability (Noise Robustness):")
    print(f"  Mean correlation (clean vs noisy): {mean_stability:.4f} ± {std_stability:.4f}")
    if mean_stability > 0.85:
        print("  ✓ Excellent stability — CAM is robust to noise")
    elif mean_stability > 0.7:
        print("  ⚠️  Good stability — CAM is reasonably robust")
    else:
        print("  ✗ Poor stability — gradients may be noisy or unstable")

    return result


def save_run_report(
    output_dir: str,
    metrics: Dict,
    calib: Dict,
    modality_contrib: Dict,
    hard_examples: pd.DataFrame,
):
    """Save a comprehensive run report in JSON format."""
    import json
    import numpy as np

    def convert_to_python_types(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    report = {
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "calibration": convert_to_python_types(calib),
        "modality_contribution": convert_to_python_types(modality_contrib),
        "hard_examples_summary": {
            "total_samples": int(len(hard_examples)),
            "n_hard": int(((hard_examples["correct"] == False) | (hard_examples["confidence"] < 0.6)).sum()),
        },
    }

    report_path = Path(output_dir) / "run_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[Analysis] Run report saved → {report_path}")
