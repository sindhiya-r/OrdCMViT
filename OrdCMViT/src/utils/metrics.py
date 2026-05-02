"""
src/utils/metrics.py — BI-RADS evaluation metrics

This module defines the evaluation metrics used throughout the project
and documents their formulae and location.  The primary quantities are:

Metric               | Formula / description                                 | Why used
-------------------- | ----------------------------------------------------- | -----------------------------
Quadratic Weighted   | Cohen\'s \kappa with quadratic weights              | penalises rank-distant mistakes
Kappa (QWK)          |                                                       | BI-RADS ordinal labels require
                     |                                                       | strong penalty for large errors
Accuracy             | correct / total                                       | simple baseline readability
AUC (Macro OvR)      | macro average of one-vs-rest ROC AUC across 5 classes | threshold-independent
Per-class accuracy   | count(correct within class) / count(class)           | identifies weak grades
CAM entropy          | \sum p log p over normalized CAM                      | self-consistency measure
                     |                                                       | (higher BI-RADS should focus more)

For more information see the slide notes in the project documentation.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    roc_auc_score, classification_report,
    confusion_matrix,
)
from typing import Dict, List, Optional


def compute_metrics(
    labels: List[int],
    preds: List[int],
    probs: Optional[List] = None,  # [N, K] probabilities
) -> Dict:
    labels = np.array(labels)
    preds  = np.array(preds)

    acc = accuracy_score(labels, preds)
    # Quadratic Weighted Kappa — penalises rank-distant errors more
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")

    metrics = {"acc": float(acc), "qwk": float(qwk)}

    # Per-class accuracy
    for cls in range(5):
        mask = labels == cls
        if mask.sum() > 0:
            metrics[f"acc_birads{cls+1}"] = float((preds[mask] == cls).mean())

    # AUC (macro OvR) — needs probabilities
    if probs is not None:
        probs_arr = np.array(probs)  # [N, K] ordinal probabilities
        try:
            # Convert ordinal probs to class probs:
            # P(y=1) = 1 - P(y≥2)          = 1 - probs[:,0]
            # P(y=k) = P(y≥k) - P(y≥k+1)   for k=2,3,4
            # P(y=5) = P(y≥5)               = probs[:,3]
            N, K = probs_arr.shape
            class_probs = np.zeros((N, 5))
            class_probs[:, 0] = 1 - probs_arr[:, 0]
            for k in range(1, 4):
                class_probs[:, k] = probs_arr[:, k-1] - probs_arr[:, k]
            class_probs[:, 4] = probs_arr[:, 3]
            class_probs = np.clip(class_probs, 0, 1)

            # Only compute AUC if at least 2 classes present in batch
            unique_cls = np.unique(labels)
            if len(unique_cls) > 1:
                auc = roc_auc_score(
                    labels, class_probs,
                    multi_class="ovr", average="macro",
                    labels=list(unique_cls),
                )
                metrics["auc_macro"] = float(auc)
        except Exception:
            pass

    return metrics


def print_metrics(metrics: Dict, split: str = ""):
    prefix = f"[{split}] " if split else ""
    print(f"{prefix}Acc: {metrics.get('acc',0):.4f} | "
          f"QWK: {metrics.get('qwk',0):.4f} | "
          f"AUC: {metrics.get('auc_macro',0):.4f}")

def cam_entropy(heatmap: np.ndarray) -> float:
    """
    Compute entropy of CAM heatmap.
    Lower entropy = more concentrated activation (better localization).
    Higher entropy = scattered/noisy activation (worse localization).
    
    Args:
        heatmap: [H, W] normalized CAM in range [0, 1]
    
    Returns:
        entropy: float in [0, log(H*W)]
    """
    h = heatmap.flatten()
    h = h / (h.sum() + 1e-10)
    entropy = -np.sum(h * np.log(np.clip(h, 1e-10, 1.0)))
    return float(entropy)


def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 5) -> float:
    """
    Expected Calibration Error — measures if model confidence matches accuracy.
    For medical imaging, ECE < 0.05 is desired (high confidence ≈ high correctness).
    
    Args:
        labels: [N] true class indices (0-4)
        probs: [N, K] predicted class probabilities (K=5 for BI-RADS)
        n_bins: number of bins for calibration curve
    
    Returns:
        ece: float in [0, 1]
    """
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    
    ece_sum = 0.0
    for bin_idx in range(n_bins):
        bin_lower = bin_idx / n_bins
        bin_upper = (bin_idx + 1) / n_bins
        
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        if in_bin.sum() == 0:
            continue
        
        accuracy = (preds[in_bin] == labels[in_bin]).mean()
        avg_confidence = confidences[in_bin].mean()
        
        ece_sum += in_bin.sum() / len(labels) * abs(accuracy - avg_confidence)
    
    return float(ece_sum)


def class_wise_analysis(labels: np.ndarray, preds: np.ndarray) -> Dict:
    """
    Analyze per-class and per-transition errors.
    Useful for understanding which BI-RADS grades are confusing.
    
    Args:
        labels: [N] true class indices (0-4)
        preds: [N] predicted class indices (0-4)
    
    Returns:
        analysis: Dict with 'error_severity', 'confusion_patterns', etc.
    """
    labels = np.array(labels)
    preds = np.array(preds)
    
    cm = confusion_matrix(labels, preds, labels=list(range(5)))
    
    analysis = {
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": {},
        "error_transitions": {},
    }
    
    # Per-class accuracy
    for cls in range(5):
        mask = labels == cls
        if mask.sum() > 0:
            acc = (preds[mask] == cls).mean()
            analysis["per_class_accuracy"][f"birads_{cls+1}"] = float(acc)
    
    # Error transitions (BI-RADS i → j mistakes)
    for i in range(5):
        for j in range(5):
            if i != j and cm[i, j] > 0:
                error_severity = abs(i - j)
                key = f"birads_{i+1}_to_{j+1}"
                analysis["error_transitions"][key] = {
                    "count": int(cm[i, j]),
                    "severity_delta": error_severity,  # ±1 acceptable, ±4 bad
                }
    
    return analysis