"""
src/utils/metrics.py — BI-RADS evaluation metrics
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

            # Only compute AUC if all classes present
            unique_cls = np.unique(labels)
            if len(unique_cls) > 1:
                auc = roc_auc_score(
                    labels, class_probs[:, :len(unique_cls)],
                    multi_class="ovr", average="macro",
                    labels=list(range(len(unique_cls))),
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
