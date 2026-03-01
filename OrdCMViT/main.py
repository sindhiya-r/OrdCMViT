"""
main.py
────────
OrdCMViT: Ordinal Cross-Modal Vision Transformer for BCMID Breast Cancer BI-RADS Grading

SINGLE ENTRY POINT — run once:
    python main.py --data_root /path/to/BCMID --labels_csv /path/to/BCMID_labels.csv

What this script does end-to-end:
  1.  Parse args + override config
  2.  Set global seed (reproducibility)
  3.  Build data pipeline (BCMID dataset + augmentations + dataloaders)
  4.  Build OrdCMViT model + transfer to device
  5.  Build loss function (CORAL + auxiliary + entropy)
  6.  Train with early stopping + checkpointing + tensorboard
  7.  Load best checkpoint
  8.  Evaluate on held-out test set (full report)
  9.  Generate weakly supervised co-localization CAMs for N test patients
  10. Save all results to CSV + JSON
  11. Print final summary

Expected output directory structure:
  runs/ordcmvit/
    ├── checkpoints/
    │     ├── best.pth
    │     └── last.pth
    ├── visualizations/
    │     ├── P001_cam.png   (2×3 grid: US + CAM, Mammo + CAM, prediction)
    │     └── ...
    ├── logs/                (TensorBoard event files)
    └── test_results.csv
"""

import os
import sys
import json
import argparse
import warnings
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, DataConfig, ModelConfig, TrainingConfig, VisualizationConfig
from src.data.bcmid import build_dataloaders
from src.data.transforms import (
    build_train_transform_us, build_train_transform_mm,
    build_val_transform_us, build_val_transform_mm,
)
from src.models.ordcmvit import OrdCMViT, OrdinalHead
from src.losses.losses import TotalLoss
from src.engine.trainer import Trainer, evaluate
from src.utils.seed import set_seed, get_device
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.visualization import generate_all_cams


# ─────────────────────────────────────────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="OrdCMViT: BCMID Breast Cancer BI-RADS Grading")

    # Required
    parser.add_argument("--data_root",   type=str, default="./BCMID",
                        help="Root directory of BCMID dataset")
    parser.add_argument("--labels_csv",  type=str, default="./BCMID/BCMID_labels.csv",
                        help="Path to BCMID_labels.csv")
    parser.add_argument("--output_dir",  type=str, default="./runs/ordcmvit")

    # Training overrides
    parser.add_argument("--max_epochs",  type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--val_fold",    type=int, default=0,
                        help="Which fold (0-4) to use as validation")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no_pretrain", action="store_true",
                        help="Disable ImageNet pretrained weights (for debugging)")

    # Execution flags
    parser.add_argument("--skip_train",  action="store_true",
                        help="Skip training, only evaluate (requires best.pth)")
    parser.add_argument("--no_vis",      action="store_true",
                        help="Skip CAM visualization (faster)")
    parser.add_argument("--n_vis",       type=int, default=20,
                        help="Number of test patients to visualize")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Config builder from args
# ─────────────────────────────────────────────────────────────────────────────

def build_config(args) -> Config:
    cfg = Config()
    cfg.data.data_root   = args.data_root
    cfg.data.labels_csv  = args.labels_csv
    cfg.data.output_dir  = args.output_dir
    cfg.data.val_fold    = args.val_fold
    cfg.data.batch_size  = args.batch_size
    cfg.data.random_seed = args.seed
    cfg.training.max_epochs = args.max_epochs
    cfg.model.pretrained = not args.no_pretrain
    cfg.vis.n_vis_samples = args.n_vis
    # Re-create output dirs
    for d in ["checkpoints", "visualizations", "logs"]:
        os.makedirs(os.path.join(cfg.data.output_dir, d), exist_ok=True)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Final evaluation: full test set report
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_test_evaluation(model, test_loader, device, cfg, loss_fn):
    """
    Full evaluation on held-out test set:
      - Per-class accuracy
      - QWK (Quadratic Weighted Kappa)
      - AUC macro
      - Confusion matrix plot
      - Per-patient predictions CSV
    """
    model.eval()
    tc = cfg.training
    from torch.cuda.amp import autocast

    all_pids, all_preds, all_labels, all_probs = [], [], [], []

    for batch in test_loader:
        us      = batch["us"].to(device)
        mm      = batch["mm"].to(device)
        ordinal = batch["ordinal"].to(device)
        labels  = batch["label"].to(device)

        with autocast(enabled=tc.use_amp):
            out  = model(us, mm)

        preds = OrdinalHead.predict(out["main_logits"]).cpu()
        probs = torch.sigmoid(out["main_logits"]).cpu().numpy()

        all_pids.extend(batch["patient_id"])
        all_preds.extend(preds.numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

    # ── Metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    print("\n" + "="*60)
    print("FINAL TEST SET RESULTS")
    print("="*60)
    print_metrics(metrics, split="TEST")
    print("\nClassification Report (BI-RADS 1-5):")
    print(classification_report(
        all_labels, all_preds,
        target_names=[f"BI-RADS {i+1}" for i in range(5)],
        labels=list(range(5)),
        zero_division=0,
    ))

    # ── Save per-patient CSV ───────────────────────────────────────────
    df_out = pd.DataFrame({
        "patient_id": all_pids,
        "true_birads": [l+1 for l in all_labels],
        "pred_birads": [p+1 for p in all_preds],
        "correct": [l==p for l,p in zip(all_labels, all_preds)],
    })
    csv_path = os.path.join(cfg.data.output_dir, "test_results.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nPer-patient results saved → {csv_path}")

    # ── Confusion matrix plot ──────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(5)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[f"Pred\nBI-RADS {i+1}" for i in range(5)],
        yticklabels=[f"True\nBI-RADS {i+1}" for i in range(5)],
        ax=ax,
    )
    ax.set_title(f"OrdCMViT — Test Set Confusion Matrix\n"
                 f"QWK={metrics.get('qwk',0):.4f} | AUC={metrics.get('auc_macro',0):.4f}")
    plt.tight_layout()
    cm_path = os.path.join(cfg.data.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved → {cm_path}")

    # ── Save metrics JSON ──────────────────────────────────────────────
    metrics_path = os.path.join(cfg.data.output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: round(v, 6) for k, v in metrics.items() if isinstance(v, float)}, f, indent=2)
    print(f"Metrics JSON saved → {metrics_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    cfg  = build_config(args)

    print("\n" + "="*60)
    print("OrdCMViT — Breast Cancer BI-RADS Grading (BCMID)")
    print("="*60)
    print(f"Data root:    {cfg.data.data_root}")
    print(f"Output dir:   {cfg.data.output_dir}")
    print(f"Max epochs:   {cfg.training.max_epochs}")
    print(f"Batch size:   {cfg.data.batch_size} × {cfg.training.grad_accum_steps} = "
          f"{cfg.data.batch_size * cfg.training.grad_accum_steps} effective")
    print(f"Val fold:     {cfg.data.val_fold} / {cfg.data.n_folds}")

    # ── 1. Setup ──────────────────────────────────────────────────────
    set_seed(cfg.data.random_seed)
    device = get_device()

    # ── 2. Data pipeline ─────────────────────────────────────────────
    print("\n[Data] Building BCMID dataloaders...")
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        data_root   = cfg.data.data_root,
        labels_csv  = cfg.data.labels_csv,
        cfg         = cfg,
        transform_train_us = build_train_transform_us(cfg),
        transform_train_mm = build_train_transform_mm(cfg),
        transform_val_us   = build_val_transform_us(cfg),
        transform_val_mm   = build_val_transform_mm(cfg),
    )
    class_weights = class_weights.to(device)

    # ── 3. Model ──────────────────────────────────────────────────────
    print("\n[Model] Building OrdCMViT...")
    model = OrdCMViT(cfg).to(device)

    # ── 4. Loss function ──────────────────────────────────────────────
    loss_fn = TotalLoss(
        class_weights    = class_weights if cfg.training.use_class_weights else None,
        label_smoothing  = cfg.model.label_smoothing,
        aux_loss_weight  = cfg.model.aux_loss_weight,
        use_aux          = cfg.model.use_aux_heads,
        entropy_loss_weight = 0.01,
    )

    best_ckpt_path = os.path.join(cfg.data.output_dir, "checkpoints", "best.pth")

    # ── 5. Training ───────────────────────────────────────────────────
    if not args.skip_train:
        print("\n[Train] Starting training...")
        trainer = Trainer(model, train_loader, val_loader, loss_fn, cfg, device)
        best_val_metrics = trainer.train()
        print("\n[Train] Best validation metrics:")
        print_metrics(best_val_metrics, split="VAL")
    else:
        print("\n[Train] Skipped (--skip_train flag)")

    # ── 6. Load best checkpoint ───────────────────────────────────────
    if os.path.exists(best_ckpt_path):
        print(f"\n[Eval] Loading best checkpoint: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        saved_epoch = ckpt.get("epoch", "?")
        print(f"[Eval] Checkpoint from epoch {saved_epoch}")
    else:
        print(f"[Eval] WARNING: No checkpoint found at {best_ckpt_path}")
        print("[Eval] Using current model weights for evaluation")

    # ── 7. Test evaluation ────────────────────────────────────────────
    print("\n[Eval] Running test set evaluation...")
    test_metrics = run_test_evaluation(model, test_loader, device, cfg, loss_fn)

    # ── 8. Visualizations ─────────────────────────────────────────────
    if not args.no_vis:
        print(f"\n[CAM] Generating weakly supervised co-localization maps "
              f"for {cfg.vis.n_vis_samples} test patients...")
        generate_all_cams(model, test_loader, device, cfg, cfg.vis.n_vis_samples)
    else:
        print("\n[CAM] Skipped (--no_vis flag)")

    # ── 9. Final summary ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Test Accuracy:        {test_metrics.get('acc', 0):.4f}")
    print(f"Test QWK:             {test_metrics.get('qwk', 0):.4f}")
    print(f"Test AUC (macro):     {test_metrics.get('auc_macro', 0):.4f}")
    print(f"\nAll outputs in:  {cfg.data.output_dir}")
    print("  checkpoints/best.pth     — best model weights")
    print("  test_results.csv         — per-patient predictions")
    print("  test_metrics.json        — evaluation metrics")
    print("  confusion_matrix.png     — confusion matrix")
    print("  visualizations/*.png     — co-localization CAM maps")
    print("  logs/                    — TensorBoard logs (run: tensorboard --logdir logs/)")


if __name__ == "__main__":
    main()
