"""
src/utils/visualization.py
───────────────────────────
Gradient × Cross-Modal Attention CAM (GXCAM) for weakly supervised
lesion co-localization on BOTH US and Mammogram from BI-RADS labels only.

Algorithm:
  1. Forward pass with return_attn=True → patch tokens + cross-attn weights
  2. Backward pass through predicted BI-RADS logit → gradients w.r.t. patch tokens
  3. Gradient-weighted patch activations = feature map for US and Mammo
  4. CAM_us  = ReLU(Σ_c α_c^us · h_c^us)    (Grad-CAM style on US patch tokens)
  5. CAM_mm  = cross_attn^T · softmax(CAM_us) (propagate US saliency to Mammo space)
  6. Upsample both maps to original image resolution
  7. Overlay on original image with jet colormap

This solves the "co-localization" requirement:
  → US heatmap shows which US patches are most associated with BI-RADS grade
  → Mammo heatmap shows corresponding suspicious regions found via cross-attention
  → NO pixel annotations needed — only the BI-RADS patient label
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Optional, Tuple, List


def denormalize(tensor: torch.Tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """Convert normalized tensor [3,H,W] → uint8 RGB array [H,W,3]."""
    t = tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def compute_gradcam_patch(
    model,
    us: torch.Tensor,   # [1, 3, 224, 224]
    mm: torch.Tensor,   # [1, 3, 384, 384]
    target_class: int,  # predicted or ground-truth BI-RADS index (0-4)
    device: torch.device,
    us_grid: int = 14,
    mm_grid: int = 24,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute GXCAM activation maps for US and Mammogram.

    Returns:
        cam_us:  [us_grid, us_grid]  float [0,1]
        cam_mm:  [mm_grid, mm_grid]  float [0,1]  (via cross-attn propagation)
        cross_attn: [us_grid², mm_grid²] raw attention matrix
    """
    model.eval()
    us = us.to(device)
    mm = mm.to(device)
    us.requires_grad = False
    mm.requires_grad = False

    # Forward with attention and gradient tracking
    out = model(us, mm, return_attn=True)

    # Hook on patch tokens for gradient computation
    us_patches  = out["us_patches"]    # [1, N_us, D]
    mm_patches  = out["mm_patches"]    # [1, N_mm, D]
    cross_attn  = out["cross_attn"]    # [1, N_us, N_mm]
    main_logits = out["main_logits"]   # [1, K]

    # Register hooks to get gradients w.r.t patch tokens
    us_grads = []
    mm_grads = []

    def hook_us(grad): us_grads.append(grad)
    def hook_mm(grad): mm_grads.append(grad)

    us_patches.retain_grad()
    mm_patches.retain_grad()

    # Backward from target class logit
    score = main_logits[0, target_class]
    score.backward(retain_graph=True)

    # ── US Grad-CAM ────────────────────────────────────────────────────
    if us_patches.grad is not None:
        us_grad = us_patches.grad[0]       # [N_us, D]
        us_feat = us_patches[0].detach()   # [N_us, D]

        # Channel-wise weights = global average of gradients
        alpha_us = us_grad.mean(dim=-1, keepdim=True)   # [N_us, 1]
        cam_us_flat = F.relu(
            (alpha_us * us_feat).sum(dim=-1)             # [N_us]
        ).cpu().numpy()
        cam_us = cam_us_flat.reshape(us_grid, us_grid)  # [14, 14]
        # Normalize
        if cam_us.max() > 0:
            cam_us = cam_us / cam_us.max()
    else:
        cam_us = np.zeros((us_grid, us_grid))

    # ── Mammo Co-localization via Cross-Attention Propagation ─────────
    # Idea: which Mammo patches contributed most to the high-activation US patches?
    # cam_mm = cross_attn^T · normalized(cam_us)
    ca = cross_attn[0].detach().cpu().numpy()   # [N_us, N_mm]
    cam_us_flat_norm = cam_us.flatten()
    cam_us_flat_norm = cam_us_flat_norm / (cam_us_flat_norm.sum() + 1e-8)

    # Weighted sum: each Mammo token's importance = how much it attended to hot US tokens
    cam_mm_flat = ca.T @ cam_us_flat_norm          # [N_mm]
    cam_mm_flat = np.maximum(cam_mm_flat, 0)
    if cam_mm_flat.max() > 0:
        cam_mm_flat = cam_mm_flat / cam_mm_flat.max()

    # Mammo patches were pooled from mm_grid²=576 to us_grid²=196
    # We have cam for the pooled version → upsample back to mm_grid
    cam_mm_pooled = cam_mm_flat.reshape(us_grid, us_grid)  # [14, 14]
    cam_mm = cv2.resize(cam_mm_pooled, (mm_grid, mm_grid),
                        interpolation=cv2.INTER_LINEAR)     # [24, 24]

    return cam_us, cam_mm, ca


def overlay_cam(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay normalized CAM heatmap on RGB image."""
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_LINEAR)
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heatmap   = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay   = (alpha * heatmap + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay


def visualize_patient(
    patient_id: str,
    us_tensor: torch.Tensor,        # [1,3,224,224]
    mm_tensor: torch.Tensor,        # [1,3,384,384]
    cam_us: np.ndarray,             # [14, 14]
    cam_mm: np.ndarray,             # [24, 24]
    true_birads: int,
    pred_birads: int,
    save_path: str,
    cfg,
):
    """
    Creates a 2×3 figure:
      Row 1: Original US | US + CAM | Cross-attention matrix
      Row 2: Original Mammo | Mammo + CAM | Prediction summary
    """
    us_img = denormalize(us_tensor[0])
    mm_img = denormalize(mm_tensor[0])

    us_overlay = overlay_cam(us_img, cam_us, alpha=cfg.vis.cam_alpha)
    mm_overlay = overlay_cam(mm_img, cam_mm, alpha=cfg.vis.cam_alpha)

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

    # ── Row 1: Ultrasound ──────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(us_img)
    ax0.set_title("Ultrasound (Original)", fontsize=11)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(us_overlay)
    ax1.set_title("US — Weakly Supervised CAM\n(Lesion Localization)", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(cam_us, cmap="jet", aspect="auto")
    ax2.set_title(f"US CAM ({cam_us.shape[0]}×{cam_us.shape[1]} grid)", fontsize=11)
    ax2.axis("off")

    # ── Row 2: Mammogram ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(mm_img)
    ax3.set_title("Mammogram (Original)", fontsize=11)
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(mm_overlay)
    ax4.set_title("Mammo — Cross-Attn Co-localization\n(propagated from US CAM)", fontsize=11)
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 2])
    birads_labels = ["1\n(Benign)", "2\n(Benign)", "3\n(Probably\nBenign)",
                     "4\n(Suspicious)", "5\n(Malignant)"]
    colors = ["#27AE60", "#2ECC71", "#F39C12", "#E67E22", "#E74C3C"]
    bars = ax5.bar(birads_labels, [1]*5, color=[c + "40" for c in colors],
                   edgecolor=colors, linewidth=2)
    bars[pred_birads].set_facecolor(colors[pred_birads] + "CC")
    ax5.set_ylim(0, 1.5)
    ax5.set_title(
        f"Patient: {patient_id}\n"
        f"True BI-RADS: {true_birads+1} | Pred BI-RADS: {pred_birads+1}\n"
        f"{'✓ CORRECT' if true_birads == pred_birads else '✗ INCORRECT'}",
        fontsize=11,
        color="green" if true_birads == pred_birads else "red",
    )
    ax5.set_ylabel("Predicted class")

    plt.suptitle(
        f"OrdCMViT — Weakly Supervised Cross-Modal Co-localization",
        fontsize=13, fontweight="bold"
    )
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def generate_all_cams(
    model,
    test_loader,
    device: torch.device,
    cfg,
    n_samples: int = 20,
):
    """
    Run CAM generation on test set and save visualizations.
    Called ONCE after loading best checkpoint.
    """
    vis_dir = os.path.join(cfg.data.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    model.eval()
    count = 0

    for batch in test_loader:
        if count >= n_samples:
            break

        us     = batch["us"].to(device)
        mm     = batch["mm"].to(device)
        labels = batch["label"]
        pids   = batch["patient_id"]

        # Get prediction
        with torch.no_grad():
            out  = model(us, mm)
            pred = out["pred"][0].item()

        true_label = labels[0].item()

        # Compute CAM (requires grad)
        try:
            cam_us, cam_mm, _ = compute_gradcam_patch(
                model, us, mm,
                target_class=pred,
                device=device,
                us_grid=cfg.data.us_size // cfg.data.patch_size,
                mm_grid=cfg.data.mm_size // cfg.data.patch_size,
            )
        except Exception as e:
            print(f"[CAM] Failed for {pids[0]}: {e}")
            continue

        if cam_us is None:
            continue

        save_path = os.path.join(vis_dir, f"{pids[0]}_cam.png")
        visualize_patient(
            patient_id=pids[0],
            us_tensor=us.cpu(),
            mm_tensor=mm.cpu(),
            cam_us=cam_us,
            cam_mm=cam_mm,
            true_birads=true_label,
            pred_birads=pred,
            save_path=save_path,
            cfg=cfg,
        )
        count += 1
        print(f"[CAM] Saved: {save_path}")

    print(f"\n[CAM] Generated {count} co-localization visualizations → {vis_dir}")


def plot_training_history(history_csv_path: str, output_dir: str, test_metrics_path: Optional[str] = None):
    """
    Plot training, validation, and (optionally) test metrics from training history CSV.
    Generates subplots for loss, QWK, accuracy, and AUC over epochs plus test results.

    Args:
        history_csv_path: path to training_history.csv (generated by Trainer.save_history)
        output_dir: directory to save the PDF/PNG plots
        test_metrics_path: optional path to test_metrics.json for additional comparison
    """
    import pandas as pd
    from pathlib import Path
    import json

    if not os.path.exists(history_csv_path):
        print(f"[Plot] Training history file not found: {history_csv_path}")
        return

    df = pd.read_csv(history_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test metrics if available
    test_metrics = {}
    if test_metrics_path and os.path.exists(test_metrics_path):
        with open(test_metrics_path, "r") as f:
            test_metrics = json.load(f)

    # Determine which metrics are available
    train_cols = [c for c in df.columns if c.startswith("train_")]
    val_cols = [c for c in df.columns if c.startswith("val_")]

    # Group metrics by type
    metric_types = {}
    for col in train_cols:
        metric_name = col.replace("train_", "")
        if metric_name not in metric_types:
            metric_types[metric_name] = {}
        metric_types[metric_name]["train"] = col

    for col in val_cols:
        metric_name = col.replace("val_", "")
        if metric_name not in metric_types:
            metric_types[metric_name] = {}
        metric_types[metric_name]["val"] = col

    # Create figure with subplots
    n_metrics = len(metric_types)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    axes_flat = axes.flatten()

    # Plot each metric
    for idx, (metric_name, cols) in enumerate(sorted(metric_types.items())):
        ax = axes_flat[idx]

        if "train" in cols:
            ax.plot(
                df["epoch"], df[cols["train"]],
                marker="o", linewidth=2, label="Train", alpha=0.7
            )
        if "val" in cols:
            ax.plot(
                df["epoch"], df[cols["val"]],
                marker="s", linewidth=2, label="Val", alpha=0.7
            )

        # Add test metric as horizontal line if available
        test_key = metric_name
        if test_key in test_metrics and test_metrics[test_key] is not None:
            test_val = float(test_metrics[test_key])
            ax.axhline(
                y=test_val, color="red", linestyle="--", linewidth=2,
                label=f"Test ({test_val:.4f})", alpha=0.8
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=11)
        ax.set_title(f"{metric_name.replace('_', ' ').title()} over Epochs", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(metric_types), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    # Save figure
    png_path = output_dir / "training_curves.png"
    pdf_path = output_dir / "training_curves.pdf"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Training curves saved → {png_path}")
    print(f"[Plot] Training curves saved → {pdf_path}")
    if test_metrics:
        print(f"[Plot] Test metrics overlaid on training curves (red dashed lines)")


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    title: str = "t-SNE Visualization of Learned Embeddings",
):
    """
    Generate t-SNE visualization of embeddings colored by BI-RADS grade.
    Useful for understanding if the model separates classes in latent space.

    Args:
        embeddings: [N, D] array of embedding vectors
        labels: [N] array of class labels (0-4 for BI-RADS 1-5)
        output_dir: directory to save the plot
        title: plot title
    """
    from sklearn.manifold import TSNE

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Plot] Computing t-SNE for {len(embeddings)} samples...")
    
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color map for BI-RADS grades
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    birads_labels = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]

    # Plot each class
    for cls in range(5):
        mask = labels == cls
        if mask.sum() > 0:
            ax.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=colors[cls], label=birads_labels[cls],
                s=50, alpha=0.7, edgecolors="k", linewidth=0.5
            )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    png_path = output_dir / "tsne_embeddings.png"
    pdf_path = output_dir / "tsne_embeddings.pdf"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"[Plot] t-SNE plot saved → {png_path}")
    print(f"[Plot] t-SNE plot saved → {pdf_path}")

    # Compute class separation metric (mean pairwise distance between classes)
    print(f"[Plot] Class separation analysis:")
    for cls in range(5):
        mask = labels == cls
        if mask.sum() > 0:
            class_center = embeddings_2d[mask].mean(axis=0)
            class_spread = np.sqrt(((embeddings_2d[mask] - class_center) ** 2).sum(axis=1)).mean()
            print(f"  BI-RADS {cls+1}: {mask.sum()} samples, spread = {class_spread:.3f}")


def plot_tsne_per_modality(
    us_embeddings: np.ndarray,
    mm_embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
):
    """
    Generate side-by-side t-SNE plots for US and Mammo modality embeddings.
    Helps diagnose if one modality dominates or if they provide complementary info.

    Args:
        us_embeddings: [N, D] ultrasound embeddings
        mm_embeddings: [N, D] mammogram embeddings
        labels: [N] class labels
        output_dir: directory to save plots
    """
    from sklearn.manifold import TSNE

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Plot] Computing t-SNE for per-modality embeddings...")

    # Separate t-SNE for each modality
    tsne_us = TSNE(n_components=2, random_state=42, perplexity=min(30, len(us_embeddings) - 1))
    us_2d = tsne_us.fit_transform(us_embeddings)

    tsne_mm = TSNE(n_components=2, random_state=42, perplexity=min(30, len(mm_embeddings) - 1))
    mm_2d = tsne_mm.fit_transform(mm_embeddings)

    # Create side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    birads_labels = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]

    # Plot US
    for cls in range(5):
        mask = labels == cls
        if mask.sum() > 0:
            axes[0].scatter(
                us_2d[mask, 0], us_2d[mask, 1],
                c=colors[cls], label=birads_labels[cls],
                s=50, alpha=0.7, edgecolors="k", linewidth=0.5
            )
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=11)
    axes[0].set_title("Ultrasound Embeddings", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot Mammo
    for cls in range(5):
        mask = labels == cls
        if mask.sum() > 0:
            axes[1].scatter(
                mm_2d[mask, 0], mm_2d[mask, 1],
                c=colors[cls], label=birads_labels[cls],
                s=50, alpha=0.7, edgecolors="k", linewidth=0.5
            )
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=11)
    axes[1].set_title("Mammogram Embeddings", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    png_path = output_dir / "tsne_per_modality.png"
    pdf_path = output_dir / "tsne_per_modality.pdf"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Per-modality t-SNE plot saved → {png_path}")
    print(f"[Plot] Per-modality t-SNE plot saved → {pdf_path}")
