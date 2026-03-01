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
from typing import Dict, Optional, Tuple


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
