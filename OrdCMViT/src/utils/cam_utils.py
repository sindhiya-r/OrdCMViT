"""
src/utils/cam_utils.py
──────────────────────
CAM (Class Activation Map) utilities for ordinal/CORAL models.

This module provides functions to:
  1. Convert CORAL ordinal logits to class probabilities for standard Grad-CAM
  2. Generate Grad-CAM visualizations
  3. Handle cross-modal CAM fusion
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import cv2
import numpy as np


def ordinal_to_class_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORAL ordinal logits to class logits/probabilities.
    
    CORAL (Consistent RAnk Logits) outputs K-1 binary sigmoid outputs for K classes.
    These represent:
        f_k = sigmoid logit for "y ≥ k+1"
        
    To recover class probabilities:
        P(y=1) = 1 - P(y≥2)
        P(y=2) = P(y≥2) - P(y≥3)
        P(y=3) = P(y≥3) - P(y≥4)
        P(y=4) = P(y≥4) - P(y≥5)
        P(y=5) = P(y≥5)
    
    Args:
        logits: [B, 4] tensor of CORAL ordinal logits (B batches, 4 thresholds)
    
    Returns:
        class_probs: [B, 5] tensor of class probabilities (one per BI-RADS class)
    
    Example:
        >>> logits = torch.randn(2, 4)
        >>> class_probs = ordinal_to_class_logits(logits)
        >>> print(class_probs.shape)
        torch.Size([2, 5])
        >>> print(class_probs.sum(dim=1))  # should be close to [1, 1]
    """
    # Convert logits to probabilities: sigmoid is already applied in ordinal head
    # logits are raw (before sigmoid), so we apply sigmoid here
    probs = torch.sigmoid(logits)  # [B, 4] → P(y≥2), P(y≥3), P(y≥4), P(y≥5)
    
    # P(y ≥ k+1) for k = 0,1,2,3 (shorthand: cum_probs[k] = P(y ≥ k+2))
    # p0 = P(y=1) = 1 - P(y≥2)
    # p1 = P(y=2) = P(y≥2) * (1 - P(y≥3))
    # p2 = P(y=3) = P(y≥3) * (1 - P(y≥4))
    # p3 = P(y=4) = P(y≥4) * (1 - P(y≥5))
    # p4 = P(y=5) = P(y≥5)
    
    p0 = 1 - probs[:, 0]                              # P(y=1)
    p1 = probs[:, 0] * (1 - probs[:, 1])              # P(y=2)
    p2 = probs[:, 1] * (1 - probs[:, 2])              # P(y=3)
    p3 = probs[:, 2] * (1 - probs[:, 3])              # P(y=4)
    p4 = probs[:, 3]                                  # P(y=5)
    
    class_probs = torch.stack([p0, p1, p2, p3, p4], dim=1)  # [B, 5]
    
    return class_probs


def compute_grad_cam_ordinal(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    feature_layer: Optional[str] = None,
) -> np.ndarray:
    """
    Compute Grad-CAM for ordinal model, converting ordinal output to class logits first.
    
    Args:
        model: PyTorch model that outputs ordinal logits [B, 4]
        input_tensor: [B, 3, H, W] input image
        target_class_idx: which class to visualize (0-4 for BI-RADS 1-5)
        feature_layer: name of feature layer to extract gradients from (optional)
    
    Returns:
        cam: [H, W] heatmap array (normalized 0-1)
    """
    # This is a template function — full Grad-CAM implementation depends on
    # your model's forward hook structure, which may be in engine/trainer.py
    # or a separate visualization module.
    
    # For now, this helper provides the ordinal→class conversion needed by CAM:
    # 
    # Usage in trainer:
    #   output = model(input)  # [B, 4] ordinal logits
    #   class_probs = ordinal_to_class_logits(output)  # [B, 5] class probs
    #   target = class_probs[:, target_class_idx]  # select class logits
    #   # Now use standard Grad-CAM on 'target'
    
    pass


def blend_cam_on_image(
    image: np.ndarray,
    cam_heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay CAM heatmap on image with blending.
    
    Args:
        image: [H, W, 3] RGB image array, values 0-255
        cam_heatmap: [H, W] or [H, W, 1] normalized heatmap (0-1)
        alpha: blending factor (0=image only, 1=heatmap only)
        colormap: OpenCV colormap (default: JET)
    
    Returns:
        blended: [H, W, 3] RGB image with CAM overlay
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Ensure heatmap is single channel, 0-255
    if cam_heatmap.ndim == 3:
        cam_heatmap = cam_heatmap[:, :, 0]
    cam_heatmap = (cam_heatmap * 255).astype(np.uint8)
    
    # Resize heatmap to match image size if needed
    if cam_heatmap.shape != image.shape[:2]:
        cam_heatmap = cv2.resize(cam_heatmap, (image.shape[1], image.shape[0]))
    
    # Convert grayscale heatmap to BGR colormap
    cam_colored = cv2.applyColorMap(cam_heatmap, colormap)
    # OpenCV uses BGR, convert to RGB for consistency
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    # Blend: output = (1-alpha)*image + alpha*cam_colored
    blended = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
    
    return blended


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """
    Normalize CAM heatmap to 0-1 range.
    
    Args:
        cam: [H, W] raw heatmap (any range)
    
    Returns:
        normalized: [H, W] heatmap normalized to [0, 1]
    """
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max - cam_min < 1e-8:
        return np.zeros_like(cam)
    return (cam - cam_min) / (cam_max - cam_min)
