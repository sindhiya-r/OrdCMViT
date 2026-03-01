"""
src/losses/losses.py
─────────────────────
Loss functions for OrdCMViT:
  1. CoralLoss       — ordinal regression (main task)
  2. AttentionEntropyLoss — prevent diffuse/collapsed cross-modal attention (Case B)
  3. TotalLoss       — orchestrates all losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  1. CORAL Ordinal Loss
# ─────────────────────────────────────────────────────────────────────────────

class CoralLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) loss for ordinal classification.

    Reference: Cao et al. "Rank Consistent Ordinal Regression for Neural Networks"
               Pattern Recognition Letters, 2020.

    Given:
      logits: [B, K]    — raw sigmoid inputs, one per rank threshold
      ordinal_targets: [B, K] — binary: 1 if y ≥ (k+2), else 0

    Loss:
      L = -(1/K) Σ_k [ t_k * log σ(f_k) + (1-t_k) * log(1-σ(f_k)) ]

    With optional:
      - class weighting (handle BI-RADS imbalance)
      - label smoothing (0.05 — reduces overconfidence on small dataset)

    WHY ordinal vs categorical:
      BI-RADS error BI-RADS3→BI-RADS5 is clinically worse than BI-RADS3→BI-RADS4.
      Standard CE treats both equally. CORAL penalises rank-jumping more.
    """

    def __init__(
        self,
        label_smoothing: float = 0.05,
        class_weights: Optional[torch.Tensor] = None,  # [num_classes]
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        logits: torch.Tensor,           # [B, K]
        ordinal_targets: torch.Tensor,  # [B, K]  float: {0.0, 1.0}
        class_labels: Optional[torch.Tensor] = None,  # [B] int: 0-4 for weighting
    ) -> torch.Tensor:
        B, K = logits.shape

        # Label smoothing: soft targets
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            targets_smooth = ordinal_targets * (1 - eps) + eps * 0.5
        else:
            targets_smooth = ordinal_targets

        # BCE per rank threshold
        loss_per_rank = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction="none"
        )  # [B, K]

        # Average over rank thresholds
        loss_per_sample = loss_per_rank.mean(dim=1)  # [B]

        # Optional: weight by class
        if self.class_weights is not None and class_labels is not None:
            weights = self.class_weights[class_labels]  # [B]
            loss = (loss_per_sample * weights).mean()
        else:
            loss = loss_per_sample.mean()

        return loss


# ─────────────────────────────────────────────────────────────────────────────
#  2. Attention Entropy Regularization
# ─────────────────────────────────────────────────────────────────────────────

class AttentionEntropyLoss(nn.Module):
    """
    Penalises cross-modal attention that is either:
      (A) Too uniform (diffuse) → entropy too HIGH → model attends everywhere = random
      (B) Too peaked → entropy too LOW → model fixates on one patch always

    Clinically motivated target entropy range:
      - Lesion typically occupies 5-20% of image area
      - At 14×14 grid = 196 tokens, 5-20% = 10-40 tokens
      - Ideal entropy for attention over 20/196 tokens ≈ 3.0-3.5 nats (natural log)
      - We define a target entropy range [entropy_min, entropy_max]
      - Loss = max(0, entropy - max)² + max(0, min - entropy)²
        → penalises both extremes, allows entropy in [min, max] range

    This directly addresses Case B (diffuse attention collapse).
    """

    def __init__(
        self,
        entropy_min: float = 2.5,   # minimum acceptable cross-modal attention entropy
        entropy_max: float = 4.5,   # maximum acceptable cross-modal attention entropy
        weight: float = 0.01,       # small weight — this is a regularizer, not primary loss
    ):
        super().__init__()
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        self.weight = weight

    def forward(self, cross_attn: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            cross_attn: [B, N_us, N_mm] — raw attention weights (before softmax normalization)
                        OR None (if return_attn was False)
        Returns:
            scalar entropy regularization loss
        """
        if cross_attn is None:
            return torch.tensor(0.0)

        # Normalize each US token's attention distribution over Mammo tokens
        attn_probs = F.softmax(cross_attn, dim=-1)  # [B, N_us, N_mm]

        # Compute entropy per US token per sample
        # H = -Σ p log(p + ε)
        eps = 1e-9
        entropy = -(attn_probs * torch.log(attn_probs + eps)).sum(dim=-1)  # [B, N_us]

        # Average over US tokens and batch
        mean_entropy = entropy.mean()

        # Hinge loss for entropy range
        loss_high = F.relu(mean_entropy - self.entropy_max) ** 2
        loss_low  = F.relu(self.entropy_min - mean_entropy) ** 2

        return self.weight * (loss_high + loss_low)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Total Loss Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TotalLoss(nn.Module):
    """
    Combines all losses:
      L_total = L_main                         (CORAL on fused CLS)
              + lambda_aux * (L_aux_us + L_aux_mm)  (auxiliary heads per modality)
              + L_entropy                       (attention entropy regularization)

    Auxiliary head losses (lambda_aux = 0.3):
      WHY: Enforce both modalities independently learn discriminative features.
      If aux losses are absent: gradient flows only through fusion path.
      US branch can become a dead-end encoder that outputs constant features.
      Auxiliary heads keep both branches "alive" throughout training.

    Output dict keys:
      "total", "main", "aux_us", "aux_mm", "entropy"
      All scalar tensors — logged to tensorboard.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        aux_loss_weight: float = 0.3,
        use_aux: bool = True,
        entropy_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.coral = CoralLoss(label_smoothing, class_weights)
        self.entropy_reg = AttentionEntropyLoss(weight=entropy_loss_weight)
        self.aux_loss_weight = aux_loss_weight
        self.use_aux = use_aux

    def forward(
        self,
        model_out: Dict,
        ordinal_targets: torch.Tensor,    # [B, 4]
        class_labels: torch.Tensor,       # [B] int
    ) -> Dict[str, torch.Tensor]:
        # Main CORAL loss on fused representation
        loss_main = self.coral(
            model_out["main_logits"], ordinal_targets, class_labels
        )

        loss_dict = {"main": loss_main}
        total = loss_main

        # Auxiliary modality losses
        if self.use_aux and "aux_logits_us" in model_out:
            loss_aux_us = self.coral(
                model_out["aux_logits_us"], ordinal_targets, class_labels
            )
            loss_aux_mm = self.coral(
                model_out["aux_logits_mm"], ordinal_targets, class_labels
            )
            loss_dict["aux_us"] = loss_aux_us
            loss_dict["aux_mm"] = loss_aux_mm
            total = total + self.aux_loss_weight * (loss_aux_us + loss_aux_mm)
        else:
            loss_dict["aux_us"] = torch.tensor(0.0)
            loss_dict["aux_mm"] = torch.tensor(0.0)

        # Attention entropy regularization
        cross_attn = model_out.get("cross_attn", None)
        loss_entropy = self.entropy_reg(cross_attn)
        loss_dict["entropy"] = loss_entropy
        total = total + loss_entropy

        loss_dict["total"] = total
        return loss_dict
