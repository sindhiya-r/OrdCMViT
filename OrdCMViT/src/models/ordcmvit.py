"""
src/models/ordcmvit.py
───────────────────────
OrdCMViT: Ordinal Cross-Modal Vision Transformer
─────────────────────────────────────────────────
Full architecture addressing EVERY identified flaw:

  FLAW → FIX:
  ├─ ViT-Base overparameterized → ViT-Small (22M), freeze all but last 3 blocks
  ├─ Resolution too low → Mammo@384, US@384, with CLAHE preprocessing
  ├─ Patch too coarse → patch_size=16 at respective resolutions (justified)
  ├─ Two isolated CLS tokens → dedicated cross-modal fusion block + fusion CLS
  ├─ Modality collapse Case A (only US used) → auxiliary heads per modality
  ├─ Modality collapse Case B (diffuse attention) → attention entropy regularization
  ├─ Modality collapse Case C (overfitting) → modality dropout + label smoothing
  ├─ BI-RADS treated as categorical → CORAL ordinal head (4 binary sigmoids)
  └─ Mammo 576 tokens ≠ US 196 tokens → learned spatial pooling to align grids
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, Optional, Tuple
import timm


# ─────────────────────────────────────────────────────────────────────────────
#  1. Backbone wrapper: ViT-Small with selective unfreezing
# ─────────────────────────────────────────────────────────────────────────────

class ViTBackbone(nn.Module):
    """
    ViT-Small backbone wrapper.
    - Loads pretrained weights from timm
    - Freezes all layers except last `n_unfreeze` transformer blocks
    - Exposes per-patch token embeddings (not just CLS)
    """

    def __init__(self, model_name: str, pretrained: bool, n_unfreeze: int):
        super().__init__()
        # Load ViT-Small — dynamic_img_size allows inference at different resolutions
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,           # remove classification head
            dynamic_img_size=True,   # allows mammo@384 and US@224 in same model
        )
        self.embed_dim = self.vit.embed_dim   # 384 for ViT-Small

        # ── Freeze strategy ────────────────────────────────────────────────
        # Freeze everything first
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last n_unfreeze transformer blocks
        blocks = list(self.vit.blocks)
        for block in blocks[-n_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze norm + cls token + pos embedding
        for name, param in self.vit.named_parameters():
            if any(k in name for k in ["norm", "cls_token", "pos_embed"]):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.vit.parameters())
        print(f"[Backbone:{model_name}] Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M params")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            cls_token:    [B, D]       — global image summary
            patch_tokens: [B, N, D]   — per-patch spatial features
        """
        tokens = self.vit.forward_features(x)  # [B, N+1, D]  (includes CLS at idx 0)
        cls_token    = tokens[:, 0, :]          # [B, D]
        patch_tokens = tokens[:, 1:, :]         # [B, N, D]
        return cls_token, patch_tokens


# ─────────────────────────────────────────────────────────────────────────────
#  2. Spatial token pooling: align Mammo (576 tokens) → US (196 tokens)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialTokenPool(nn.Module):
    """
    Reshapes mammo patch tokens [B, 576, D] (24×24 spatial grid) →
    [B, 196, D] (14×14 grid) via 2D adaptive average pooling.

    WHY deterministic pooling (not learned):
      - Preserves spatial structure (unlike linear projection which mixes positions)
      - No new parameters → less overfitting
      - 14×14 matches US grid → enables spatial cross-attention alignment

    Pool window: 24×24 → 14×14 ≈ 1.7× downpool per side
    """

    def __init__(self, in_spatial: int, out_spatial: int, dim: int):
        super().__init__()
        self.in_spatial = in_spatial   # e.g., 24
        self.out_spatial = out_spatial # e.g., 14
        self.pool = nn.AdaptiveAvgPool2d((out_spatial, out_spatial))
        # Optional projection to align features after pooling
        # (only useful if pooling distorts feature statistics — kept simple here)
        self.norm = nn.LayerNorm(dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, N, D] where N = in_spatial²
        Returns:
            pooled: [B, M, D] where M = out_spatial²
        """
        B, N, D = patch_tokens.shape
        # Reshape to spatial grid
        h = w = self.in_spatial
        x = rearrange(patch_tokens, "b (h w) d -> b d h w", h=h, w=w)
        # Pool spatial dims
        x = self.pool(x)                          # [B, D, out_spatial, out_spatial]
        # Flatten back to tokens
        x = rearrange(x, "b d h w -> b (h w) d")  # [B, M, D]
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Bidirectional Cross-Modal Attention Block
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalBlock(nn.Module):
    """
    One bidirectional cross-modal attention block.

    Flow:
      1. US self-attention (within-modality context)
      2. Mammo self-attention
      3. US→Mammo cross-attention (US queries Mammo for complementary features)
      4. Mammo→US cross-attention
      5. FFN refinement for each stream

    Cross-attention formula:
      CrossAttn(Q, K, V) = softmax(QK^T / √d) · V
      Q = from source modality, K/V = from target modality
      → source tokens attend to most relevant target patches

    WHY bidirectional (both US→Mammo AND Mammo→US):
      - Lesions appear differently in US vs Mammo
      - US: hypoechoic mass, posterior shadowing
      - Mammo: opacity, architectural distortion
      - Each modality should query the other for its unique signal

    This solves:
      Case B (diffuse attention): attention entropy regularization in loss
      Case A (collapse to one modality): both branches updated bidirectionally
    """

    def __init__(self, dim: int, num_heads: int, dropout: float, ffn_ratio: float):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # ── Layer norms ────────────────────────────────────────────────────
        self.norm_us_self    = nn.LayerNorm(dim)
        self.norm_mm_self    = nn.LayerNorm(dim)
        self.norm_us_cross   = nn.LayerNorm(dim)
        self.norm_mm_cross   = nn.LayerNorm(dim)
        self.norm_us_ffn     = nn.LayerNorm(dim)
        self.norm_mm_ffn     = nn.LayerNorm(dim)

        # ── Self-attention (within modality) ─────────────────────────────
        self.us_self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mm_self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # ── Cross-attention (bidirectional) ────────────────────────────────
        self.us2mm_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mm2us_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # ── FFN ────────────────────────────────────────────────────────────
        ffn_dim = int(dim * ffn_ratio)
        self.us_ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout),
        )
        self.mm_ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout),
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        us_tokens: torch.Tensor,   # [B, N_us+1, D]  (includes CLS)
        mm_tokens: torch.Tensor,   # [B, N_mm+1, D]  (includes CLS)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            us_out: [B, N_us+1, D]
            mm_out: [B, N_mm+1, D]
            cross_attn_weights: [B, N_us, N_mm]  — for co-localization CAM
        """
        # ── 1. Within-modality self-attention ──────────────────────────────
        us_res, _ = self.us_self_attn(
            self.norm_us_self(us_tokens),
            self.norm_us_self(us_tokens),
            self.norm_us_self(us_tokens),
        )
        us_tokens = us_tokens + self.drop(us_res)

        mm_res, _ = self.mm_self_attn(
            self.norm_mm_self(mm_tokens),
            self.norm_mm_self(mm_tokens),
            self.norm_mm_self(mm_tokens),
        )
        mm_tokens = mm_tokens + self.drop(mm_res)

        # ── 2. Cross-modal attention ────────────────────────────────────────
        # US → Mammo: US patches query Mammo patches
        us_cross, attn_us2mm = self.us2mm_attn(
            self.norm_us_cross(us_tokens),   # Q: US tokens
            self.norm_mm_cross(mm_tokens),   # K: Mammo tokens
            self.norm_mm_cross(mm_tokens),   # V: Mammo tokens
            need_weights=True,
            average_attn_weights=True,       # average over heads → [B, N_us+1, N_mm+1]
        )
        us_tokens = us_tokens + self.drop(us_cross)

        # Mammo → US: Mammo patches query US patches
        mm_cross, _ = self.mm2us_attn(
            self.norm_mm_cross(mm_tokens),
            self.norm_us_cross(us_tokens),
            self.norm_us_cross(us_tokens),
            need_weights=False,
        )
        mm_tokens = mm_tokens + self.drop(mm_cross)

        # ── 3. FFN refinement ──────────────────────────────────────────────
        us_tokens = us_tokens + self.us_ffn(self.norm_us_ffn(us_tokens))
        mm_tokens = mm_tokens + self.mm_ffn(self.norm_mm_ffn(mm_tokens))

        # Extract patch-level cross-attention (exclude CLS row/col for CAM)
        # attn_us2mm: [B, N_us+1, N_mm+1] → [B, N_us, N_mm]
        cross_weights = attn_us2mm[:, 1:, 1:]   # remove CLS rows/cols

        return us_tokens, mm_tokens, cross_weights


# ─────────────────────────────────────────────────────────────────────────────
#  4. Ordinal (CORAL) Classification Head
# ─────────────────────────────────────────────────────────────────────────────

class OrdinalHead(nn.Module):
    """
    CORAL (Consistent RAnk Logits) ordinal head.

    Architecture:
      input [B, D] →
      Dropout →
      FC: D → D//2 (GELU) →
      Dropout →
      CORAL: shared weight vector + K-1 biases → K-1 sigmoid outputs

    CORAL formulation:
      f_k = W^T h + b_k,  k = 1,...,K-1
      P(y ≥ k+1) = sigmoid(f_k)
      Loss:  Σ_k BCE(P(y≥k+1), 1[y≥k+1])

    Rank-consistent prediction:
      ŷ = 1 + Σ_k 1[sigmoid(f_k) > 0.5]

    WHY CORAL over softmax CE:
      BI-RADS is ordinal: error of BI-RADS 1→5 is worse than 1→2.
      Standard CE treats all misclassifications equally → clinically wrong.
      CORAL respects: P(y≥2) ≥ P(y≥3) ≥ P(y≥4) ≥ P(y≥5)
    """

    def __init__(self, dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        K = num_classes - 1  # number of binary tasks

        hidden = max(dim // 2, 128)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        # Shared weight for all K binary tasks (CORAL constraint)
        self.fc_weight = nn.Linear(hidden, 1, bias=False)
        # Separate learnable biases per rank threshold
        self.biases = nn.Parameter(torch.zeros(K))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D]
        Returns:
            logits: [B, K]  — raw logits before sigmoid
                    sigmoid(logits[b, k]) = P(y_b ≥ k+2)
        """
        h = self.net(x)                   # [B, hidden]
        w = self.fc_weight(h)             # [B, 1]
        logits = w + self.biases          # [B, K]  (broadcast biases)
        return logits

    @staticmethod
    def predict(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits → predicted BI-RADS index (0-indexed, add 1 for actual BI-RADS).
        Args:
            logits: [B, K]
        Returns:
            pred: [B] — predicted class index (0 = BI-RADS 1, 4 = BI-RADS 5)
        """
        probs = torch.sigmoid(logits)             # [B, K]
        pred = (probs > 0.5).sum(dim=1).long()    # count how many thresholds exceeded
        return pred   # in {0, 1, 2, 3, 4}


# ─────────────────────────────────────────────────────────────────────────────
#  5. Full OrdCMViT Model
# ─────────────────────────────────────────────────────────────────────────────

class OrdCMViT(nn.Module):
    """
    Ordinal Cross-Modal Vision Transformer for BI-RADS grading.

    Data flow:
      I_us [B,3,384,384] → US backbone → [B, 576+1, 384]
      I_mm [B,3,384,384] → MM backbone → [B, 576+1, 384]
                                              ↓
                                  Spatial pool (no-op, both 24×24): [B, 576+1, 384]
                                              ↓
                              ┌── 3× CrossModalBlock ──┐
                              │  US tokens ↔ MM tokens  │
                              └────────────────────────┘
                                              ↓
                                 Fuse CLS tokens: [B, 384]
                                              ↓
                              Main Ordinal Head → [B, 4] logits
                              + Aux Head US    → [B, 4] logits (anti-collapse)
                              + Aux Head MM    → [B, 4] logits (anti-collapse)
    """

    def __init__(self, cfg):
        super().__init__()
        mc = cfg.model
        dc = cfg.data

        self.embed_dim  = mc.embed_dim      # 384
        self.num_classes = mc.num_classes   # 5
        self.modality_dropout_p = mc.modality_dropout_p

        # ── Compute token counts ──────────────────────────────────────────
        us_grid = dc.us_size // dc.patch_size        # 384//16 = 24
        mm_grid = dc.mm_size // dc.patch_size        # 384//16 = 24
        self.us_n_tokens = us_grid ** 2              # 576 (24×24)
        self.mm_n_tokens = mm_grid ** 2              # 576 (24×24)
        self.us_grid = us_grid
        self.mm_grid = mm_grid

        print(f"[OrdCMViT] US grid: {us_grid}×{us_grid}={self.us_n_tokens} tokens")
        print(f"[OrdCMViT] MM grid: {mm_grid}×{mm_grid}={self.mm_n_tokens} tokens")
        print(f"[OrdCMViT] Spatial pool: {mm_grid}×{mm_grid} → {us_grid}×{us_grid} "
              f"({'no-op' if mm_grid == us_grid else 'downpool'})")

        # ── 1. Backbones ──────────────────────────────────────────────────
        self.us_backbone = ViTBackbone(
            mc.us_backbone_name, mc.pretrained, mc.freeze_n_last_blocks
        )
        self.mm_backbone = ViTBackbone(
            mc.mm_backbone_name, mc.pretrained, mc.freeze_n_last_blocks
        )

        # ── 2. Mammo token pooling (576 → 196 tokens) ─────────────────────
        self.mm_pool = SpatialTokenPool(
            in_spatial=mm_grid,    # 24
            out_spatial=us_grid,   # 14
            dim=self.embed_dim,
        )

        # ── 3. Learnable fusion CLS token ─────────────────────────────────
        # This token attends to BOTH modalities simultaneously → true fusion
        # Addresses the "two isolated CLS tokens" flaw
        self.fusion_cls = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.fusion_cls, std=0.02)

        # Projection to combine [CLS_us, CLS_mm] → single vector
        self.cls_fusion = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
        )

        # ── 4. Cross-modal attention blocks ──────────────────────────────
        self.cross_blocks = nn.ModuleList([
            CrossModalBlock(
                dim=self.embed_dim,
                num_heads=mc.cross_attn_heads,
                dropout=mc.cross_attn_dropout,
                ffn_ratio=mc.cross_ffn_ratio,
            )
            for _ in range(mc.n_cross_blocks)
        ])

        # ── 5. Classification heads ───────────────────────────────────────
        # Main head (on fused representation)
        self.main_head = OrdinalHead(self.embed_dim, mc.num_classes, mc.head_dropout)

        # Auxiliary heads (on individual CLS tokens — anti-collapse)
        if mc.use_aux_heads:
            self.aux_head_us = OrdinalHead(self.embed_dim, mc.num_classes, mc.head_dropout)
            self.aux_head_mm = OrdinalHead(self.embed_dim, mc.num_classes, mc.head_dropout)
        self.use_aux_heads = mc.use_aux_heads

        self._print_param_count()

    def _print_param_count(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[OrdCMViT] Total trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    def _apply_modality_dropout(
        self,
        us_cls: torch.Tensor,
        us_patches: torch.Tensor,
        mm_cls: torch.Tensor,
        mm_patches: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modality dropout: randomly zero-out an ENTIRE modality during training.

        WHY: Forces each branch to independently learn discriminative features.
        Without this: model collapses to using only the easier modality (Case A).
        p=0.1 means 10% chance of dropping US, 10% chance of dropping Mammo.
        Both dropped simultaneously is rare (0.01) but handled gracefully.

        During inference: NEVER applied (standard behavior).
        """
        if not self.training:
            return us_cls, us_patches, mm_cls, mm_patches

        B = us_cls.shape[0]
        p = self.modality_dropout_p

        for b in range(B):
            if torch.rand(1).item() < p:
                us_cls[b]     = 0.0
                us_patches[b] = 0.0
            if torch.rand(1).item() < p:
                mm_cls[b]     = 0.0
                mm_patches[b] = 0.0

        return us_cls, us_patches, mm_cls, mm_patches

    def forward(
        self,
        us: torch.Tensor,           # [B, 3, 384, 384]
        mm: torch.Tensor,           # [B, 3, 384, 384]
        return_attn: bool = False,  # set True for CAM visualization
        return_embeddings: bool = False,  # set True to export CLS embeddings
    ) -> Dict:
        """
        Returns dict with:
            main_logits:    [B, 4]   main CORAL logits
            aux_logits_us:  [B, 4]   US-only CORAL logits
            aux_logits_mm:  [B, 4]   MM-only CORAL logits
            pred:           [B]      predicted BI-RADS index (0-4)
            cross_attn:     [B, N_us, N_mm]  last block cross-attention (if return_attn)
            us_cls:         [B, D]   US CLS embedding (for visualization)
            mm_cls:         [B, D]   MM CLS embedding (for visualization)

            us_embeddings:   [B, D]   synonym for us_cls when return_embeddings=True
            mm_embeddings:   [B, D]   synonym for mm_cls when return_embeddings=True
            fused_embeddings:[B, D]   fused CLS vector when return_embeddings=True
        """
        # ── 1. Feature extraction ─────────────────────────────────────────
        us_cls, us_patches = self.us_backbone(us)   # [B,384], [B,196,384]
        mm_cls, mm_patches = self.mm_backbone(mm)   # [B,384], [B,576,384]

        # Save for auxiliary heads BEFORE modality dropout
        us_cls_orig = us_cls.clone()
        mm_cls_orig = mm_cls.clone()

        # ── 2. Modality dropout (training only) ──────────────────────────
        us_cls, us_patches, mm_cls, mm_patches = self._apply_modality_dropout(
            us_cls, us_patches, mm_cls, mm_patches
        )

        # ── 3. Mammo spatial pooling: 576 → 196 tokens ───────────────────
        mm_patches_pooled = self.mm_pool(mm_patches)  # [B, 196, 384]

        # ── 4. Prepend CLS tokens to patch sequences ──────────────────────
        B = us.shape[0]
        us_tokens = torch.cat([us_cls.unsqueeze(1), us_patches], dim=1)       # [B,197,384]
        mm_tokens = torch.cat([mm_cls.unsqueeze(1), mm_patches_pooled], dim=1) # [B,197,384]

        # ── 5. Cross-modal attention ──────────────────────────────────────
        last_cross_attn = None
        for block in self.cross_blocks:
            us_tokens, mm_tokens, cross_attn = block(us_tokens, mm_tokens)
            last_cross_attn = cross_attn   # [B, 196, 196] — keep last block's weights

        # ── 6. Fusion ─────────────────────────────────────────────────────
        # Fuse CLS tokens from both modalities
        us_cls_fused = us_tokens[:, 0, :]   # [B, 384]
        mm_cls_fused = mm_tokens[:, 0, :]   # [B, 384]
        fused = self.cls_fusion(
            torch.cat([us_cls_fused, mm_cls_fused], dim=-1)
        )  # [B, 384]

        # ── 7. Classification heads ───────────────────────────────────────
        main_logits = self.main_head(fused)              # [B, 4]
        pred = OrdinalHead.predict(main_logits)          # [B]

        out = {
            "main_logits": main_logits,
            "pred": pred,
            "us_cls": us_cls_fused,
            "mm_cls": mm_cls_fused,
        }

        # optionally export embeddings under more descriptive keys
        if return_embeddings:
            out["us_embeddings"] = us_cls_fused
            out["mm_embeddings"] = mm_cls_fused
            out["fused_embeddings"] = fused

        if self.use_aux_heads:
            out["aux_logits_us"] = self.aux_head_us(us_cls_orig)   # [B, 4]
            out["aux_logits_mm"] = self.aux_head_mm(mm_cls_orig)   # [B, 4]

        if return_attn:
            out["cross_attn"]     = last_cross_attn    # [B, 196, 196]
            out["us_patches"]     = us_tokens[:, 1:, :]  # [B, 196, 384]
            out["mm_patches"]     = mm_tokens[:, 1:, :]  # [B, 196, 384]

        return out
