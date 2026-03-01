"""
src/engine/trainer.py
──────────────────────
Training engine with:
  - Differential learning rates (backbone vs new layers)
  - Gradient accumulation (effective batch = 32)
  - Mixed precision (AMP)
  - Gradient clipping
  - Cosine LR schedule with warmup
  - Early stopping
  - Checkpoint saving (best + last)
  - Tensorboard logging
"""

import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple
import numpy as np

from ..models.ordcmvit import OrdCMViT, OrdinalHead
from ..losses.losses import TotalLoss
from ..utils.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Optimizer builder — differential learning rates
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: OrdCMViT, cfg) -> torch.optim.AdamW:
    """
    Differential LR:
      - Backbone unfrozen blocks: backbone_lr = 5e-6  (pretrained → small LR)
      - New layers (cross-attention, heads): new_layers_lr = 1e-4

    WHY differential LR:
      Unfrozen backbone blocks have good pretrained features → small updates preserve them.
      New cross-modal layers and heads start from scratch → need larger LR to converge.
      Applying same LR to both would either:
        - Destroy pretrained features (LR too high) OR
        - Fail to train new layers (LR too low)
    """
    tc = cfg.training

    backbone_params = []
    new_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            new_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": tc.backbone_lr,   "name": "backbone"},
        {"params": new_params,      "lr": tc.new_layers_lr, "name": "new_layers"},
    ]

    n_backbone = sum(p.numel() for p in backbone_params)
    n_new      = sum(p.numel() for p in new_params)
    print(f"[Optimizer] Backbone params: {n_backbone/1e6:.2f}M @ lr={tc.backbone_lr}")
    print(f"[Optimizer] New layer params: {n_new/1e6:.2f}M @ lr={tc.new_layers_lr}")

    return torch.optim.AdamW(
        param_groups,
        betas=tc.betas,
        weight_decay=tc.weight_decay,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Warmup-Cosine Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup for warmup_epochs, then cosine decay to 0."""

    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, min_lr_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._epoch = 0

    def step(self):
        self._epoch += 1
        epoch = self._epoch

        if epoch <= self.warmup_epochs:
            scale = epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + np.cos(np.pi * progress)
            )

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
#  Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = None
        self.counter   = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────────────────────────────────────
#  Train / Eval loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: OrdCMViT,
    loader,
    loss_fn: TotalLoss,
    optimizer: torch.optim.AdamW,
    scaler: GradScaler,
    device: torch.device,
    cfg,
    epoch: int,
) -> Dict:
    model.train()
    tc = cfg.training

    all_preds, all_labels = [], []
    epoch_losses = {k: 0.0 for k in ["total", "main", "aux_us", "aux_mm", "entropy"]}
    n_steps = 0

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        us    = batch["us"].to(device, non_blocking=True)
        mm    = batch["mm"].to(device, non_blocking=True)
        ordinal = batch["ordinal"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)

        # ── Forward + loss ────────────────────────────────────────────────
        # Enable cross_attn only every 10 steps to reduce memory (entropy loss)
        compute_entropy = (step % 10 == 0)

        with autocast(enabled=tc.use_amp):
            out = model(us, mm, return_attn=compute_entropy)
            if not compute_entropy:
                out["cross_attn"] = None
            loss_dict = loss_fn(out, ordinal, labels)
            loss = loss_dict["total"] / tc.grad_accum_steps  # scale for accumulation

        # ── Backward ─────────────────────────────────────────────────────
        scaler.scale(loss).backward()

        if (step + 1) % tc.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ── Accumulate metrics ────────────────────────────────────────────
        for k in epoch_losses:
            epoch_losses[k] += loss_dict[k].item()
        n_steps += 1

        preds = OrdinalHead.predict(out["main_logits"]).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    # Handle remaining steps if not divisible by grad_accum_steps
    if n_steps % tc.grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    metrics = compute_metrics(all_labels, all_preds)
    metrics.update({f"loss_{k}": v / n_steps for k, v in epoch_losses.items()})
    return metrics


@torch.no_grad()
def evaluate(
    model: OrdCMViT,
    loader,
    loss_fn: TotalLoss,
    device: torch.device,
    cfg,
    split: str = "val",
) -> Dict:
    model.eval()
    tc = cfg.training

    all_preds, all_labels, all_probs = [], [], []
    epoch_losses = {k: 0.0 for k in ["total", "main", "aux_us", "aux_mm", "entropy"]}
    n_steps = 0

    for batch in loader:
        us    = batch["us"].to(device, non_blocking=True)
        mm    = batch["mm"].to(device, non_blocking=True)
        ordinal = batch["ordinal"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=tc.use_amp):
            out = model(us, mm, return_attn=False)
            loss_dict = loss_fn(out, ordinal, labels)

        for k in epoch_losses:
            epoch_losses[k] += loss_dict[k].item()
        n_steps += 1

        preds = OrdinalHead.predict(out["main_logits"]).cpu().numpy()
        probs = torch.sigmoid(out["main_logits"]).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics.update({f"loss_{k}": v / n_steps for k, v in epoch_losses.items()})
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Main Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model: OrdCMViT,
        train_loader,
        val_loader,
        loss_fn: TotalLoss,
        cfg,
        device: torch.device,
    ):
        self.model       = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.loss_fn      = loss_fn
        self.cfg          = cfg
        self.device       = device

        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=cfg.training.warmup_epochs,
            max_epochs=cfg.training.max_epochs,
        )
        self.scaler = GradScaler(enabled=cfg.training.use_amp)
        self.early_stopper = EarlyStopping(
            patience=cfg.training.patience,
            min_delta=cfg.training.min_delta,
            mode="max",
        )

        ckpt_dir = os.path.join(cfg.data.output_dir, "checkpoints")
        log_dir  = os.path.join(cfg.data.output_dir, "logs")
        self.writer = SummaryWriter(log_dir)
        self.ckpt_dir = ckpt_dir
        self.best_metric = -1.0

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_epoch": self.scheduler._epoch,
            "metrics": metrics,
            "cfg": self.cfg,
        }
        if is_best:
            path = os.path.join(self.ckpt_dir, "best.pth")
            torch.save(state, path)
            print(f"  ✓ Best model saved → {path}")
        if self.cfg.training.save_last:
            torch.save(state, os.path.join(self.ckpt_dir, "last.pth"))

    def _log(self, split: str, metrics: Dict, epoch: int):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"{split}/{k}", v, epoch)

    def train(self) -> Dict:
        tc = self.cfg.training
        print(f"\n{'='*60}")
        print(f"Training OrdCMViT for {tc.max_epochs} epochs")
        print(f"Early stopping patience: {tc.patience}")
        print(f"{'='*60}\n")

        best_metrics = {}
        for epoch in range(1, tc.max_epochs + 1):
            t0 = time.time()

            # ── Train ──────────────────────────────────────────────────
            train_metrics = train_one_epoch(
                self.model, self.train_loader, self.loss_fn,
                self.optimizer, self.scaler, self.device, self.cfg, epoch
            )
            self._log("train", train_metrics, epoch)

            # ── Validate ──────────────────────────────────────────────
            val_metrics = evaluate(
                self.model, self.val_loader, self.loss_fn,
                self.device, self.cfg, split="val"
            )
            self._log("val", val_metrics, epoch)

            # ── LR schedule ────────────────────────────────────────────
            self.scheduler.step()
            lrs = self.scheduler.get_last_lr()
            self.writer.add_scalar("lr/backbone",   lrs[0], epoch)
            self.writer.add_scalar("lr/new_layers", lrs[1], epoch)

            # ── Monitor & checkpoint ───────────────────────────────────
            monitor = val_metrics.get("qwk", val_metrics.get("acc", 0.0))
            is_best = monitor > self.best_metric
            if is_best:
                self.best_metric = monitor
                best_metrics = val_metrics.copy()
            self.save_checkpoint(epoch, val_metrics, is_best)

            # ── Print ──────────────────────────────────────────────────
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{tc.max_epochs} | {elapsed:.1f}s | "
                f"Train loss: {train_metrics['loss_total']:.4f} | "
                f"Val QWK: {val_metrics.get('qwk', 0):.4f} | "
                f"Val AUC: {val_metrics.get('auc_macro', 0):.4f} | "
                f"{'★ BEST' if is_best else ''}"
            )

            # ── Early stopping ─────────────────────────────────────────
            if self.early_stopper(monitor):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        self.writer.close()
        print(f"\nBest Val QWK: {self.best_metric:.4f}")
        return best_metrics
