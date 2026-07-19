"""
Optimizer + scheduler builders per plan Section 6.

Uses discriminative / layer-wise LR decay across the 3-phase training
schedule:
  Phase 1: frozen backbone, head-only training (high LR on head).
  Phase 2: full unfreeze, low uniform LR (or gentle layer-wise decay).
  Phase 3: APTOS fine-tune, very low LR, heavier augmentation.
"""
import torch


def build_optimizer(model, phase: str, head_lr: float = 1e-3, backbone_lr: float = 1e-5,
                     weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """
    phase: "phase1_frozen", "phase2_finetune", "phase3_aptos".

    Phase 1: only head params have requires_grad=True (backbone frozen via
             model.freeze_backbone()), so param groups collapse naturally.
    Phase 2/3: full model unfrozen — head keeps head_lr, backbone gets the
               (lower) backbone_lr.
    """
    head_params = list(model.head.parameters())
    if model.use_cbam:
        cbam_params = list(model.cbam_modules.parameters())
    else:
        cbam_params = []
    backbone_params = list(model.backbone.parameters())

    if phase == "phase1_frozen":
        # Backbone/CBAM are frozen (requires_grad=False), so we can safely
        # include them in a param group at LR 0 without affecting training —
        # but simplest is to just optimize head params only.
        param_groups = [{"params": head_params, "lr": head_lr}]
    else:
        param_groups = [
            {"params": head_params, "lr": head_lr},
            {"params": cbam_params, "lr": head_lr * 0.5},  # CBAM: between head and backbone LR
            {"params": backbone_params, "lr": backbone_lr},
        ]

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(optimizer, num_training_steps: int, num_warmup_steps: int = 0,
                     scheduler_type: str = "cosine"):
    """
    scheduler_type: "cosine" (cosine annealing with optional linear warmup)
                    or "plateau" (ReduceLROnPlateau, useful for phase 3
                    fine-tune where step-count planning is less reliable).
    """
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    if scheduler_type == "cosine":
        if num_warmup_steps > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=num_warmup_steps
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(num_training_steps - num_warmup_steps, 1)
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[num_warmup_steps]
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

    raise ValueError(f"Unknown scheduler_type: {scheduler_type!r}")
