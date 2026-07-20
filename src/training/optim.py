"""
Optimizer + scheduler builders per plan Section 6.

Uses discriminative / layer-wise LR decay across the 3-phase training
schedule:
  Phase 1: frozen backbone, head-only training (high LR on head).
  Phase 2: full unfreeze, low uniform LR (or gentle layer-wise decay).
  Phase 3: APTOS fine-tune, very low LR, heavier augmentation.

Weight-decay exemption (plan Section 6): weight_decay applied only to
Convolution and Linear weights.  weight_decay = 0.0 for all Biases,
BatchNorm, and LayerNorm parameters — standard practice, prevents
unnecessary regularization pressure on normalization/bias terms.
"""
import torch

# Keywords in parameter names that indicate bias/norm params
# (should be exempt from weight decay).
_NO_DECAY_KEYWORDS = ("bias", ".bn", "norm", "_bn")


def _split_decay_groups(params_with_names, lr, weight_decay):
    """Split named parameters into weight-decay and no-decay groups.

    Returns a list of param-group dicts (may contain 0-2 groups depending
    on whether decay/no-decay params exist).
    """
    decay_params = []
    no_decay_params = []

    for name, param in params_with_names:
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in _NO_DECAY_KEYWORDS):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    return groups


def build_optimizer(model, phase: str, head_lr: float = 1e-3, backbone_lr: float = 1e-5,
                     weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """
    phase: "phase1_frozen", "phase2_full_training", "phase3_aptos", etc.

    Phase 1: only head params have requires_grad=True (backbone frozen via
             model.freeze_backbone()), so param groups collapse naturally.
    Phase 2/3: full model unfrozen — head keeps head_lr, backbone gets the
               (lower) backbone_lr.

    Weight-decay exemption per plan Section 6: bias, BatchNorm, and
    LayerNorm parameters get weight_decay=0.0 in every phase.
    """
    # Classify each parameter by its owning module.
    # torch.compile wraps param names with '_orig_mod.' prefix — strip it
    # so that module-based grouping (head. / cbam / backbone.) still works.
    def _strip(name: str) -> str:
        return name.removeprefix("_orig_mod.")

    head_named = [(n, p) for n, p in model.named_parameters() if _strip(n).startswith("head.")]
    cbam_named = [(n, p) for n, p in model.named_parameters() if _strip(n).startswith("cbam")]
    backbone_named = [(n, p) for n, p in model.named_parameters()
                      if not _strip(n).startswith("head.") and not _strip(n).startswith("cbam")]

    if phase == "phase1_frozen":
        # Backbone/CBAM are frozen (requires_grad=False). Only optimize head.
        param_groups = _split_decay_groups(head_named, head_lr, weight_decay)
    else:
        param_groups = (
            _split_decay_groups(head_named, head_lr, weight_decay)
            + _split_decay_groups(cbam_named, head_lr * 0.5, weight_decay)
            + _split_decay_groups(backbone_named, backbone_lr, weight_decay)
        )

    # Safety: filter out empty groups (e.g. no CBAM params in baseline config)
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer, num_training_steps: int, num_warmup_steps: int = 0,
                     scheduler_type: str = "cosine"):
    """
    scheduler_type: "cosine" (cosine annealing with optional linear warmup)
                    or "plateau" (ReduceLROnPlateau, useful for phase 3
                    fine-tune where step-count planning is less reliable).

    IMPORTANT: For cosine/warmup, the returned scheduler must be stepped
    PER BATCH (inside train_one_epoch), not per epoch.  T_max equals the
    total number of optimizer steps across the phase.  For plateau, it is
    stepped per epoch in run_phase() with val_loss.
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
