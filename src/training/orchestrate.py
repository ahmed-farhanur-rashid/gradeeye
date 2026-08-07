"""
orchestrate: top-level training orchestration.

Library code only — no argparse / CLI. The CLI entry point is
`python -m src.train` (see `src/train.py`).

The training driver is a thin convenience layer over the per-phase
functions in `trainer.py`. The job of orchestration is to:

  1. Load run config + build model + EMA + val_model.
  2. Discover the ordered list of phases (phase1, phase2, phase3) in the config.
  3. Run each phase in sequence, threading global_step + best_qwk across.
  4. Restore best checkpoint at each phase boundary so the next phase
     starts from the best model state, not the (potentially overfit) last epoch.

Critically, this module is **single-pipeline**: the cleanup plan reduced
the legacy 3-stage (EyePACS pretrain -> EyePACS finetune -> APTOS finetune)
to a single-stage LODO schedule (`phase1_frozen` head warmup +
`phase2_full_training` on the pooled train sources). The Phase 3 APTOS
fine-tune is gone — the LODO fold holds out one source at the *data*
level, no sequential cross-source training is needed.
"""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.augmentation.transforms import build_eval_transforms, build_train_transforms
from src.data._05_dataset import DRDataset
from src.losses.class_weights import compute_class_counts, compute_corn_per_threshold_weights
from src.losses.ce_baseline import compute_5class_inverse_sqrt_weights
from src.models.dr_model import build_model
from src.training.checkpoint import (
    DivergenceGuard,
    append_csv_log,
    load_checkpoint,
    rolling_checkpoint_cleanup,
    save_checkpoint,
)
from src.training.ema import ModelEMA
from src.training.optim import build_optimizer, build_scheduler
from src.training.trainer import train_one_epoch, validate_one_epoch
from src.training import progress as ui

NUM_CLASSES = 5
NUM_DATALOADER_WORKERS = min(4, os.cpu_count() or 1)


def set_seed(seed: int = 42) -> None:
    """Set all RNGs for reproducibility. Call before model build."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(manifest_train, manifest_val, aug_strength, batch_size,
                       seg_dir=None, img_size=None, use_sqrt_sampler=False):
    """Build train/val dataloaders. Optionally uses sqrt-frequency WeightedRandomSampler."""
    train_ds = DRDataset(manifest_train, transform=build_train_transforms(aug_strength, img_size=img_size),
                          seg_dir=seg_dir)
    val_ds = DRDataset(manifest_val, transform=build_eval_transforms(img_size=img_size),
                        seg_dir=seg_dir)

    sampler = None
    shuffle = True
    if use_sqrt_sampler:
        labels = train_ds.get_labels().astype(np.int64)
        class_counts = np.bincount(labels, minlength=NUM_CLASSES)
        class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
        sample_weights = class_weights[labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=NUM_DATALOADER_WORKERS,
        pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_DATALOADER_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader, train_ds


def _discover_phase_names(config: dict) -> list[str]:
    """Return the ordered list of phase names present in this config."""
    phases = config.get("phases", {})
    ordered = []
    for prefix in ("phase1", "phase2", "phase3"):
        for key in phases:
            if key.startswith(prefix):
                ordered.append(key)
                break
    return ordered


def _build_model_for_device(config: dict, device: str) -> tuple[Any, Any]:
    """Build train + val models. Apply torch.compile if CUDA + channels_last."""
    model = build_model(config).to(device)
    val_model = build_model(config).to(device)
    if device == "cuda":
        arch = config.get("model", {}).get("arch", "convnext_tiny")
        from src.models.backbone import get_arch_spec
        if get_arch_spec(arch).channels_last:
            model = model.to(memory_format=torch.channels_last)
        model = torch.compile(model)
    return model, val_model


def _run_phase(model, phase_name: str, phase_cfg: dict, run_cfg: dict, device,
               checkpoint_dir: str, log_dir: str, run_name: str, global_step: int,
               ema: ModelEMA, best_qwk: float, global_epoch_idx: list,
               phase_idx: int, total_phases: int, start_epoch: int = 0,
               val_model=None):
    """Run a single training phase (frozen-head, full-finetune, or LODO)."""
    if val_model is None:
        val_model = model

    from src.eval.metrics import quadratic_weighted_kappa

    manifest_train = phase_cfg["manifest_train"]
    manifest_val = phase_cfg["manifest_val"]
    aug_strength = phase_cfg.get("aug_strength", "light")
    batch_size = phase_cfg.get("batch_size", 32)
    num_epochs = phase_cfg.get("num_epochs", 10)
    seg_dir = phase_cfg.get("seg_dir", None)
    img_size = phase_cfg.get("img_size") or run_cfg.get("model", {}).get("img_size")
    use_sqrt_sampler = phase_cfg.get("use_sqrt_sampler", False)

    train_loader, val_loader, train_ds = build_dataloaders(
        manifest_train, manifest_val, aug_strength, batch_size,
        seg_dir=seg_dir, img_size=img_size, use_sqrt_sampler=use_sqrt_sampler,
    )

    freeze = phase_cfg.get("freeze_backbone", False)
    freeze_backbone_only = phase_cfg.get("freeze_backbone_only", False)
    for m in [model, val_model]:
        if freeze:
            m.freeze_backbone()
        elif freeze_backbone_only:
            m.freeze_backbone_only()
        else:
            m.unfreeze_backbone()

    # Reset BN running stats at every phase transition. The stats accumulated
    # during Phase 1 (frozen backbone, head-only gradients) were estimated
    # under a near-frozen weight distribution. When Phase 2 unfreezes the
    # backbone, those stats are immediately stale and cause val_loss to
    # explode. LayerNorm backbones have no running stats to reset, so this is
    # a no-op for them.
    for m in [model, val_model]:
        for sub in m.modules():
            if isinstance(sub, nn.BatchNorm2d):
                sub.running_mean.zero_()
                sub.running_var.fill_(1.0)
                sub.num_batches_tracked.zero_()

    if ema is not None:
        ema.reset(model)

    best_qwk = -1.0

    optimizer = build_optimizer(
        model, phase=phase_name,
        head_lr=phase_cfg.get("head_lr", 1e-3),
        backbone_lr=phase_cfg.get("backbone_lr", 1e-5),
        weight_decay=phase_cfg.get("weight_decay", 1e-4),
    )
    total_steps = num_epochs * len(train_loader)
    warmup_epochs = phase_cfg.get("warmup_epochs", 0)
    warmup_steps = warmup_epochs * len(train_loader)

    scheduler = build_scheduler(optimizer, total_steps,
                                num_warmup_steps=warmup_steps,
                                scheduler_type=phase_cfg.get("scheduler", "cosine"))

    loss_type = run_cfg.get("loss_type", "corn")
    use_class_weighting = run_cfg.get("use_class_weighting", False)

    per_threshold_weights, ce_class_weights = None, None
    if use_class_weighting:
        labels = train_ds.get_labels()
        class_counts = compute_class_counts(labels, NUM_CLASSES)
        if loss_type == "corn":
            per_threshold_weights = compute_corn_per_threshold_weights(
                labels, NUM_CLASSES, method="inverse_sqrt"
            )
        else:
            ce_class_weights = compute_5class_inverse_sqrt_weights(class_counts).to(device)

    divergence_guard = DivergenceGuard(patience=3)
    mixup_enabled = run_cfg.get("use_mixup", False)
    log_path = os.path.join(log_dir, f"{run_name}_{phase_name}_train_log.csv")
    epoch_log_path = os.path.join(log_dir, f"{run_name}_epoch_log.csv")

    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    step_scheduler = None if is_plateau else scheduler

    overfitting_patience = phase_cfg.get("overfitting_patience", 10)
    min_epochs = phase_cfg.get("min_epochs", 8)
    consecutive_val_loss_increases = 0
    best_val_loss_so_far = float("inf")

    ui.print_phase_start(phase_name, phase_idx, total_phases,
                         num_epochs, batch_size, freeze or freeze_backbone_only)

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch, global_step,
            loss_type=loss_type, per_threshold_weights=per_threshold_weights,
            class_weights=ce_class_weights, mixup_enabled=mixup_enabled,
            ema=ema, divergence_guard=divergence_guard, log_path=log_path,
            checkpoint_dir=checkpoint_dir, run_name=run_name,
            scheduler=step_scheduler,
            camera_jitter=phase_cfg.get("camera_jitter", False),
            camera_jitter_channel_std=phase_cfg.get("camera_jitter_channel_std", 0.08),
            camera_jitter_gamma_range=tuple(phase_cfg.get("camera_jitter_gamma_range", [0.85, 1.15])),
        )

        if ema is not None:
            ema.apply_to(val_model)
        else:
            sd = model.state_dict()
            val_sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
            val_model.load_state_dict(val_sd)

        val_loss, val_acc, val_preds, val_labels = validate_one_epoch(
            val_model, val_loader, device, epoch, loss_type=loss_type,
            per_threshold_weights=per_threshold_weights, class_weights=ce_class_weights,
        )
        val_qwk = quadratic_weighted_kappa(val_labels.numpy(), val_preds.numpy())

        is_new_best = val_qwk > best_qwk
        if is_new_best:
            best_qwk = val_qwk

        current_lr = optimizer.param_groups[0]["lr"]
        best_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt") if is_new_best else None

        ui.print_epoch_summary(
            epoch, num_epochs, train_loss, train_acc,
            val_loss, val_acc, val_qwk, current_lr,
            best_qwk, is_new_best, best_path,
        )

        append_csv_log(
            epoch_log_path,
            {"phase": phase_name, "phase_epoch": epoch, "global_epoch_idx": global_epoch_idx[0],
             "train_loss": train_loss, "train_acc": train_acc,
             "val_loss": val_loss, "val_acc": val_acc, "val_qwk": val_qwk,
             "lr": current_lr},
            fieldnames=["phase", "phase_epoch", "global_epoch_idx", "train_loss", "train_acc",
                         "val_loss", "val_acc", "val_qwk", "lr"],
        )
        global_epoch_idx[0] += 1

        if is_plateau:
            scheduler.step(val_loss)

        ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_step{global_step}.pt")
        save_checkpoint(
            ckpt_path, model, optimizer, scheduler, epoch, global_step,
            config=run_cfg, ema_state_dict=ema.state_dict(), best_metric=best_qwk,
            phase_name=phase_name,
        )
        rolling_checkpoint_cleanup(checkpoint_dir, run_name, keep_last_n=3)

        if is_new_best:
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch, global_step,
                config=run_cfg, ema_state_dict=ema.state_dict(), best_metric=best_qwk,
                phase_name=phase_name,
            )

        if overfitting_patience > 0:
            if val_loss > best_val_loss_so_far:
                consecutive_val_loss_increases += 1
            else:
                consecutive_val_loss_increases = 0
                best_val_loss_so_far = val_loss

            if (epoch >= min_epochs
                    and consecutive_val_loss_increases >= overfitting_patience):
                ui.log(
                    f"Overfitting detected: val_loss increased for "
                    f"{overfitting_patience} consecutive epochs "
                    f"(best_val_loss={best_val_loss_so_far:.4f}, "
                    f"current={val_loss:.4f}). Stopping phase.",
                    style="bold yellow",
                )
                break

    # Always restore best checkpoint at phase end.
    best_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
    if os.path.exists(best_ckpt_path):
        best_ckpt = load_checkpoint(best_ckpt_path, map_location=str(device))
        model.load_state_dict(best_ckpt["model_state_dict"])
        clean_sd = {k.removeprefix("_orig_mod."): v for k, v in best_ckpt["model_state_dict"].items()}
        val_model.load_state_dict(clean_sd)
        if ema is not None and best_ckpt.get("ema_state_dict"):
            ema.load_state_dict(best_ckpt["ema_state_dict"])
        ui.log(
            f"Phase {phase_name} complete — restored best checkpoint "
            f"(QWK={best_qwk:.4f})",
            style="bold green",
        )

    return global_step, best_qwk


def run_training(config: dict, *, resume_from: str | None = None) -> None:
    """Top-level driver: run all phases defined in `config.phases` in order.

    Args:
        config: run config dict (YAML loaded into a dict).
        resume_from: optional path to a checkpoint to resume from.
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    run_name = config["run_name"]
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_dir = config.get("checkpoint_dir", "saved/checkpoints")
    log_dir = config.get("log_dir", "saved/logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    ui.print_header(run_name, device, config.get("loss_type", "corn"))

    model, val_model = _build_model_for_device(config, device)
    ema = ModelEMA(model, decay=config.get("ema_decay", 0.999))

    global_step = 0
    best_qwk = -1.0
    global_epoch_idx = [0]

    resume_phase = None
    resume_epoch = 0

    if resume_from:
        ui.log(f"Resuming from {resume_from}...", style="bold yellow")
        ckpt = load_checkpoint(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        clean_sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
        val_model.load_state_dict(clean_sd)
        if ema and ckpt.get("ema_state_dict"):
            ema.load_state_dict(ckpt["ema_state_dict"])
        global_step = ckpt.get("global_step", 0)
        best_qwk = ckpt.get("best_metric", -1.0)
        if best_qwk is None:
            best_qwk = -1.0
        resume_phase = ckpt.get("phase_name")
        resume_epoch = ckpt.get("epoch", -1) + 1

    phases_to_run = _discover_phase_names(config)
    total_phases = len(phases_to_run)

    if resume_phase in phases_to_run:
        phases_to_run = phases_to_run[phases_to_run.index(resume_phase):]
        skipped = 0
        for p in _discover_phase_names(config):
            if p == resume_phase:
                skipped += resume_epoch
                break
            if p in config["phases"]:
                skipped += config["phases"][p].get("num_epochs", 10)
        global_epoch_idx[0] = skipped

    for i, phase_name in enumerate(phases_to_run, start=total_phases - len(phases_to_run) + 1):
        if phase_name not in config["phases"]:
            continue
        phase_cfg = config["phases"][phase_name]
        start_epoch = resume_epoch if phase_name == resume_phase else 0

        global_step, best_qwk = _run_phase(
            model, phase_name, phase_cfg, config, device,
            checkpoint_dir, log_dir, run_name, global_step, ema, best_qwk,
            global_epoch_idx, phase_idx=i, total_phases=total_phases,
            start_epoch=start_epoch,
            val_model=val_model,
        )

    ui.print_training_complete(best_qwk, run_name)
