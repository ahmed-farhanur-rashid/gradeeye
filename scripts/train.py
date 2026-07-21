"""
Top-level training driver — orchestrates the 3-phase schedule
(EyePACS frozen-head -> EyePACS full-unfreeze -> APTOS fine-tune) per
plan Section 6, for a single named run configuration from the run matrix
in plan Section 5.

Usage:
    python scripts/train.py --config configs/full_method.yaml
"""
import argparse
import json
import os
import sys

# Allow running as `python scripts/train.py` (not just `python -m scripts.train`)
# by ensuring the repo root is on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
import yaml
from torch.utils.data import DataLoader
import cv2

# CRITICAL FIX for DataLoader deadlock on Linux: prevent OpenCV from spawning
# its own threads inside the PyTorch multiprocessing workers!
cv2.setNumThreads(0)

from src.augmentation.transforms import build_eval_transforms, build_train_transforms
from src.data.datasets import DRDataset
from src.losses.class_weights import compute_class_counts, compute_corn_per_threshold_weights
from src.losses.ce_baseline import compute_5class_inverse_sqrt_weights
from src.models.dr_model import build_model
from src.training.checkpoint import DivergenceGuard, append_csv_log, save_checkpoint
from src.training.ema import ModelEMA
from src.training.optim import build_optimizer, build_scheduler
from src.training.trainer import train_one_epoch, validate_one_epoch
from src.training import progress as ui

NUM_CLASSES = 5
NUM_DATALOADER_WORKERS = min(6, os.cpu_count() or 1)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(manifest_train, manifest_val, norm_stats_path, aug_strength, batch_size):
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    train_ds = DRDataset(manifest_train, norm_stats, transform=build_train_transforms(aug_strength))
    val_ds = DRDataset(manifest_val, norm_stats, transform=build_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=NUM_DATALOADER_WORKERS,
                               pin_memory=True, drop_last=True,
                               persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_DATALOADER_WORKERS,
                             pin_memory=True,
                             persistent_workers=True)
    return train_loader, val_loader, train_ds


def run_phase(model, phase_name: str, phase_cfg: dict, run_cfg: dict, device,
              checkpoint_dir: str, log_dir: str, run_name: str, global_step: int,
              ema: ModelEMA, best_qwk: float, global_epoch_idx: list,
              phase_idx: int, total_phases: int, start_epoch: int = 0):
    """Run a single training phase (frozen-head, full-finetune, or APTOS)."""
    from src.eval.metrics import quadratic_weighted_kappa

    manifest_train = phase_cfg["manifest_train"]
    manifest_val = phase_cfg["manifest_val"]
    norm_stats_path = phase_cfg["norm_stats"]
    aug_strength = phase_cfg.get("aug_strength", "light")
    batch_size = phase_cfg.get("batch_size", 32)
    num_epochs = phase_cfg.get("num_epochs", 10)

    train_loader, val_loader, train_ds = build_dataloaders(
        manifest_train, manifest_val, norm_stats_path, aug_strength, batch_size
    )

    freeze = phase_cfg.get("freeze_backbone", False)
    freeze_backbone_only = phase_cfg.get("freeze_backbone_only", False)
    if freeze:
        model.freeze_backbone()
    elif freeze_backbone_only:
        model.freeze_backbone_only()
    else:
        model.unfreeze_backbone()

    # Reset EMA at each phase transition so shadow weights start fresh.
    if ema is not None:
        ema.reset(model)

    # Reset best_qwk per phase — each phase may use a DIFFERENT validation
    # set (EyePACS in Phase 1/2 vs APTOS in Phase 3), so carrying best_qwk
    # across phases compares apples to oranges and prevents Phase 3 from
    # ever saving a "best" checkpoint.
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
            per_threshold_weights = compute_corn_per_threshold_weights(labels, NUM_CLASSES)
        else:
            ce_class_weights = compute_5class_inverse_sqrt_weights(class_counts).to(device)

    divergence_guard = DivergenceGuard(patience=3)
    mixup_enabled = run_cfg.get("use_mixup", False)
    log_path = os.path.join(log_dir, f"{run_name}_{phase_name}_train_log.csv")
    epoch_log_path = os.path.join(log_dir, f"{run_name}_epoch_log.csv")

    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    step_scheduler = None if is_plateau else scheduler

    # ── Overfitting detection (replaces QWK-based early stopping) ──
    # QWK is too noisy on small val sets (caused premature Phase 3 stop
    # at epoch 9 when val_loss was still decreasing 0.512→0.397).
    # Instead, track consecutive val_loss INCREASES — this catches real
    # overfitting (memorization) not metric noise.
    #
    # Guard only fires after `min_epochs` AND after `overfitting_patience`
    # consecutive val_loss increases.  Set overfitting_patience=0 to disable.
    overfitting_patience = phase_cfg.get("overfitting_patience", 10)
    min_epochs = phase_cfg.get("min_epochs", 8)
    consecutive_val_loss_increases = 0
    best_val_loss_so_far = float("inf")

    # ── Phase banner ──
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
        )

        train_state = None
        if ema is not None:
            train_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)

        val_loss, val_acc, val_preds, val_labels = validate_one_epoch(
            model, val_loader, device, epoch, loss_type=loss_type,
            per_threshold_weights=per_threshold_weights, class_weights=ce_class_weights,
        )
        val_qwk = quadratic_weighted_kappa(val_labels.numpy(), val_preds.numpy())

        if train_state is not None:
            model.load_state_dict(train_state)

        # ── Best checkpoint tracking ──
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

        # ── CSV log ──
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

        # ── Checkpoint ──
        from src.training.checkpoint import rolling_checkpoint_cleanup
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

        # ── Overfitting detection ──
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

    # ── Always restore best checkpoint at phase end ──
    # Ensures the next phase starts from the best model state, not the
    # last (potentially overfit) epoch. Previously this only happened
    # on early stop, causing Phase 3 to inherit degraded Phase 2 weights.
    best_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
    if os.path.exists(best_ckpt_path):
        from src.training.checkpoint import load_checkpoint
        best_ckpt = load_checkpoint(best_ckpt_path, map_location=str(device))
        model.load_state_dict(best_ckpt["model_state_dict"])
        if ema is not None and best_ckpt.get("ema_state_dict"):
            ema.load_state_dict(best_ckpt["ema_state_dict"])
        ui.log(
            f"Phase {phase_name} complete — restored best checkpoint "
            f"(QWK={best_qwk:.4f})",
            style="bold green",
        )

    return global_step, best_qwk


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


def main():
    parser = argparse.ArgumentParser(description="Train DR grading model per run config.")
    parser.add_argument("--config", required=True, help="Path to run config YAML.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_name = config["run_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_dir = config.get("checkpoint_dir", "saved/checkpoints")
    log_dir = config.get("log_dir", "saved/logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ── Header ──
    ui.print_header(run_name, device, config.get("loss_type", "corn"))

    model = build_model(config).to(device)
    if device == "cuda":
        arch = config.get("model", {}).get("arch", "convnext_tiny")
        # channels_last + reduce-overhead for ConvNeXt. reduce-overhead
        # uses CUDA graphs for lower kernel-launch overhead and leaves more
        # VRAM for larger batches than max-autotune (which also warns
        # "Not enough SMs" on RTX 4070 Super anyway).
        if arch == "convnext_tiny":
            model = model.to(memory_format=torch.channels_last)
            model = torch.compile(model, mode="reduce-overhead")
        else:
            model = torch.compile(model)
    ema = ModelEMA(model, decay=config.get("ema_decay", 0.999))

    global_step = 0
    best_qwk = -1.0
    global_epoch_idx = [0]

    resume_phase = None
    resume_epoch = 0

    if args.resume:
        from src.training.checkpoint import load_checkpoint
        ui.log(f"Resuming from {args.resume}...", style="bold yellow")
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
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

        global_step, best_qwk = run_phase(
            model, phase_name, phase_cfg, config, device,
            checkpoint_dir, log_dir, run_name, global_step, ema, best_qwk,
            global_epoch_idx, phase_idx=i, total_phases=total_phases,
            start_epoch=start_epoch,
        )

    ui.print_training_complete(best_qwk, run_name)


if __name__ == "__main__":
    main()
