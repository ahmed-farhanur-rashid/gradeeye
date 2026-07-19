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
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentation.transforms import build_eval_transforms, build_train_transforms
from src.data.datasets import DRDataset
from src.losses.class_weights import compute_class_counts, compute_corn_per_threshold_weights
from src.losses.ce_baseline import compute_5class_inverse_sqrt_weights
from src.models.dr_model import build_model
from src.training.checkpoint import DivergenceGuard, append_csv_log, save_checkpoint
from src.training.ema import ModelEMA
from src.training.optim import build_optimizer, build_scheduler
from src.training.trainer import train_one_epoch, validate_one_epoch

NUM_CLASSES = 5


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(manifest_train, manifest_val, norm_stats_path, aug_strength, batch_size):
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    train_ds = DRDataset(manifest_train, norm_stats, transform=build_train_transforms(aug_strength))
    val_ds = DRDataset(manifest_val, norm_stats, transform=build_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_ds


def run_phase(model, phase_name: str, phase_cfg: dict, run_cfg: dict, device,
              checkpoint_dir: str, log_dir: str, run_name: str, global_step: int,
              ema: ModelEMA, best_qwk: float, global_epoch_idx: list, start_epoch: int = 0):
    """global_epoch_idx: single-element list used as a mutable counter shared
    across phase1/phase2/phase3 calls, so the per-epoch CSV log has one
    continuous epoch axis spanning the full training run (for plotting)."""
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

    if phase_cfg.get("freeze_backbone", False):
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    optimizer = build_optimizer(
        model, phase=phase_name,
        head_lr=phase_cfg.get("head_lr", 1e-3),
        backbone_lr=phase_cfg.get("backbone_lr", 1e-5),
        weight_decay=phase_cfg.get("weight_decay", 1e-4),
    )
    total_steps = num_epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps, scheduler_type=phase_cfg.get("scheduler", "cosine"))

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

    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc=f"[{phase_name}] Epochs", position=0)
    for epoch in epoch_pbar:
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch, global_step,
            loss_type=loss_type, per_threshold_weights=per_threshold_weights,
            class_weights=ce_class_weights, mixup_enabled=mixup_enabled,
            ema=ema, divergence_guard=divergence_guard, log_path=log_path,
            checkpoint_dir=checkpoint_dir, run_name=run_name,
        )

        val_loss, val_acc, val_preds, val_labels = validate_one_epoch(
            model, val_loader, device, epoch, loss_type=loss_type,
            per_threshold_weights=per_threshold_weights, class_weights=ce_class_weights,
        )
        val_qwk = quadratic_weighted_kappa(val_labels.numpy(), val_preds.numpy())

        tqdm.write(f"[{phase_name}] Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                   f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_qwk={val_qwk:.4f}")

        # Per-epoch log spanning ALL phases in one file (global_epoch_idx keeps
        # a continuous x-axis for plotting the full training curve across
        # phase1 -> phase2 -> phase3, since plot_training_curves.py reads
        # this single file per run).
        current_lr = optimizer.param_groups[0]["lr"]
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

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Save checkpoint at the end of every epoch
        from src.training.checkpoint import rolling_checkpoint_cleanup
        ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_step{global_step}.pt")
        save_checkpoint(
            ckpt_path, model, optimizer, scheduler, epoch, global_step,
            config=run_cfg, ema_state_dict=ema.state_dict(), best_metric=best_qwk,
            phase_name=phase_name,
        )
        rolling_checkpoint_cleanup(checkpoint_dir, run_name, keep_last_n=3)

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch, global_step,
                config=run_cfg, ema_state_dict=ema.state_dict(), best_metric=best_qwk,
                phase_name=phase_name,
            )
            tqdm.write(f"  -> New best QWK {best_qwk:.4f}, saved to {best_path}")

    return global_step, best_qwk


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

    model = build_model(config).to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
        model = torch.compile(model, mode="max-autotune")
    ema = ModelEMA(model, decay=config.get("ema_decay", 0.999))

    global_step = 0
    best_qwk = -1.0
    global_epoch_idx = [0]  # mutable counter, shared/incremented across phase calls

    resume_phase = None
    resume_epoch = 0

    if args.resume:
        from src.training.checkpoint import load_checkpoint
        print(f"Resuming weights from {args.resume}...")
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ema and ckpt.get("ema_state_dict"):
            ema.load_state_dict(ckpt["ema_state_dict"])
        global_step = ckpt.get("global_step", 0)
        best_qwk = ckpt.get("best_metric", -1.0)
        if best_qwk is None:
            best_qwk = -1.0
            
        resume_phase = ckpt.get("phase_name")
        resume_epoch = ckpt.get("epoch", -1) + 1  # start at the NEXT epoch
        
        if resume_phase:
            skipped = 0
            for p in ["phase1_frozen", "phase2_finetune", "phase3_aptos"]:
                if p == resume_phase:
                    skipped += resume_epoch
                    break
                if p in config["phases"]:
                    skipped += config["phases"][p].get("num_epochs", 10)
            global_epoch_idx[0] = skipped

    phases_to_run = ["phase1_frozen", "phase2_finetune", "phase3_aptos"]
    if resume_phase in phases_to_run:
        phases_to_run = phases_to_run[phases_to_run.index(resume_phase):]

    for phase_name in phases_to_run:
        if phase_name not in config["phases"]:
            continue
        phase_cfg = config["phases"][phase_name]
        
        start_epoch = resume_epoch if phase_name == resume_phase else 0
        
        global_step, best_qwk = run_phase(
            model, phase_name, phase_cfg, config, device,
            checkpoint_dir, log_dir, run_name, global_step, ema, best_qwk, global_epoch_idx,
            start_epoch=start_epoch
        )

    print(f"\nTraining complete. Best val QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
