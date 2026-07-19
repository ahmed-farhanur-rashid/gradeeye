"""
Checkpointing/resilience, modeled after the existing bangla-gamba training
infra pattern (plan Section 6 / handoff note 9) — reuse the pattern, don't
rebuild from scratch:

  - Step-based checkpoint saves at regular intervals.
  - Resume-from-checkpoint support (instant batch-skip resume, no wasted I/O).
  - Safe CSV logging in append mode (never overwrite on resume).
  - EMA shadow weights (see ema.py).
  - Rolling checkpoint retention (last N + best.pt).
  - Divergence guard (NaN loss -> abort trial, don't crash multi-run sweep).
  - Self-contained checkpoints: architecture config + class_names saved
    directly inside the .pt file so evaluate.py can load without the
    original training YAML present.
"""
import glob
import math
import os

import torch

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, global_step: int,
                     config: dict, ema_state_dict: dict | None = None,
                     best_metric: float | None = None):
    """Self-contained checkpoint: includes config + class names for standalone loading."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "class_names": CLASS_NAMES,
        "ema_state_dict": ema_state_dict,
        "best_metric": best_metric,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)  # atomic write — avoids corrupt checkpoint on crash mid-save


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)


def rolling_checkpoint_cleanup(checkpoint_dir: str, run_name: str, keep_last_n: int = 3):
    """
    Keeps only the last N step checkpoints plus best.pt (best.pt is named
    separately and never touched by this cleanup).
    """
    pattern = os.path.join(checkpoint_dir, f"{run_name}_step*.pt")
    checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)

    if len(checkpoints) <= keep_last_n:
        return

    for ckpt_path in checkpoints[:-keep_last_n]:
        os.remove(ckpt_path)


class DivergenceGuard:
    """
    NaN-loss divergence guard: aborts the current trial/run rather than
    crashing an unattended multi-run sweep (plan Section 6).
    """

    def __init__(self, patience: int = 3):
        self.patience = patience
        self.nan_count = 0

    def check(self, loss_value: float) -> bool:
        """Returns True if training should abort due to divergence."""
        if math.isnan(loss_value) or math.isinf(loss_value):
            self.nan_count += 1
            if self.nan_count >= self.patience:
                return True
        else:
            self.nan_count = 0
        return False


def append_csv_log(log_path: str, row: dict, fieldnames: list[str]):
    """
    Safe CSV logging in append mode — never overwrites on resume. Writes
    the header only if the file doesn't already exist.
    """
    import csv
    file_exists = os.path.exists(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def compute_resume_batch_offset(global_step: int, steps_per_epoch: int) -> tuple[int, int]:
    """
    Given a resumed global_step, compute (epoch, batch_offset_within_epoch)
    so the dataloader can be fast-forwarded without re-reading already-seen
    data from disk (instant batch-skip resume).
    """
    epoch = global_step // steps_per_epoch
    batch_offset = global_step % steps_per_epoch
    return epoch, batch_offset
