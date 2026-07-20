"""
Main training loop — wires together model, loss, optimizer, checkpointing,
EMA, and MixUp across the 3-phase schedule (plan Sections 5-6).
"""
import time

import torch
from torch.utils.data import DataLoader

from src.augmentation.transforms import maybe_apply_mixup
from src.losses.corn_loss import corn_loss
from src.losses.ce_baseline import ce_loss
from src.models.corn import corn_predict
from src.training.checkpoint import (
    DivergenceGuard,
    append_csv_log,
)
from src.training.ema import ModelEMA
from src.training.progress import make_batch_progress

NUM_CLASSES = 5


def train_one_epoch(model, dataloader: DataLoader, optimizer, device, epoch: int,
                     global_step: int, loss_type: str = "corn", per_threshold_weights=None,
                     class_weights=None, mixup_enabled: bool = False, mixup_alpha: float = 0.2,
                     ema: ModelEMA | None = None, divergence_guard: DivergenceGuard | None = None,
                     grad_clip_norm: float = 5.0, log_path: str | None = None,
                     checkpoint_dir: str | None = None, run_name: str = "run",
                     checkpoint_every_n_steps: int = 500,
                     scheduler=None):
    """Train for one epoch.

    scheduler: optional step-based scheduler (cosine, warmup, etc.) to step
        after every optimizer step. Do NOT pass ReduceLROnPlateau here —
        plateau schedulers need val_loss and are stepped per-epoch in
        run_phase() instead.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    progress = make_batch_progress()
    with progress:
        task = progress.add_task(
            f"[cyan]Epoch {epoch}[/cyan] train", total=len(dataloader), status=""
        )
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            images, labels_a, labels_b, lam = maybe_apply_mixup(
                images, labels, alpha=mixup_alpha, enabled=mixup_enabled
            )

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
                logits = model(images)

            # CRITICAL: cast logits to float32 before loss computation.
            # BCE_with_logits (used by CORN) is numerically unstable in
            # bfloat16 — it computes log(sigmoid(x)) which underflows,
            # causing loss explosion (46k+) especially for EfficientNet.
            logits = logits.float()

            if loss_type == "corn":
                loss_a = corn_loss(logits, labels_a, NUM_CLASSES, per_threshold_weights)
                loss_b = corn_loss(logits, labels_b, NUM_CLASSES, per_threshold_weights)
            elif loss_type == "ce":
                loss_a = ce_loss(logits, labels_a, class_weights)
                loss_b = ce_loss(logits, labels_b, class_weights)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type!r}")

            loss = lam * loss_a + (1 - lam) * loss_b

            if divergence_guard is not None and divergence_guard.check(loss.item()):
                raise RuntimeError(
                    f"Training diverged (NaN/Inf loss) at epoch {epoch}, step {global_step}. "
                    f"Aborting this run so a multi-run sweep can continue past it."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            # Step-based schedulers step after each optimizer step.
            if scheduler is not None:
                scheduler.step()

            if ema is not None:
                ema.update(model)

            with torch.no_grad():
                if loss_type == "corn":
                    preds = corn_predict(logits)
                else:
                    preds = logits.argmax(dim=1)
                running_correct += (preds == labels_a).sum().item()
                running_total += labels_a.size(0)

            running_loss += loss.item() * images.size(0)
            global_step += 1

            acc = running_correct / max(running_total, 1)
            progress.update(
                task, advance=1,
                status=f"loss={loss.item():.4f}  acc={acc:.4f}",
            )

            if log_path is not None:
                append_csv_log(
                    log_path,
                    {"epoch": epoch, "step": global_step, "batch_loss": loss.item(),
                     "running_acc": acc, "timestamp": time.time()},
                    fieldnames=["epoch", "step", "batch_loss", "running_acc", "timestamp"],
                )

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_correct / max(running_total, 1)
    return epoch_loss, epoch_acc, global_step


@torch.no_grad()
def validate_one_epoch(model, dataloader: DataLoader, device, epoch: int,
                        loss_type: str = "corn", per_threshold_weights=None, class_weights=None):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    all_preds = []
    all_labels = []

    progress = make_batch_progress()
    with progress:
        task = progress.add_task(
            f"[green]Epoch {epoch}[/green] val  ", total=len(dataloader), status=""
        )
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
                logits = model(images)

            # Cast to float32 for numerically stable loss (see train_one_epoch).
            logits = logits.float()

            if loss_type == "corn":
                loss = corn_loss(logits, labels, NUM_CLASSES, per_threshold_weights)
                preds = corn_predict(logits)
            else:
                loss = ce_loss(logits, labels, class_weights)
                preds = logits.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            acc = running_correct / max(running_total, 1)
            progress.update(
                task, advance=1,
                status=f"loss={loss.item():.4f}  acc={acc:.4f}",
            )

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_correct / max(running_total, 1)
    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    return epoch_loss, epoch_acc, preds_cat, labels_cat
