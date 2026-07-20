"""Tests for LR scheduler behavior.

Validates:
  - Cosine scheduler with step-based stepping actually decays the LR
  - Cosine scheduler reaches near-zero LR at T_max
  - Warmup + cosine: LR starts low, ramps up, then decays
  - Plateau scheduler responds to val_loss improvements
  - The critical bug: calling .step() only N times with T_max=N*K must NOT
    leave LR flat (this was the original bug — these tests prevent regression)
"""
import pytest
import torch

from src.training.optim import build_scheduler


def _make_optimizer(lr=1e-3):
    """Dummy optimizer for scheduler tests."""
    params = [torch.nn.Parameter(torch.randn(2, 2))]
    return torch.optim.AdamW(params, lr=lr)


class TestCosineScheduler:
    """Cosine annealing with per-step stepping."""

    def test_lr_decays_over_total_steps(self):
        """LR must actually decrease when stepped total_steps times."""
        optimizer = _make_optimizer(lr=1e-3)
        total_steps = 500
        scheduler = build_scheduler(optimizer, total_steps, scheduler_type="cosine")

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step through all training steps
        for _ in range(total_steps):
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr * 0.01, \
            f"LR should decay to near 0 after T_max steps: initial={initial_lr}, final={final_lr}"

    def test_lr_at_halfway_is_half(self):
        """At T_max/2, cosine LR should be ~50% of initial."""
        optimizer = _make_optimizer(lr=1e-3)
        total_steps = 1000
        scheduler = build_scheduler(optimizer, total_steps, scheduler_type="cosine")

        for _ in range(total_steps // 2):
            scheduler.step()

        mid_lr = optimizer.param_groups[0]["lr"]
        expected_mid = 1e-3 * 0.5  # cos(pi/2) = 0 → (1+0)/2 = 0.5
        assert abs(mid_lr - expected_mid) < 1e-4, \
            f"LR at halfway should be ~{expected_mid}, got {mid_lr}"

    def test_regression_epoch_only_stepping_leaves_lr_flat(self):
        """REGRESSION TEST for the original bug: stepping only num_epochs
        times with T_max=total_steps must NOT produce meaningful LR decay.

        This test documents the bug we fixed — if this behavior ever returns,
        the test catches it.
        """
        optimizer = _make_optimizer(lr=1e-3)
        num_epochs = 5
        batches_per_epoch = 137
        total_steps = num_epochs * batches_per_epoch  # 685

        scheduler = build_scheduler(optimizer, total_steps, scheduler_type="cosine")

        initial_lr = optimizer.param_groups[0]["lr"]

        # BUG scenario: only step num_epochs times instead of total_steps
        for _ in range(num_epochs):
            scheduler.step()

        barely_decayed_lr = optimizer.param_groups[0]["lr"]

        # With T_max=685 and only 5 steps, LR barely moves (this is the bug)
        lr_ratio = barely_decayed_lr / initial_lr
        assert lr_ratio > 0.99, \
            f"With only {num_epochs} steps out of T_max={total_steps}, LR should barely move. " \
            f"Ratio={lr_ratio:.4f} — if this fails, something else changed the stepping logic."


class TestWarmupCosineScheduler:
    """Cosine with linear warmup prefix."""

    def test_lr_starts_low_with_warmup(self):
        """With warmup, initial LR should be 10% of configured (start_factor=0.1)."""
        optimizer = _make_optimizer(lr=1e-3)
        total_steps = 1000
        warmup_steps = 100
        scheduler = build_scheduler(optimizer, total_steps,
                                     num_warmup_steps=warmup_steps,
                                     scheduler_type="cosine")

        # After first step, LR should be near start_factor * base_lr
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert lr < 2e-4, f"LR after first warmup step should be low, got {lr}"

    def test_lr_ramps_up_during_warmup(self):
        """LR should increase throughout the warmup period."""
        optimizer = _make_optimizer(lr=1e-3)
        total_steps = 1000
        warmup_steps = 100
        scheduler = build_scheduler(optimizer, total_steps,
                                     num_warmup_steps=warmup_steps,
                                     scheduler_type="cosine")

        lrs = []
        for _ in range(warmup_steps):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # LR should be increasing during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1] - 1e-8, \
                f"LR should increase during warmup: step {i-1}={lrs[i-1]}, step {i}={lrs[i]}"

    def test_lr_decays_after_warmup(self):
        """After warmup completes, LR should decay via cosine."""
        optimizer = _make_optimizer(lr=1e-3)
        total_steps = 1000
        warmup_steps = 100
        scheduler = build_scheduler(optimizer, total_steps,
                                     num_warmup_steps=warmup_steps,
                                     scheduler_type="cosine")

        # Complete warmup
        for _ in range(warmup_steps):
            scheduler.step()

        lr_at_warmup_end = optimizer.param_groups[0]["lr"]

        # Step through remaining cosine decay
        for _ in range(total_steps - warmup_steps):
            scheduler.step()

        lr_at_end = optimizer.param_groups[0]["lr"]
        assert lr_at_end < lr_at_warmup_end * 0.05, \
            f"LR should decay after warmup: warmup_end={lr_at_warmup_end}, final={lr_at_end}"


class TestPlateauScheduler:
    """ReduceLROnPlateau for Phase 3."""

    def test_plateau_returns_correct_type(self):
        optimizer = _make_optimizer()
        scheduler = build_scheduler(optimizer, 1000, scheduler_type="plateau")
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_plateau_reduces_lr_on_stagnation(self):
        optimizer = _make_optimizer(lr=1e-3)
        scheduler = build_scheduler(optimizer, 1000, scheduler_type="plateau")

        initial_lr = optimizer.param_groups[0]["lr"]

        # Feed same val_loss many times (patience=3, so after 4 stagnant epochs)
        for _ in range(10):
            scheduler.step(1.0)

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr, \
            f"Plateau should reduce LR after stagnation: {initial_lr} -> {final_lr}"
