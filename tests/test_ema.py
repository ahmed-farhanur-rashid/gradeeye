"""Tests for EMA (Exponential Moving Average) shadow weights.

Validates:
  - Warmup schedule: early updates have low effective decay (rapid tracking)
  - Steady-state: after many updates, decay approaches configured value
  - reset() re-snapshots shadow weights and restarts warmup
  - state_dict / load_state_dict roundtrip preserves shadow weights
  - apply_to loads shadow weights into a model correctly
"""
import copy

import pytest
import torch
import torch.nn as nn

from src.training.ema import ModelEMA


class SimpleModel(nn.Module):
    """Tiny model for fast EMA tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, x):
        return self.bn(self.linear(x))


class TestEMAWarmup:
    """Verify the warmup schedule d = min(decay, (1+updates)/(10+updates))."""

    def _param_keys(self, model):
        """Return state_dict keys that correspond to nn.Parameter (not buffers)."""
        param_names = {n for n, _ in model.named_parameters()}
        return param_names

    def test_first_update_has_low_decay(self):
        """After 1 update, effective decay should be (1+1)/(10+1) = 2/11 ≈ 0.18,
        much lower than the configured 0.999."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)

        # Modify model weights to something different
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(10.0)

        ema.update(model)

        # After 1 update with d ≈ 0.18, shadow should be close to model (10.0)
        # shadow = 0.18 * old_shadow + 0.82 * 10.0
        param_keys = self._param_keys(model)
        for key, shadow_val in ema.shadow.items():
            if key in param_keys and shadow_val.dtype.is_floating_point:
                # Shadow should have moved significantly toward 10.0
                assert shadow_val.mean().item() > 5.0, \
                    f"Shadow[{key}] didn't move fast enough during warmup: {shadow_val.mean().item()}"

    def test_warmup_increases_monotonically(self):
        """Effective decay should increase with each update."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)

        decays = []
        for i in range(50):
            ema.updates = i
            d = min(ema.decay, (1 + (i + 1)) / (10 + (i + 1)))
            decays.append(d)

        for i in range(1, len(decays)):
            assert decays[i] >= decays[i - 1], \
                f"Decay not monotonically increasing at step {i}: {decays[i-1]} -> {decays[i]}"

    def test_decay_reaches_configured_value(self):
        """After enough updates, effective decay should equal configured decay."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)

        # After many updates: (1+N)/(10+N) → 1.0, so min(0.999, ~1.0) = 0.999
        ema.updates = 9999
        d = min(ema.decay, (1 + 10000) / (10 + 10000))
        assert abs(d - 0.999) < 0.001, f"Decay should be ~0.999, got {d}"


class TestEMAReset:
    """Verify reset() re-snapshots shadow and restarts warmup."""

    def test_reset_copies_current_model_weights(self):
        """After reset(), shadow should exactly match current model state."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)

        # Train for a while so EMA diverges from model
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(42.0)

        for _ in range(100):
            ema.update(model)

        # Now change model weights again
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(7.0)

        # Reset should snapshot the current model (all 7.0)
        ema.reset(model)

        model_state = model.state_dict()
        for key in ema.shadow:
            if ema.shadow[key].dtype.is_floating_point:
                assert torch.allclose(ema.shadow[key], model_state[key]), \
                    f"Shadow[{key}] doesn't match model after reset"

    def test_reset_restarts_warmup_counter(self):
        """After reset(), updates counter should be 0."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)

        for _ in range(100):
            ema.update(model)
        assert ema.updates == 100

        ema.reset(model)
        assert ema.updates == 0, f"Expected updates=0 after reset, got {ema.updates}"

    def test_reset_enables_fast_tracking_again(self):
        """After reset, the first update should use low decay (warmup restarts)."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)

        # Exhaust warmup
        for _ in range(1000):
            ema.update(model)

        # Reset with current model
        ema.reset(model)

        # Change model dramatically
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(99.0)

        # First update after reset should track aggressively (low decay)
        ema.update(model)

        param_keys = {n for n, _ in model.named_parameters()}
        for key, shadow_val in ema.shadow.items():
            if key in param_keys and shadow_val.dtype.is_floating_point:
                # Should be close to 99.0 due to low warmup decay
                assert shadow_val.mean().item() > 50.0, \
                    f"Shadow[{key}] didn't track quickly after reset: {shadow_val.mean().item()}"


class TestEMAStateDictRoundtrip:
    """state_dict / load_state_dict preservation."""

    def test_roundtrip_preserves_shadow(self):
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)

        # Update a few times so shadow diverges from init
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        for _ in range(10):
            ema.update(model)

        # Save and restore
        saved = copy.deepcopy(ema.state_dict())
        ema2 = ModelEMA(model, decay=0.999)  # fresh EMA
        ema2.load_state_dict(saved)

        for key in ema.shadow:
            assert torch.allclose(ema.shadow[key], ema2.shadow[key]), \
                f"Shadow[{key}] not preserved after roundtrip"


class TestEMAApplyTo:
    """apply_to() loads shadow weights into a model."""

    def test_apply_to_overwrites_model_weights(self):
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)

        # Make shadow different from model
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(100.0)
        for _ in range(50):
            ema.update(model)

        # Save current shadow
        shadow_copy = {k: v.clone() for k, v in ema.shadow.items()}

        # Reset model to zeros
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()

        # apply_to should overwrite model with shadow
        ema.apply_to(model)

        for key in shadow_copy:
            assert torch.allclose(model.state_dict()[key], shadow_copy[key]), \
                f"apply_to didn't set model[{key}] to shadow values"
