"""Shared pytest fixtures for gradeeye tests."""
import os
import sys

import pytest
import torch

# Ensure repo root is on sys.path so `src.*` imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def device():
    """Use CPU for deterministic, CI-friendly tests."""
    return torch.device("cpu")


@pytest.fixture
def dummy_model():
    """Build a tiny DRGradingModel (CORN head) for unit tests.

    Uses pretrained=False to avoid downloading weights in CI.
    """
    from src.models.dr_model import DRGradingModel

    return DRGradingModel(
        pretrained=False,
        use_cbam=True,
        cbam_num_stages=2,
        num_thresholds=4,
        head_hidden_dim=64,  # small for speed
        dropout=0.1,
        output_mode="corn",
    )


@pytest.fixture
def dummy_model_ce():
    """Build a tiny DRGradingModel (CE / softmax head) for unit tests."""
    from src.models.dr_model import DRGradingModel

    return DRGradingModel(
        pretrained=False,
        use_cbam=False,
        cbam_num_stages=0,
        num_thresholds=4,
        head_hidden_dim=64,
        dropout=0.1,
        output_mode="softmax",
    )


@pytest.fixture
def dummy_batch(device):
    """(images, labels) batch with correct shapes for 384×384 CORN model."""
    B = 8
    images = torch.randn(B, 3, 384, 384, device=device)
    labels = torch.randint(0, 5, (B,), device=device)
    return images, labels


@pytest.fixture
def small_batch(device):
    """Smaller 64×64 batch for fast integration tests."""
    B = 4
    images = torch.randn(B, 3, 64, 64, device=device)
    labels = torch.randint(0, 5, (B,), device=device)
    return images, labels
