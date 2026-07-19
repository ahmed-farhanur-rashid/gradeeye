"""
Projection head (regularization funnel), per plan Section 4:

  GlobalAvgPool -> BatchNorm1d -> Dropout -> Linear(512) -> Mish ->
  BatchNorm1d -> Dropout -> Linear(4) -> CORN conditional-probability layer

The final Linear(4) here produces the 4 raw logits consumed by the CORN
layer (see corn.py) — this is NOT a standard Linear(num_classes=5) + softmax.
"""
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 512, num_thresholds: int = 4,
                 dropout: float = 0.3, output_mode: str = "corn"):
        """
        output_mode: "corn" -> final Linear outputs num_thresholds raw values
                     (num_classes - 1), consumed by the CORN layer.
                     "softmax" -> final Linear outputs num_thresholds + 1
                     raw values (num_classes), consumed by plain CE. Used
                     only by the baseline/ablation runs in plan Section 5's
                     run matrix that don't use ordinal structure.
        """
        super().__init__()
        if output_mode not in ("corn", "softmax"):
            raise ValueError(f"output_mode must be 'corn' or 'softmax', got {output_mode!r}")

        out_dim = num_thresholds if output_mode == "corn" else num_thresholds + 1
        self.output_mode = output_mode

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Dropout(dropout),
            nn.Linear(in_channels, hidden_dim),
            nn.Mish(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.gap(x).flatten(1)  # (B, in_channels, H, W) -> (B, in_channels)
        return self.net(x)  # (B, out_dim) raw logits -- CORN layer or CE, per output_mode
