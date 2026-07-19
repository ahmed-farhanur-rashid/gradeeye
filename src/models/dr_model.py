"""
Full model: ConvNeXt-Tiny backbone + CBAM (last 1-2 stages only) +
projection head + CORN output layer.

Assembles backbone.py, cbam.py, head.py, corn.py per plan Section 4.
"""
import torch.nn as nn

from src.models.backbone import build_backbone
from src.models.cbam import CBAM


class DRGradingModel(nn.Module):
    def __init__(self, pretrained: bool = True, use_cbam: bool = True,
                 cbam_num_stages: int = 2, num_thresholds: int = 4,
                 head_hidden_dim: int = 512, dropout: float = 0.3,
                 output_mode: str = "corn"):
        """
        use_cbam: toggle for the baseline run (plan Section 5 run matrix —
                  baseline has attention=None).
        cbam_num_stages: how many of the LAST backbone stages get CBAM
                         inserted (default 2, per plan Section 4).
        """
        super().__init__()
        self.backbone = build_backbone(pretrained=pretrained)
        num_stages = len(self.backbone.out_channels)
        cbam_num_stages = min(cbam_num_stages, num_stages)

        self.use_cbam = use_cbam
        if use_cbam:
            # Attach CBAM only to the last `cbam_num_stages` stages.
            self.cbam_modules = nn.ModuleDict()
            for stage_idx in range(num_stages - cbam_num_stages, num_stages):
                channels = self.backbone.out_channels[stage_idx]
                self.cbam_modules[str(stage_idx)] = CBAM(channels)
        else:
            self.cbam_modules = None

        final_channels = self.backbone.out_channels[-1]
        from src.models.head import ProjectionHead
        self.head = ProjectionHead(
            in_channels=final_channels,
            hidden_dim=head_hidden_dim,
            num_thresholds=num_thresholds,
            dropout=dropout,
            output_mode=output_mode,
        )

    def forward(self, x):
        stage_features = self.backbone(x)  # list of feature maps, shallow -> deep

        if self.use_cbam:
            for stage_idx, feat in enumerate(stage_features):
                key = str(stage_idx)
                if key in self.cbam_modules:
                    stage_features[stage_idx] = self.cbam_modules[key](feat)

        final_features = stage_features[-1]
        logits = self.head(final_features)  # (B, num_thresholds) CORN logits
        return logits

    def freeze_backbone(self):
        """
        For Phase 1: frozen-backbone, head-only training. 
        Note: We intentionally never freeze self.head here because Phase 1 is designed to train the head.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        if self.use_cbam:
            for param in self.cbam_modules.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """For Phase 2/3: full unfreeze."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        if self.use_cbam:
            for param in self.cbam_modules.parameters():
                param.requires_grad = True


def build_model(config: dict) -> DRGradingModel:
    """
    Build model from a config dict (see configs/*.yaml). output_mode is
    derived from loss_type: "corn" loss needs a CORN-shaped head,
    "ce" loss needs a standard softmax-shaped head.
    """
    model_cfg = config.get("model", {})
    output_mode = "corn" if config.get("loss_type", "corn") == "corn" else "softmax"
    return DRGradingModel(
        pretrained=model_cfg.get("pretrained", True),
        use_cbam=model_cfg.get("use_cbam", True),
        cbam_num_stages=model_cfg.get("cbam_num_stages", 2),
        num_thresholds=model_cfg.get("num_thresholds", 4),
        head_hidden_dim=model_cfg.get("head_hidden_dim", 512),
        dropout=model_cfg.get("dropout", 0.3),
        output_mode=output_mode,
    )
