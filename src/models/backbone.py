"""
ConvNeXt-Tiny backbone, ImageNet-pretrained via timm.

LOCKED per plan Section 4 / handoff note 1 — do not substitute
ResNet/EfficientNet/ViT. This was a deliberate literature-benchmarking
choice: DeiT/EfficientNet-class transformers underperform slightly on
APTOS at this data scale, and architecture swaps alone show diminishing
returns past VGG-era models on this dataset. The novelty lives in the
CBAM + CORN combination, not the backbone.
"""
import timm
import torch.nn as nn


class ConvNeXtTinyBackbone(nn.Module):
    """
    Feature extractor wrapper. Exposes intermediate stage feature maps
    (needed for CBAM insertion into the last 1-2 stages) as well as the
    final pooled feature vector.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # features_only=True gives access to per-stage feature maps for CBAM insertion.
        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            features_only=True,
        )
        self.feature_info = self.model.feature_info
        self.out_channels = [info["num_chs"] for info in self.feature_info]

    def forward(self, x):
        """Returns a list of feature maps, one per stage (shallow -> deep)."""
        return self.model(x)


def build_backbone(pretrained: bool = True) -> ConvNeXtTinyBackbone:
    return ConvNeXtTinyBackbone(pretrained=pretrained)
