"""
Backbone feature extractors via timm.

Supports ConvNeXt-Tiny (primary, plan Section 4) and EfficientNetV2-S
(ensemble member, plan_extension Section 2).

All backbones expose the same contract:
  - .out_channels: list[int]  (shallow → deep)
  - forward(x) → list[Tensor]  (per-stage feature maps, shallow → deep)
"""
import timm
import torch.nn as nn


class TimmBackbone(nn.Module):
    """Generic timm features_only wrapper. Exposes per-stage feature maps
    for CBAM insertion and reports channel counts for downstream modules."""

    def __init__(self, arch: str, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            arch,
            pretrained=pretrained,
            features_only=True,
        )
        self.feature_info = self.model.feature_info
        self.out_channels = [info["num_chs"] for info in self.feature_info]

    def forward(self, x):
        """Returns a list of feature maps, one per stage (shallow -> deep)."""
        return self.model(x)


# Supported architectures — add new timm model names here.
_SUPPORTED_ARCHS = {
    "convnext_tiny": "convnext_tiny",
    "efficientnetv2_s": "tf_efficientnetv2_s",
}


def build_backbone(pretrained: bool = True, arch: str = "convnext_tiny") -> TimmBackbone:
    timm_name = _SUPPORTED_ARCHS.get(arch)
    if timm_name is None:
        raise ValueError(
            f"Unknown backbone arch: {arch!r}. "
            f"Supported: {list(_SUPPORTED_ARCHS.keys())}"
        )
    return TimmBackbone(timm_name, pretrained=pretrained)
