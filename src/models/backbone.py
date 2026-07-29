"""
Backbone feature extractors via timm.

Supports ConvNeXt-Tiny, EfficientNetV2-S, and Swin-Tiny@384 (plan Section 4 /
Section 6). All backbones expose the same contract:
  - .out_channels: list[int]  (shallow -> deep)
  - forward(x) -> list[Tensor]  (per-stage feature maps, shallow -> deep)

The ARCH_REGISTRY is the single source of truth for architecture metadata.
Adding a new architecture is a one-line change here, plus a YAML config.
"""
from dataclasses import dataclass
from typing import Optional

import timm
import torch.nn as nn


@dataclass(frozen=True)
class ArchSpec:
    """Per-architecture metadata used by training/inference plumbing."""
    timm_name: str                 # timm.create_model name
    channels_last: bool = False    # whether to use channels_last memory format
    supports_cbam: bool = True     # whether CBAM insertion makes architectural sense
    needs_img_size_kwarg: bool = False  # whether timm requires img_size for this arch


class TimmBackbone(nn.Module):
    """Generic timm features_only wrapper. Exposes per-stage feature maps
    for CBAM insertion and reports channel counts for downstream modules."""

    def __init__(self, arch: str, pretrained: bool = True, img_size: int | None = None,
                 in_chans: int = 3):
        super().__init__()
        kwargs = dict(pretrained=pretrained, features_only=True, in_chans=in_chans)
        if img_size is not None:
            kwargs["img_size"] = (img_size, img_size)
        self.model = timm.create_model(arch, **kwargs)
        self.feature_info = self.model.feature_info
        self.out_channels = [info["num_chs"] for info in self.feature_info]

    def forward(self, x):
        """Returns a list of feature maps, one per stage (shallow -> deep)."""
        return self.model(x)


# Single source of truth for supported architectures.
# Add a new architecture = add one line here + create a YAML config.
ARCH_REGISTRY: dict[str, ArchSpec] = {
    "convnext_tiny": ArchSpec(
        timm_name="convnext_tiny",
        channels_last=True,   # depthwise convs benefit from channels_last
    ),
    "efficientnetv2_s": ArchSpec(
        timm_name="tf_efficientnetv2_s",
        channels_last=False,  # BN-heavy; standard contiguous works fine
    ),
    # SwinV2-Tiny @ 256: timm ships pretrained weights as
    # `swinv2_tiny_window8_256.ms_in1k`. The _ns and _cr variants @ 384
    # don't have downloadable weights in the registry this environment
    # can fetch, so we use the standard 256 SwinV2 (window8). Training
    # input must be 256x256 — set `model.img_size: 256` in the YAML and
    # the train pipeline handles the resize from 384-preprocessed images.
    "swin_tiny_patch4_window7_384": ArchSpec(
        timm_name="swinv2_tiny_window8_256",
        channels_last=False,
        needs_img_size_kwarg=False,
    ),
}


def get_arch_spec(arch: str) -> ArchSpec:
    """Look up architecture metadata, raising a clear error on unknown archs."""
    spec = ARCH_REGISTRY.get(arch)
    if spec is None:
        raise ValueError(
            f"Unknown backbone arch: {arch!r}. "
            f"Supported: {list(ARCH_REGISTRY.keys())}. "
            f"Add a new entry to ARCH_REGISTRY in src/models/backbone.py to register a new architecture."
        )
    return spec


# Backwards-compatible alias for code that imported the old dict name.
_SUPPORTED_ARCHS = {alias: spec.timm_name for alias, spec in ARCH_REGISTRY.items()}


def build_backbone(pretrained: bool = True, arch: str = "convnext_tiny",
                   in_chans: int = 3, img_size: int | None = None,
                   use_groupnorm: bool = False) -> TimmBackbone:
    """
    in_chans: number of input channels. Default 3 (RGB). Set to 4 to accept
              an optional 4th-channel segmentation mask concatenated to RGB.
              timm handles in_chans != 3 by resizing the first conv weight
              (initializes new channels as the mean of existing RGB channels,
              so pretrained RGB features are preserved).
    img_size: optional override of input spatial size. Required for Swin-Tiny
              because the pretrained variant is at 224 and our preprocessing
              produces 384x384 images.
    use_groupnorm: if True, replace every BatchNorm2d in the backbone with
                   GroupNorm. This is used for EffNetV2-S where BN's running
                   statistics were unstable at the phase transition.
    """
    spec = get_arch_spec(arch)
    img_size_arg = img_size if spec.needs_img_size_kwarg else None
    backbone = TimmBackbone(spec.timm_name, pretrained=pretrained,
                             img_size=img_size_arg, in_chans=in_chans)
    if use_groupnorm:
        from src.models.groupnorm_swap import swap_bn_to_gn
        n = swap_bn_to_gn(backbone)
        print(f"  Replaced {n} BatchNorm2d with GroupNorm in {arch}")
    return backbone