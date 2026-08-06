"""
Backbone feature extractors via timm.

Five LayerNorm-native architectures for the journal ensemble:
  - ConvNeXt-Tiny / ConvNeXt-Small (within-family scaling)
  - MaxViT-Tiny @ 384 (hybrid CNN-transformer)
  - DeiT-III-Small @ 384 (pure transformer)
  - SwinV2-Base @ 384 (hierarchical transformer)

All backbones expose the same contract:
  - .out_channels: list[int]  (shallow -> deep)
  - forward(x) -> list[Tensor]  (per-stage feature maps, shallow -> deep)

The ARCH_REGISTRY is the single source of truth for architecture metadata.
Adding a new architecture is a one-line change here, plus a YAML config.

NOTE: EfficientNetV2-S was deliberately excluded from the journal ensemble
after its BatchNorm running-statistics divergence was identified as the
root cause of repeated eval-mode logit explosions at the frozen->unfrozen
phase transition (see saved/logs/effnetv2_root_cause.md). Replacing BN
with GN fixes the explosion but caused class collapse, so the architecture
is excluded entirely rather than used with a workaround.
"""
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

# timm is imported lazily inside TimmBackbone.__init__ to avoid forcing
# the torchvision import chain (which is broken on some torch/torchvision
# version combos, e.g. PyTorch 2.x on Python 3.14). The metadata
# definitions below don't need timm at all — only model construction does.


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
        import timm  # lazy: avoid torchvision import chain at module import time
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
# NOTE: All LayerNorm-native for journal-publishable 5-model ensemble.
ARCH_REGISTRY: dict[str, ArchSpec] = {
    # --- 5-model LayerNorm-native journal ensemble ---
    # All accept 384x384 input; three have native 384x384 pretrained weights.
    # EfficientNetV2-S deliberately excluded: 109 BatchNorm2d layers cause
    # stale running-stats at the frozen->unfrozen phase transition
    # (see saved/logs/effnetv2_root_cause.md).
    "convnext_tiny": ArchSpec(
        timm_name="convnext_tiny",
        channels_last=True,   # depthwise convs benefit from channels_last
    ),
    "convnext_small": ArchSpec(
        timm_name="convnext_small.fb_in22k_ft_in1k",  # 22k pretrain, much stronger
        channels_last=True,
    ),
    "maxvit_tiny_tf_384": ArchSpec(
        timm_name="maxvit_tiny_tf_384",
        channels_last=False,
    ),
    "deit3_small_patch16_384": ArchSpec(
        timm_name="deit3_small_patch16_384.fb_in1k",
        channels_last=False,
        needs_img_size_kwarg=True,  # DeiT requires explicit img_size
    ),
    # SwinV2-Base @ 384 (native ImageNet-22k pretrained at 384, NOT resized):
    # Microsoft released window12to24 192to384 finetuned checkpoints that
    # natively train at 384x384 with continuous relative position bias. This
    # is the strongest LayerNorm-native transformer option at 384 we can
    # actually download pretrained weights for in this environment.
    "swinv2_base_window12to24_192to384": ArchSpec(
        timm_name="swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        channels_last=False,
        needs_img_size_kwarg=True,
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