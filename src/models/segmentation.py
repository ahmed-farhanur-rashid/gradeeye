"""
Lightweight U-Net for vessel/lesion segmentation.

Built directly on torch + torchvision (no segmentation_models_pytorch
dependency), using a timm encoder backbone. Used by Option A segmentation
to pre-compute vessel masks over EyePACS/APTOS/Messidor.

For inference we typically load weights pretrained on DRIVE (40 retinal
vessel segmentation images) — see `src.data._03_segmentation` and
`python -m src.data.cli segmentation` for the fine-tune + inference
pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    """Double-conv block used throughout the U-Net decoder."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetVessel(nn.Module):
    """
    Minimal U-Net with a timm encoder. Outputs a single-channel sigmoid
    vessel/lesion probability map at the same spatial size as the input.

    encoder: timm backbone name (e.g. "efficientnet_b0"). Only used in
             features_only mode so we get per-stage feature maps for skip
             connections.
    """

    def __init__(self, encoder: str = "efficientnet_b4", pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            encoder, pretrained=pretrained, features_only=True,
        )
        enc_channels = self.encoder.feature_info.channels()  # e.g. [32, 24, 40, 112, 320] for B0

        # Decoder: progressively upsample and concat with encoder skip features.
        # Encoder reductions are [2, 4, 8, 16, 32] so 5 upsamples recover input res.
        self.up5 = nn.ConvTranspose2d(enc_channels[4], 256, 2, stride=2)
        self.dec5 = ConvBlock(256 + enc_channels[3], 256)

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = ConvBlock(128 + enc_channels[2], 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(64 + enc_channels[1], 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(32 + enc_channels[0], 32)

        # Final upsample to match input resolution (no skip from encoder —
        # we already exhausted all 5 levels).
        self.up1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec1 = ConvBlock(32, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        skips = self.encoder(x)  # list of 5 feature maps, shallow -> deep
        # Bottleneck = deepest features (index 4)
        d = self.up5(skips[4])
        d = self.dec5(torch.cat([d, skips[3]], dim=1))

        d = self.up4(d)
        d = self.dec4(torch.cat([d, skips[2]], dim=1))

        d = self.up3(d)
        d = self.dec3(torch.cat([d, skips[1]], dim=1))

        d = self.up2(d)
        d = self.dec2(torch.cat([d, skips[0]], dim=1))

        d = self.up1(d)
        d = self.dec1(d)

        return self.final(d)  # (B, 1, H, W) logits; apply sigmoid externally


def build_unet_vessel(pretrained: bool = True, encoder: str = "efficientnet_b4") -> UNetVessel:
    """Convenience constructor — defaults match the EfficientNet-B4 IDRiD-fine-tuned weights."""
    return UNetVessel(encoder=encoder, pretrained=pretrained)