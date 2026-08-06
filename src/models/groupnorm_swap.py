"""
Convert an EfficientNetV2-S checkpoint from BatchNorm to GroupNorm.

The motivation: BatchNorm2d's running_mean/running_var accumulated during
Phase 1 (frozen backbone, head-only gradients) is stale by the time Phase 2
unfreezes the backbone. The result is eval-mode logits that explode
(-68k to +367k) while train-mode logits are bounded (-3 to +3).

GroupNorm has no running statistics, so the phase transition can't break it.

This is a strict swap: BatchNorm2d → GroupNorm(num_groups, num_features).
Affine parameters (weight/bias) carry over 1:1. running_mean/running_var
are dropped. num_batches_tracked is dropped.

Usage:
    python -m src.models.groupnorm_swap \
        --input saved/checkpoints/full_method_multidomain_effnetv2_best.pt \
        --output saved/checkpoints/full_method_multidomain_effnetv2_gn_best.pt
"""
import argparse
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def num_groups_for_channels(channels: int) -> int:
    """Choose a sensible num_groups for GroupNorm given a channel count.

    Rules:
      - Channels < 32 → groups = channels (effectively InstanceNorm, but
        with learnable affine; matches the small-channel regime where
        InstanceNorm is known to be effective).
      - Otherwise, target 8 channels per group (a common starting point),
        clipped to a power of 2 if needed so it divides channels.
    """
    if channels < 32:
        return channels
    # Aim for ~8 channels per group, but pick a divisor of `channels`.
    target = max(1, channels // 8)
    # Common GroupNorm group counts to try, descending
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0 and g <= target * 2:
            return g
    return 1


def swap_bn_to_gn(model: nn.Module) -> int:
    """In-place swap all BatchNorm2d submodules to GroupNorm.

    Returns the number of swaps performed.
    """
    swaps = 0
    # Detect the device of the existing BN params so the new GN layer
    # lands on the same device.
    device = next(model.parameters()).device
    # Collect (parent, name, child) so we can setattr.
    to_replace = []
    for module in model.modules():
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                to_replace.append((module, name, child))
    for parent, name, bn in to_replace:
        channels = bn.num_features
        gn = nn.GroupNorm(
            num_groups=num_groups_for_channels(channels),
            num_channels=channels,
            eps=bn.eps,
            affine=True,
        ).to(device)
        # Carry over affine parameters
        with torch.no_grad():
            if bn.affine:
                gn.weight.copy_(bn.weight)
                gn.bias.copy_(bn.bias)
        setattr(parent, name, gn)
        swaps += 1
    return swaps


def main():
    parser = argparse.ArgumentParser(description="Swap BN→GN in an EffNetV2-S checkpoint.")
    parser.add_argument("--input", required=True, help="Path to input .pt checkpoint")
    parser.add_argument("--output", required=True, help="Path to write converted .pt")
    parser.add_argument("--arch", default="efficientnetv2_s",
                        help="Backbone arch (default: efficientnetv2_s)")
    parser.add_argument("--in-chans", type=int, default=4,
                        help="Input channels (default: 4)")
    parser.add_argument("--img-size", type=int, default=384)
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    ck = torch.load(args.input, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("ema_state_dict", {}))
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    # Build unconverted model just to inspect structure
    from src.models.dr_model import DRGradingModel
    model = DRGradingModel(
        pretrained=False, use_cbam=True, cbam_num_stages=2, num_thresholds=4,
        head_hidden_dim=512, dropout=0.4, output_mode="corn",
        arch=args.arch, in_chans=args.in_chans, img_size=args.img_size,
    )
    pre_swaps = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    print(f"BN2d layers found: {pre_swaps}")

    # Try to load state dict BEFORE the swap — at this stage the model
    # still has BN layers so the keys match.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        # filter out the BN running stats that we'll drop anyway
        missing = [k for k in missing if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        if missing:
            print(f"Missing (other): {missing[:5]}...")
    if unexpected:
        unexpected = [k for k in unexpected if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        if unexpected:
            print(f"Unexpected (other): {unexpected[:5]}...")

    # Now do the swap in-place
    swaps = swap_bn_to_gn(model)
    print(f"Swapped {swaps} BN2d → GroupNorm layers")

    # Sanity check: forward pass on a random batch
    model.eval()
    x = torch.randn(2, args.in_chans, args.img_size, args.img_size)
    with torch.no_grad():
        out = model(x)
    print(f"Sanity forward: logits shape={tuple(out.shape)}, "
          f"min={out.min().item():.3f}, max={out.max().item():.3f}")

    # Save converted state dict (and update ema_state_dict the same way)
    new_sd = model.state_dict()
    out_ck = dict(ck)
    if "model_state_dict" in out_ck:
        out_ck["model_state_dict"] = new_sd
    if "ema_state_dict" in out_ck and out_ck["ema_state_dict"] is not None:
        # EMA shadow was keyed by the compiled (BN) version. Just overwrite
        # with the new GN keys; the EMA buffer will re-accumulate from the
        # first training step.
        out_ck["ema_state_dict"] = new_sd

    # Drop optimizer state — won't match the new parameter layout
    out_ck.pop("optimizer_state_dict", None)
    out_ck.pop("scheduler_state_dict", None)
    # Reset step counters and best metric — fresh training
    out_ck["global_step"] = 0
    out_ck["epoch"] = 0
    out_ck["best_metric"] = -1.0
    out_ck["phase_name"] = None
    # Annotate the conversion in the config so training logs show it
    if "config" in out_ck and isinstance(out_ck["config"], dict):
        out_ck["config"]["_converted_from_bn"] = True

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(out_ck, args.output)
    print(f"Wrote: {args.output}")
    print("Note: this checkpoint will load a fresh EMA at the first "
          "training step. Optimizer state was reset.")


if __name__ == "__main__":
    main()