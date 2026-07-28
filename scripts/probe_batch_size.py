"""
Probe the maximum batch size that fits in VRAM for each backbone.

Runs a forward + backward + optimizer step with synthetic data at
increasing batch sizes and reports the largest size that completed
without OOM, plus peak VRAM usage.

Run:
    python scripts/probe_batch_size.py
    python scripts/probe_batch_size.py --arch swin_tiny_patch4_window7_384

This is intentionally a one-off tool — it lives in scripts/ but is not
imported by the rest of the codebase.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dr_model import DRGradingModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _try_one(arch: str, in_chans: int, batch_size: int, frozen: bool, img_size: int):
    """Try a single fwd+bwd at the given bs. Returns (ok: bool, peak_gb: float).
    img_size: backbone-native input size (384 for ConvNeXt, 256 for Swin)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = DRGradingModel(
            pretrained=False,  # avoid network download in probe
            arch=arch,
            in_chans=in_chans,
            use_cbam=True,
            cbam_num_stages=2,
            num_thresholds=4,
            head_hidden_dim=512,
            dropout=0.4,
            img_size=img_size,
        ).to(DEVICE)
        # Apply channels_last if the arch's spec says so (matches train.py).
        from src.models.backbone import get_arch_spec
        if get_arch_spec(arch).channels_last:
            model = model.to(memory_format=torch.channels_last)
        # NOTE: torch.compile is NOT applied here — the probe is meant to
        # give a quick upper bound; compile-time memory pools vary across
        # inductor versions and can mask the real fit. Train.py still uses
        # compile in production.
        if frozen:
            model.freeze_backbone()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )
        x = torch.randn(batch_size, in_chans, img_size, img_size, device=DEVICE)
        if get_arch_spec(arch).channels_last:
            x = x.to(memory_format=torch.channels_last)
        y = torch.randint(0, 5, (batch_size,), device=DEVICE)

        # Match training-time autocast behavior so the probe is representative.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        # CORN-style 4-way logistic loss
        loss = logits.sum()
        loss.backward()
        opt.step()

        peak = torch.cuda.max_memory_allocated() / 1024**3
        del model, opt, x, y, logits, loss
        return True, peak
    except torch.cuda.OutOfMemoryError:
        return False, float("nan")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False, float("nan")
        raise


def probe_arch(arch: str, in_chans: int = 3, img_size: int = 384):
    """Walk batch sizes 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384 for both
    frozen-backbone and unfrozen scenarios; report the largest size that fits."""
    print(f"\n=== {arch} (in_chans={in_chans}, img_size={img_size}) ===")
    sizes = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    for frozen in (True, False):
        phase = "Phase 1 frozen" if frozen else "Phase 2 unfrozen"
        print(f"\n  {phase}:")
        best = None
        for bs in sizes:
            ok, peak = _try_one(arch, in_chans, bs, frozen=frozen, img_size=img_size)
            status = "OK" if ok else "OOM"
            print(f"    bs={bs:<4}  {status:<4}  peak={peak:.2f} GB")
            if ok:
                best = (bs, peak)
            else:
                break
        if best is not None:
            print(f"  -> {phase} max bs = {best[0]} (peak {best[1]:.2f} GB)")
        else:
            print(f"  -> {phase}: NO batch size fit!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", action="append",
                        help="Specific arch to probe (can repeat). Default: all registered.")
    parser.add_argument("--in-chans", type=int, default=3,
                        help="Input channels (3 RGB, or 4 RGB+vessel).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA available; nothing to probe.")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    archs = args.arch
    if not archs:
        from src.models.backbone import ARCH_REGISTRY
        archs = list(ARCH_REGISTRY.keys())

    for arch in archs:
        try:
            img_size = 256 if "swin" in arch else 384
            probe_arch(arch, in_chans=args.in_chans, img_size=img_size)
        except Exception as e:
            print(f"  Skipped {arch}: {e}")


if __name__ == "__main__":
    main()