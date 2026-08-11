"""Generate config.json, modeling.py, and README.md for each staged HF repo.

This writes only under staging/hf/<repo>/ and does not contact Hugging Face.
All descriptions use the verified fourth-channel semantics:
  soft    = raw BCE+Dice U-Net sigmoid probability map
  tversky = raw Tversky U-Net sigmoid probability map
  morph   = BCE+Dice soft map after 3x3 opening, 3x3 closing, Gaussian blur sigma .5
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGING = REPO / "staging/hf"
CHECKPOINT_ROOT = REPO / "saved/checkpoints"
GITHUB_URL = "https://github.com/ahmed-farhanur-rashid/gradeeye"

VARIANT_EXTRA = {
    "four-ch-soft": {
        "source": "segmentation_pooled_soft",
        "producer": "gradeeye/seg-unet-bcedice",
        "description": "Raw sigmoid probability map from the BCE+Dice-trained U-Net; uint16 PNG divided by 65535 to [0,1]. No morphological post-processing.",
    },
    "four-ch-morph": {
        "source": "segmentation_pooled_morph",
        "producer": "gradeeye/seg-unet-bcedice",
        "description": "The BCE+Dice U-Net raw sigmoid probability map after 3x3 elliptical morphological opening, 3x3 elliptical closing, and Gaussian blur (sigma=0.5). It remains a continuous [0,1] soft-probability map, not a morphological gradient or edge channel.",
    },
    "four-ch-tversky": {
        "source": "segmentation_pooled_tversky",
        "producer": "gradeeye/seg-unet-tversky",
        "description": "Raw sigmoid probability map from the Tversky-trained U-Net; uint16 PNG divided by 65535 to [0,1]. No morphological post-processing.",
    },
}


def read_meta(path: Path) -> dict:
    return json.loads(path.read_text())


def classifier_configs(repo: str, files: list[Path]) -> dict:
    # Each checkpoint gets a config alongside its weights for baseline repos.
    # Four-channel repos share one root config; use the first metadata row.
    first_meta = read_meta(sorted(files)[0])
    cfg = first_meta["training_config"]
    variant = repo
    in_chans = cfg["in_chans"]
    payload = {
        "model_type": "DRGradingModel",
        "task": "diabetic-retinopathy-grading",
        "architecture": cfg["arch"],
        "backbone_timm_name": cfg["arch"],
        "in_channels": in_chans,
        "image_size": cfg["img_size"],
        "num_classes": 5,
        "ordinal_head": "CORN",
        "num_thresholds": cfg.get("num_thresholds", 4),
        "use_cbam": cfg.get("use_cbam", True),
        "cbam_num_stages": cfg.get("cbam_num_stages", 2),
        "head_hidden_dim": cfg["head_hidden_dim"],
        "dropout": cfg["dropout"],
        "channels_last": cfg.get("channels_last"),
        "ema_decay": cfg.get("ema_decay"),
        "loss_type": cfg.get("loss_type", "corn"),
        "use_class_weighting": cfg.get("use_class_weighting"),
        "use_mixup": cfg.get("use_mixup"),
        "primary_weights": "*_best_ema.safetensors (EMA; reproduces paper metrics)",
        "secondary_weights": "*_best.safetensors (raw model_state_dict)",
        "normalization": {
            "rgb": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "extra_channel": "Already in [0,1]; appended after RGB normalization; not ImageNet-normalized.",
        },
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
    }
    if in_chans == 4:
        payload["extra_channel"] = VARIANT_EXTRA[variant]
    return payload


def seg_config(repo: str, meta: dict) -> dict:
    return {
        "model_type": "UNetVessel",
        "task": "retinal-lesion-segmentation",
        "encoder": meta["encoder"],
        "pretrained_encoder": True,
        "in_channels": 3,
        "out_channels": 1,
        "image_size": meta["img_size"],
        "loss": meta["loss"],
        "training_source": meta["training_source"],
        "primary_weights": "best.safetensors",
        "best_val_dice": meta["best_val_dice"],
        "best_epoch": meta["best_epoch"],
        "output": "Logits; apply sigmoid to obtain a probability map.",
        "downstream_consumers": [
            "gradeeye/four-ch-soft",
            "gradeeye/four-ch-morph",
        ] if repo == "seg-unet-bcedice" else ["gradeeye/four-ch-tversky"],
    }


def classifier_modeling() -> str:
    return '''"""Minimal GradeEye classifier loader for Hugging Face Hub.

Requires the GradeEye source package on PYTHONPATH plus torch, timm, and
safetensors. The architecture is the same DRGradingModel used during training.
EMA files are the primary weights and reproduce the paper evaluation protocol.
"""
from __future__ import annotations

from pathlib import Path
import sys
import torch

# For local source checkout usage. Users may instead install the GradeEye package.
try:
    from src.models.dr_model import DRGradingModel
except ImportError as exc:
    raise ImportError(
        "Install/clone GradeEye and make its repository root available on PYTHONPATH."
    ) from exc
from safetensors.torch import load_file


def load_model(weights_path: str | Path, config: dict, device: str = "cpu") -> DRGradingModel:
    """Instantiate DRGradingModel and strictly load a .safetensors state dict."""
    model = DRGradingModel(
        pretrained=False,
        use_cbam=config["use_cbam"],
        cbam_num_stages=config["cbam_num_stages"],
        num_thresholds=config["num_thresholds"],
        head_hidden_dim=config["head_hidden_dim"],
        dropout=config["dropout"],
        output_mode="corn",
        arch=config["architecture"],
        in_chans=config["in_channels"],
        img_size=config["image_size"],
    )
    state_dict = load_file(str(weights_path), device="cpu")
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"State-dict mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return model.to(device).eval()
'''


def seg_modeling() -> str:
    return '''"""Minimal GradeEye U-Net loader for Hugging Face Hub.

Requires the GradeEye source package on PYTHONPATH plus torch, timm, and
safetensors. The output is a single logit channel; apply torch.sigmoid().
"""
from __future__ import annotations

from pathlib import Path
import torch
from safetensors.torch import load_file

try:
    from src.models.segmentation import UNetVessel
except ImportError as exc:
    raise ImportError(
        "Install/clone GradeEye and make its repository root available on PYTHONPATH."
    ) from exc


def load_model(weights_path: str | Path, config: dict, device: str = "cpu") -> UNetVessel:
    model = UNetVessel(encoder=config["encoder"], pretrained=False)
    state_dict = load_file(str(weights_path), device="cpu")
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"State-dict mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return model.to(device).eval()
'''


def classifier_readme(repo: str, config: dict, metas: list[dict]) -> str:
    is_four = config["in_channels"] == 4
    archs = sorted({m["training_config"]["arch"] for m in metas})
    if len(archs) == 1:
        arch_blurb = f"the `{archs[0]}` backbone"
    else:
        arch_blurb = f"three backbones ({', '.join(f'`{a}`' for a in archs)})"
    lines = [
        "---",
        "license: cc-by-nc-4.0",
        "tags:",
        "  - medical-imaging",
        "  - diabetic-retinopathy",
        "  - domain-generalization",
        "  - pytorch",
        "pipeline_tag: image-classification",
        "library_name: pytorch",
        "---",
        "",
        f"# GradeEye {repo}",
        "",
        "This repository contains GradeEye CORN ordinal diabetic-retinopathy "
        f"classifiers using {arch_blurb} at {config['image_size']}x{config['image_size']} resolution, with "
        f"{config['in_channels']}-channel input and a {config['num_thresholds']}-threshold ordinal head. "
        "The `_ema.safetensors` file is the **primary** artifact: the paper's "
        "reported evaluation metrics were generated using the EMA state dict. "
        "The unsuffixed `.safetensors` file is the corresponding raw "
        "`model_state_dict` secondary artifact.",
        "",
        "## Checkpoints",
        "",
        "| Primary EMA weights | Raw secondary weights | Architecture | Held-out fold | Best QWK | Epoch |",
        "|---|---|---|---|---:|---:|",
    ]
    for m in sorted(metas, key=lambda x: x["checkpoint_source"]):
        src = Path(m["checkpoint_source"])
        stem = src.stem
        fold = next((p for p in src.parts if p in {"aptos", "ddr", "eyepacs", "messidor2"}), "varies")
        arch = m["training_config"]["arch"]
        lines.append(
            f"| `{stem}_ema.safetensors` | `{stem}.safetensors` | `{arch}` | {fold} | "
            f"{m['best_metric']:.4f} | {m['epoch']} |"
        )
    lines += [
        "",
        "## Preprocessing",
        "",
        "1. Resize the RGB fundus image to 384x384 using the same offline preprocessing pipeline.",
        "2. Convert RGB to float in [0,1] and apply ImageNet normalization: "
        "mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).",
    ]
    if is_four:
        extra = config["extra_channel"]
        lines += [
            "3. Append one auxiliary channel in [0,1] after RGB normalization. "
            "It is not ImageNet-normalized.",
            "",
            "### Fourth-channel construction",
            "",
            f"- Source pool: `{extra['source']}`.",
            f"- Producer: `{extra['producer']}`.",
            f"- Exact method: {extra['description']}",
            "- The resulting tensor is `(4, 384, 384)` in channel-first layout.",
        ]
        if repo == "four-ch-soft":
            lines.append("- No opening, closing, or Gaussian blur is applied in this variant.")
        elif repo == "four-ch-tversky":
            lines.append("- No opening, closing, or Gaussian blur is applied in this variant; only the segmenter training loss differs from the BCE+Dice producer.")
        else:
            lines.append("- This is the only one of the three published four-channel variants with morphological post-processing; it is not a gradient or edge channel.")
    lines += [
        "",
        "## Loading",
        "",
        "```python",
        "import json",
        "from modeling import load_model",
        "",
        "config = json.load(open('config.json'))",
        "model = load_model('lodo_eyepacs_convnext_tiny_best_ema.safetensors', config)",
        "# model(x) returns CORN logits with shape (batch, 4)",
        "```",
        "",
        "Install `torch`, `timm`, and `safetensors`, and make the GradeEye source repository available on `PYTHONPATH`.",
        "",
        "## Intended use and limitations",
        "",
        "These weights are released for research and reproducibility only. They are not validated for clinical diagnosis or treatment decisions. Performance varies substantially by held-out dataset and should not be interpreted as clinical-grade generalization.",
        "",
        f"Source code and paper materials: [{GITHUB_URL}]({GITHUB_URL}).",
        "",
    ]
    return "\n".join(lines)


def seg_readme(repo: str, config: dict, meta: dict) -> str:
    consumers = ", ".join(f"[{x}](https://huggingface.co/{x})" for x in config["downstream_consumers"])
    loss_desc = "BCE+Dice" if config["loss"] == "bce_dice" else "Tversky"
    lines = [
        "---",
        "license: cc-by-nc-4.0",
        "tags:",
        "  - medical-imaging",
        "  - diabetic-retinopathy",
        "  - pytorch",
        "  - image-segmentation",
        "pipeline_tag: image-segmentation",
        "library_name: pytorch",
        "---",
        "",
        f"# GradeEye {repo}",
        "",
        f"This repository contains a retinal lesion segmentation `UNetVessel` with an EfficientNet-B4 encoder, trained at {config['image_size']}x{config['image_size']} using {loss_desc} loss on {config['training_source']}. The model outputs one logit channel; apply sigmoid to obtain a continuous probability map.",
        "",
        "## Checkpoint",
        "",
        "| File | Encoder | Image size | Loss | Best validation Dice | Best epoch |",
        "|---|---|---:|---|---:|---:|",
        f"| `best.safetensors` | `{config['encoder']}` | {config['image_size']} | `{config['loss']}` | {config['best_val_dice']:.4f} | {config['best_epoch']} |",
        "",
        "## Preprocessing and output",
        "",
        "Use RGB input, scale to [0,1], apply ImageNet mean=(0.485, 0.456, 0.406) and std=(0.229, 0.224, 0.225), and resize to 384x384. The output is `(batch, 1, 384, 384)` logits; apply `torch.sigmoid` for probabilities.",
        "",
        f"The masks produced by this model feed: {consumers}.",
        "",
        "## Loading",
        "",
        "```python",
        "import json",
        "import torch",
        "from modeling import load_model",
        "",
        "config = json.load(open('config.json'))",
        "model = load_model('best.safetensors', config)",
        "with torch.no_grad():",
        "    probability = torch.sigmoid(model(rgb_tensor))",
        "```",
        "",
        "Install `torch`, `timm`, and `safetensors`, and make the GradeEye source repository available on `PYTHONPATH`.",
        "",
        "## Intended use and limitations",
        "",
        "These weights are released for research and reproducibility only. They are not validated for clinical diagnosis or treatment decisions.",
        "",
        f"Source code and paper materials: [{GITHUB_URL}]({GITHUB_URL}).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if not STAGING.is_dir():
        raise SystemExit(f"Missing {STAGING}; run convert_to_safetensors.py first")

    for repo_dir in sorted(p for p in STAGING.iterdir() if p.is_dir()):
        repo = repo_dir.name
        modeling = seg_modeling() if repo.startswith("seg-unet-") else classifier_modeling()
        (repo_dir / "modeling.py").write_text(modeling)

        metas = sorted(repo_dir.rglob("*.meta.json"))
        if repo.startswith("seg-unet-"):
            meta = read_meta(metas[0])
            cfg = seg_config(repo, meta)
            (repo_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
            (repo_dir / "README.md").write_text(seg_readme(repo, cfg, meta))
        else:
            cfg = classifier_configs(repo, metas)
            # Root-level config for flat four-channel repos; per-checkpoint
            # configs for nested baseline repos are written below.
            if repo.startswith("four-ch-"):
                (repo_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
            else:
                for meta_path in metas:
                    meta = read_meta(meta_path)
                    source = Path(meta["checkpoint_source"])
                    # source = saved/checkpoints/<folder>/<fold>/<arch>/<file>
                    # Weights live at repo_dir/<fold>/<arch>/<files>
                    # The config should sit beside them.
                    source_parts = source.parts
                    rel_dir = Path(*source_parts[3:-1])
                    local_dir = repo_dir / rel_dir
                    local_dir.mkdir(parents=True, exist_ok=True)
                    local_cfg = dict(cfg)
                    local_cfg["architecture"] = meta["training_config"]["arch"]
                    local_cfg["backbone_timm_name"] = local_cfg["architecture"]
                    local_cfg["in_channels"] = meta["training_config"]["in_chans"]
                    local_cfg["image_size"] = meta["training_config"]["img_size"]
                    local_cfg["head_hidden_dim"] = meta["training_config"]["head_hidden_dim"]
                    local_cfg["dropout"] = meta["training_config"]["dropout"]
                    local_cfg["channels_last"] = meta["training_config"].get("channels_last")
                    local_cfg["held_out_fold"] = next((p for p in source.parts if p in {"aptos", "ddr", "eyepacs", "messidor2"}), None)
                    local_cfg["primary_weights"] = f"{source.stem}_ema.safetensors"
                    local_cfg["secondary_weights"] = f"{source.stem}.safetensors"
                    (local_dir / "config.json").write_text(json.dumps(local_cfg, indent=2) + "\n")
            (repo_dir / "README.md").write_text(classifier_readme(repo, cfg, [read_meta(p) for p in metas]))

    print(f"Generated metadata/model cards under {STAGING.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
