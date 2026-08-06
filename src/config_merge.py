"""
Config merge helpers — pure-Python, no torch / torchvision / timm imports.

Lives outside `src/train.py` so it can be unit-tested and imported by other
tools (e.g. config linters, dry-run scripts) without dragging the full
training dependency tree.

The single non-trivial function is `merge_configs(model_cfg, lodo_cfg)`,
which composes a model-axis yaml with a LODO-axis yaml into the unified
config dict that `src.training.orchestrate.run_training` expects.

The legacy monolithic `lodo_<holdout>_<arch>.yaml` files had a flat
shape (everything in one file). The split form achieves the same thing
by overlaying LODO fields onto the model config + injecting manifest
paths into both phases + deriving `run_name`. The merge is constructed
so that the output is **deep-equal** to the legacy monolithic dict for
any (arch, holdout) pair — see tests/test_config_merge.py.
"""
from __future__ import annotations


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — `override` wins on scalar conflicts,
    nested dicts are merged key-wise."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def merge_configs(model_cfg: dict, lodo_cfg: dict) -> dict:
    """Merge one model-axis config with one LODO-axis config.

    Steps:
      1. Start with the model config as base.
      2. Overlay LODO-axis fields (holdout_source, manifest paths).
      3. Inject LODO manifest paths into BOTH phases' manifest_train/val.
      4. Derive `run_name` from holdout + arch (not stored in either yaml).

    The merged dict matches the legacy monolithic `lodo_<holdout>_<arch>.yaml`
    shape — `src.training.orchestrate.run_training` doesn't know or care
    which form produced it.
    """
    merged = _deep_merge(model_cfg, lodo_cfg)

    holdout = merged["holdout_source"]
    arch = merged["arch"]
    manifest_train = lodo_cfg.get("manifest_train",
                                   f"data/splits/lodo_{holdout}_train.csv")
    manifest_val = lodo_cfg.get("manifest_val",
                                 f"data/splits/lodo_{holdout}_val.csv")

    for phase_name in list(merged.get("phases", {}).keys()):
        merged["phases"][phase_name]["manifest_train"] = manifest_train
        merged["phases"][phase_name]["manifest_val"] = manifest_val

    merged["run_name"] = f"lodo_{holdout}_{arch}"
    return merged
