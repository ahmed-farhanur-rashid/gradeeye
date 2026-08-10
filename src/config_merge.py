"""
Config merge helpers — pure-Python, no torch / torchvision / timm imports.

Lives outside `src/train.py` so it can be unit-tested and imported by other
tools (e.g. config linters, dry-run scripts) without dragging the full
training dependency tree.

The merge is constructed to support a three-axis experimental matrix:

    1. Model axis  (`configs/models/<arch>.yaml`)     — what the backbone is.
    2. Variant axis (`configs/variants/<variant>.yaml`) — auxiliary channels
       (4ch / 5ch variants, in_chans: 4 or 5, plus seg_dir). Optional;
       absent for the 3-channel baseline.
    3. LODO axis   (`configs/lodo/<experiment>/<holdout>.yaml`) — which source
       is held out for the LODO fold.

`merge_configs(model_cfg, lodo_cfg, variant_cfg=None, experiment_name=None)`
returns a unified config dict in the shape that
`src.training.orchestrate.run_training` expects.

The merge order is: model → variant → LODO. LODO wins on conflicts because
its fields (holdout_source, manifest paths, run_name) are the most run-specific
and must override anything inherited. The variant overlays only `model.in_chans`,
`phases.*.seg_dir`, `log_dir`, and `checkpoint_dir` — it never overrides the
backbone (arch, batch_size, lr, etc.).

`run_name` is derived deterministically at invocation time:
`f"lodo_{holdout}_{arch}"` — never stored in any yaml.

Output isolation (so different experiments never overwrite each other):

- If a LODO config lives under `configs/<experiment_name>/`, the trainer
  scopes logs and checkpoints under that experiment_name:
  `saved/logs/<experiment_name>/<holdout>/<arch>/`.
- If a variant config sets its own `log_dir` / `checkpoint_dir`, those
  override the LODO-derived path because the variant is the more specific
  axis: variants share a single directory across all four LODO folds,
  distinguished by `run_name` in the filename.
- Pass `--experiment-name` to override the auto-detected parent dir.
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


# Fields that a variant config is allowed to set. Anything else in a
# variant yaml is ignored — variant configs are intentionally minimal so
# they can compose with any backbone without overriding its hyperparams.
_VARIANT_ALLOWED_TOP_LEVEL = {"model", "phases", "log_dir", "checkpoint_dir"}
_VARIANT_ALLOWED_MODEL_KEYS = {"in_chans"}
_VARIANT_ALLOWED_PHASE_KEYS = {"seg_dir"}


def _filter_variant(variant_cfg: dict) -> dict:
    """Drop any fields from a variant config that aren't in the allow-list.

    A variant config is intentionally minimal: it may only override
    `model.in_chans`, `phases.*.seg_dir`, `log_dir`, and `checkpoint_dir`.
    Anything else (arch, batch_size, lr, augmentation, etc.) is the
    backbone's responsibility and the variant must not override it.
    """
    out: dict = {}
    for k, v in variant_cfg.items():
        if k in ("log_dir", "checkpoint_dir"):
            out[k] = v
        elif k == "model" and isinstance(v, dict):
            m = {mk: mv for mk, mv in v.items() if mk in _VARIANT_ALLOWED_MODEL_KEYS}
            if m:
                out["model"] = m
        elif k == "phases" and isinstance(v, dict):
            p = {}
            for phase_name, phase_cfg in v.items():
                if isinstance(phase_cfg, dict):
                    p[phase_name] = {
                        pk: pv for pk, pv in phase_cfg.items()
                        if pk in _VARIANT_ALLOWED_PHASE_KEYS
                    }
            if p:
                out["phases"] = p
    return out


def merge_configs(model_cfg: dict, lodo_cfg: dict,
                  variant_cfg: dict | None = None,
                  experiment_name: str | None = None) -> dict:
    """Merge one model-axis config with one LODO-axis config and (optionally)
    one variant-axis config.

    Steps:
      1. Start with the model config as the base.
      2. If a variant config is provided, filter it to the allow-list
         (only `model.in_chans`, `phases.*.seg_dir`, `log_dir`,
         `checkpoint_dir`) and overlay it on top of the model config.
      3. Overlay LODO-axis fields (holdout_source, manifest paths).
      4. Inject LODO manifest paths into BOTH phases' manifest_train/val.
      5. Derive `run_name` from holdout + arch (not stored in any yaml).
      6. Set per-fold matrix layout for log/checkpoint dirs:
           If a variant config sets `log_dir` / `checkpoint_dir`, those
           win (variant is the most specific axis for output paths).
           Else if `experiment_name` is set, scope to
           `saved/logs/<experiment_name>/<holdout>/<arch>/`.
           Else (legacy), scope to `saved/logs/<holdout>/<arch>/`.
      7. Inject per-fold `seg_dir` if the LODO yaml provides one.
    """
    if variant_cfg is not None:
        variant_cfg = _filter_variant(variant_cfg)

    merged = _deep_merge(model_cfg, variant_cfg or {})
    merged = _deep_merge(merged, lodo_cfg)

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

    # Path resolution: variant > LODO-experiment > legacy.
    variant_set_log = variant_cfg is not None and "log_dir" in variant_cfg
    variant_set_ckpt = variant_cfg is not None and "checkpoint_dir" in variant_cfg
    if variant_set_log:
        # Variant's path is canonical; do not override.
        pass
    elif experiment_name:
        if merged.get("log_dir", "saved/logs") == "saved/logs":
            merged["log_dir"] = f"saved/logs/{experiment_name}/{holdout}/{arch}"
    else:
        if merged.get("log_dir", "saved/logs") == "saved/logs":
            merged["log_dir"] = f"saved/logs/{holdout}/{arch}"

    if variant_set_ckpt:
        pass
    elif experiment_name:
        if merged.get("checkpoint_dir", "saved/checkpoints") == "saved/checkpoints":
            merged["checkpoint_dir"] = (
                f"saved/checkpoints/{experiment_name}/{holdout}/{arch}"
            )
    else:
        if merged.get("checkpoint_dir", "saved/checkpoints") == "saved/checkpoints":
            merged["checkpoint_dir"] = f"saved/checkpoints/{holdout}/{arch}"

    # Per-fold seg_dir: optional in the LODO yaml.
    if "seg_dir" in lodo_cfg:
        seg_dir = lodo_cfg["seg_dir"]
        for phase_name in list(merged.get("phases", {}).keys()):
            merged["phases"][phase_name]["seg_dir"] = seg_dir

    return merged
