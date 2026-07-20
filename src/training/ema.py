"""EMA (Exponential Moving Average) shadow weights, per plan Section 6."""
import copy

import torch

_COMPILE_PREFIX = "_orig_mod."


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Strip torch.compile's '_orig_mod.' prefix from state_dict keys.

    torch.compile wraps models so all parameter names get prefixed with
    '_orig_mod.'.  EMA shadow weights are captured from the compiled model
    during training, so they carry this prefix.  At eval time the model is
    NOT compiled, so loading these keys directly fails.

    This function normalises keys so checkpoints are portable regardless
    of whether torch.compile was used at training time.
    """
    return {k.removeprefix(_COMPILE_PREFIX): v for k, v in state_dict.items()}


class ModelEMA:
    """
    Maintains a shadow copy of model weights updated via EMA after each
    optimizer step. Used for eval (typically more stable/generalizing than
    raw weights) and saved alongside the main checkpoint.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        for v in self.shadow.values():
            v.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.updates += 1
        # Warmup: rapidly update shadow weights early in training
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        
        model_state = model.state_dict()
        for key, shadow_val in self.shadow.items():
            model_val = model_state[key]
            if shadow_val.dtype.is_floating_point:
                shadow_val.mul_(d).add_(model_val.detach(), alpha=1 - d)
            else:
                # non-float buffers (e.g. BatchNorm num_batches_tracked) — just copy
                shadow_val.copy_(model_val)

    def state_dict(self) -> dict:
        """Return shadow weights with compile prefix stripped for portability."""
        return _strip_compile_prefix(self.shadow)

    def load_state_dict(self, state_dict: dict):
        self.shadow = copy.deepcopy(_strip_compile_prefix(state_dict))

    def reset(self, model: torch.nn.Module):
        """Re-snapshot shadow weights from current model and restart warmup.

        Must be called at phase transitions (e.g. Phase 1 → Phase 2) so the
        EMA doesn't drag stale frozen-backbone weights into the fine-tuning
        phase.  Without this, the warmup counter is already maxed out from
        Phase 1 and the decay stays at 0.999, making the shadow effectively
        stuck on Phase 1 weights.
        """
        self.shadow = copy.deepcopy(model.state_dict())
        for v in self.shadow.values():
            v.requires_grad_(False)
        self.updates = 0

    def apply_to(self, model: torch.nn.Module):
        """Load EMA shadow weights into a model (e.g. for eval).

        Handles both compiled (_orig_mod. prefix) and non-compiled models
        by matching keys from the shadow to whatever the model expects.
        """
        model_keys = set(model.state_dict().keys())
        shadow_clean = _strip_compile_prefix(self.shadow)

        # If model expects _orig_mod. keys (compiled), re-add the prefix
        if model_keys and next(iter(model_keys)).startswith(_COMPILE_PREFIX):
            sd = {_COMPILE_PREFIX + k: v for k, v in shadow_clean.items()}
        else:
            sd = shadow_clean

        model.load_state_dict(sd)
