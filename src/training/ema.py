"""EMA (Exponential Moving Average) shadow weights, per plan Section 6."""
import copy

import torch


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
        return self.shadow

    def load_state_dict(self, state_dict: dict):
        self.shadow = copy.deepcopy(state_dict)

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
        """Load EMA shadow weights into a model (e.g. for eval)."""
        model.load_state_dict(self.shadow)
