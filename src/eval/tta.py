import torch
from torchvision.transforms import functional as F


def tta_forward(model, images, output_mode="corn"):
    """
    Test-time augmentation: run the model on raw, h-flipped, v-flipped, and
    180-rotated copies. Decode each pass into FULL class probabilities, then
    average the class distributions.

    Why decode-then-average (not average-then-decode):
    Averaging conditional probabilities P(y>k | y>k-1) across TTA passes
    and then doing cumprod to recover unconditional P(y>k) is mathematically
    wrong — it breaks CORN's rank-consistency guarantee because the
    averaged conditional probas may no longer correspond to any valid joint
    distribution. The safer path is: each TTA pass independently produces
    its own valid class distribution, then average those distributions.
    """
    from src.models.corn import corn_predict_probas

    all_class_probs = []

    # 1. Original
    all_class_probs.append(corn_predict_probas(model(images)) if output_mode == "corn"
                           else torch.softmax(model(images), dim=1))

    # 2. Horizontal Flip
    all_class_probs.append(corn_predict_probas(model(F.hflip(images))) if output_mode == "corn"
                           else torch.softmax(model(F.hflip(images)), dim=1))

    # 3. Vertical Flip
    all_class_probs.append(corn_predict_probas(model(F.vflip(images))) if output_mode == "corn"
                           else torch.softmax(model(F.vflip(images)), dim=1))

    # 4. Rotation 180 (helpful for fundus images)
    all_class_probs.append(corn_predict_probas(model(F.rotate(images, 180))) if output_mode == "corn"
                           else torch.softmax(model(F.rotate(images, 180)), dim=1))

    avg_class_probs = torch.stack(all_class_probs).mean(dim=0)
    # Re-normalize for floating-point drift.
    avg_class_probs = avg_class_probs.clamp(min=0.0)
    avg_class_probs = avg_class_probs / avg_class_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return avg_class_probs


def tta_predict(avg_class_probs, output_mode="corn"):
    """Argmax on the already-averaged class distribution."""
    return avg_class_probs.argmax(dim=1)


def tta_predict_probas(avg_class_probs, output_mode="corn"):
    """Already a class distribution — return as-is."""
    return avg_class_probs
