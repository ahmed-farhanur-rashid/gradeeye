import torch
from torchvision.transforms import functional as F

def tta_forward(model, images, output_mode="corn"):
    """
    Test-time augmentation: averages probabilities across raw, horizontal flip,
    vertical flip, and 180 degree rotation.
    Returns the averaged conditional (CORN) or marginal (CE) probabilities.
    """
    # 1. Original
    logits_orig = model(images)
    
    # 2. Horizontal Flip
    images_hflip = F.hflip(images)
    logits_hflip = model(images_hflip)
    
    # 3. Vertical Flip
    images_vflip = F.vflip(images)
    logits_vflip = model(images_vflip)
    
    # 4. Rotation 180 (helpful for fundus images)
    images_rot = F.rotate(images, 180)
    logits_rot = model(images_rot)
    
    if output_mode == "corn":
        from src.models.corn import corn_logits_to_probas
        prob_orig = corn_logits_to_probas(logits_orig)
        prob_hflip = corn_logits_to_probas(logits_hflip)
        prob_vflip = corn_logits_to_probas(logits_vflip)
        prob_rot = corn_logits_to_probas(logits_rot)
    else:
        prob_orig = torch.softmax(logits_orig, dim=1)
        prob_hflip = torch.softmax(logits_hflip, dim=1)
        prob_vflip = torch.softmax(logits_vflip, dim=1)
        prob_rot = torch.softmax(logits_rot, dim=1)
        
    avg_prob = (prob_orig + prob_hflip + prob_vflip + prob_rot) / 4.0
    return avg_prob

def tta_predict(avg_probas, output_mode="corn"):
    """
    Converts averaged probabilities back into predicted classes.
    """
    if output_mode == "corn":
        unconditional_probas = torch.cumprod(avg_probas, dim=1)
        return (unconditional_probas > 0.5).sum(dim=1)
    else:
        return avg_probas.argmax(dim=1)
        
def tta_predict_probas(avg_probas, output_mode="corn"):
    """
    Converts averaged conditional probas into full class distributions.
    """
    if output_mode == "corn":
        unconditional_probas = torch.cumprod(avg_probas, dim=1)
        batch_size = avg_probas.size(0)
        num_thresholds = avg_probas.size(1)
        num_classes = num_thresholds + 1

        class_probas = torch.zeros(batch_size, num_classes, device=avg_probas.device)
        class_probas[:, 0] = 1 - unconditional_probas[:, 0]
        for k in range(1, num_thresholds):
            class_probas[:, k] = unconditional_probas[:, k - 1] - unconditional_probas[:, k]
        class_probas[:, num_thresholds] = unconditional_probas[:, num_thresholds - 1]

        return class_probas.clamp(min=0.0)
    else:
        return avg_probas
