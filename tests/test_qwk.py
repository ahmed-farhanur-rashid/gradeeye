import os
import sys
import torch
import json
from collections import Counter

sys.path.insert(0, "/home/farhan/my-projects/gradeeye")

from src.models.dr_model import build_model
from src.data.datasets import DRDataset
from torch.utils.data import DataLoader
from src.augmentation.transforms import build_eval_transforms
from src.losses.corn_loss import corn_loss
from src.models.corn import corn_predict
from src.eval.metrics import quadratic_weighted_kappa

def main():
    device = "cpu"
    ckpt_path = "/home/farhan/my-projects/gradeeye/saved/checkpoints/ensemble_effnetv2_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    
    model = build_model(config).to(device)
    # Ensure backbone is in eval mode since we froze it
    model.freeze_backbone()
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()})
    model.eval()
    
    val_manifest = "/home/farhan/my-projects/gradeeye/data/splits/eyepacs_val.csv"
    norm_stats_path = "/home/farhan/my-projects/gradeeye/data/processed/eyepacs_norm_stats.json"
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    dataset = DRDataset(val_manifest, norm_stats, transform=build_eval_transforms())
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            if idx >= 5:
                break
            images = images.to(device)
            logits = model(images)
            preds = corn_predict(logits)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    preds_cat = torch.cat(all_preds).numpy()
    labels_cat = torch.cat(all_labels).numpy()
    
    print("Preds distribution:", Counter(preds_cat))
    print("Labels distribution:", Counter(labels_cat))
    
    qwk = quadratic_weighted_kappa(labels_cat, preds_cat)
    print(f"Calculated QWK: {qwk:.6f}")

if __name__ == "__main__":
    main()
