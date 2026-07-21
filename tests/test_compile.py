import os
import sys
import torch
import json

sys.path.insert(0, "/home/farhan/my-projects/gradeeye")

from src.models.dr_model import build_model
from src.data.datasets import DRDataset
from torch.utils.data import DataLoader
from src.augmentation.transforms import build_eval_transforms
from src.losses.corn_loss import corn_loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "loss_type": "corn",
        "model": {
            "arch": "efficientnetv2_s",
            "pretrained": True,
            "use_cbam": True,
            "cbam_num_stages": 2,
            "num_thresholds": 4,
            "head_hidden_dim": 512,
            "dropout": 0.4
        }
    }
    
    # 1. Test without compile
    print("Testing WITHOUT torch.compile:")
    model = build_model(config).to(device)
    model.freeze_backbone() # freeze backbone
    
    val_manifest = "/home/farhan/my-projects/gradeeye/data/splits/eyepacs_val.csv"
    norm_stats_path = "/home/farhan/my-projects/gradeeye/data/processed/eyepacs_norm_stats.json"
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    dataset = DRDataset(val_manifest, norm_stats, transform=build_eval_transforms())
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Eval mode before any training
    model.eval()
    with torch.no_grad():
        logits_eval_init = model(images)
    print("  Initial Eval Logits - Min:", logits_eval_init.min().item(), "Max:", logits_eval_init.max().item())
    
    # Let's run a few mock training steps
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for step in range(10):
        optimizer.zero_grad()
        logits = model(images)
        loss = corn_loss(logits.float(), labels, 5)
        loss.backward()
        optimizer.step()
        print(f"  Step {step} Train Loss: {loss.item():.4f}")
        
    model.eval()
    with torch.no_grad():
        logits_eval_after = model(images)
    print("  After 10 Steps Eval Logits - Min:", logits_eval_after.min().item(), "Max:", logits_eval_after.max().item())

    # 2. Test WITH compile
    print("\nTesting WITH torch.compile:")
    model_c = build_model(config).to(device)
    model_c.freeze_backbone()
    model_c = torch.compile(model_c)
    
    model_c.eval()
    with torch.no_grad():
        logits_eval_init_c = model_c(images)
    print("  Initial Compiled Eval Logits - Min:", logits_eval_init_c.min().item(), "Max:", logits_eval_init_c.max().item())
    
    model_c.train()
    optimizer_c = torch.optim.Adam(model_c.parameters(), lr=0.001)
    for step in range(10):
        optimizer_c.zero_grad()
        logits = model_c(images)
        loss = corn_loss(logits.float(), labels, 5)
        loss.backward()
        optimizer_c.step()
        print(f"  Step {step} Compiled Train Loss: {loss.item():.4f}")
        
    model_c.eval()
    with torch.no_grad():
        logits_eval_after_c = model_c(images)
    print("  After 10 Steps Compiled Eval Logits - Min:", logits_eval_after_c.min().item(), "Max:", logits_eval_after_c.max().item())

if __name__ == "__main__":
    main()

