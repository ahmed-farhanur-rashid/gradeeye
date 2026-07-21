import os
import sys
import torch
import json

sys.path.insert(0, "/home/farhan/my-projects/gradeeye")

from src.models.dr_model import build_model
from src.data.datasets import DRDataset
from torch.utils.data import DataLoader
from src.augmentation.transforms import build_eval_transforms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "/home/farhan/my-projects/gradeeye/saved/checkpoints/ensemble_effnetv2_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    
    model = build_model(config).to(device)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()})
    
    val_manifest = "/home/farhan/my-projects/gradeeye/data/splits/eyepacs_val.csv"
    norm_stats_path = "/home/farhan/my-projects/gradeeye/data/processed/eyepacs_norm_stats.json"
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    dataset = DRDataset(val_manifest, norm_stats, transform=build_eval_transforms())
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    images, labels = next(iter(loader))
    images = images.to(device)
    
    # We want to trace:
    # 1. model.backbone(images) outputs
    # 2. CBAM outputs (if used)
    # 3. GAP output (input to head.net)
    
    def print_tensor_stats(name, t):
        print(f"{name}:")
        print(f"  shape: {list(t.shape)}")
        print(f"  min:   {t.min().item():.4f}")
        print(f"  max:   {t.max().item():.4f}")
        print(f"  mean:  {t.mean().item():.4f}")

    # TRAIN mode
    model.train()
    with torch.no_grad():
        stage_features = model.backbone(images)
        final_features = stage_features[-1]
        if model.use_cbam:
            final_features = model.cbam_modules[str(len(stage_features)-1)](final_features)
        gap_out = model.head.gap(final_features).flatten(1)
        logits = model.head.net(gap_out)
        
    print("=== TRAIN MODE ===")
    print_tensor_stats("Backbone Final Output", stage_features[-1])
    print_tensor_stats("CBAM Output", final_features)
    print_tensor_stats("GAP Output (Input to Head Net)", gap_out)
    print_tensor_stats("Head Net Logits", logits)
    
    # EVAL mode
    model.eval()
    with torch.no_grad():
        stage_features_e = model.backbone(images)
        final_features_e = stage_features_e[-1]
        if model.use_cbam:
            final_features_e = model.cbam_modules[str(len(stage_features_e)-1)](final_features_e)
        gap_out_e = model.head.gap(final_features_e).flatten(1)
        logits_e = model.head.net(gap_out_e)
        
    print("\n=== EVAL MODE ===")
    print_tensor_stats("Backbone Final Output", stage_features_e[-1])
    print_tensor_stats("CBAM Output", final_features_e)
    print_tensor_stats("GAP Output (Input to Head Net)", gap_out_e)
    print_tensor_stats("Head Net Logits", logits_e)

if __name__ == "__main__":
    main()
