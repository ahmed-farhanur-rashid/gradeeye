import torch

def main():
    ckpt_path = "/home/farhan/my-projects/gradeeye/saved/checkpoints/ensemble_effnetv2_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    
    print("Checking running stats in model_state_dict:")
    var_keys = [k for k in sd.keys() if "running_var" in k]
    mean_keys = [k for k in sd.keys() if "running_mean" in k]
    
    print(f"Total running_var buffers: {len(var_keys)}")
    print(f"Total running_mean buffers: {len(mean_keys)}")
    
    # Check for NaNs or Infs
    nan_vars = 0
    inf_vars = 0
    zero_vars = 0
    small_vars = 0
    huge_vars = 0
    
    for k in var_keys:
        v = sd[k]
        if torch.isnan(v).any():
            nan_vars += 1
        if torch.isinf(v).any():
            inf_vars += 1
        if (v == 0).any():
            zero_vars += 1
        if (v < 1e-5).any():
            small_vars += 1
        if (v > 1e5).any():
            huge_vars += 1
            
    print(f"NaN vars: {nan_vars}")
    print(f"Inf vars: {inf_vars}")
    print(f"Zero vars: {zero_vars}")
    print(f"Vars < 1e-5: {small_vars}")
    print(f"Vars > 1e5: {huge_vars}")
    
    # Print some actual values from a few random layers
    import random
    random.seed(42)
    sample_keys = random.sample(var_keys, min(5, len(var_keys)))
    for k in sample_keys:
        mean_k = k.replace("running_var", "running_mean")
        var_val = sd[k]
        mean_val = sd[mean_k]
        print(f"\nLayer: {k}")
        print(f"  Var:  min={var_val.min().item():.6f}, max={var_val.max().item():.6f}, mean={var_val.mean().item():.6f}")
        print(f"  Mean: min={mean_val.min().item():.6f}, max={mean_val.max().item():.6f}, mean={mean_val.mean().item():.6f}")

if __name__ == "__main__":
    main()

