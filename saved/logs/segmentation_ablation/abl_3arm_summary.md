# Segmentation 3-Arm Ablation — 5-fold CV on IDRiD

All arms share: U-Net (EfficientNet-B0 encoder, ImageNet-pretrained), Dice+BCE loss, 384×384 input, AdamW lr=1e-4, cosine schedule, gradient clip 1.0. Only the loss aggregation + dataset pool differ.

Metrics: mean ± std across 5 folds. Per-lesion Dice/IoU is computed against the model's single sigmoid output (threshold=0.5), compared to each lesion type's separate ground truth (MA / HE / EX / SE) and the union mask.

| Arm | n_train (drive+idrid) | Union Dice | Union IoU | MA Dice | MA IoU | HE Dice | HE IoU | EX Dice | EX IoU | SE Dice | SE IoU |
|-----|----------------------|-----------|-----------|---------|---------|---------|---------|---------|---------|---------|---------|
| baseline | 63 | 0.061±0.016 | 0.032±0.009 | 0.006±0.002 | 0.003±0.001 | 0.036±0.013 | 0.019±0.007 | 0.018±0.007 | 0.009±0.004 | 0.004±0.002 | 0.002±0.001 |
| reweighted | 63 | 0.059±0.020 | 0.032±0.012 | 0.003±0.001 | 0.001±0.001 | 0.030±0.016 | 0.016±0.010 | 0.025±0.008 | 0.013±0.004 | 0.005±0.003 | 0.003±0.001 |
| idrid_only | 43 | 0.049±0.015 | 0.026±0.009 | 0.003±0.001 | 0.002±0.001 | 0.022±0.011 | 0.012±0.006 | 0.020±0.006 | 0.010±0.003 | 0.005±0.003 | 0.003±0.002 |
