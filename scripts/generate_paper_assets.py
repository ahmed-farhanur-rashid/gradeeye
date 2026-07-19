"""
Generate all conference-paper figures and tables from trained checkpoints
and logs. Run this after training all 3 run-matrix configs
(baseline / ablation_ce_weighted_cbam / full_method).

Usage:
    python scripts/generate_paper_assets.py \
        --checkpoints saved/checkpoints/baseline_best.pt saved/checkpoints/ablation_ce_weighted_cbam_best.pt saved/checkpoints/full_method_best.pt \
        --test-manifest data/splits/aptos_test.csv \
        --norm-stats data/processed/aptos_norm_stats.json

Outputs land in paper_assets/{figures,tables}/.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.evaluate import evaluate_checkpoint
from src.eval.figures import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_run_comparison,
    plot_training_curves,
)
from src.eval.latex_tables import (
    dataset_summary_table,
    per_class_metrics_table,
    run_comparison_table,
)


def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures + tables.")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                         help="Paths to *_best.pt checkpoints, one per run.")
    parser.add_argument("--test-manifest", required=True,
                         help="Manifest CSV to evaluate all runs on (typically aptos_test.csv).")
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--log-dir", default="saved/logs")
    parser.add_argument("--out-dir", default="paper_assets")
    parser.add_argument("--manifests-for-distribution", nargs="*", default=[],
                         help="Optional: source_name=path pairs for the class "
                              "distribution figure, e.g. EyePACS=data/processed/eyepacs_manifest.csv")
    args = parser.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures")
    table_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    all_results = {}
    per_run_eval_output = {}

    def extract_run_name(ckpt_path: str) -> str:
        """
        Expects the naming convention scripts/train.py uses when saving
        checkpoints: "{run_name}_best.pt". Warns loudly rather than
        silently producing a wrong/confusing run name if a checkpoint
        doesn't follow that convention (e.g. a manually renamed file).
        """
        basename = os.path.basename(ckpt_path)
        if not basename.endswith("_best.pt"):
            print(f"WARNING: {basename} doesn't follow the '{{run_name}}_best.pt' naming "
                  f"convention used by scripts/train.py. Using the full filename (minus "
                  f".pt) as the run name, which may look wrong in figures/tables — "
                  f"consider renaming the checkpoint to match the convention.")
            return basename.removesuffix(".pt")
        return basename.removesuffix("_best.pt")

    for ckpt_path in args.checkpoints:
        run_name = extract_run_name(ckpt_path)
        print(f"\nEvaluating {run_name} on {args.test_manifest}...")

        result = evaluate_checkpoint(ckpt_path, args.test_manifest, args.norm_stats)
        all_results[run_name] = result["metrics"]
        per_run_eval_output[run_name] = result

        # Confusion matrix figure, per run.
        cm_paths = plot_confusion_matrix(result["confusion_matrix"], fig_dir, run_name)
        print(f"  Confusion matrix: {cm_paths}")

        # Per-class metrics LaTeX table, per run.
        table_path = os.path.join(table_dir, f"{run_name}_per_class_metrics.tex")
        per_class_metrics_table(result["metrics"], table_path, run_name=run_name)
        print(f"  Per-class table: {table_path}")

        # Training curves, if the epoch log exists for this run.
        epoch_log_path = os.path.join(args.log_dir, f"{run_name}_epoch_log.csv")
        if os.path.exists(epoch_log_path):
            curve_paths = plot_training_curves(epoch_log_path, fig_dir, run_name)
            print(f"  Training curves: {curve_paths}")
        else:
            print(f"  Skipping training curves for {run_name}: {epoch_log_path} not found.")

    # ROC curves need raw y_true/y_proba, re-derive from evaluate_checkpoint's
    # internals — simplest is to call evaluate_checkpoint again with a small
    # wrapper that also returns proba arrays. To avoid re-running inference,
    # we instead recompute directly here for each run using the same helper
    # logic evaluate_checkpoint uses internally.
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.augmentation.transforms import build_eval_transforms
    from src.data.datasets import DRDataset
    from src.models.corn import corn_predict_probas
    from src.models.dr_model import DRGradingModel
    from src.training.checkpoint import load_checkpoint

    for ckpt_path in args.checkpoints:
        run_name = extract_run_name(ckpt_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = load_checkpoint(ckpt_path, map_location=device)
        config = checkpoint["config"]
        model_cfg = config.get("model", {}) if config else {}

        # Only CORN-mode runs (full_method) have a well-formed per-class
        # probability distribution via corn_predict_probas; CE-mode runs
        # (baseline/ablation) already produce softmax probabilities directly
        # in evaluate_checkpoint's y_proba, which was captured above.
        loss_type = config.get("loss_type", "corn") if config else "corn"
        output_mode = "corn" if loss_type == "corn" else "softmax"

        model = DRGradingModel(
            pretrained=False, use_cbam=model_cfg.get("use_cbam", True),
            cbam_num_stages=model_cfg.get("cbam_num_stages", 2),
            num_thresholds=model_cfg.get("num_thresholds", 4),
            head_hidden_dim=model_cfg.get("head_hidden_dim", 512),
            dropout=model_cfg.get("dropout", 0.3), output_mode=output_mode,
        )
        state_key = "ema_state_dict" if checkpoint.get("ema_state_dict") else "model_state_dict"
        model.load_state_dict(checkpoint[state_key])
        model.to(device).eval()

        with open(args.norm_stats) as f:
            norm_stats = json.load(f)
        dataset = DRDataset(args.test_manifest, norm_stats, transform=build_eval_transforms())
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

        all_labels, all_probas = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                if output_mode == "corn":
                    probas = corn_predict_probas(logits)
                else:
                    probas = torch.softmax(logits, dim=1)
                all_labels.append(labels.numpy())
                all_probas.append(probas.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_proba = np.concatenate(all_probas)
        roc_paths = plot_roc_curves(y_true, y_proba, fig_dir, run_name)
        print(f"  ROC curves: {roc_paths}")

    # Cross-run comparison figure + table.
    if len(all_results) > 1:
        comparison_fig_paths = plot_run_comparison(all_results, fig_dir, metric="qwk")
        print(f"\nRun comparison figure: {comparison_fig_paths}")

        comparison_table_path = os.path.join(table_dir, "run_comparison.tex")
        run_comparison_table(all_results, comparison_table_path)
        print(f"Run comparison table: {comparison_table_path}")

    # Optional dataset class-distribution figure.
    if args.manifests_for_distribution:
        manifest_map = {}
        for pair in args.manifests_for_distribution:
            source, path = pair.split("=", 1)
            manifest_map[source] = path
        dist_paths = plot_class_distribution(manifest_map, fig_dir)
        print(f"\nClass distribution figure: {dist_paths}")

    # Dump raw metrics as JSON too, for anyone who wants to build custom tables later.
    metrics_json_path = os.path.join(table_dir, "all_run_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nRaw metrics JSON: {metrics_json_path}")

    print(f"\nAll paper assets written to {args.out_dir}/")


if __name__ == "__main__":
    main()
