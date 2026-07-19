"""
LaTeX table generation for the conference paper's results section.

Produces booktabs-style tables (\\toprule/\\midrule/\\bottomrule), the
standard convention for ML conference papers (NeurIPS/CVPR/MICCAI-style).
Assumes \\usepackage{booktabs} in the paper's preamble.
"""
import os

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def _escape_latex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def run_comparison_table(results: dict, out_path: str,
                          metrics_to_show: list = None,
                          caption: str = "Comparison of model configurations on the held-out APTOS test set.",
                          label: str = "tab:run_comparison") -> str:
    """
    results: dict of {run_name: metrics_dict}, where metrics_dict is the
    output of src/eval/metrics.py:compute_all_metrics.

    metrics_to_show: which metric keys to include as columns, in order.
    Defaults to the standard set for a DR grading paper.
    """
    if metrics_to_show is None:
        metrics_to_show = ["qwk", "accuracy", "macro_f1", "macro_auc_roc"]

    header_labels = {
        "qwk": "QWK", "accuracy": "Accuracy", "macro_f1": "Macro F1",
        "macro_precision": "Macro Prec.", "macro_recall": "Macro Rec.",
        "macro_auc_roc": "Macro AUC-ROC",
    }

    col_spec = "l" + "c" * len(metrics_to_show)
    header_row = "Configuration & " + " & ".join(
        header_labels.get(m, m.replace("_", " ").title()) for m in metrics_to_show
    ) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_row,
        r"\midrule",
    ]

    # Bold the best value per column.
    best_per_metric = {}
    for m in metrics_to_show:
        vals = [results[r].get(m, float("nan")) for r in results]
        best_per_metric[m] = max(vals) if vals else None

    for run_name, metrics in results.items():
        cells = []
        for m in metrics_to_show:
            val = metrics.get(m, float("nan"))
            val_str = f"{val:.3f}"
            if best_per_metric[m] is not None and abs(val - best_per_metric[m]) < 1e-9:
                val_str = rf"\textbf{{{val_str}}}"
            cells.append(val_str)
        row = _escape_latex(run_name.replace("_", " ")) + " & " + " & ".join(cells) + r" \\"
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table_str = "\n".join(lines)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table_str + "\n")

    return table_str


def per_class_metrics_table(metrics: dict, out_path: str, run_name: str = "",
                             caption: str = None,
                             label: str = "tab:per_class_metrics") -> str:
    """
    Single-run per-class breakdown table (precision/recall/F1 per ICDR grade),
    from the per-class keys in compute_all_metrics' output
    (f1_{class}, recall_{class}, precision_{class}).
    """
    if caption is None:
        caption = f"Per-class performance{' for ' + run_name.replace('_', ' ') if run_name else ''}."

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Grade & Precision & Recall & F1 \\",
        r"\midrule",
    ]

    for name in CLASS_NAMES:
        key_suffix = name.replace(" ", "_")
        precision = metrics.get(f"precision_{key_suffix}", float("nan"))
        recall = metrics.get(f"recall_{key_suffix}", float("nan"))
        f1 = metrics.get(f"f1_{key_suffix}", float("nan"))
        lines.append(f"{name} & {precision:.3f} & {recall:.3f} & {f1:.3f} " + r"\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table_str = "\n".join(lines)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table_str + "\n")

    return table_str


def dataset_summary_table(source_stats: dict, out_path: str,
                           caption: str = "Dataset summary.",
                           label: str = "tab:dataset_summary") -> str:
    """
    source_stats: dict of {source_name: {"n_images": int, "role": str,
                  "class_counts": [n0,n1,n2,n3,n4]}}, e.g.:
        {"EyePACS": {"n_images": 35126, "role": "Pretrain",
                     "class_counts": [25810, 2443, 5292, 873, 708]}, ...}
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Source & Role & N & " + " & ".join(CLASS_NAMES) + r" \\",
        r"\midrule",
    ]

    for source, stats in source_stats.items():
        counts = stats["class_counts"]
        row = f"{_escape_latex(source)} & {stats['role']} & {stats['n_images']} & " + \
              " & ".join(str(c) for c in counts) + r" \\"
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table_str = "\n".join(lines)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table_str + "\n")

    return table_str
