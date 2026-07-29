"""
Generate all figures for the LaTeX docs.

Outputs to /home/farhan/my-projects/gradeeye/docs/latex/figures/:
  - fig1_training_curves.png         ConvNeXt multi-domain training curves
  - fig2_confusion_matrices.png      Per-source confusion matrices (ConvNeXt)
  - fig3_qwk_comparison.png          Per-source test QWK comparison (4 models)
  - fig4_effnetv2_loss_explosion.png EffNetV2-S loss explosion plot
  - fig5_swin_training_curves.png    SwinV2-Tiny training curves
  - fig6_pipeline.png                (skipped — schematic in text)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "latex", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: ConvNeXt multi-domain training curves
# ─────────────────────────────────────────────────────────────────────────────
def fig1_convnext_curves():
    df = pd.read_csv("saved/logs/full_method_multidomain_epoch_log.csv")
    # Phase 1 has 5 epochs; Phase 2 follows. The "best" epoch was 11 (global).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # Panel A: loss
    ax = axes[0]
    p1 = df[df.phase == "phase1_frozen"]
    p2 = df[df.phase == "phase2_full_training"]
    ax.plot(p1.global_epoch_idx, p1.train_loss, "o-", color="#4477AA", label="Train (Phase 1)")
    ax.plot(p1.global_epoch_idx, p1.val_loss, "s-", color="#4477AA", label="Val (Phase 1)", linestyle="--")
    ax.plot(p2.global_epoch_idx, p2.train_loss, "o-", color="#CC6677", label="Train (Phase 2)")
    ax.plot(p2.global_epoch_idx, p2.val_loss, "s-", color="#CC6677", label="Val (Phase 2)", linestyle="--")
    ax.axvline(4.5, color="k", linestyle=":", alpha=0.5)
    ax.text(4.6, ax.get_ylim()[1] * 0.95, "Phase 2 →", fontsize=8, color="gray")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("ConvNeXt-Tiny training loss (multi-domain, 4-channel)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)

    # Panel B: QWK
    ax = axes[1]
    ax.plot(p1.global_epoch_idx, p1.val_qwk, "s-", color="#4477AA", label="Val QWK (Phase 1)")
    ax.plot(p2.global_epoch_idx, p2.val_qwk, "s-", color="#CC6677", label="Val QWK (Phase 2)")
    ax.axvline(4.5, color="k", linestyle=":", alpha=0.5)
    # Best epoch marker
    best_idx = p2.val_qwk.idxmax()
    ax.scatter(p2.global_epoch_idx[best_idx], p2.val_qwk[best_idx], s=180, marker="*",
               color="gold", edgecolor="black", zorder=5,
               label=f"Best: {p2.val_qwk[best_idx]:.4f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation QWK")
    ax.set_title("ConvNeXt-Tiny validation QWK")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax.set_ylim(0.0, 0.85)

    fig.suptitle("ConvNeXt-Tiny + CBAM (4-channel, multi-domain) training trajectory",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_training_curves.png"))
    plt.close(fig)
    print("Wrote fig1_training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-source confusion matrices (ConvNeXt 4-channel best)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_confusion_matrices():
    # Three confusion matrices from RESULTS.md
    cms = {
        "EyePACS (n=8,871)": np.array([
            [6301,  43, 183,   3,   5],
            [ 522,  19,  79,   1,   0],
            [ 504,  27, 706,  73,   5],
            [  11,   2, 107,  86,   3],
            [  13,   1,  57,  23,  97],
        ]),
        "APTOS 2019 (n=367)": np.array([
            [178,  0,   3,  0,  0],
            [  7,  6,  24,  0,  0],
            [  1,  4,  87,  6,  2],
            [  0,  0,  13,  4,  2],
            [  0,  0,  13,  2, 15],
        ]),
        "Messidor-2 (n=227)": np.array([
            [101,  0,   1,  0,  0],
            [ 25,  2,   0,  0,  0],
            [  2,  3,  27,  3,  0],
            [  0,  0,   6,  2,  0],
            [  0,  0,   2,  0,  1],
        ]),
    }
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for ax, (title, cm) in zip(axes, cms.items()):
        # Normalize rows to percentages for color, but show raw counts
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(class_names, rotation=0)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontsize=11)
        for i in range(5):
            for j in range(5):
                v = cm[i, j]
                color = "white" if cm_pct[i, j] > 50 else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        color=color, fontsize=8)
    fig.suptitle("ConvNeXt-Tiny + CBAM (4-channel, TTA) — per-source confusion matrices",
                 fontsize=12, y=1.02)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Row-normalized %", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_confusion_matrices.png"))
    plt.close(fig)
    print("Wrote fig2_confusion_matrices.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Per-source test QWK comparison — 4 model variants
# ─────────────────────────────────────────────────────────────────────────────
def fig3_qwk_comparison():
    """Side-by-side bars for ConvNeXt 4-ch vs Swin 4-ch vs Ensemble vs legacy."""
    models = ["ConvNeXt\n4-ch (headline)", "SwinV2-Tiny\n4-ch @256", "ConvNeXt+Swin\nensemble",
              "Legacy 3-phase\n(RGB only)"]
    eyepacs = [0.7196, 0.5496, 0.6411, None]
    aptos   = [0.8848, 0.7677, 0.8716, 0.8697]
    messi   = [0.8376, 0.5994, 0.6741, 0.6146]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(models))
    w = 0.27

    def safe_plot(vals, color, label):
        # legacy has no EyePACS number — show None
        plot_vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + ({"EyePACS": -w, "APTOS": 0, "Messidor-2": w}[label] - 0),
                      plot_vals, w, color=color, label=label, edgecolor="black",
                      linewidth=0.4)
        for bar, v in zip(bars, plot_vals):
            if v == 0:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "N/A",
                        ha="center", va="bottom", fontsize=8, color="gray")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)

    safe_plot(eyepacs, "#4477AA", "EyePACS")
    safe_plot(aptos, "#CC6677", "APTOS")
    safe_plot(messi, "#117733", "Messidor-2")

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Test QWK")
    ax.set_title("Per-source test QWK comparison")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_qwk_comparison.png"))
    plt.close(fig)
    print("Wrote fig3_qwk_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: EffNetV2-S loss explosion plot
# ─────────────────────────────────────────────────────────────────────────────
def fig4_effnetv2_explosion():
    df = pd.read_csv("saved/logs/full_method_multidomain_effnetv2_epoch_log.csv")
    p1 = df[df.phase == "phase1_frozen"]
    p2 = df[df.phase == "phase2_full_training"]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(p1.global_epoch_idx, p1.train_loss, "o-", color="#4477AA", label="Train (Phase 1)")
    ax.plot(p1.global_epoch_idx, p1.val_loss, "s-", color="#4477AA", label="Val (Phase 1)", linestyle="--")
    ax.plot(p2.global_epoch_idx, p2.train_loss, "o-", color="#CC6677", label="Train (Phase 2)")
    ax.plot(p2.global_epoch_idx, p2.val_loss, "s-", color="#CC6677", label="Val (Phase 2)", linestyle="--")
    ax.axvline(4.5, color="k", linestyle=":", alpha=0.5)
    ax.text(2.2, 0.55, "Phase 1 (frozen backbone)", ha="center", fontsize=9, color="gray")
    ax.text(8.0, 0.55, "Phase 2 (full fine-tune)", ha="center", fontsize=9, color="gray")
    # Highlight the explosion — point at the bad val_loss point
    bad_x = p2.global_epoch_idx.iloc[0]
    bad_y = p2.val_loss.iloc[0]
    ax.scatter([bad_x], [bad_y], s=180, marker="*", color="red", edgecolor="black", zorder=5)
    ax.annotate(f"Phase 2 epoch 0:\nval_loss = {bad_y:,.0f}",
                xy=(bad_x, bad_y), xytext=(2.0, 5e3),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=10, color="red", fontweight="bold", ha="center")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (symlog)")
    ax.set_title("EfficientNetV2-S: Phase 2 val_loss explosion")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_effnetv2_loss_explosion.png"))
    plt.close(fig)
    print("Wrote fig4_effnetv2_loss_explosion.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: SwinV2-Tiny training curves
# ─────────────────────────────────────────────────────────────────────────────
def fig5_swin_curves():
    df = pd.read_csv("saved/logs/full_method_multidomain_swint_epoch_log.csv")
    p1 = df[df.phase == "phase1_frozen"]
    p2 = df[df.phase == "phase2_full_training"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    ax.plot(p1.global_epoch_idx, p1.train_loss, "o-", color="#4477AA", label="Train (Phase 1)")
    ax.plot(p1.global_epoch_idx, p1.val_loss, "s-", color="#4477AA", label="Val (Phase 1)", linestyle="--")
    ax.plot(p2.global_epoch_idx, p2.train_loss, "o-", color="#CC6677", label="Train (Phase 2)")
    ax.plot(p2.global_epoch_idx, p2.val_loss, "s-", color="#CC6677", label="Val (Phase 2)", linestyle="--")
    ax.axvline(4.5, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("SwinV2-Tiny training loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(p1.global_epoch_idx, p1.val_qwk, "s-", color="#4477AA", label="Val QWK (Phase 1)")
    ax.plot(p2.global_epoch_idx, p2.val_qwk, "s-", color="#CC6677", label="Val QWK (Phase 2)")
    ax.axvline(4.5, color="k", linestyle=":", alpha=0.5)
    best_idx = p2.val_qwk.idxmax()
    ax.scatter(p2.global_epoch_idx[best_idx], p2.val_qwk[best_idx], s=180, marker="*",
               color="gold", edgecolor="black", zorder=5,
               label=f"Best: {p2.val_qwk[best_idx]:.4f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation QWK")
    ax.set_title("SwinV2-Tiny validation QWK"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0.0, 0.85)

    fig.suptitle("SwinV2-Tiny + CBAM (4-channel @256, multi-domain) training trajectory",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_swin_training_curves.png"))
    plt.close(fig)
    print("Wrote fig5_swin_training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Pipeline schematic (described in text, drawn as a simple flowchart)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_pipeline_schematic():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    ax.axis("off")

    # Top: raw RGB
    boxes = [
        (0.5, 4.0, "Fundus\nRGB", "#AABBCC"),
        (2.5, 4.0, "Border crop\n+ pad + 384²", "#AABBCC"),
        (4.5, 4.0, "Ben Graham\n+ green CLAHE", "#AABBCC"),
        (6.5, 4.0, "Circular mask\n+ per-source norm", "#AABBCC"),
        (8.5, 4.0, "Preprocessed\ntensor (3, 384, 384)", "#88CC88"),
        # Bottom row: U-Net
        (0.5, 1.5, "Preprocessed\nfundus", "#AABBCC"),
        (2.5, 1.5, "U-Net\n(EfficientNet-B0)", "#CC8888"),
        (4.5, 1.5, "Seg mask\n(384, 384)", "#CC8888"),
        # Concatenate
        (6.5, 2.75, "Concat\n4ch", "#8888CC"),
        (8.5, 2.75, "ConvNeXt-Tiny\n+ CBAM", "#88CC88"),
        (10.5, 2.75, "CORN\nlogits", "#DDAA88"),
    ]
    for (x, y, label, color) in boxes:
        ax.add_patch(plt.Rectangle((x - 0.7, y - 0.5), 1.4, 1.0,
                                    facecolor=color, edgecolor="black", linewidth=1.0))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)

    # Arrows
    arrow_pairs = [
        ((1.2, 4.0), (1.8, 4.0)),
        ((3.2, 4.0), (3.8, 4.0)),
        ((5.2, 4.0), (5.8, 4.0)),
        ((7.2, 4.0), (7.8, 4.0)),
        ((1.2, 1.5), (1.8, 1.5)),
        ((3.2, 1.5), (3.8, 1.5)),
        ((5.2, 1.5), (5.85, 2.35)),
        ((7.85, 4.0), (7.85, 3.3)),
        ((7.2, 2.75), (7.8, 2.75)),
        ((9.2, 2.75), (9.8, 2.75)),
    ]
    for (x0, y0), (x1, y1) in arrow_pairs:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))

    # Labels
    ax.text(6.0, 5.5, "RGB preprocessing pipeline", ha="center", fontsize=11, fontweight="bold")
    ax.text(2.5, 0.0, "U-Net segmentation pipeline", ha="center", fontsize=11, fontweight="bold")
    ax.text(8.5, 0.0, "Grader (4-channel)", ha="center", fontsize=11, fontweight="bold")

    ax.set_title("GradeEye pipeline overview", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_pipeline.png"))
    plt.close(fig)
    print("Wrote fig6_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Multi-domain training set size comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig7_dataset_composition():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # Panel A: total images per source
    ax = axes[0]
    sources = ["EyePACS", "APTOS 2019", "Messidor-2"]
    train = [56502, 2343, 1452]
    val = [7058, 293, 180]
    test = [7058, 293, 184]
    x = np.arange(len(sources))
    w = 0.27
    ax.bar(x - w, train, w, color="#4477AA", label="Train", edgecolor="black", linewidth=0.4)
    ax.bar(x,     val,  w, color="#117733", label="Val",   edgecolor="black", linewidth=0.4)
    ax.bar(x + w, test, w, color="#CC6677", label="Test",  edgecolor="black", linewidth=0.4)
    for i, src in enumerate(sources):
        for j, (data, name) in enumerate(zip([train, val, test], ["Train", "Val", "Test"])):
            v = data[i]
            ax.text(i + (j - 1) * w, v + 500, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(sources)
    ax.set_ylabel("Number of images")
    ax.set_title("Per-source stratified splits (80/10/10)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # Panel B: class distribution per source
    ax = axes[1]
    classes = ["No DR\n(0)", "Mild\n(1)", "Moderate\n(2)", "Severe\n(3)", "Prolif.\n(4)"]
    eyepacs_dist_pct = [73.0, 7.5, 14.5, 2.5, 2.5]
    aptos_dist_pct   = [49.0, 9.0, 27.0, 7.0, 8.0]
    messi_dist_pct   = [57.0, 12.0, 25.0, 3.0, 3.0]
    x = np.arange(len(classes))
    w = 0.27
    ax.bar(x - w, eyepacs_dist_pct, w, color="#4477AA", label="EyePACS",  edgecolor="black", linewidth=0.4)
    ax.bar(x,     aptos_dist_pct,   w, color="#CC6677", label="APTOS",    edgecolor="black", linewidth=0.4)
    ax.bar(x + w, messi_dist_pct,   w, color="#117733", label="Messidor-2", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("Class fraction (%)")
    ax.set_title("Class distribution per source")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Multi-domain dataset composition", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig7_dataset_composition.png"))
    plt.close(fig)
    print("Wrote fig7_dataset_composition.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: CORN monotonicity illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig8_corn_illustration():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # Panel A: 4 conditional logits, sigmoid, cumulative product
    ax = axes[0]
    logits = np.array([-2.5, 1.8, 0.5, -1.0])
    sig = 1.0 / (1.0 + np.exp(-logits))
    cum = np.cumprod(sig)
    x = np.arange(4)
    ax.plot(x, sig, "o-", color="#4477AA", label="$P(y > k | y \\geq k)$", linewidth=2)
    ax.plot(x, cum, "s-", color="#CC6677", label="$P(y > k) = \\prod$", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"$k={i}$" for i in range(4)])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Threshold index"); ax.set_ylabel("Probability")
    ax.set_title("CORN conditional vs unconditional probabilities")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)

    # Panel B: predicted class derivation
    ax = axes[1]
    k = np.arange(4)
    exceeded = cum > 0.5
    ax.bar(k, exceeded.astype(float), color=("#CC6677" if exceeded.any() else "#4477AA"),
           edgecolor="black", linewidth=0.5)
    for i, c in enumerate(cum):
        ax.text(i, 0.5, f"{c:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(k); ax.set_xticklabels([f"$k={i}$" for i in range(4)])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["False", "True", "True"])
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Threshold index"); ax.set_ylabel("$P(y > k) > 0.5$?")
    ax.set_title(f"Predicted class = #{exceeded.sum()} (sum of exceeded thresholds)")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("CORN ordinal inference rule", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig8_corn_inference.png"))
    plt.close(fig)
    print("Wrote fig8_corn_inference.png")


if __name__ == "__main__":
    fig1_convnext_curves()
    fig2_confusion_matrices()
    fig3_qwk_comparison()
    fig4_effnetv2_explosion()
    fig5_swin_curves()
    fig6_pipeline_schematic()
    fig7_dataset_composition()
    fig8_corn_illustration()
    print("\nAll figures written to", OUT_DIR)
