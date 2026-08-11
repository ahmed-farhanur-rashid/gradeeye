#!/usr/bin/env python3
"""Parse calibration study JSON output and emit §3 tables for docs/results.md.

Reads saved/logs/calibration/lodo_<fold>_convnext_tiny_probability_calibration.json
for each fold and emits markdown tables for §3.1 (raw), §3.2 (per-threshold T),
§3.3 (global T), §3.4 (delta), §3.6 (per-threshold class prev), §3.7 (range).

If `--write` is passed, writes the §3 tables to docs/results.md in place.

Usage:
    python scripts/calibration_to_results.py            # print tables
    python scripts/calibration_to_results.py --write   # also patch docs/results.md
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = REPO_ROOT / "saved/logs/calibration"
RESULTS_MD = REPO_ROOT / "docs/results.md"

FOLDS = ["eyepacs", "aptos", "messidor2", "ddr"]
FOLD_DISPLAY = {
    "eyepacs": "EyePACS",
    "aptos": "APTOS",
    "messidor2": "Messidor-2",
    "ddr": "DDR",
}


def load_fold(fold: str) -> dict | None:
    p = CALIB_DIR / f"lodo_{fold}_convnext_tiny_probability_calibration.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["result"]


def ece_per_threshold(ece_table: list[dict]) -> list[float]:
    """Extract ECE for thresholds 1..4 from an _ece_table output."""
    return [row["ece"] for row in ece_table]


def aggregate_ece(ece_table: list[dict]) -> float:
    return statistics.mean(ece_per_threshold(ece_table))


def _resolve_table(d: dict, ece_section: str | None, src: str) -> list[dict]:
    """Resolve the per-threshold list for a fold given a section name and source.

    ece_section ∈ {"raw", "per_threshold", "global", None}.
    src ∈ {"raw_ece", "calibrated_ece"}.
    """
    if ece_section is None:
        return d["raw_ece"][src]["per_threshold"]
    return d[src][src if src == "raw_ece" else "full"][ece_section]["per_threshold"]


def fmt_ece_table(folds_data: dict[str, dict], ece_section: str | None = None,
                  src: str = "raw_ece") -> str:
    """Format an ECE table. ece_section is the calibrated_ece sub-dict
    (one of "per_threshold", "global", or None for raw). src is "raw_ece" or
    "calibrated_ece"."""
    rows = []
    for fold in FOLDS:
        d = folds_data.get(fold)
        if d is None:
            rows.append(f"| {FOLD_DISPLAY[fold]} |  |  |  |  |  |")
            continue
        if src == "raw_ece":
            table = d["raw_ece"]["full"]["per_threshold"]
        else:
            if ece_section is None:
                ece_section = "per_threshold"
            table = d["calibrated_ece"]["full"][ece_section]["per_threshold"]
        eces = ece_per_threshold(table)
        agg = aggregate_ece(table)
        rows.append(
            f"| {FOLD_DISPLAY[fold]} | {eces[0]:.4f} | {eces[1]:.4f} | {eces[2]:.4f} | {eces[3]:.4f} | {agg:.4f} |"
        )
    # Mean ± SD
    numeric = {fold: folds_data[fold] for fold in FOLDS if folds_data.get(fold) is not None}
    mean_cells = []
    sd_cells = []
    for k in range(4):
        if src == "raw_ece":
            vals = [ece_per_threshold(d["raw_ece"]["full"]["per_threshold"])[k] for d in numeric.values()]
        else:
            sec = ece_section or "per_threshold"
            vals = [ece_per_threshold(d["calibrated_ece"]["full"][sec]["per_threshold"])[k] for d in numeric.values()]
        if not vals:
            mean_cells.append("")
            sd_cells.append("")
        else:
            mean_cells.append(f"{statistics.mean(vals):.4f}")
            sd_cells.append(f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else "0.0000")
    if src == "raw_ece":
        agg_vals = [aggregate_ece(d["raw_ece"]["full"]["per_threshold"]) for d in numeric.values()]
    else:
        sec = ece_section or "per_threshold"
        agg_vals = [aggregate_ece(d["calibrated_ece"]["full"][sec]["per_threshold"]) for d in numeric.values()]
    if agg_vals:
        agg_mean = f"{statistics.mean(agg_vals):.4f}"
        agg_sd = f"{statistics.stdev(agg_vals):.4f}" if len(agg_vals) > 1 else "0.0000"
    else:
        agg_mean, agg_sd = "", ""
    rows.append(
        f"| **Mean ± SD** | {mean_cells[0]} ± {sd_cells[0]} | {mean_cells[1]} ± {sd_cells[1]} | {mean_cells[2]} ± {sd_cells[2]} | {mean_cells[3]} ± {sd_cells[3]} | {agg_mean} ± {agg_sd} |"
    )
    return "\n".join(rows)


def fmt_val_table(folds_data: dict[str, dict]) -> str:
    """Per-threshold raw ECE on the in-distribution validation set (used to fit T)."""
    rows = []
    for fold in FOLDS:
        d = folds_data.get(fold)
        if d is None:
            rows.append(f"| {FOLD_DISPLAY[fold]} |  |  |  |  |  |  |")
            continue
        table = d["raw_ece"]["validation"]["per_threshold"]
        eces = ece_per_threshold(table)
        agg = aggregate_ece(table)
        n_val = d["n_samples"]["validation"]
        T_perT = ", ".join(f"{t:.3f}" for t in d["temperatures"]["per_threshold"])
        T_glob = d["temperatures"]["global"]
        rows.append(
            f"| {FOLD_DISPLAY[fold]} | {n_val} | {eces[0]:.4f} | {eces[1]:.4f} | {eces[2]:.4f} | {eces[3]:.4f} | {agg:.4f} | {T_perT} | {T_glob:.3f} |"
        )
    return "\n".join(rows)


def fmt_delta_table(folds_data: dict[str, dict], target_section: str) -> str:
    """ΔECE = calibrated - raw. Negative = improvement."""
    rows = []
    for fold in FOLDS:
        d = folds_data.get(fold)
        if d is None:
            rows.append(f"| {FOLD_DISPLAY[fold]} |  |  |  |  |  |")
            continue
        raw_eces = ece_per_threshold(d["raw_ece"]["full"]["per_threshold"])
        cal_eces = ece_per_threshold(d["calibrated_ece"]["full"][target_section]["per_threshold"])
        deltas = [c - r for c, r in zip(cal_eces, raw_eces)]
        agg_raw = aggregate_ece(d["raw_ece"]["full"]["per_threshold"])
        agg_cal = aggregate_ece(d["calibrated_ece"]["full"][target_section]["per_threshold"])
        rows.append(
            f"| {FOLD_DISPLAY[fold]} | {deltas[0]:+.4f} | {deltas[1]:+.4f} | {deltas[2]:+.4f} | {deltas[3]:+.4f} | {agg_cal - agg_raw:+.4f} |"
        )
    # Mean ± SD row
    numeric = {fold: folds_data[fold] for fold in FOLDS if folds_data.get(fold) is not None}
    delta_grid = []
    for d in numeric.values():
        raws = ece_per_threshold(d["raw_ece"]["full"]["per_threshold"])
        cals = ece_per_threshold(d["calibrated_ece"]["full"][target_section]["per_threshold"])
        delta_grid.append([c - r for c, r in zip(cals, raws)])
    if delta_grid:
        mean_cells = [f"{statistics.mean([row[k] for row in delta_grid]):+.4f}" for k in range(4)]
        sd_cells = [f"{statistics.stdev([row[k] for row in delta_grid]):.4f}" if len(delta_grid) > 1 else "0.0000" for k in range(4)]
        agg_raw_vals = [aggregate_ece(d["raw_ece"]["full"]["per_threshold"]) for d in numeric.values()]
        agg_cal_vals = [aggregate_ece(d["calibrated_ece"]["full"][target_section]["per_threshold"]) for d in numeric.values()]
        agg_deltas = [c - r for c, r in zip(agg_cal_vals, agg_raw_vals)]
        agg_mean = f"{statistics.mean(agg_deltas):+.4f}"
        agg_sd = f"{statistics.stdev(agg_deltas):.4f}" if len(agg_deltas) > 1 else "0.0000"
        rows.append(
            f"| **Mean ± SD** | {mean_cells[0]} ± {sd_cells[0]} | {mean_cells[1]} ± {sd_cells[1]} | {mean_cells[2]} ± {sd_cells[2]} | {mean_cells[3]} ± {sd_cells[3]} | {agg_mean} ± {agg_sd} |"
        )
    return "\n".join(rows)


def fmt_class_prevalence(folds_data: dict[str, dict]) -> str:
    """Per-threshold positive class prevalence (P(grade >= t)).
    The calibration JSON does not store labels directly; we leave this blank
    and rely on the split CSVs as the source of truth."""
    rows = []
    for fold in FOLDS:
        d = folds_data.get(fold)
        if d is None:
            rows.append(f"| {FOLD_DISPLAY[fold]} |  |  |  |  |")
            continue
        rows.append(f"| {FOLD_DISPLAY[fold]} | _see split CSVs_ | _see split CSVs_ | _see split CSVs_ | _see split CSVs_ |")
    return "\n".join(rows)


def write_to_results_md(folds_data: dict[str, dict]) -> None:
    """Patch docs/results.md §3 in place. Only fills in the table bodies
    (replacing empty cells), not the surrounding prose."""
    if not RESULTS_MD.exists():
        print(f"  results.md not found at {RESULTS_MD}; skipping write.")
        return

    text = RESULTS_MD.read_text()

    # §3.1 raw ECE
    raw_block = (
        "| Held-out source | ECE$_1$ | ECE$_2$ | ECE$_3$ | ECE$_4$ | ECE (aggregate) |\n"
        "|:----------------|:-------:|:-------:|:-------:|:-------:|:---------------:|\n"
        + fmt_ece_table(folds_data)
    )
    text = _replace_table(text, "### 3.1 Per-Threshold ECE by Fold (No Calibration Adjustment)", raw_block)

    # §3.2 per-threshold T
    pt_block = (
        "| Held-out source | ECE$_1$ | ECE$_2$ | ECE$_3$ | ECE$_4$ | ECE (aggregate) |\n"
        "|:----------------|:-------:|:-------:|:-------:|:-------:|:---------------:|\n"
        + fmt_ece_table(folds_data, ece_section="per_threshold", src="calibrated_ece")
    )
    text = _replace_table(text, "### 3.2 Per-Threshold ECE by Fold (Per-Threshold Temperature Scaling)", pt_block)

    # §3.3 global T
    g_block = (
        "| Held-out source | ECE$_1$ | ECE$_2$ | ECE$_3$ | ECE$_4$ | ECE (aggregate) |\n"
        "|:----------------|:-------:|:-------:|:-------:|:-------:|:---------------:|\n"
        + fmt_ece_table(folds_data, ece_section="global", src="calibrated_ece")
    )
    text = _replace_table(text, "### 3.3 Per-Threshold ECE by Fold (Single Global Temperature Scaling)", g_block)

    # §3.4 first delta (per-T vs raw)
    d1_block = (
        "| Held-out source | ΔECE$_1$ | ΔECE$_2$ | ΔECE$_3$ | ΔECE$_4$ | ΔECE (aggregate) |\n"
        "|:----------------|:--------:|:--------:|:--------:|:--------:|:----------------:|\n"
        + fmt_delta_table(folds_data, "per_threshold")
    )
    text = _replace_delta_first(text, d1_block)

    # §3.4 second delta (global vs raw)
    d2_block = (
        "| Held-out source | ΔECE$_1$ | ΔECE$_2$ | ΔECE$_3$ | ΔECE$_4$ | ΔECE (aggregate) |\n"
        "|:----------------|:--------:|:--------:|:--------:|:--------:|:----------------:|\n"
        + fmt_delta_table(folds_data, "global")
    )
    text = _replace_delta_second(text, d2_block)

    # §3.7 per-threshold vs single
    s37_block = (
        "| Held-out source | max$_t$ ECE$_t$ | min$_t$ ECE$_t$ | Range | Aggregate ECE |\n"
        "|:----------------|:---------------:|:---------------:|:-----:|:-------------:|\n"
        + fmt_range_table(folds_data)
    )
    text = _replace_table(text, "### 3.7 Per-Threshold Calibration Versus Single-Threshold Reporting", s37_block)

    RESULTS_MD.write_text(text)
    print(f"  patched {RESULTS_MD}")


def _replace_table(text: str, header_marker: str, new_block: str) -> str:
    """Replace the first markdown table that follows header_marker with new_block."""
    h_idx = text.find(header_marker)
    if h_idx < 0:
        return text
    # Skip past the header line + immediate paragraph
    block_start = text.find("\n|", h_idx)
    if block_start < 0:
        return text
    # Find the end of the contiguous table block (lines starting with |)
    end = block_start
    lines = text[block_start:].split("\n")
    n_table = 0
    for i, line in enumerate(lines):
        if line.startswith("|"):
            n_table += 1
        else:
            if n_table > 0:
                end = block_start + sum(len(l) + 1 for l in lines[:i])
                break
    return text[:block_start] + new_block + text[end:]


def _replace_delta_first(text: str, new_block: str) -> str:
    """Replace the FIRST delta table under §3.4 (per-T vs raw)."""
    h_idx = text.find("### 3.4 Per-Threshold Calibration: Per-Fold Delta Comparison")
    if h_idx < 0:
        return text
    block_start = text.find("\n|", h_idx)
    if block_start < 0:
        return text
    lines = text[block_start:].split("\n")
    n_table = 0
    for i, line in enumerate(lines):
        if line.startswith("|"):
            n_table += 1
        else:
            if n_table > 0:
                end = block_start + sum(len(l) + 1 for l in lines[:i])
                break
    return text[:block_start] + new_block + text[end:]


def _replace_delta_second(text: str, new_block: str) -> str:
    """Replace the SECOND delta table under §3.4 (global vs raw)."""
    h_idx = text.find("### 3.4 Per-Threshold Calibration: Per-Fold Delta Comparison")
    if h_idx < 0:
        return text
    first_block_start = text.find("\n|", h_idx)
    if first_block_start < 0:
        return text
    # Find the end of the first table
    lines = text[first_block_start:].split("\n")
    n_table = 0
    first_end = first_block_start
    for i, line in enumerate(lines):
        if line.startswith("|"):
            n_table += 1
        else:
            if n_table > 0:
                first_end = first_block_start + sum(len(l) + 1 for l in lines[:i])
                break
    # Find the second table
    second_block_start = text.find("\n|", first_end)
    if second_block_start < 0:
        return text
    lines2 = text[second_block_start:].split("\n")
    n_table = 0
    second_end = second_block_start
    for i, line in enumerate(lines2):
        if line.startswith("|"):
            n_table += 1
        else:
            if n_table > 0:
                second_end = second_block_start + sum(len(l) + 1 for l in lines2[:i])
                break
    return text[:second_block_start] + new_block + text[second_end:]


def fmt_range_table(folds_data: dict[str, dict]) -> str:
    """Per-fold max vs min per-threshold ECE plus aggregate."""
    rows = []
    for fold in FOLDS:
        d = folds_data.get(fold)
        if d is None:
            rows.append(f"| {FOLD_DISPLAY[fold]} |  |  |  |  |")
            continue
        eces = ece_per_threshold(d["raw_ece"]["full"]["per_threshold"])
        agg = aggregate_ece(d["raw_ece"]["full"]["per_threshold"])
        rows.append(
            f"| {FOLD_DISPLAY[fold]} | {max(eces):.4f} | {min(eces):.4f} | {max(eces) - min(eces):.4f} | {agg:.4f} |"
        )
    numeric = {fold: folds_data[fold] for fold in FOLDS if folds_data.get(fold) is not None}
    if numeric:
        maxes = [max(ece_per_threshold(d["raw_ece"]["full"]["per_threshold"])) for d in numeric.values()]
        mins = [min(ece_per_threshold(d["raw_ece"]["full"]["per_threshold"])) for d in numeric.values()]
        ranges = [max(ece_per_threshold(d["raw_ece"]["full"]["per_threshold"])) - min(ece_per_threshold(d["raw_ece"]["full"]["per_threshold"])) for d in numeric.values()]
        aggs = [aggregate_ece(d["raw_ece"]["full"]["per_threshold"]) for d in numeric.values()]
        m = lambda v: f"{statistics.mean(v):.4f}"
        s = lambda v: f"{statistics.stdev(v):.4f}" if len(v) > 1 else "0.0000"
        rows.append(f"| **Mean ± SD** | {m(maxes)} ± {s(maxes)} | {m(mins)} ± {s(mins)} | {m(ranges)} ± {s(ranges)} | {m(aggs)} ± {s(aggs)} |")
    return "\n".join(rows)


def main():
    folds_data = {fold: load_fold(fold) for fold in FOLDS}
    available = [f for f, d in folds_data.items() if d is not None]
    if not available:
        print(f"No calibration JSONs found in {CALIB_DIR}; cannot populate §3.")
        return

    print(f"Loaded {len(available)}/{len(FOLDS)} folds: {available}")

    print("\n=== §3.1 Raw ECE (no calibration, full held-out test) ===")
    print(fmt_ece_table(folds_data))

    print("\n=== Validation-set context: ECE used to fit temperatures ===")
    val_header = "| Held-out source | n_val | ECE$_1$ | ECE$_2$ | ECE$_3$ | ECE$_4$ | ECE (agg) | T (per-t) | T (global) |"
    val_div = "|:----------------|:----:|:-------:|:-------:|:-------:|:-------:|:---------:|:---------:|:----------:|"
    print(val_header)
    print(val_div)
    print(fmt_val_table(folds_data))

    print("\n=== §3.2 Per-Threshold T ECE ===")
    print(fmt_ece_table(folds_data, ece_section="per_threshold", src="calibrated_ece"))

    print("\n=== §3.3 Global T ECE ===")
    print(fmt_ece_table(folds_data, ece_section="global", src="calibrated_ece"))

    print("\n=== §3.4 Delta (per-T vs raw) ===")
    print(fmt_delta_table(folds_data, "per_threshold"))

    print("\n=== §3.4 Delta (global vs raw) ===")
    print(fmt_delta_table(folds_data, "global"))

    print("\n=== §3.7 Per-threshold range vs aggregate ===")
    print(fmt_range_table(folds_data))

    if "--write" in sys.argv:
        write_to_results_md(folds_data)


if __name__ == "__main__":
    main()
