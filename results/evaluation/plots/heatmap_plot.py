#!/usr/bin/env python3
"""
Per-column heatmap for experiment deltas (adapted − base).

- One figure with columns = metrics, rows = experiments (test_name).
- Each column (metric) has its OWN diverging color scale centered at 0,
  shown as a mini colorbar above the column.
- Cells are annotated with the raw delta (Δ).
- Optionally sort rows by an overall "goodness" score to cluster stronger settings.

CSV format:
    test_name,CodeBLEU,BLEU,ROUGE-1,ROUGE-2,ROUGE-L,BERTScore
    v2_adapted_12b_core-..., -0.05, -0.3, ...

Notes:
- Colors are only comparable WITHIN a column. The caption/title makes this explicit.
- Column limits are chosen robustly (95th percentile of |Δ| per metric), then made symmetric.
  You can switch to 'max_abs' if you prefer, or set manual limits per metric.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm

# ========= USER CONSTANTS =========
CSV_PATH = "heatmap_data.csv"  # <-- set your input CSV path here
OUTPUT_PATH = "per_column_heatmap.png"
DPI = 200

# Which columns to visualize (order is kept)
METRICS = ["CodeBLEU", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]

# Column scaling strategy: "p95_abs" (recommended), "max_abs", or "manual"
COLUMN_SCALE_MODE = "p95_abs"

# Manual limits (only used if COLUMN_SCALE_MODE == "manual")
# Limits must be symmetric around 0; provide the POSITIVE half-range for each metric.
# Example: {"BLEU": 3.5, "ROUGE-L": 2.6, "BERTScore": 1.0}
MANUAL_HALF_RANGES = {}

# Show annotation text inside cells?
ANNOTATE_CELLS = True
ANNOTATION_FMT = "{:+.2f}"  # e.g., +1.57 / -0.34
ANNOTATION_MIN_ABS = 0.00   # hide annotation if |Δ| < threshold

# Sort rows (experiments) by an overall score to improve readability?
# "mean_scaled" uses per-metric scaled deltas (Δ / column_limit) and sorts descending.
ROW_SORT_MODE = "none"  # "none" or "mean_scaled"

# Visually separate generation vs summarization columns?
# Provide a list of indices AFTER which to draw a thin vertical gap (0-based, inclusive).
# With METRICS as above, CodeBLEU is gen; the rest are summarization -> draw a divider after index 0.
COLUMN_DIVIDERS_AFTER = [0]  # [] for none

# Appearance
CMAP = "RdBu_r"  # diverging colormap centered at 0
FIG_LEFT_LABELS = True       # show row labels only on the far-left column
ROW_LABEL_FONTSIZE = 8
COL_LABEL_FONTSIZE = 9
ANNOT_FONTSIZE = 7
TICK_LABEL_FONTSIZE = 7
TITLE = "Per-metric heatmap of Δ (adapted − base); colors are not comparable across metrics"
SUPTITLES = True  # print small note about column-wise color scales
# ==================================


def compute_half_range(series: pd.Series, mode: str, manual_val: float | None = None) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().values
    if s.size == 0:
        return 1.0
    if mode == "p95_abs":
        half = np.quantile(np.abs(s), 0.95)
    elif mode == "max_abs":
        half = float(np.max(np.abs(s)))
    elif mode == "manual":
        if manual_val is None:
            raise ValueError("Manual half-range requested but not provided for a metric.")
        half = float(manual_val)
    else:
        raise ValueError(f"Unknown COLUMN_SCALE_MODE: {mode}")
    # Avoid zero range
    if half == 0:
        half = 1.0
    return half


def build_row_order(df: pd.DataFrame, metrics: list[str], half_ranges: dict[str, float]) -> list[int]:
    if ROW_SORT_MODE == "none":
        return list(range(len(df)))
    elif ROW_SORT_MODE == "mean_scaled":
        # Scale each metric by its own half-range, then average per row
        scaled = []
        for m in metrics:
            hr = half_ranges[m]
            scaled.append(df[m] / hr)
        scaled_mat = np.vstack([s.values for s in scaled]).T  # shape (rows, metrics)
        scores = scaled_mat.mean(axis=1)
        order = np.argsort(-scores)  # descending
        return order.tolist()
    else:
        raise ValueError(f"Unknown ROW_SORT_MODE: {ROW_SORT_MODE}")


def draw_per_column_heatmap(df: pd.DataFrame):
    # Compute half-ranges per column
    half_ranges = {}
    for m in METRICS:
        manual_val = MANUAL_HALF_RANGES.get(m) if COLUMN_SCALE_MODE == "manual" else None
        half_ranges[m] = compute_half_range(df[m], COLUMN_SCALE_MODE, manual_val)

    # Row order
    row_order = build_row_order(df, METRICS, half_ranges)
    df_sorted = df.iloc[row_order].reset_index(drop=True)

    n_rows = df_sorted.shape[0]
    n_cols = len(METRICS)

    # Figure sizing: height scales with rows; width scales with columns
    fig_h = max(4.5, min(12, 0.28 * n_rows + 1.5))
    fig_w = max(7.5, min(18, 1.8 * n_cols + 1.5))

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI, constrained_layout=False)
    # Create a GridSpec with 2 rows: row 0 for mini colorbars, row 1 for heat tiles
    gs = GridSpec(nrows=2, ncols=n_cols, height_ratios=[0.25, 1.0], hspace=0.2, wspace=0.15, figure=fig)

    # Draw each column as its own 1-column heatmap
    for j, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[1, j])

        # Data for this column: shape (n_rows, 1)
        col_vals = df_sorted[metric].to_numpy().reshape(n_rows, 1)

        half = half_ranges[metric]
        norm = TwoSlopeNorm(vmin=-half, vcenter=0.0, vmax=half)

        im = ax.imshow(
            col_vals,
            cmap=CMAP,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )

        # Row labels only on the left-most column
        if FIG_LEFT_LABELS and j == 0:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(df_sorted["test_name"].tolist(), fontsize=ROW_LABEL_FONTSIZE)
        else:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels([""] * n_rows)

        # Column label
        ax.set_xticks([0])
        ax.set_xticklabels([metric], fontsize=COL_LABEL_FONTSIZE, rotation=45, ha="right")

        # Draw grid lines between rows for readability
        for r in range(1, n_rows):
            ax.axhline(r - 0.5, color="black", linewidth=0.2, alpha=0.3)

        # Remove x-axis ticks (we show the column label instead)
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

        # Optional cell annotations
        if ANNOTATE_CELLS:
            for r in range(n_rows):
                val = col_vals[r, 0]
                if abs(val) >= ANNOTATION_MIN_ABS:
                    ax.text(
                        0, r, ANNOTATION_FMT.format(val),
                        ha="center", va="center",
                        fontsize=ANNOT_FONTSIZE,
                        color="black" if abs(val) < half * 0.65 else "white",
                    )

        # Mini colorbar for this column (above)
        cax = fig.add_subplot(gs[0, j])
        cb = plt.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7)
        # Tight ticks: use 3 tick points (−half, 0, +half)
        cb.set_ticks([-half, 0.0, half])
        cb.set_ticklabels([f"{-half:.2f}", "0", f"{half:.2f}"])
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        cax.set_title(metric, fontsize=8, pad=2)
        # Hide the little column title below (we already labeled above)
        ax.set_xlabel("")

        # Optional vertical divider after certain columns
        if j in COLUMN_DIVIDERS_AFTER and j != n_cols - 1:
            # draw a thin white spacer by overlaying a narrow rectangle in figure coords
            # easier: draw a vertical line on the right edge of this axis
            ax.vlines(x=0.5, ymin=-0.5, ymax=n_rows - 0.5, colors="k", linewidth=0.6, alpha=0.4)

    # Big title and caption note
    fig.suptitle(TITLE, fontsize=11, y=0.995)
    if SUPTITLES:
        fig.text(
            0.5, 0.965,
            "Note: Each metric column uses its own symmetric color scale centered at 0; "
            "colors are not comparable across columns. Numbers are raw Δ.",
            ha="center", va="top", fontsize=9
        )

    # Tight layout and save
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.28 if FIG_LEFT_LABELS else 0.06, right=0.98)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


def main():
    df_raw = pd.read_csv(CSV_PATH)
    # Validate needed columns
    required = ["test_name"] + METRICS
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Ensure column types are numeric for metrics
    for m in METRICS:
        df_raw[m] = pd.to_numeric(df_raw[m], errors="coerce")

    draw_per_column_heatmap(df_raw)


if __name__ == "__main__":
    main()
