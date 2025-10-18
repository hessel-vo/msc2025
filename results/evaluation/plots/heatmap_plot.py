#!/usr/bin/env python3
"""
Per-column heatmap for experiment deltas (adapted − base).

Updates:
- Color scales stay ABOVE each column as THIN, VERTICAL bars.
- Narrower columns and tighter vertical spacing.
- Removed overall plot title/suptitle code.
- Removed manual half-range support.

CSV format:
    test_name,CodeBLEU,BLEU,ROUGE-1,ROUGE-2,ROUGE-L,BERTScore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ========= USER CONSTANTS =========
CSV_PATH = "heatmap_data.csv"         # <-- set your input CSV path here
OUTPUT_PATH = "per_column_heatmap.png"
DPI = 200

# Which columns to visualize (order is kept)
METRICS = ["CodeBLEU", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]

# Column scaling strategy: "p95_abs" (recommended) or "max_abs"
COLUMN_SCALE_MODE = "p95_abs"

# Show annotation text inside cells?
ANNOTATE_CELLS = True
ANNOTATION_FMT = "{:+.2f}"  # e.g., +1.57 / -0.34
ANNOTATION_MIN_ABS = 0.00   # hide annotation if |Δ| < threshold

# Visually separate generation vs summarization columns?
COLUMN_DIVIDERS_AFTER = [0]  # [] for none

# Appearance
CMAP = "RdBu_r"                  # diverging colormap centered at 0
FIG_LEFT_LABELS = True           # show row labels only on the far-left column
ROW_LABEL_FONTSIZE = 8
COL_LABEL_FONTSIZE = 9
ANNOT_FONTSIZE = 7
TICK_LABEL_FONTSIZE = 7

# —— layout sizing (narrow) ——
HEIGHT_PER_ROW = 0.20
BASE_FIG_H = 1.5
WIDTH_PER_COL = 1
BASE_FIG_W = 0.9
GRID_WSPACE = 0.10     # spacing between columns

# —— thin vertical colorbar (above each column) ——
CBAR_REL_WIDTH = 0.10   # fraction of the top cell width (0–1). Smaller = thinner bar.
CBAR_REL_HEIGHT = 0.50  # fraction of the top cell height (0–1)
CBAR_TICK_FONTSIZE = 7
CBAR_TITLE_FONTSIZE = 8

# —— tight vertical spacing controls ——
TOP_MARGIN = 0.96       # overall top margin
GRID_HSPACE_ROWS = 0.00 # gap between colorbars row and heatmap row
TOPROW_HEIGHT_RATIO = 0.10  # relative height of the top row
# ==================================


def compute_half_range(series: pd.Series, mode: str) -> float:
    """Compute symmetric half-range for a metric column."""
    s = pd.to_numeric(series, errors="coerce").dropna().values
    if s.size == 0:
        return 1.0
    if mode == "p95_abs":
        half = float(np.quantile(np.abs(s), 0.95))
    elif mode == "max_abs":
        half = float(np.max(np.abs(s)))
    else:
        raise ValueError(f"Unknown COLUMN_SCALE_MODE: {mode}")
    return half or 1.0

def draw_per_column_heatmap(df: pd.DataFrame):
    # Half-ranges per column
    half_ranges = {m: compute_half_range(df[m], COLUMN_SCALE_MODE) for m in METRICS}

    # Row order
    row_order = list(range(len(df)))
    df_sorted = df.iloc[row_order].reset_index(drop=True)

    n_rows = df_sorted.shape[0]
    n_cols = len(METRICS)

    # Figure sizing
    fig_h = max(4.5, min(12, HEIGHT_PER_ROW * n_rows + BASE_FIG_H))
    fig_w = max(6.6, min(18, WIDTH_PER_COL * n_cols + BASE_FIG_W))

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI, constrained_layout=False)
    # 2 rows: row 0 = vertical colorbars area; row 1 = heatmap tiles
    gs = GridSpec(
        nrows=2, ncols=n_cols,
        height_ratios=[TOPROW_HEIGHT_RATIO, 1.0],
        hspace=GRID_HSPACE_ROWS,
        wspace=GRID_WSPACE,
        figure=fig
    )

    for j, metric in enumerate(METRICS):
        # — Heatmap cell (bottom row) —
        ax = fig.add_subplot(gs[1, j])

        col_vals = df_sorted[metric].to_numpy().reshape(n_rows, 1)
        half = half_ranges[metric]
        norm = TwoSlopeNorm(vmin=-half, vcenter=0.0, vmax=half)

        im = ax.imshow(
            col_vals, cmap=CMAP, norm=norm,
            aspect="auto", interpolation="nearest", origin="upper",
        )

        # Row labels only on left-most col
        if FIG_LEFT_LABELS and j == 0:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(df_sorted["test_name"].tolist(), fontsize=ROW_LABEL_FONTSIZE)
        else:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels([""] * n_rows)

        # Column label at bottom
        ax.set_xticks([0])
        ax.set_xticklabels([metric], fontsize=COL_LABEL_FONTSIZE, rotation=45, ha="right")

        # Row separators
        for r in range(1, n_rows):
            ax.axhline(r - 0.5, color="black", linewidth=0.2, alpha=0.3)

        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

        # Cell annotations (raw Δ)
        if ANNOTATE_CELLS:
            for r in range(n_rows):
                val = col_vals[r, 0]
                if abs(val) >= ANNOTATION_MIN_ABS:
                    ax.text(
                        0, r, ANNOTATION_FMT.format(val),
                        ha="center", va="center", fontsize=ANNOT_FONTSIZE,
                        color="black" if abs(val) < half * 0.65 else "white",
                    )

        # Optional vertical divider after certain columns
        if j in COLUMN_DIVIDERS_AFTER and j != n_cols - 1:
            ax.vlines(x=0.5, ymin=-0.5, ymax=n_rows - 0.5, colors="k", linewidth=0.6, alpha=0.4)

        # — Thin vertical colorbar ABOVE this column —
        holder_ax = fig.add_subplot(gs[0, j])
        holder_ax.axis("off")
        cax = inset_axes(
            holder_ax,
            width=f"{CBAR_REL_WIDTH*100:.1f}%",   # % of holder width
            height=f"{CBAR_REL_HEIGHT*100:.1f}%", # % of holder height
            loc="center"
        )
        cb = plt.colorbar(im, cax=cax, orientation="vertical")
        cb.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
        cb.set_ticks([-half, 0.0, half])
        cb.set_ticklabels([f"{-half:.2f}", "0", f"{half:.2f}"])
        cax.set_title(metric, fontsize=CBAR_TITLE_FONTSIZE, pad=2)

    # Tight overall margins
    plt.subplots_adjust(
        top=TOP_MARGIN,
        bottom=0.05,
        left=0.28 if FIG_LEFT_LABELS else 0.06,
        right=0.985
    )

    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


def main():
    df_raw = pd.read_csv(CSV_PATH)
    required = ["test_name"] + METRICS
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    for m in METRICS:
        df_raw[m] = pd.to_numeric(df_raw[m], errors="coerce")
    draw_per_column_heatmap(df_raw)


if __name__ == "__main__":
    main()
