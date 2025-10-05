#!/usr/bin/env python3
"""
Minimal plotting script for Hugging Face Trainer 'trainer_state.json' files.

What it does:
- Iterates runs R1..R14 and opens 'R{idx}_trainer_state.json' from INPUT_DIR
- Extracts training loss (loss vs step) and eval loss (eval_loss vs step)
- Plots:
    - loss_by_step__train_all_runs.png  (training losses across runs)
    - loss_by_step__eval_all_runs.png   (eval losses across runs)
    - best_run_bar.png                  (bar chart of best metric per run; y-axis truncated for visibility)
- Writes:
    - runs_summary.csv                  (columns: run_id, best_metric, best_eval_step, last_step, num_eval_points)

Constraints:
- No CLI args; constants at top for input/output dirs
- Steps only on X-axis (no epoch variants)
- No smoothing, no de-duplication, no vertical markers
- No regex/glob; filenames are constructed R1..R14 explicitly

Dependencies:
    pip install python-dotenv pandas matplotlib
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt


# ========= CONFIG =========
load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
if not project_root_str:
    raise RuntimeError(
        "Environment variable PROJECT_ROOT not set. "
        "Create a .env with PROJECT_ROOT=/path/to/your/project."
    )

PROJECT_ROOT = Path(project_root_str)

# Adjust these two paths if needed:
INPUT_DIR = PROJECT_ROOT / "results" / "training"               # folder containing R{idx}_trainer_state.json
OUTPUT_DIR = PROJECT_ROOT / "results" / "training" / "plots"    # output folder for plots & CSV

RUN_START = 1
RUN_END = 14  # inclusive

TRAIN_PLOT_NAME = "loss_by_step__train_all_runs.png"
EVAL_PLOT_NAME = "loss_by_step__eval_all_runs.png"
BAR_PLOT_NAME = "best_run_bar.png"
SUMMARY_CSV_NAME = "runs_summary.csv"
# ==========================


def load_trainer_state(fp: Path) -> Dict[str, Any]:
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_points(state: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        train_df: columns [step, epoch, loss, learning_rate, grad_norm]
        eval_df:  columns [step, epoch, eval_loss]
    """
    logs = state.get("log_history", []) or []

    train_rows = []
    eval_rows = []

    for rec in logs:
        if "loss" in rec and "step" in rec:
            train_rows.append(
                {
                    "step": rec["step"],
                    "epoch": rec.get("epoch", None),
                    "loss": rec["loss"],
                    "learning_rate": rec.get("learning_rate", None),
                    "grad_norm": rec.get("grad_norm", None),
                }
            )
        if "eval_loss" in rec and "step" in rec:
            eval_rows.append(
                {
                    "step": rec["step"],
                    "epoch": rec.get("epoch", None),
                    "eval_loss": rec["eval_loss"],
                }
            )

    train_df = pd.DataFrame(train_rows).sort_values("step").reset_index(drop=True)
    eval_df  = pd.DataFrame(eval_rows).sort_values("step").reset_index(drop=True)
    return train_df, eval_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_train = []
    all_eval = []
    summary_rows = []

    for i in range(RUN_START, RUN_END + 1):
        run_id = f"R{i}"
        filename = f"{run_id}_trainer_state.json"
        fp = INPUT_DIR / filename

        if not fp.exists():
            print(f"[WARN] Missing file: {fp} — skipping.")
            continue

        try:
            state = load_trainer_state(fp)
        except Exception as e:
            print(f"[ERROR] Failed to load {fp}: {e}")
            continue

        train_df, eval_df = collect_points(state)

        if not train_df.empty:
            train_df = train_df.assign(run_id=run_id)
            all_train.append(train_df)

        if not eval_df.empty:
            eval_df = eval_df.assign(run_id=run_id)
            all_eval.append(eval_df)

        # CSV fields (only the requested ones)
        best_metric = state.get("best_metric", None)
        best_eval_step = state.get("best_global_step", None)  # from file
        num_eval_points = int(len(eval_df)) if not eval_df.empty else 0

        last_step_candidates = []
        if not train_df.empty:
            last_step_candidates.append(train_df["step"].max())
        if not eval_df.empty:
            last_step_candidates.append(eval_df["step"].max())
        last_step = int(max(last_step_candidates)) if last_step_candidates else None

        summary_rows.append(
            {
                "run_id": run_id,
                "best_metric": best_metric,
                "best_eval_step": best_eval_step,
                "last_step": last_step,
                "num_eval_points": num_eval_points,
            }
        )

    # Concatenate for line plots
    train_all_df = pd.concat(all_train, ignore_index=True) if all_train else pd.DataFrame(columns=["run_id","step","loss"])
    eval_all_df  = pd.concat(all_eval,  ignore_index=True) if all_eval  else pd.DataFrame(columns=["run_id","step","eval_loss"])

    # ---------------- CSV (write first) ----------------
    summary_df = pd.DataFrame(summary_rows, columns=["run_id","best_metric","best_eval_step","last_step","num_eval_points"])
    summary_path = OUTPUT_DIR / SUMMARY_CSV_NAME
    summary_df.to_csv(summary_path, index=False)
    print(f"[OK] Wrote summary CSV → {summary_path}")

    # ---------------- Plots ----------------
    # A) Training loss across runs
    if not train_all_df.empty:
        plt.figure(figsize=(10, 6))
        for run_id, g in train_all_df.groupby("run_id"):
            plt.plot(g["step"], g["loss"], label=run_id)
        plt.xlabel("Step")
        plt.ylabel("Training loss")
        plt.title("Training loss by step (all runs)")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Run", fontsize="small")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / TRAIN_PLOT_NAME, dpi=150)
        plt.close()
        print(f"[OK] Saved training plot → {OUTPUT_DIR / TRAIN_PLOT_NAME}")
    else:
        print("[INFO] No training data found to plot.")

    # B) Eval loss across runs
    if not eval_all_df.empty:
        plt.figure(figsize=(10, 6))
        for run_id, g in eval_all_df.groupby("run_id"):
            plt.plot(g["step"], g["eval_loss"], marker="o", linestyle="-", label=run_id)
        plt.xlabel("Step")
        plt.ylabel("Validation (eval) loss")
        plt.title("Validation (eval) loss by step (all runs)")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Run", fontsize="small")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / EVAL_PLOT_NAME, dpi=150)
        plt.close()
        print(f"[OK] Saved eval plot → {OUTPUT_DIR / EVAL_PLOT_NAME}")
    else:
        print("[INFO] No eval data found to plot.")

    # C) Best-metric bar chart (values from summary CSV)
    try:
        chart_df = pd.read_csv(summary_path)
        chart_df["best_metric"] = pd.to_numeric(chart_df["best_metric"], errors="coerce")
        chart_df = chart_df.dropna(subset=["best_metric"])

        if not chart_df.empty:
            # Sort so "best" (usually lowest for loss) appears first
            # chart_df = chart_df.sort_values("best_metric", ascending=True)

            vmin = float(chart_df["best_metric"].min())
            vmax = float(chart_df["best_metric"].max())
            # Tight y-limits for visibility; allow non-zero baseline
            if vmin == vmax:
                margin = (abs(vmin) * 0.01) or 1e-6
            else:
                margin = (vmax - vmin) * 0.10  # 10% padding

            y_bottom = vmin - margin
            y_top = vmax + margin

            plt.figure(figsize=(10, 6))
            plt.bar(chart_df["run_id"], chart_df["best_metric"])
            plt.xlabel("Run")
            plt.ylabel("Best metric (from summary CSV)")
            plt.title("Best metric per run — y-axis truncated for visibility")
            plt.ylim(y_bottom, y_top)  # <- truncated axis (may start above 0)
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / BAR_PLOT_NAME, dpi=150)
            plt.close()
            print(f"[OK] Saved best-run bar chart → {OUTPUT_DIR / BAR_PLOT_NAME}")
        else:
            print("[INFO] No numeric best_metric values in summary CSV; skipping bar chart.")
    except Exception as e:
        print(f"[WARN] Could not render bar chart from summary CSV: {e}")


if __name__ == "__main__":
    main()
