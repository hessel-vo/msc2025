import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt


load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)

INPUT_DIR = PROJECT_ROOT / "results" / "training"
OUTPUT_DIR = PROJECT_ROOT / "results" / "training" / "plots"

RUN_START = 1
RUN_END = 14

TRAIN_PLOT_NAME = "loss_by_step__train_all_runs.png"
EVAL_PLOT_NAME = "loss_by_step__eval_all_runs.png"
BAR_PLOT_NAME = "best_run_bar.png"
SUMMARY_CSV_NAME = "runs_summary.csv"


def load_trainer_state(fp: Path) -> Dict[str, Any]:
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_points(state: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:

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
            print(f"Warning- Missing file: {fp} — skipping.")
            continue

        try:
            state = load_trainer_state(fp)
        except Exception as e:
            print(f"Error- Failed to load {fp}: {e}")
            continue

        train_df, eval_df = collect_points(state)

        if not train_df.empty:
            train_df = train_df.assign(run_id=run_id)
            all_train.append(train_df)

        if not eval_df.empty:
            eval_df = eval_df.assign(run_id=run_id)
            all_eval.append(eval_df)

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

    summary_df = pd.DataFrame(summary_rows, columns=["run_id","best_metric","best_eval_step","last_step","num_eval_points"])
    summary_path = OUTPUT_DIR / SUMMARY_CSV_NAME
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote summary CSV → {summary_path}")

    all_run_ids = sorted(
        list(set(train_all_df["run_id"]).union(set(eval_all_df["run_id"]))),
        key=lambda rid: int(rid[1:]) if isinstance(rid, str) and rid[1:].isdigit() else 10**9
    )
    # tab20 palette, simple mapping so colors are consistent across plots
    palette = plt.cm.tab20.colors
    color_map = {rid: palette[i % len(palette)] for i, rid in enumerate(all_run_ids)}

    # A) Training loss all runs
    if not train_all_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_prop_cycle(color=palette)
        for rid in all_run_ids:
            g = train_all_df[train_all_df["run_id"] == rid]
            if g.empty:
                continue
            ax.plot(g["step"], g["loss"], label=rid, color=color_map[rid])
        ax.set_xlabel("Step")
        ax.set_ylabel("Training loss")
        ax.set_title("Training loss by step (all runs)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Run", fontsize="small")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / TRAIN_PLOT_NAME, dpi=150)
        plt.close(fig)
        print(f"Saved training plot → {OUTPUT_DIR / TRAIN_PLOT_NAME}")
    else:
        print("No training data found to plot.")

    # B) Eval loss all runs
    if not eval_all_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_prop_cycle(color=palette)
        for rid in all_run_ids:
            g = eval_all_df[eval_all_df["run_id"] == rid]
            if g.empty:
                continue
            ax.plot(g["step"], g["eval_loss"], marker="o", linestyle="-", label=rid, color=color_map[rid])
        ax.set_xlabel("Step")
        ax.set_ylabel("Validation loss")
        ax.set_title("Validation loss by step (all runs)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Run", fontsize="small")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / EVAL_PLOT_NAME, dpi=150)
        plt.close(fig)
        print(f"Saved eval plot → {OUTPUT_DIR / EVAL_PLOT_NAME}")
    else:
        print("No eval data found to plot.")

    # C) Best-metric bar chart
    try:
        chart_df = pd.read_csv(summary_path)
        chart_df["best_metric"] = pd.to_numeric(chart_df["best_metric"], errors="coerce")
        chart_df = chart_df.dropna(subset=["best_metric"])

        if not chart_df.empty:
            vmin = float(chart_df["best_metric"].min())
            vmax = float(chart_df["best_metric"].max())
            margin = (vmax - vmin) * 0.10 if vmin != vmax else (abs(vmin) * 0.01 or 1e-6)
            y_bottom, y_top = vmin - margin, vmax + margin

            # match colors to line plots
            bar_colors = [color_map.get(rid, palette[0]) for rid in chart_df["run_id"].tolist()]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(chart_df["run_id"], chart_df["best_metric"], color=bar_colors)
            ax.set_xlabel("Run")
            ax.set_ylabel("Best_metric")
            ax.set_title("Best_metric per run — y-axis truncated for visibility")
            ax.set_ylim(y_bottom, y_top)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / BAR_PLOT_NAME, dpi=150)
            plt.close(fig)
            print(f"Saved best-run bar chart → {OUTPUT_DIR / BAR_PLOT_NAME}")
        else:
            print("No numeric best_metric values in summary CSV; skipping bar chart.")
    except Exception as e:
        print(f"Warning: Could not render bar chart from summary CSV: {e}")


if __name__ == "__main__":
    main()
