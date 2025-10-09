import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt


# ── Constants ──────────────────────────────────────────────────────────────────
load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
if not project_root_str:
    raise RuntimeError("PROJECT_ROOT environment variable is not set.")
PROJECT_ROOT = Path(project_root_str)

DATASET_TYPE = "core"
MODEL_SIZE = "1b"

INPUT_DIR = PROJECT_ROOT / "results" / "training"
OUTPUT_DIR = PROJECT_ROOT / "results" / "training" / "plots" / MODEL_SIZE / DATASET_TYPE
TRAINER_STATE_FILENAME = f"{MODEL_SIZE}_{DATASET_TYPE}_trainer_state.json"  # adjust if your file is named differently

TRAIN_PLOT_NAME = "loss_by_step__train.png"
EVAL_PLOT_NAME = "loss_by_step__eval.png"
BOTH_PLOT_NAME = "loss_by_step__train_and_eval.png"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_trainer_state(fp: Path) -> Dict[str, Any]:
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_points(state: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract training and evaluation points from Hugging Face Trainer 'log_history'.
    Returns two DataFrames with columns:
      train_df: step, epoch, loss, learning_rate, grad_norm
      eval_df : step, epoch, eval_loss
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
    eval_df = pd.DataFrame(eval_rows).sort_values("step").reset_index(drop=True)
    return train_df, eval_df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fp = INPUT_DIR / TRAINER_STATE_FILENAME
    if not fp.exists():
        print(f"Error: Missing file: {fp}")
        return

    try:
        state = load_trainer_state(fp)
    except Exception as e:
        print(f"Error: Failed to load {fp}: {e}")
        return

    train_df, eval_df = collect_points(state)

    if train_df.empty and eval_df.empty:
        print("No training or evaluation records found in log_history; nothing to plot.")
        return

    # A) Training loss
    if not train_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_df["step"], train_df["loss"], linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Training loss")
        ax.set_title("Training loss by step")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = OUTPUT_DIR / TRAIN_PLOT_NAME
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved training plot → {out_path}")
    else:
        print("No training data found to plot.")

    # B) Evaluation loss
    if not eval_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eval_df["step"], eval_df["eval_loss"], marker="o", linestyle="-", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Validation loss")
        ax.set_title("Validation loss by step")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = OUTPUT_DIR / EVAL_PLOT_NAME
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved eval plot → {out_path}")
    else:
        print("No evaluation data found to plot.")

    # C) Combined plot (train + eval)
    if not train_df.empty or not eval_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        lines = []
        labels = []
        if not train_df.empty:
            (ln_train,) = ax.plot(train_df["step"], train_df["loss"], linewidth=1.5)
            lines.append(ln_train)
            labels.append("Training loss")
        if not eval_df.empty:
            (ln_eval,) = ax.plot(eval_df["step"], eval_df["eval_loss"], marker="o", linestyle="-", linewidth=1.5)
            lines.append(ln_eval)
            labels.append("Validation loss")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss by step (train + validation)")
        ax.grid(True, alpha=0.3)
        if lines:
            ax.legend(lines, labels, fontsize="small", title="Series")
        fig.tight_layout()
        out_path = OUTPUT_DIR / BOTH_PLOT_NAME
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved combined plot → {out_path}")


if __name__ == "__main__":
    main()
