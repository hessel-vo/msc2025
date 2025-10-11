import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)

MODEL_SIZE = "12b"
RESULT_TYPE = "adapted"  # "baseline" or "adapted"
DATASET_TYPE = "core"

if RESULT_TYPE == "adapted":
    MODEL_NAME = f"adapted_{MODEL_SIZE}_{DATASET_TYPE}"
    INPUT_PATH = PROJECT_ROOT / "results" / "evaluation" / f"{RESULT_TYPE}" / f"{MODEL_NAME}"
else:
    MODEL_NAME = f"base_{MODEL_SIZE}"
    INPUT_PATH = PROJECT_ROOT / "results" / "evaluation" / f"{RESULT_TYPE}" / f"{MODEL_NAME}"

# ----------------------------
# CLI args & validation
# ----------------------------
def main():

    if len(sys.argv) != 6:
        print("Usage: python extract_extremes_problems.py <tasktype> <source> <summarylength> <shotcount> <rci_y_or_n>")
        print("Example: python extract_extremes_problems.py generation xl short zero y")
        sys.exit(1)

    tasktype = sys.argv[1].strip().lower()
    source = sys.argv[2].strip().lower()
    summarylength = sys.argv[3].strip().lower()
    shotcount = sys.argv[4].strip().lower()
    rci_flag = sys.argv[5].strip().lower()

    if tasktype not in ["summarization", "generation"]:
        print(f"Error: Invalid tasktype '{tasktype}'. Must be 'summarization' or 'generation'.")
        sys.exit(1)
    if source not in ["xl", "auto"]:
        print(f"Error: Invalid source '{source}'. Must be 'xl' or 'auto'.")
        sys.exit(1)
    if summarylength not in ["short", "long"]:
        print(f"Error: Invalid summarylength '{summarylength}'. Must be 'short' or 'long'.")
        sys.exit(1)
    if shotcount not in ["zero", "one", "three"]:
        print(f"Error: Invalid shotcount '{shotcount}'. Must be 'zero', 'one', or 'three'.")
        sys.exit(1)
    if rci_flag not in ["y", "n"]:
        print(f"Error: Invalid rci flag '{rci_flag}'. Must be 'y' or 'n'.")
        sys.exit(1)

    is_rci = (rci_flag == "y")

    # Per-problem evaluation CSV:
    # baseline: evaluation_results_{MODEL_NAME}_{TASK}_{source}_{summarylength}_{shotcount}[ _rci].csv
    # adapted:  evaluation_results_{MODEL_NAME}_{DATASET_TYPE}_{TASK}_{source}_{summarylength}_{shotcount}[ _rci].csv
    TASK = tasktype

    if RESULT_TYPE == "adapted":
        base_name = f"evaluation_results_{MODEL_NAME}_{TASK}_{source}_{summarylength}_{shotcount}"
    else:
        base_name = f"evaluation_results_{MODEL_NAME}_{TASK}_{source}_{summarylength}_{shotcount}"

    input_filename = f"{base_name}{'_rci' if is_rci else ''}.csv"
    input_filepath = INPUT_PATH / input_filename

    print(f"Loading per-problem evaluation results from: {input_filepath}")
    df = pd.read_csv(input_filepath)

    # ----------------------------
    # Metric selection & hygiene
    # ----------------------------
    if tasktype == "summarization":
        metric_col = "bertscore_f1"
    else:  # generation
        metric_col = "codebleu"

    # Which generated column to output?
    generated_col = "generated_rci" if is_rci else "generated"
    for required in ["id", metric_col, generated_col, "reference"]:
        if required not in df.columns:
            print(f"Error: Input CSV missing required column: '{required}'")
            sys.exit(1)

    # language is optional but nice to include if present
    has_language = "language" in df.columns

    # Ensure numeric metric (coerce errors to NaN and drop)
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df_clean = df.dropna(subset=[metric_col]).copy()
    if df_clean.empty:
        print(f"Error: No valid '{metric_col}' values found.")
        sys.exit(1)

    # ----------------------------
    # Find top-5 and bottom-5
    # ----------------------------
    k = 5
    n = len(df_clean)

    top_k = df_clean.nlargest(k, metric_col).copy()
    bottom_k = df_clean.nsmallest(k, metric_col).copy()

    top_k["category"] = "top"
    bottom_k["category"] = "bottom"

    top_k = top_k.sort_values([metric_col, "id"], ascending=[False, True]).reset_index(drop=True)
    bottom_k = bottom_k.sort_values([metric_col, "id"], ascending=[True, True]).reset_index(drop=True)

    top_k["rank"] = top_k.index + 1
    bottom_k["rank"] = bottom_k.index + 1

    cols_out = ["category", "rank", "id", "reference", generated_col, metric_col]
    if has_language:
        cols_out.insert(2, "language")  # after rank

    extremes = pd.concat([top_k[cols_out], bottom_k[cols_out]], ignore_index=True)

    # ----------------------------
    # Save output
    # ----------------------------
    out_suffix = "_rci" if is_rci else ""
    out_filename = f"all_extremes_{MODEL_NAME}_{TASK}_{source}_{summarylength}_{shotcount}{out_suffix}.csv"
    out_path = INPUT_PATH / out_filename
    extremes.to_csv(out_path, index=False)

    print("\n=== Summary ===")
    print(f"Task: {TASK} | Source: {source} | Summary: {summarylength} | Shots: {shotcount} | RCI: {is_rci}")
    print(f"Metric: {metric_col} | Output text column: {generated_col}")
    print(f"Problems available: {n} | Reported per side: {k}")
    print(f"Saved extremes CSV to: {out_path}")

if __name__ == "__main__":
    main()
