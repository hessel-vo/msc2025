import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))

RESULTS_SUBFOLDER = "adapted"
MODEL_SIZE = "12b"
DATASET_TYPE = "extended"

INPUT_FOLDER = f"results/benchmark/{RESULTS_SUBFOLDER}/to_process/processed_results"
OUTPUT_FOLDER = "results/evaluation/rci_diff"
OVERVIEW_OUTPUT_FILENAME = "rci_diff_overview.csv"

NORMALIZE_WHITESPACE = False
REMOVE_CODE_FENCES = False

def compare_changed(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    both_empty = (series_a == "") & (series_b == "")
    return (series_a != series_b) & (~both_empty)


def process_single_csv(csv_path: Path, out_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Compute per-row change
    df["_rci_changed"] = compare_changed(df["generated"], df["generated_rci"])

    # Per-language aggregation
    per_lang = (
        df.groupby("language", dropna=False)
          .agg(
              total=("language", "size"),
              changed=("_rci_changed", "sum")
          )
          .reset_index()
    )

    print(per_lang)
    # Compute percentages
    per_lang["percentage different"] = per_lang.apply(
        lambda r: (100.0 * r["changed"] / r["total"]) if r["total"] else 0.0,
        axis=1
    )

    print(per_lang)

    # Rename and order columns for output
    per_lang_out = per_lang.rename(columns={
        "language": "language",
        "changed": "no. RCI different"
    })[["language", "no. RCI different", "percentage different"]]

    # Write per-file output

    if RESULTS_SUBFOLDER == "adapted":
        input_filename = csv_path.stem.split("_")
        task_type = input_filename[3]
        source = input_filename[4]
        summary_length = input_filename[5]
        shot_count = input_filename[6]
        test_name = f"adapted_{MODEL_SIZE}_{DATASET_TYPE}_{task_type}_{source}_{summary_length}_{shot_count}"
    else:
        input_filename = csv_path.stem.split("_")
        task_type = input_filename[1]
        source = input_filename[2]
        summary_length = input_filename[3]
        shot_count = input_filename[4]        
        test_name = f"base_{MODEL_SIZE}_{task_type}_{source}_{summary_length}_{shot_count}"

    per_file_name = f"{test_name}_rci_diff_by_language.csv"
    per_file_path = out_dir / per_file_name
    per_lang_out.to_csv(per_file_path, index=False)

    # File-level overview row
    total_rows = len(df)
    total_changed = int(df["_rci_changed"].sum())
    pct_changed = (100.0 * total_changed / total_rows) if total_rows else 0.0

    overview_row = pd.DataFrame([{
        "test_name": test_name,
        "no. RCI different": total_changed,
        "percentage different": pct_changed
    }])

    print(f"Processed: {csv_path.name} | Changed: {total_changed}/{total_rows} ({pct_changed:.2f}%)")
    print(f"  ↳ Per-language output: {per_file_path}")

    return overview_row


def main():

    input_path = PROJECT_ROOT / INPUT_FOLDER
    output_path = PROJECT_ROOT / OUTPUT_FOLDER

    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_path.glob("*.csv"))

    overview_rows = []
    print(f"Searching for input CSVs in: {input_path}")

    for csv_file in csv_files:
        try:
            overview_rows.append(process_single_csv(csv_file, output_path))
        except Exception as e:
            print(f"Could not process '{csv_file.name}': {e}")

    overview_df = pd.concat(overview_rows, ignore_index=True)
    overview_out_path = output_path / OVERVIEW_OUTPUT_FILENAME
    overview_df.to_csv(overview_out_path, index=False, float_format="%.2f")

    print("\n---")
    print(f"Successfully processed {len(overview_rows)} file(s).")
    print(f"Overview saved to: {overview_out_path}")
    print("Preview:")
    print(overview_df.head())
    print("---")


if __name__ == "__main__":
    main()
