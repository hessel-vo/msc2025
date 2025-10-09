import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))

RESULTS_SUBFOLDER = "baseline"
INPUT_FOLDER = f"results/benchmark/{RESULTS_SUBFOLDER}/to_process/processed_results"
OUTPUT_FOLDER = "results/evaluation/rci_diff"
OVERVIEW_OUTPUT_FILENAME = "rci_diff_overview.csv"

NORMALIZE_WHITESPACE = False
REMOVE_CODE_FENCES = False


EXPECTED_COLUMNS = [
    "id",
    "language",
    "reference",
    "generated",
    "generated_rci",
]


def _strip_code_fences(text: str) -> str:
    """
    Removes leading/trailing triple backtick code fences if present, with or without a language hint.
    """
    if text is None:
        return ""
    s = str(text).strip()

    # Leading fence
    if s.startswith("```"):
        # Drop the first fence line (could be ``` or ```lang)
        lines = s.splitlines()
        if lines:
            lines = lines[1:]
            s = "\n".join(lines).strip()

    # Trailing fence
    if s.endswith("```"):
        # Remove last fence line
        lines = s.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
            s = "\n".join(lines).strip()

    return s


def _normalize(text: str) -> str:
    """
    Normalizes text for a more forgiving equality check:
    - Optionally strips code fences
    - Trims leading/trailing whitespace
    - Collapses trailing spaces on each line
    - Normalizes line endings
    """
    if text is None:
        print("Error: Encountered None text during normalization.")
        sys.exit(1)
        return ""
    s = str(text)

    if REMOVE_CODE_FENCES:
        s = _strip_code_fences(s)

    if NORMALIZE_WHITESPACE:
        # Normalize line endings and trim per line
        lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        # Re-join and strip outer whitespace
        s = "\n".join(lines).strip()
    else:
        s = s.strip()

    return s


def compare_changed(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """
    Returns a boolean Series where True indicates 'generated_rci' differs from 'generated',
    after normalization. If both are empty after normalization, counts as NOT different.
    """
    a_norm = series_a.fillna("").map(_normalize)
    b_norm = series_b.fillna("").map(_normalize)
    both_empty = (a_norm == "") & (b_norm == "")
    return (a_norm != b_norm) & (~both_empty)


def process_single_csv(csv_path: Path, out_dir: Path) -> pd.DataFrame:
    """
    Reads one CSV, computes per-language counts and percentages of changed solutions,
    writes a per-file breakdown CSV, and returns a single-row DataFrame for the overview.
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    # Compute per-row change boolean
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
    # Compute percentages safely
    per_lang["percentage different"] = per_lang.apply(
        lambda r: (100.0 * r["changed"] / r["total"]) if r["total"] else 0.0,
        axis=1
    )

    # Rename to requested columns and order
    per_lang_out = per_lang.rename(columns={
        "language": "language",
        "changed": "no. RCI different"
    })[["language", "no. RCI different", "percentage different"]]

    # Write per-file output
    per_file_name = f"{csv_path.stem}_rci_diff_by_language.csv"
    per_file_path = out_dir / per_file_name
    per_lang_out.to_csv(per_file_path, index=False)

    # File-level overview row
    total_rows = len(df)
    total_changed = int(df["_rci_changed"].sum())
    pct_changed = (100.0 * total_changed / total_rows) if total_rows else 0.0

    overview_row = pd.DataFrame([{
        "test_name": csv_path.stem,
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
    overview_df.to_csv(overview_out_path, index=False)

    print("\n---")
    print(f"Successfully processed {len(overview_rows)} file(s).")
    print(f"Overview saved to: {overview_out_path}")
    print("Preview:")
    print(overview_df.head())
    print("---")


if __name__ == "__main__":
    main()
