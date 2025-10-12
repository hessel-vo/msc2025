import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configuration
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ""))

RCI_DIFF_INPUT_ROOT = "results/evaluation/rci_diff"

CODEGEN_NAME_TOKENS = "generation"
CODESUM_NAME_TOKENS = "summarization"

# Output folder for the aggregated summaries
SUMMARY_OUTPUT_FOLDER = "results/evaluation/rci_diff/summary"

# Filenames for outputs
CODEGEN_OUT = "rci_diff_summary_code_generation.csv"
CODESUM_OUT = "rci_diff_summary_code_summarization.csv"
COMBINED_OUT = "rci_diff_summary_combined.csv"


def _find_task_files(root: Path, task_name) -> list[Path]:
    candidates = []
    for p in sorted(root.glob("*_rci_diff_by_language.csv")):
        name_lower = p.name.lower()
        if task_name in name_lower:
            candidates.append(p)
    return candidates


def _read_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["no. RCI different"] = pd.to_numeric(df["no. RCI different"], errors="coerce")
    df["percentage different"] = pd.to_numeric(df["percentage different"], errors="coerce")
    df["language"] = df["language"]
    return df


def _summarize_task(task_name: str, files: list[Path]) -> pd.DataFrame:

    frames = []
    for f in files:
        try:
            df = _read_and_validate(f)
            frames.append(df[["language", "no. RCI different", "percentage different"]])
        except Exception as e:
            print(f"Could not process '{f.name}': {e}")

    print(frames)

    all_rows = pd.concat(frames, ignore_index=True)

    # Per-language averages across files/rows
    per_lang = (
        all_rows.groupby("language", dropna=False)
        .agg(
            **{
                "avg no. RCI different": ("no. RCI different", "mean"),
                "avg percentage different": ("percentage different", "mean"),
            }
        )
        .reset_index()
    )

    # Overall macro-averages across all rows
    overall_avg_count = all_rows["no. RCI different"].mean()
    overall_avg_pct = all_rows["percentage different"].mean()

    overall_row = pd.DataFrame(
        [{
            "language": "all_languages",
            "avg no. RCI different": overall_avg_count,
            "avg percentage different": overall_avg_pct,
        }]
    )

    out = pd.concat([per_lang, overall_row], ignore_index=True)
    
    out["avg no. RCI different"] = out["avg no. RCI different"]
    out["avg percentage different"] = out["avg percentage different"]

    return out


def main():

    input_root = PROJECT_ROOT / RCI_DIFF_INPUT_ROOT
    output_root = PROJECT_ROOT / SUMMARY_OUTPUT_FOLDER
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.is_dir():
        print(f"Error: Input root '{input_root}' does not exist.")
        return

    # Locate files for each task
    codegen_files = _find_task_files(input_root, CODEGEN_NAME_TOKENS)
    codesum_files = _find_task_files(input_root, CODESUM_NAME_TOKENS)

    print(f"Found {len(codegen_files)} code-generation file(s).")
    print(f"Found {len(codesum_files)} code-summarization file(s).")

    # Build summaries
    codegen_summary = _summarize_task("generation", codegen_files)
    codesum_summary = _summarize_task("summarization", codesum_files)

    # Write per-task CSVs
    codegen_out_path = output_root / CODEGEN_OUT
    codesum_out_path = output_root / CODESUM_OUT
    if not codegen_summary.empty:
        codegen_summary.to_csv(codegen_out_path, index=False, float_format="%.2f")
        print(f"[generation] Summary written to: {codegen_out_path}")
    if not codesum_summary.empty:
        codesum_summary.to_csv(codesum_out_path, index=False, float_format="%.2f")
        print(f"[summarization] Summary written to: {codesum_out_path}")

    # Combined CSV with a 'task' column
    combined = []
    if not codegen_summary.empty:
        df = codegen_summary.copy()
        df.insert(0, "task", "generation")
        combined.append(df)
    if not codesum_summary.empty:
        df = codesum_summary.copy()
        df.insert(0, "task", "summarization")
        combined.append(df)

    if combined:
        combined_df = pd.concat(combined, ignore_index=True)
        combined_out_path = output_root / COMBINED_OUT
        combined_df.to_csv(combined_out_path, index=False, float_format="%.2f")
        print(f"[combined] Summary written to: {combined_out_path}")


if __name__ == "__main__":
    main()
