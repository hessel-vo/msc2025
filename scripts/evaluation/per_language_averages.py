import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configuration
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))

RESULTS_SUBFOLDER = "baseline"
MODEL_SIZE = "12b"
DATASET_TYPE = "core"

if RESULTS_SUBFOLDER == "adapted":
    MODEL = f"adapted_{MODEL_SIZE}_{DATASET_TYPE}"
    INPUT_ROOT = PROJECT_ROOT / f"results/evaluation/{RESULTS_SUBFOLDER}/{MODEL}_{DATASET_TYPE}"
else:
    MODEL = f"base_{MODEL_SIZE}"
    INPUT_ROOT = PROJECT_ROOT / f"results/evaluation/{RESULTS_SUBFOLDER}/{MODEL}"

SUMMARY_OUTPUT_FOLDER = INPUT_ROOT / "per_language_averages"

# Output filenames
SUMMARIZATION_OUT = "corpus_score_summary_summarization.csv"
GENERATION_OUT    = "corpus_score_summary_generation.csv"
COMBINED_OUT      = "corpus_score_summary_combined.csv"

# Glob patterns
SUMMARIZATION_GLOB = "corpus_score_by_language_*_summarization*.csv"
GENERATION_GLOB    = "corpus_score_*_generation*.csv"

# Metrics by task
SUMMARIZATION_METRICS = [
    "bleu_score",
    "corpus_rouge1", "corpus_rouge2", "corpus_rougeL",
    "corpus_bertscore_f1",
]

GENERATION_METRICS = [
    "corpus_codebleu",
    "corpus_ngram_match_score",
    "corpus_weighted_ngram_match_score",
    "corpus_syntax_match_score",
    "corpus_dataflow_match_score",
]

def _find_task_files(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob(pattern))


def _read_and_validate(path: Path, metrics: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)

    keep_cols = ["language"] + metrics
    df = df[keep_cols].copy()

    # Filter out average rows from generation files
    df["language"] = df["language"].astype(str)
    df = df[df["language"].str.lower() != "average"]

    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    return df


def _summarize_files(task_name: str, files: list[Path], metrics: list[str]) -> pd.DataFrame:

    frames = []
    for f in files:
        try:
            frames.append(_read_and_validate(f, metrics))
        except Exception as e:
            print(f"[{task_name}] Skipping '{f.name}': {e}")

    all_rows = pd.concat(frames, ignore_index=True)

    # Per-language macro averages
    per_lang = (
        all_rows.groupby("language", dropna=False)
        .agg({m: "mean" for m in metrics})
        .reset_index()
    )

    return per_lang


def _write_csv(df: pd.DataFrame, path: Path, task_label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.2f")
    print(f"[{task_label}] Summary written to: {path}")


def main():
    SUMMARY_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    summarization_files = _find_task_files(INPUT_ROOT, SUMMARIZATION_GLOB)
    generation_files    = _find_task_files(INPUT_ROOT,    GENERATION_GLOB)

    generation_files = [p for p in generation_files if not p.name.endswith("_combined.csv")]


    print(f"Found {len(summarization_files)} summarization file(s) in '{INPUT_ROOT}'.")
    print(f"Found {len(generation_files)} generation file(s) in '{INPUT_ROOT}'.")

    sum_summary = _summarize_files("summarization", summarization_files, SUMMARIZATION_METRICS)
    gen_summary = _summarize_files("generation", generation_files, GENERATION_METRICS)

    # Write per-task CSVs
    _write_csv(sum_summary, SUMMARY_OUTPUT_FOLDER / SUMMARIZATION_OUT, "summarization")
    _write_csv(gen_summary, SUMMARY_OUTPUT_FOLDER / GENERATION_OUT, "generation")

    # Combined CSV with a 'task' column
    if not sum_summary.empty:
        df = sum_summary.copy()
        df.insert(0, "task", "summarization")
    if not gen_summary.empty:
        df = gen_summary.copy()
        df.insert(0, "task", "generation")

if __name__ == "__main__":
    main()
