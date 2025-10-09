import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np

# Load environment variables from .env file
load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
os.environ['HF_HOME'] = str(HF_CACHE_DIR)

from codebleu import calc_codebleu

# Constants for the model and results folder
MODEL_NAME = "gemma-3-12b-it"
RESULTS_SUBFOLDER = "baseline"
TASK = "generation"

def validate_arguments(args):
    if len(args) < 4:
        print("Usage: python run_generation_evaluation.py <source> <summary_length> <shot_count> [subset]")
        print("Example: python run_generation_evaluation.py xl short zero")
        sys.exit(1)

    source, summary_length, shot_count, subset = args[1], args[2], args[3], args[4] if len(args) == 5 else None

    if source not in ["xl", "auto"]:
        print(f"Error: Invalid source '{source}'. Must be 'xl' or 'auto'.")
        sys.exit(1)
    if summary_length not in ["short", "long"]:
        print(f"Error: Invalid summary_length '{summary_length}'. Must be 'short' or 'long'.")
        sys.exit(1)
    if shot_count not in ["zero", "one", "three"]:
        print(f"Error: Invalid shot_count '{shot_count}'. Must be 'zero', 'one', or 'three'.")
        sys.exit(1)

    return source, summary_length, shot_count, subset

def _to_0_100_and_round(x):
    """Rescale to 0–100 if value looks like 0–1; then round to 2 decimals."""
    try:
        val = float(x)
    except Exception:
        return x
    if val <= 1.0:
        val *= 100.0
    return round(val, 2)

def calculate_codebleu_scores_for_column(df, pred_col, desc_suffix=""):
    """
    Compute per-problem CodeBLEU metrics for the given prediction column,
    and corpus-level metrics per language. All metrics are rescaled to 0–100
    (if needed) and rounded to 2 decimals.
    """
    problem_scores = {}
    weights = (0.25, 0.25, 0.25, 0.25)

    print(f"Calculating problem-level CodeBLEU scores for each row{desc_suffix}...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Problem-Level Scores{desc_suffix}"):
        prediction = str(row[pred_col])
        reference = str(row["reference"])
        lang = row["language"]
        p_id = row["id"]

        codebleu_results = calc_codebleu(
            references=[[reference]],
            predictions=[prediction],
            lang=lang,
            weights=weights
        )

        # Rescale/round every value we store
        problem_scores[p_id] = {
            'codebleu': _to_0_100_and_round(codebleu_results['codebleu']),
            'ngram_match_score': _to_0_100_and_round(codebleu_results['ngram_match_score']),
            'weighted_ngram_match_score': _to_0_100_and_round(codebleu_results['weighted_ngram_match_score']),
            'syntax_match_score': _to_0_100_and_round(codebleu_results['syntax_match_score']),
            'dataflow_match_score': _to_0_100_and_round(codebleu_results['dataflow_match_score']),
        }

    # Corpus-Level Calculation by Language
    corpus_scores_by_lang = {}
    languages = df['language'].unique()
    print(f"\nFound languages for corpus evaluation{desc_suffix}: {list(languages)}")

    for lang in languages:
        print(f"Calculating corpus-level CodeBLEU score for '{lang}'{desc_suffix}...")
        lang_df = df[df['language'] == lang]

        all_predictions = lang_df[pred_col].astype(str).tolist()
        all_references = [[str(ref)] for ref in lang_df["reference"].tolist()]

        corpus_result = calc_codebleu(
            references=all_references,
            predictions=all_predictions,
            lang=lang,
            weights=weights
        )

        # Rescale + prefix keys with corpus_
        corpus_scores_by_lang[lang] = {
            f"corpus_{k}": _to_0_100_and_round(v) for k, v in corpus_result.items()
        }

    return problem_scores, corpus_scores_by_lang

def _save_per_problem(df, problem_scores, output_path):
    metrics_df = pd.DataFrame.from_dict(problem_scores, orient='index').reset_index().rename(columns={'index': 'id'})
    output_df = pd.merge(df, metrics_df, on='id')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"\nDetailed problem-level results saved to: {output_path}")

def _rows_with_average(test_name, corpus_scores_by_lang):
    """Build rows per language + an 'average' row, with rounding preserved."""
    corpus_rows = []
    for lang, scores in corpus_scores_by_lang.items():
        row_data = {'test_name': test_name, 'language': lang}
        row_data.update(scores)
        corpus_rows.append(row_data)

    corpus_df = pd.DataFrame(corpus_rows)
    # Average numeric columns over the languages we just added
    avg_scores = corpus_df.select_dtypes(include='number').mean().apply(lambda v: round(float(v), 2))
    average_row = {'test_name': test_name, 'language': 'average'}
    average_row.update(avg_scores.to_dict())
    return corpus_rows + [average_row]

def main():
    source, summary_length, shot_count, subset = validate_arguments(sys.argv)
    print(f"Starting code generation evaluation for: [Source: {source}, Summary: {summary_length}, Shots: {shot_count}]")

    input_filename = f"{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}_results.csv"
    input_filepath = PROJECT_ROOT / "results" / "benchmark" / RESULTS_SUBFOLDER / "to_process/processed_results" / input_filename

    if subset:
        input_filepath = PROJECT_ROOT / "results" / "benchmark" / RESULTS_SUBFOLDER / f"{MODEL_NAME}_{TASK}_subset_results.csv"

    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded input from: {input_filepath}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        sys.exit(1)

    required_cols = {"id", "language", "reference", "generated", "generated_rci"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing required column(s): {missing}")
        sys.exit(1)

    df['generated'] = df['generated'].astype(str).str.strip('"')
    df['generated_rci'] = df['generated_rci'].astype(str).str.strip('"')
    df['reference'] = df['reference'].astype(str).str.strip('"')

    # ----- Evaluate: generated -----
    problem_gen, corpus_gen_by_lang = calculate_codebleu_scores_for_column(df, "generated", desc_suffix=" (generated)")

    # ----- Evaluate: generated_rci -----
    problem_rci, corpus_rci_by_lang = calculate_codebleu_scores_for_column(df, "generated_rci", desc_suffix=" (generated_rci)")

    # ---------- Save per-problem results ----------
    OUTPUT_ROOT = PROJECT_ROOT / "results" / "evaluation" / RESULTS_SUBFOLDER
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # generated
    out_file = f"evaluation_results_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}.csv"
    out_path = OUTPUT_ROOT / out_file
    if subset:
        out_path = OUTPUT_ROOT / "subset" / f"evaluation_results_{MODEL_NAME}_{TASK}_subset.csv"
    _save_per_problem(df, problem_gen, out_path)

    # generated_rci (with _rci suffix)
    if subset:
        out_path_rci = OUTPUT_ROOT / "subset" / f"evaluation_results_{MODEL_NAME}_{TASK}_subset_rci.csv"
    else:
        out_path_rci = OUTPUT_ROOT / f"evaluation_results_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}_rci.csv"
    _save_per_problem(df, problem_rci, out_path_rci)

    # ---------- Save corpus-level results ----------
    if RESULTS_SUBFOLDER == "adapted":
        test_model = f"adapted_{MODEL_NAME.split('-')[2]}"
    else:
        test_model = f"base_{MODEL_NAME.split('-')[2]}"

    if shot_count == "zero":
        test_name = f"{test_model}-{summary_length}-{shot_count}"
    else:
        test_name = f"{test_model}-{source}-{summary_length}-{shot_count}"

    rows = []
    rows += _rows_with_average(test_name, corpus_gen_by_lang)            # generated rows (+ average)
    rows += _rows_with_average(f"{test_name}-rci", corpus_rci_by_lang)   # rci rows (+ average)

    corpus_df = pd.DataFrame(rows)

    corpus_out_file = f"corpus_score_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}.csv"
    corpus_out_path = OUTPUT_ROOT / corpus_out_file
    if subset:
        corpus_out_path = OUTPUT_ROOT / "subset" / f"corpus_score_{MODEL_NAME}_{TASK}_subset.csv"
        corpus_out_path.parent.mkdir(parents=True, exist_ok=True)

    corpus_df.to_csv(corpus_out_path, index=False)
    print(f"Overall corpus scores by language (including RCI) saved to: {corpus_out_path}")

    # ----- Print summaries -----
    print("\n--- Corpus Scores Summary (generated) ---")
    for lang, scores in corpus_gen_by_lang.items():
        main_score = scores.get('corpus_codebleu', 0.0)
        print(f"\nLanguage: {lang.upper()}")
        print(f"  Corpus CodeBLEU: {main_score:.2f}")
        for key, value in scores.items():
            if key != 'corpus_codebleu':
                print(f"  {key.replace('_', ' ').title()}: {value:.2f}")

    print("\n--- Corpus Scores Summary (generated_rci) ---")
    for lang, scores in corpus_rci_by_lang.items():
        main_score = scores.get('corpus_codebleu', 0.0)
        print(f"\nLanguage: {lang.upper()}")
        print(f"  Corpus CodeBLEU: {main_score:.2f}")
        for key, value in scores.items():
            if key != 'corpus_codebleu':
                print(f"  {key.replace('_', ' ').title()}: {value:.2f}")

if __name__ == "__main__":
    main()
