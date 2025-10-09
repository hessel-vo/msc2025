import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
os.environ['HF_HOME'] = str(HF_CACHE_DIR)

import evaluate

# Constants for the model and results folder
MODEL_NAME = "gemma-3-12b-it"
RESULTS_SUBFOLDER = "baseline"
TASK = "summarization"

def validate_arguments(args):
    if len(args) < 4:
        print("Usage: python run_summarization_evaluation.py <xl|auto> <summary_length> <shot_count> [subset]")
        print("Example: python run_summarization_evaluation.py xl short one")
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

def _round2(x):
    return round(float(x), 2)

def calculate_bleu_scores(predictions, references, ids, bleu_metric):
    """
    SacreBLEU returns scores in 0-100 already. We only round to 2 decimals.
    """
    problem_scores = {}

    # Problem-level
    for p, r, pid in zip(predictions, references, ids):
        bleu_results = bleu_metric.compute(predictions=[p], references=[[r]])
        problem_scores[pid] = {
            "bleu_score": _round2(bleu_results["score"])
        }

    # Corpus-level
    corpus_result = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    corpus_scores = {"bleu_score": _round2(corpus_result["score"])}

    return problem_scores, corpus_scores

def calculate_rouge_scores(predictions, references, ids, rouge_metric):
    """
    ROUGE from evaluate returns 0–1; rescale to 0–100 and round to 2 decimals.
    """
    # Problem-level (no aggregator)
    individual = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_aggregator=False
    )

    problem_scores = {}
    for i, pid in enumerate(ids):
        problem_scores[pid] = {
            "rouge1": _round2(individual["rouge1"][i] * 100.0),
            "rouge2": _round2(individual["rouge2"][i] * 100.0),
            "rougeL": _round2(individual["rougeL"][i] * 100.0),
        }

    # Corpus-level (aggregated)
    corpus = rouge_metric.compute(predictions=predictions, references=references)
    corpus_scores = {
        "corpus_rouge1": _round2(corpus["rouge1"] * 100.0),
        "corpus_rouge2": _round2(corpus["rouge2"] * 100.0),
        "corpus_rougeL": _round2(corpus["rougeL"] * 100.0),
        # corpus["rougeLsum"] omitted by design
    }

    return problem_scores, corpus_scores

def calculate_bertscore(predictions, references, ids, bertscore_metric):
    """
    BERTScore returns 0–1; rescale F1 to 0–100 and round to 2 decimals.
    """
    individual = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
    )

    problem_scores = {}
    for i, pid in enumerate(ids):
        problem_scores[pid] = {
            "bertscore_f1": _round2(individual["f1"][i] * 100.0)
        }

    corpus_scores = {
        "corpus_bertscore_f1": _round2(sum(individual["f1"]) / len(individual["f1"]) * 100.0)
    }

    return problem_scores, corpus_scores

def evaluate_for_column(df, pred_col, bleu_metric, rouge_metric, bertscore_metric):
    predictions = df[pred_col].astype(str).tolist()
    references = df["reference"].astype(str).tolist()
    ids = df["id"].tolist()

    problem_bleu, corpus_bleu = calculate_bleu_scores(predictions, references, ids, bleu_metric)
    problem_rouge, corpus_rouge = calculate_rouge_scores(predictions, references, ids, rouge_metric)
    print("evaluating BERTScore")
    problem_bert, corpus_bert = calculate_bertscore(predictions, references, ids, bertscore_metric)
    print("done evaluating BERTScore")

    # Per-problem dataframes
    bleu_df = pd.DataFrame.from_dict(problem_bleu, orient="index").reset_index().rename(columns={"index": "id"})
    rouge_df = pd.DataFrame.from_dict(problem_rouge, orient="index").reset_index().rename(columns={"index": "id"})
    bert_df = pd.DataFrame.from_dict(problem_bert, orient="index").reset_index().rename(columns={"index": "id"})

    per_problem = df.merge(bleu_df, on="id").merge(rouge_df, on="id").merge(bert_df, on="id")

    # Corpus dict
    corpus_scores = {**corpus_bleu, **corpus_rouge, **corpus_bert}
    return per_problem, corpus_scores

def main():
    source, summary_length, shot_count, subset = validate_arguments(sys.argv)
    print(f"Starting summarization evaluation for: [Source: {source}, Summary: {summary_length}, Shots: {shot_count}]")

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

    # Basic hygiene
    if "generated" not in df.columns:
        print("Error: 'generated' column is missing from the input CSV.")
        sys.exit(1)
    if "generated_rci" not in df.columns:
        print("Error: 'generated_rci' column is missing from the input CSV.")
        sys.exit(1)
    if "reference" not in df.columns:
        print("Error: 'reference' column is missing from the input CSV.")
        sys.exit(1)

    df["generated"] = df["generated"].astype(str).str.strip('"')
    df["generated_rci"] = df["generated_rci"].astype(str).str.strip('"')
    df["reference"] = df["reference"].astype(str).str.strip('"')

    # Load metrics
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge", seed=42)
    bertscore_metric = evaluate.load("bertscore")


    # Evaluate for original generated
    print("\n=== Evaluating: generated ===")
    per_problem_generated, corpus_generated = evaluate_for_column(df, "generated", bleu_metric, rouge_metric, bertscore_metric)

    # Evaluate for generated_rci
    print("\n=== Evaluating: generated_rci ===")
    per_problem_rci, corpus_rci = evaluate_for_column(df, "generated_rci", bleu_metric, rouge_metric, bertscore_metric)

    # ---------- Save per-problem results ----------
    OUTPUT_ROOT = PROJECT_ROOT / "results" / "evaluation" / RESULTS_SUBFOLDER
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"evaluation_results_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}.csv"
    output_filepath = OUTPUT_ROOT / output_filename
    if subset:
        output_filepath = OUTPUT_ROOT / "subset" / f"evaluation_results_{MODEL_NAME}_{TASK}_subset.csv"
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

    per_problem_generated.to_csv(output_filepath, index=False)
    print(f"\nDetailed problem-level results (generated) saved to: {output_filepath}")

    # RCI per-problem file with _rci suffix
    if subset:
        output_filepath_rci = OUTPUT_ROOT / "subset" / f"evaluation_results_{MODEL_NAME}_{TASK}_subset_rci.csv"
    else:
        output_filepath_rci = OUTPUT_ROOT / f"evaluation_results_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}_rci.csv"

    per_problem_rci.to_csv(output_filepath_rci, index=False)
    print(f"Detailed problem-level results (generated_rci) saved to: {output_filepath_rci}")

    # ---------- Save corpus-level results (two rows) ----------
    if RESULTS_SUBFOLDER == "adapted":
        test_model = f"adapted_{MODEL_NAME.split('-')[2]}"
    else:
        test_model = f"base_{MODEL_NAME.split('-')[2]}"
    test_name = f"{test_model}-{source}-{summary_length}-{shot_count}"
    row_generated = {"test_name": test_name, **corpus_generated}
    row_rci = {"test_name": f"{test_name}-rci", **corpus_rci}
    corpus_df = pd.DataFrame([row_generated, row_rci])

    corpus_output_filename = f"corpus_score_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}.csv"
    corpus_output_filepath = OUTPUT_ROOT / corpus_output_filename
    if subset:
        corpus_output_filepath = OUTPUT_ROOT / "subset" / f"corpus_score_{MODEL_NAME}_{TASK}_subset.csv"
        corpus_output_filepath.parent.mkdir(parents=True, exist_ok=True)

    corpus_df.to_csv(corpus_output_filepath, index=False)
    print(f"\nOverall corpus scores saved to: {corpus_output_filepath}")

    print("\n--- Corpus Scores Summary ---")
    print(f"[generated] {row_generated}")
    print(f"[generated_rci] {row_rci}")

if __name__ == "__main__":
    main()
