import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
if not project_root_str:
    raise ValueError("PROJECT_ROOT not found in .env file. Please set it.")
PROJECT_ROOT = Path(project_root_str)
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
os.environ['HF_HOME'] = str(HF_CACHE_DIR)

import evaluate

# Constants for the model and results folder
MODEL_NAME = "gemma-3-4b-it"
RESULTS_SUBFOLDER = "baseline"
TASK = "summarization"
# --- End Configuration ---

def validate_arguments(args):
    """Validate the command-line arguments for num_shots and summary_length."""
    if len(args) != 3:
        print("Usage: python run_summarization_evaluation.py <num_shots> <summary_length>")
        print("Example: python run_summarization_evaluation.py one short")
        sys.exit(1)

    num_shots, summary_length = args[1], args[2]

    if num_shots not in ["one", "three"]:
        print(f"Error: Invalid num_shots '{num_shots}'. Must be 'one' or 'three'.")
        sys.exit(1)
    if summary_length not in ["short", "long"]:
        print(f"Error: Invalid summary_length '{summary_length}'. Must be 'short' or 'long'.")
        sys.exit(1)

    return num_shots, summary_length

def calculate_bleu_scores(df):
    """
    Calculates problem-level and corpus-level BLEU scores by iterating.
    """
    bleu_metric = evaluate.load("sacrebleu")
    problem_scores = {}
    all_predictions = []
    all_references = []

    print("Calculating problem-level BLEU scores...")
    for _, row in df.iterrows():
        prediction = str(row["generated_summary"])
        reference = str(row["summary"])
        p_id = row["id"]

        all_predictions.append(prediction)
        all_references.append([reference])

        bleu_results = bleu_metric.compute(predictions=[prediction], references=[[reference]])
        
        problem_scores[p_id] = {
            'bleu_score': bleu_results['score'],
            'bleu_1': bleu_results['precisions'][0],
            'bleu_2': bleu_results['precisions'][1],
            'bleu_3': bleu_results['precisions'][2],
            'bleu_4': bleu_results['precisions'][3]
        }

    print("Calculating corpus-level BLEU score...")
    corpus_result = bleu_metric.compute(predictions=all_predictions, references=all_references)
    corpus_scores = {'bleu_score': corpus_result['score']}

    return problem_scores, corpus_scores

def calculate_rouge_scores(df):
    """
    Calculates problem-level and corpus-level ROUGE scores in a single batch.
    """
    rouge_metric = evaluate.load("rouge")
    
    all_predictions = df["generated_summary"].astype(str).tolist()
    all_references = df["summary"].astype(str).tolist()
    
    print("Calculating batch problem-level ROUGE scores...")
    # use_aggregator=False returns a list of scores for each input pair
    individual_results = rouge_metric.compute(
        predictions=all_predictions,
        references=all_references,
        use_aggregator=False
    )
    
    problem_scores = {}
    for i, p_id in enumerate(df["id"]):
        problem_scores[p_id] = {
            'rouge1': individual_results['rouge1'][i],
            'rouge2': individual_results['rouge2'][i],
            'rougeL': individual_results['rougeL'][i],
            'rougeLsum': individual_results['rougeLsum'][i],
        }

    print("Calculating corpus-level ROUGE scores...")
    # The default behavior (use_aggregator=True) provides the corpus-level score
    corpus_result = rouge_metric.compute(
        predictions=all_predictions,
        references=all_references
    )
    # Rename keys for clarity in the final corpus file
    corpus_scores = {f"corpus_{key}": value for key, value in corpus_result.items()}

    return problem_scores, corpus_scores

def main():
    """Main function to run the evaluation pipeline."""
    num_shots, summary_length = validate_arguments(sys.argv)
    print(f"Starting summarization evaluation for: [Shots: {num_shots}, Summary: {summary_length}]")

    input_filename = f"{MODEL_NAME}_{TASK}_{num_shots}_shot_{summary_length}_results.csv"
    input_filepath = PROJECT_ROOT / "results" / "benchmark" / RESULTS_SUBFOLDER / input_filename

    try:
        df = pd.read_csv(input_filepath)
        print(df)
        print(f"Successfully loaded input from: {input_filepath}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        sys.exit(1)

    # --- Metric Calculation ---
    problem_bleu_scores, corpus_bleu_scores = calculate_bleu_scores(df)
    problem_rouge_scores, corpus_rouge_scores = calculate_rouge_scores(df)

    # --- Save Problem-Level Results ---
    # Convert dictionaries to DataFrames
    bleu_metrics_df = pd.DataFrame.from_dict(problem_bleu_scores, orient='index')
    print(bleu_metrics_df)
    bleu_metrics_df.reset_index(inplace=True)
    print(bleu_metrics_df)
    bleu_metrics_df.rename(columns={'index': 'id'}, inplace=True)
    print(bleu_metrics_df)

    rouge_metrics_df = pd.DataFrame.from_dict(problem_rouge_scores, orient='index')
    rouge_metrics_df.reset_index(inplace=True)
    rouge_metrics_df.rename(columns={'index': 'id'}, inplace=True)

    # Merge all metrics with the original DataFrame
    output_df = pd.merge(df, bleu_metrics_df, on='id')
    output_df = pd.merge(output_df, rouge_metrics_df, on='id')
    
    output_filename = f"evaluation_results_{MODEL_NAME}_{TASK}_{num_shots}_shot_{summary_length}.csv"
    output_filepath = PROJECT_ROOT / "results" / "evaluation" / output_filename
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_filepath, index=False)
    print(f"\nDetailed problem-level results saved to: {output_filepath}")

    # --- Save Corpus-Level Results ---
    # Combine all corpus scores into a single dictionary
    all_corpus_scores = {**corpus_bleu_scores, **corpus_rouge_scores}
    corpus_df = pd.DataFrame([all_corpus_scores])
    corpus_df.insert(0, 'model_name', MODEL_NAME)
    
    corpus_output_filename = f"corpus_score_{MODEL_NAME}_{TASK}_{num_shots}_shot_{summary_length}.csv"
    corpus_output_filepath = PROJECT_ROOT / "results" / "evaluation" / corpus_output_filename
    corpus_df.to_csv(corpus_output_filepath, index=False)
    print(f"Overall corpus score saved to: {corpus_output_filepath}")
    print("\n--- Corpus Scores Summary ---")
    for key, value in all_corpus_scores.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")


if __name__ == "__main__":
    main()