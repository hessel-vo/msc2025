# scripts/evaluation/run_summarization_evaluation.py

import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
if not project_root_str:
    raise ValueError("PROJECT_ROOT not found in .env file. Please set it.")
PROJECT_ROOT = Path(project_root_str)
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
os.environ['HF_HOME'] = str(HF_CACHE_DIR)

# Import evaluate after setting the cache directory
import evaluate

# Constants for the model and results folder (adjust as needed)
MODEL_NAME = "gemma"
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

def compute_summarization_metrics(prediction, reference, metric_objects):
    """
    Compute summarization metrics for a single prediction-reference pair.
    """
    results = {}
    default_scores = {'bleu_score': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}

    if not isinstance(prediction, str) or not isinstance(reference, str):
        return default_scores

    bleu_metric = metric_objects.get("bleu")
    # This check is now secondary, as the main() function will ensure the metric exists.
    if bleu_metric:
        bleu_results = bleu_metric.compute(
            predictions=[prediction],
            references=[[reference]],
            max_order=4
        )
        results['bleu_score'] = bleu_results['bleu']
        results['bleu_1'] = bleu_results['precisions'][0]
        results['bleu_2'] = bleu_results['precisions'][1]
        results['bleu_3'] = bleu_results['precisions'][2]
        results['bleu_4'] = bleu_results['precisions'][3]
    else:
        print("Warning: BLEU metric object not found. Assigning default scores of 0.0.")
        return default_scores

    return results

def main():
    """Main function to run the evaluation pipeline."""
    num_shots, summary_length = validate_arguments(sys.argv)
    print(f"Starting summarization evaluation for: [Shots: {num_shots}, Summary: {summary_length}]")

    input_filename = f"{MODEL_NAME}_{TASK}_{num_shots}_shot_{summary_length}_results.csv"
    input_filepath = PROJECT_ROOT / "results" / "benchmark" / RESULTS_SUBFOLDER / input_filename
    
    prediction_col = "generated_summary"
    reference_col = "summary"

    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded results from: {input_filepath}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        sys.exit(1)

    # --- Robust Metric Loading ---
    metric_objects = {}
    print("Attempting to load BLEU metric from Hugging Face Hub...")
    try:
        bleu_metric = evaluate.load("bleu")
        if bleu_metric is None:
            # This handles cases where load returns None without an exception
            raise ConnectionError("evaluate.load() returned None.")
        metric_objects["bleu"] = bleu_metric
        print("BLEU metric loaded successfully.")
        print(f"Metric details: {bleu_metric}")
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to load the BLEU metric.")
        print("This is likely due to a network issue (no internet, firewall, or proxy).")
        print(f"Please check your internet connection and try again. Details: {e}")
        sys.exit(1)
    # --- End Robust Metric Loading ---
    
    all_row_metrics = []
    print("Calculating scores for each row...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        prediction = row[prediction_col]
        reference = row[reference_col]
        row_metrics = compute_summarization_metrics(prediction, reference, metric_objects)
        all_row_metrics.append(row_metrics)

    if not all_row_metrics:
        print("No metrics were computed. Exiting.")
        return

    metrics_df = pd.DataFrame(all_row_metrics)
    output_df = pd.concat([df, metrics_df], axis=1)
    output_filename = f"evaluation_results_{MODEL_NAME}_{TASK}_{num_shots}_shot_{summary_length}.csv"
    output_filepath = PROJECT_ROOT / "results" / "evaluation" / output_filename
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_filepath, index=False)
    print(f"\nEvaluation complete. Results saved to: {output_filepath}")

if __name__ == "__main__":
    main()