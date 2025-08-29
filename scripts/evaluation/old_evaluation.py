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

import evaluate


# Constants for the model and results folder (adjust as needed)
MODEL_NAME = "gemma"
RESULTS_SUBFOLDER = "baseline"
# --- End Configuration ---

def validate_arguments(args):
    """Validate the command-line arguments."""
    if len(args) != 4:
        print("Usage: python run_evaluation.py <task> <num_shots> <summary_length>")
        print("Example: python run_evaluation.py summarization one short")
        sys.exit(1)

    task, num_shots, summary_length = args[1], args[2], args[3]

    if task not in ["summarization", "generation"]:
        print(f"Error: Invalid task '{task}'. Must be 'summarization' or 'generation'.")
        sys.exit(1)
    if num_shots not in ["one", "three"]:
        print(f"Error: Invalid num_shots '{num_shots}'. Must be 'one' or 'three'.")
        sys.exit(1)
    if summary_length not in ["short", "long"]:
        print(f"Error: Invalid summary_length '{summary_length}'. Must be 'short' or 'long'.")
        sys.exit(1)

    return task, num_shots, summary_length

def compute_metrics_for_row(prediction, reference, task, metric_objects):
    """
    Compute evaluation metrics for a single prediction-reference pair.
    """
    results = {}
    
    # Ensure inputs are strings, otherwise metric will fail
    if not isinstance(prediction, str) or not isinstance(reference, str):
        if task == "summarization":
            return {'bleu_score': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
        else:
            return {}

    if task == "summarization":
        bleu_metric = metric_objects.get("bleu")
        if bleu_metric:
            bleu_results = bleu_metric.compute(
                predictions=[prediction],
                references=[[reference]], # Expects a list of lists
                max_order=4
            )
            # Store individual n-gram precisions
            results['bleu_score'] = bleu_results['bleu']
            results['bleu_1'] = bleu_results['precisions'][0]
            results['bleu_2'] = bleu_results['precisions'][1]
            results['bleu_3'] = bleu_results['precisions'][2]
            results['bleu_4'] = bleu_results['precisions'][3]

    elif task == "generation":
        # Placeholder for code generation metrics like CodeBLEU
        # In future iterations, we will add CodeBLEU here.
        pass

    return results

def main():
    """Main function to run the evaluation pipeline."""
    # 1. Parse and validate command-line arguments
    task, num_shots, summary_length = validate_arguments(sys.argv)
    print(f"Starting evaluation for: [Task: {task}, Shots: {num_shots}, Summary: {summary_length}]")

    # 2. Dynamically determine input path and column names
    input_filename = f"{MODEL_NAME}_{task}_{num_shots}_shot_{summary_length}_results.csv"
    input_filepath = PROJECT_ROOT / "results" / "benchmark" / RESULTS_SUBFOLDER / input_filename
    
    if task == "summarization":
        prediction_col = "generated_summary"
        reference_col = "summary"
    else: # task == "generation"
        prediction_col = "generated_code"
        reference_col = "code"

    # 3. Load and prepare the data
    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded results from: {input_filepath}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        sys.exit(1)

    # 4. Initialize metrics and compute scores row-by-row
    metric_objects = {}
    if task == "summarization":
        print("Loading BLEU metric...")
        metric_objects["bleu"] = evaluate.load("bleu")
    
    # A list to store the dictionary of metrics for each row
    all_row_metrics = []
    
    print("Calculating scores for each row...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        prediction = row[prediction_col]
        reference = row[reference_col]
        
        row_metrics = compute_metrics_for_row(prediction, reference, task, metric_objects)
        all_row_metrics.append(row_metrics)

    print(all_row_metrics)
    if not all_row_metrics:
        print("No metrics were computed. Exiting.")
        return

    # 5. Aggregate results and save
    # Convert the list of metric dictionaries into a new DataFrame
    metrics_df = pd.DataFrame(all_row_metrics)
    
    # Concatenate the original DataFrame with the new metrics DataFrame
    output_df = pd.concat([df, metrics_df], axis=1)

    output_filename = f"evaluation_results_{MODEL_NAME}_{task}_{num_shots}_shot_{summary_length}.csv"
    output_filepath = PROJECT_ROOT / "results" / "evaluation" / output_filename
    
    # Create the output directory if it doesn't exist
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    print(output_df)
    output_df.to_csv(output_filepath, index=False)
    print(f"\nEvaluation complete. Results saved to: {output_filepath}")


if __name__ == "__main__":
    main()