# scripts/evaluation/run_evaluation.py

import sys
import os
import pandas as pd
import evaluate
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

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

def compute_metrics(predictions, references, task):
    """
    Compute evaluation metrics based on the task.
    For this iteration, only BLEU is implemented.
    """
    results = {}
    print(f"Computing metrics for task: {task}...")

    if task == "summarization":
        # Load the BLEU metric
        bleu_metric = evaluate.load("bleu")

        if bleu_metric:
            print("WE GOT BLEU!!")

        if bleu_metric is None:
            print("Weird stuff")
        
        # Calculate BLEU score.
        # We specify max_order=4 to get precisions for 1, 2, 3, and 4-grams.
        bleu_results = bleu_metric.compute(
            predictions=predictions, 
            references=[[ref] for ref in references], # BLEU expects a list of lists for references
            max_order=4
        )

        if bleu_metric:
            print("WE GOT BLEU!!")

        print(bleu_results)
        
        # Store individual n-gram precisions for a more granular view
        results['bleu_1'] = bleu_results['precisions'][0]
        results['bleu_2'] = bleu_results['precisions'][1]
        results['bleu_3'] = bleu_results['precisions'][2]
        results['bleu_4'] = bleu_results['precisions'][3]
        results['bleu_score'] = bleu_results['bleu'] # The composite score

    elif task == "generation":
        # Placeholder for code generation metrics like CodeBLEU
        print("Metric for 'generation' task is not implemented in this iteration.")
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
    input_filepath = os.path.join(PROJECT_ROOT, "results", "benchmark", RESULTS_SUBFOLDER, input_filename)
    
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

    predictions = df[prediction_col].tolist()
    references = df[reference_col].tolist()

    if not predictions or not references or len(predictions) != len(references):
        print("Error: Predictions or references are empty or have mismatched lengths.")
        sys.exit(1)

    # 4. Compute metrics
    metrics = compute_metrics(predictions, references, task)
    
    if not metrics:
        print("No metrics were computed. Exiting.")
        return

    # 5. Aggregate and save results
    results_df = pd.DataFrame([metrics])
    # Add context columns for traceability
    results_df.insert(0, "model", MODEL_NAME)
    results_df.insert(1, "task", task)
    results_df.insert(2, "num_shots", num_shots)
    results_df.insert(3, "summary_length", summary_length)

    output_filename = f"evaluation_results_{MODEL_NAME}_{task}_{num_shots}_{summary_length}.csv"
    output_filepath = os.path.join(PROJECT_ROOT, "results", "evaluation", output_filename)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    results_df.to_csv(output_filepath, index=False)
    print(f"Evaluation complete. Results saved to: {output_filepath}")


if __name__ == "__main__":
    main()