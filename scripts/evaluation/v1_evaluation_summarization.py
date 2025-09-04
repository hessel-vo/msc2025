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
    Calculates problem-level and corpus-level BLEU scores.

    Args:
        df (pd.DataFrame): DataFrame with 'id', 'generated_summary', and 'summary' columns.

    Returns:
        tuple: A tuple containing:
            - problem_scores (dict): A dictionary with problem IDs as keys and scores as values.
            - corpus_scores (dict): A dictionary with the overall corpus BLEU score.
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

        # Store data for final corpus calculation
        all_predictions.append(prediction)
        all_references.append([reference]) # sacrebleu expects a list of references for each prediction

        # Calculate score for the individual problem
        bleu_results = bleu_metric.compute(
            predictions=[prediction],
            references=[[reference]],
        )
        
        # Initialize dictionary for the problem ID if not present
        if p_id not in problem_scores:
            problem_scores[p_id] = {}
        
        problem_scores[p_id]['bleu_score'] = bleu_results['score']
        problem_scores[p_id]['bleu_1'] = bleu_results['precisions'][0]
        problem_scores[p_id]['bleu_2'] = bleu_results['precisions'][1]
        problem_scores[p_id]['bleu_3'] = bleu_results['precisions'][2]
        problem_scores[p_id]['bleu_4'] = bleu_results['precisions'][3]

    # Calculate corpus BLEU score using all predictions and references
    print("Calculating corpus-level BLEU score...")
    corpus_result = bleu_metric.compute(
        predictions=all_predictions,
        references=all_references,
    )

    corpus_scores = {'bleu_score': corpus_result['score']}

    return problem_scores, corpus_scores

def main():
    """Main function to run the evaluation pipeline."""
    num_shots, summary_length = validate_arguments(sys.argv)
    print(f"Starting summarization evaluation for: [Shots: {num_shots}, Summary: {summary_length}]")

    input_filename = f"{MODEL_NAME}_{TASK}_{num_shots}_shot_{summary_length}_results.csv"
    input_filepath = PROJECT_ROOT / "results" / "benchmark" / RESULTS_SUBFOLDER / input_filename

    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded input from: {input_filepath}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        sys.exit(1)

    # --- Metric Calculation ---
    # Load metric once and pass it to the function
    problem_bleu_scores, corpus_bleu_scores = calculate_bleu_scores(df)

    # --- Save Problem-Level Results ---
    # Convert the problem-scores dictionary to a DataFrame
    # Merge based on 'id'
    metrics_df = pd.DataFrame.from_dict(problem_bleu_scores, orient='index')
    metrics_df.index.name = 'id'
    metrics_df.reset_index(inplace=True)
    output_df = pd.merge(df, metrics_df, on='id')
    
    output_filename = f"evaluation_results_{MODEL_NAME}_{TASK}_{num_shots}_shot_{summary_length}.csv"
    output_filepath = PROJECT_ROOT / "results" / "evaluation" / output_filename
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_filepath, index=False)
    print(f"\nDetailed problem-level results saved to: {output_filepath}")

    # --- Save Corpus-Level Results ---
    corpus_df = pd.DataFrame([corpus_bleu_scores])
    corpus_df.insert(0, 'model_name', MODEL_NAME)
    
    corpus_output_filename = f"corpus_score_{MODEL_NAME}_{TASK}_{num_shots}_shot_{summary_length}.csv"
    corpus_output_filepath = PROJECT_ROOT / "results" / "evaluation" / corpus_output_filename
    corpus_df.to_csv(corpus_output_filepath, index=False)
    print(f"Overall corpus score saved to: {corpus_output_filepath}")
    print(f"\nCorpus BLEU Score: {corpus_bleu_scores['bleu_score']:.2f}")


if __name__ == "__main__":
    main()