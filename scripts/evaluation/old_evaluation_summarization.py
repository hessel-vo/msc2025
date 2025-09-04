import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
os.environ['HF_HOME'] = str(HF_CACHE_DIR)

import evaluate

# Constants for the model and results folder
MODEL_NAME = "gemma-3-4b-it"
RESULTS_SUBFOLDER = "baseline"
TASK = "summarization"

def validate_arguments(args):
    """Validate the command-line arguments for num_shots and summary_length."""
    if len(args) != 3:
        print("Usage: python run_summarization_evaluation.py <num_shots> <summary_length>")
        sys.exit(1)

    num_shots, summary_length = args[1], args[2]

    if num_shots not in ["one", "three"]:
        print(f"Error: Invalid num_shots '{num_shots}'. Must be 'one' or 'three'.")
        sys.exit(1)
    if summary_length not in ["short", "long"]:
        print(f"Error: Invalid summary_length '{summary_length}'. Must be 'short' or 'long'.")
        sys.exit(1)

    return num_shots, summary_length

def bleu_score(df):
    results = {}

    bleu_metric = evaluate.load("sacrebleu")

    for _, row in df.iterrows():
        prediction = row["generated_summary"]
        reference = row["summary"]
        p_id = row["id"]

        bleu_results = bleu_metric.compute(
            predictions=[prediction],
            references=[[reference]],
            use_effective_order=True
        )
        results[p_id]['bleu_score'] = bleu_results['score']
        results[p_id]['bleu_1'] = bleu_results['precisions'][0]
        results[p_id]['bleu_2'] = bleu_results['precisions'][1]
        results[p_id]['bleu_3'] = bleu_results['precisions'][2]
        results[p_id]['bleu_4'] = bleu_results['precisions'][3]
    
    # Retrieve list of all predictions and references and compute corpus BLEU
    predictions=[]
    references=[]
    corpus_result = bleu_metric.compute(
        predictions=predictions,
        references=[references]
    )

    results['corpus_bleu'] = corpus_result['score']

    return results

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

    
    all_results = {}

    all_results['bleu'] = bleu_score(df)

    # Fix to align with results dictionary
    metrics_df = pd.DataFrame(all_row_metrics)
    output_df = pd.concat([df, metrics_df], axis=1)
    output_filename = f"evaluation_results_{MODEL_NAME}_{TASK}_{num_shots}_shot_{summary_length}.csv"
    output_filepath = PROJECT_ROOT / "results" / "evaluation" / output_filename
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_filepath, index=False)
    print(f"\nEvaluation complete. Results saved to: {output_filepath}")

if __name__ == "__main__":
    main()