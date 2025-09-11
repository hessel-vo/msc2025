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

# Import the specific function from the codebleu PyPI package
from codebleu import calc_codebleu

# Constants for the model and results folder
MODEL_NAME = "gemma-3-1b-it"
RESULTS_SUBFOLDER = "baseline"
TASK = "generation"
# --- End Configuration ---

def validate_arguments(args):
    """Validate the command-line arguments for source, summary_length, and shot_count."""
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

def calculate_codebleu_scores(df):
    """
    Calculates problem-level and corpus-level CodeBLEU scores using the codebleu PyPI package.
    """
    problem_scores = {}
    all_predictions = []
    all_references = []
    
    # Define parameters for calc_codebleu
    lang = "python"
    weights = (0.25, 0.25, 0.25, 0.25)

    print("Calculating problem-level CodeBLEU scores...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="CodeBLEU Scores"):
        prediction = str(row["generated"])
        reference = str(row["reference"])
        p_id = row["id"]

        all_predictions.append(prediction)
        # calc_codebleu expects references as a list of lists of strings
        all_references.append([reference])

        # Calculate score for the individual problem
        # Note the argument order: references, then predictions
        codebleu_results = calc_codebleu(
            references=[[reference]], 
            predictions=[prediction], 
            lang=lang, 
            weights=weights
        )
        
        problem_scores[p_id] = {
            'codebleu': codebleu_results['codebleu'],
            'ngram_match_score': codebleu_results['ngram_match_score'],
            'weighted_ngram_match_score': codebleu_results['weighted_ngram_match_score'],
            'syntax_match_score': codebleu_results['syntax_match_score'],
            'dataflow_match_score': codebleu_results['dataflow_match_score'],
        }

    print("Calculating corpus-level CodeBLEU score...")
    corpus_result = calc_codebleu(
        references=all_references, 
        predictions=all_predictions, 
        lang=lang, 
        weights=weights
    )
    
    # Prefix corpus scores for clarity in the final CSV
    corpus_scores = {f"corpus_{key}": value for key, value in corpus_result.items()}

    return problem_scores, corpus_scores

def main():
    """Main function to run the evaluation pipeline."""
    source, summary_length, shot_count, subset = validate_arguments(sys.argv)
    print(f"Starting code generation evaluation for: [Source: {source}, Summary: {summary_length}, Shots: {shot_count}]")

    input_filename = f"{MODEL_NAME}_{TASK}_{source}_{shot_count}_shot_{summary_length}_results.csv"
    input_filepath = PROJECT_ROOT / "results" / "benchmark" / RESULTS_SUBFOLDER / input_filename
    
    if subset:
        input_filepath = PROJECT_ROOT / "results" / "benchmark" / RESULTS_SUBFOLDER / "subset" / f"{MODEL_NAME}_{TASK}_results.csv"

    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded input from: {input_filepath}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        sys.exit(1)

    df['generated'] = df['generated'].str.strip('"')
    df['reference'] = df['reference'].str.strip('"')

    # --- Metric Calculation ---
    problem_codebleu_scores, corpus_codebleu_scores = calculate_codebleu_scores(df)

    # --- Save Problem-Level Results ---
    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame.from_dict(problem_codebleu_scores, orient='index')
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={'index': 'id'}, inplace=True)

    # Merge metrics with the original DataFrame
    output_df = pd.merge(df, metrics_df, on='id')
    
    OUTPUT_ROOT = PROJECT_ROOT / "results" / "evaluation" / RESULTS_SUBFOLDER
    output_filename = f"evaluation_results_{MODEL_NAME}_{TASK}_{source}_{shot_count}_{summary_length}_shot.csv"
    output_filepath = OUTPUT_ROOT / output_filename
    if subset:
        output_filepath = OUTPUT_ROOT / "subset" / f"evaluation_results_{MODEL_NAME}_{TASK}_results.csv"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_filepath, index=False)
    print(f"\nDetailed problem-level results saved to: {output_filepath}")

    # --- Save Corpus-Level Results ---
    corpus_df = pd.DataFrame([corpus_codebleu_scores])
    corpus_df.insert(0, 'model_name', MODEL_NAME)
    
    corpus_output_filename = f"corpus_score_{MODEL_NAME}_{TASK}_{source}_{shot_count}_{summary_length}_shot.csv"
    corpus_output_filepath = OUTPUT_ROOT / corpus_output_filename
    if subset:
        corpus_output_filepath = OUTPUT_ROOT / "subset" / f"corpus_score_{MODEL_NAME}_{TASK}_results.csv"
    corpus_df.to_csv(corpus_output_filepath, index=False)
    print(f"Overall corpus score saved to: {corpus_output_filepath}")
    print("\n--- Corpus Scores Summary ---")
    for key, value in corpus_codebleu_scores.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")


if __name__ == "__main__":
    main()