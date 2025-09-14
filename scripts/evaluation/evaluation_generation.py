import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
if not project_root_str:
    raise ValueError("PROJECT_ROOT not found in .env file. Please set it.")
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

def calculate_codebleu_scores(df):
    problem_scores = {}
    weights = (0.25, 0.25, 0.25, 0.25)

    print("Calculating problem-level CodeBLEU scores for each row...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Problem-Level Scores"):
        prediction = str(row["generated"])
        reference = str(row["reference"])
        lang = row["language"]
        p_id = row["id"]

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

    # Corpus-Level Calculation by Language
    corpus_scores_by_lang = {}
    languages = df['language'].unique()
    print(f"\nFound languages for corpus evaluation: {list(languages)}")

    for lang in languages:
        print(f"Calculating corpus-level CodeBLEU score for '{lang}'...")
        lang_df = df[df['language'] == lang]
        
        all_predictions = lang_df["generated"].astype(str).tolist()
        all_references = [[str(ref)] for ref in lang_df["reference"].tolist()]

        corpus_result = calc_codebleu(
            references=all_references, 
            predictions=all_predictions, 
            lang=lang, 
            weights=weights
        )
        
        corpus_scores_by_lang[lang] = {f"corpus_{key}": value for key, value in corpus_result.items()}

    return problem_scores, corpus_scores_by_lang

def main():
    source, summary_length, shot_count, subset = validate_arguments(sys.argv)
    print(f"Starting code generation evaluation for: [Source: {source}, Summary: {summary_length}, Shots: {shot_count}]")

    input_filename = f"{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}_results.csv"
    input_filepath = PROJECT_ROOT / "results" / "benchmark" / RESULTS_SUBFOLDER / input_filename
    
    if subset:
        input_filepath = PROJECT_ROOT / "results" / "benchmark" / RESULTS_SUBFOLDER / f"{MODEL_NAME}_{TASK}_subset_results.csv"

    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded input from: {input_filepath}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        sys.exit(1)

    df['generated'] = df['generated'].str.strip('"')
    df['reference'] = df['reference'].str.strip('"')

    problem_codebleu_scores, corpus_codebleu_scores_by_lang = calculate_codebleu_scores(df)

    # Save per-problem results
    metrics_df = pd.DataFrame.from_dict(problem_codebleu_scores, orient='index')
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={'index': 'id'}, inplace=True)
    output_df = pd.merge(df, metrics_df, on='id')
    
    OUTPUT_ROOT = PROJECT_ROOT / "results" / "evaluation" / RESULTS_SUBFOLDER
    output_filename = f"evaluation_results_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}.csv"
    output_filepath = OUTPUT_ROOT / output_filename
    if subset:
        output_filepath = OUTPUT_ROOT / "subset" / f"evaluation_results_{MODEL_NAME}_{TASK}_subset.csv"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_filepath, index=False)
    print(f"\nDetailed problem-level results saved to: {output_filepath}")

    # Save corpus-level results
    corpus_rows = []
    for lang, scores in corpus_codebleu_scores_by_lang.items():
        row_data = {'model_name': MODEL_NAME, 'language': lang}
        row_data.update(scores)
        corpus_rows.append(row_data)

    corpus_df = pd.DataFrame(corpus_rows)
    
    corpus_output_filename = f"corpus_score_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}.csv"
    corpus_output_filepath = OUTPUT_ROOT / corpus_output_filename
    if subset:
        corpus_output_filepath = OUTPUT_ROOT / "subset" / f"corpus_score_{MODEL_NAME}_{TASK}_subset.csv"
    corpus_df.to_csv(corpus_output_filepath, index=False)
    print(f"Overall corpus scores by language saved to: {corpus_output_filepath}")
    
    print("\n--- Corpus Scores Summary ---")
    for lang, scores in corpus_codebleu_scores_by_lang.items():
        main_score = scores.get('corpus_codebleu', 0.0)
        print(f"\nLanguage: {lang.upper()}")
        print(f"  Corpus Codebleu: {main_score:.4f}")
        for key, value in scores.items():
            if key != 'corpus_codebleu':
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")

if __name__ == "__main__":
    main()