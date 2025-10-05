import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

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
MODEL_NAME = "gemma-3-12b-it"
RESULTS_SUBFOLDER = "baseline"
TASK = "summarization"

def validate_arguments(args):
    if len(args) < 4:
        print("Usage: python run_summarization_evaluation.py <xl|auto> <summary_length> <shot_count>")
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

def calculate_bleu_scores(df):
    bleu_metric = evaluate.load("sacrebleu")
    problem_scores = {}
    all_predictions = []
    all_references = []

    print("Calculating problem-level BLEU scores...")
    for _, row in df.iterrows():
        prediction = str(row["generated"])
        reference = str(row["reference"])
        p_id = row["id"]

        all_predictions.append(prediction)
        all_references.append([reference])

        bleu_results = bleu_metric.compute(predictions=[prediction], references=[[reference]])
        
        problem_scores[p_id] = {
            'bleu_score': bleu_results['score'],
            # 'bleu_1': bleu_results['precisions'][0],
            # 'bleu_2': bleu_results['precisions'][1],
            # 'bleu_3': bleu_results['precisions'][2],
            # 'bleu_4': bleu_results['precisions'][3]
        }

    print("Calculating corpus-level BLEU score...")
    corpus_result = bleu_metric.compute(predictions=all_predictions, references=all_references)
    corpus_scores = {'bleu_score': corpus_result['score']}

    return problem_scores, corpus_scores

def calculate_rouge_scores(df):
    # Seeded due to bootstrapping applied when calculating corpus-level scores
    rouge_metric = evaluate.load("rouge", seed=42)
    
    all_predictions = df["generated"].astype(str).tolist()
    all_references = df["reference"].astype(str).tolist()
    
    print("Calculating problem-level ROUGE scores...")
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
            # 'rougeLsum': individual_results['rougeLsum'][i],
        }

    print("Calculating corpus-level ROUGE scores...")
    corpus_result = rouge_metric.compute(
        predictions=all_predictions,
        references=all_references
    )
    
    corpus_scores = {f"corpus_{key}": value for key, value in corpus_result.items() if key != 'rougeLsum'}

    return problem_scores, corpus_scores

def calculate_bertscore(df):

    bertscore_metric = evaluate.load("bertscore")
    
    all_predictions = df["generated"].astype(str).tolist()
    all_references = df["reference"].astype(str).tolist()

    print("Calculating batch problem-level BERTScore...")
    individual_results = bertscore_metric.compute(
        predictions=all_predictions,
        references=all_references,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
    )

    problem_scores = {}
    for i, p_id in enumerate(df["id"]):
        problem_scores[p_id] = {
            # 'bertscore_precision': individual_results['precision'][i],
            # 'bertscore_recall': individual_results['recall'][i],
            'bertscore_f1': individual_results['f1'][i]
        }

    print("Calculating corpus-level BERTScore...")
    # Average score for overall corpus score
    corpus_scores = {
        # "corpus_bertscore_precision": sum(individual_results['precision']) / len(individual_results['precision']),
        # "corpus_bertscore_recall": sum(individual_results['recall']) / len(individual_results['recall']),
        "corpus_bertscore_f1": sum(individual_results['f1']) / len(individual_results['f1']),
    }

    return problem_scores, corpus_scores

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

    df['generated'] = df['generated'].str.strip('"')
    df['reference'] = df['reference'].str.strip('"')

    problem_bleu_scores, corpus_bleu_scores = calculate_bleu_scores(df)
    problem_rouge_scores, corpus_rouge_scores = calculate_rouge_scores(df)
    problem_bert_scores, corpus_bert_scores = calculate_bertscore(df)

    # Save per-problem results
    bleu_metrics_df = pd.DataFrame.from_dict(problem_bleu_scores, orient='index')
    bleu_metrics_df.reset_index(inplace=True)
    bleu_metrics_df.rename(columns={'index': 'id'}, inplace=True)

    rouge_metrics_df = pd.DataFrame.from_dict(problem_rouge_scores, orient='index')
    rouge_metrics_df.reset_index(inplace=True)
    rouge_metrics_df.rename(columns={'index': 'id'}, inplace=True)

    bert_metrics_df = pd.DataFrame.from_dict(problem_bert_scores, orient='index')
    bert_metrics_df.reset_index(inplace=True)
    bert_metrics_df.rename(columns={'index': 'id'}, inplace=True)

    output_df = pd.merge(df, bleu_metrics_df, on='id')
    output_df = pd.merge(output_df, rouge_metrics_df, on='id')
    output_df = pd.merge(output_df, bert_metrics_df, on='id')
    

    OUTPUT_ROOT = PROJECT_ROOT / "results" / "evaluation" / RESULTS_SUBFOLDER
    output_filename = f"evaluation_results_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}.csv"
    output_filepath = OUTPUT_ROOT / output_filename
    if subset:
        output_filepath = OUTPUT_ROOT / "subset" / f"evaluation_results_{MODEL_NAME}_{TASK}_subset.csv"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_filepath, index=False)
    print(f"\nDetailed problem-level results saved to: {output_filepath}")

    # Save corpus-level results
    all_corpus_scores = {**corpus_bleu_scores, **corpus_rouge_scores, **corpus_bert_scores}
    corpus_df = pd.DataFrame([all_corpus_scores])
    corpus_df.insert(0, 'test_name', f"{MODEL_NAME}-{source}-{summary_length}-{shot_count}")
    
    corpus_output_filename = f"corpus_score_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}.csv"
    corpus_output_filepath = OUTPUT_ROOT/ corpus_output_filename
    if subset:
        corpus_output_filepath = OUTPUT_ROOT / "subset" / f"corpus_score_{MODEL_NAME}_{TASK}_subset.csv"
    corpus_df.to_csv(corpus_output_filepath, index=False)
    print(f"Overall corpus score saved to: {corpus_output_filepath}")
    print("\n--- Corpus Scores Summary ---")
    for key, value in all_corpus_scores.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")


if __name__ == "__main__":
    main()