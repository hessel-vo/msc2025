import pandas as pd
from pathlib import Path
import evaluate # The main Hugging Face evaluation library
import warnings
import nltk # Import the NLTK library

# --- HELPER FUNCTION TO MANAGE NLTK DATA (CORRECTED) ---
def download_nltk_data():
    """
    Checks if necessary NLTK data ('punkt', 'wordnet', 'omw-1.4') is available,
    and downloads it if not. This is required for the METEOR metric.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt', quiet=True)
        print("'punkt' downloaded successfully.")
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("NLTK 'wordnet' not found. Downloading...")
        nltk.download('wordnet', quiet=True)
        print("'wordnet' downloaded successfully.")
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("NLTK 'omw-1.4' not found. Downloading...")
        nltk.download('omw-1.4', quiet=True)
        print("'omw-1.4' downloaded successfully.")

# --- 1. Configuration ---
# Define the project root relative to this script's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the input file from the previous benchmarking step
INPUT_CSV_PATH = PROJECT_ROOT / "results" / "baseline" / "temp.csv"

# Define where to save the new CSV with evaluation scores
OUTPUT_CSV_PATH = PROJECT_ROOT / "results" / "evaluation_results.csv"

print("--- Project Setup Confirmation ---")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Input CSV: {INPUT_CSV_PATH}")
print(f"Output Filename: {OUTPUT_CSV_PATH}")
print("---------------------------------")


# --- 2. Main Evaluation Logic ---
def perform_evaluation():
    """
    Loads model predictions, calculates standard NLP metrics, and saves the results.
    """
    # --- Setup: Ensure NLTK data is ready before proceeding ---
    print("\nChecking for required NLTK data...")
    download_nltk_data()
    print("NLTK check complete.")
    
    # --- Load the Data ---
    print(f"\nLoading data from: {INPUT_CSV_PATH}")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found. Please ensure the file exists at the specified path.")
        return

    # Ensure required columns exist
    required_cols = ['summary', 'generated_summary']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ ERROR: Input CSV must contain the columns: {', '.join(required_cols)}")
        return
        
    predictions = df['generated_summary'].astype(str).tolist()
    references = df['summary'].astype(str).tolist()

    print(f"Found {len(predictions)} summaries to evaluate.")

    # --- Load Evaluation Metrics ---
    print("\nLoading evaluation metrics (this may download data on first run)...")
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    bertscore = evaluate.load('bertscore')
    print("Metrics loaded successfully.")

    # --- Calculate Aggregate Scores (for console report) ---
    print("\nCalculating aggregate scores for the entire dataset...")
    agg_rouge = rouge.compute(predictions=predictions, references=references)
    agg_bleu = bleu.compute(predictions=predictions, references=references)
    agg_meteor = meteor.compute(predictions=predictions, references=references)

    print("\n--- AGGREGATE RESULTS (for high-level overview) ---")
    print(f"ROUGE Scores: {agg_rouge}")
    print(f"BLEU Score: {agg_bleu['bleu']:.4f}")
    print(f"METEOR Score: {agg_meteor['meteor']:.4f}")
    print("--------------------------------------------------\n")

    # --- Calculate Row-by-Row Scores (for detailed output CSV) ---
    print("Calculating row-by-row scores for detailed analysis...")
    
    # Initialize lists to store per-item scores
    rouge1_scores, rouge2_scores, rougeL_scores, bleu_scores, meteor_scores = [], [], [], [], []

    for pred, ref in zip(predictions, references):
        # The metrics expect a list of predictions and references, even for a single item
        single_pred, single_ref = [pred], [ref]
        
        rouge_result = rouge.compute(predictions=single_pred, references=single_ref)
        bleu_result = bleu.compute(predictions=single_pred, references=single_ref)
        meteor_result = meteor.compute(predictions=single_pred, references=single_ref)
        
        # Append the specific scores to the lists
        rouge1_scores.append(rouge_result['rouge1'])
        rouge2_scores.append(rouge_result['rouge2'])
        rougeL_scores.append(rouge_result['rougeL'])
        bleu_scores.append(bleu_result['bleu'])
        meteor_scores.append(meteor_result['meteor'])
        
    # Calculate BERTScore (it's already row-by-row)
    print("Calculating row-by-row BERTScore (this may be slow)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bert_results = bertscore.compute(
            predictions=predictions, 
            references=references, 
            lang="en",
            model_type="distilbert-base-uncased"
        )
    
    print("All score calculations complete.")

    # --- Add all calculated scores to the DataFrame ---
    df['bleu'] = bleu_scores
    df['meteor'] = meteor_scores
    df['rouge1'] = rouge1_scores
    df['rouge2'] = rouge2_scores
    df['rougeL'] = rougeL_scores
    df['bertscore_f1'] = bert_results['f1']

    # --- Save the Results ---
    print(f"\nSaving detailed results with all scores to: {OUTPUT_CSV_PATH}")
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\n✅ Evaluation complete.")

# --- Script Entry Point ---
if __name__ == "__main__":
    perform_evaluation()