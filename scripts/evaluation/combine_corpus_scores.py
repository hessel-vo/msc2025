import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)

# --- Configuration ---
MODEL_NAME = "gemma-3-12b-it"
RESULTS_FOLDER = "results/evaluation/baseline"  # <-- IMPORTANT: SET THIS TO YOUR RESULTS FOLDER
OUTPUT_FILENAME = f"combined_scores_{MODEL_NAME}.csv"
# ---------------------

def parse_filename(filename):
    """
    Parses the filename to extract the test conditions.
    Expected format: corpus_score_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}.csv
    """
    parts = filename.stem.split('_')
    if len(parts) >= 7 and parts[0] == 'corpus' and parts[1] == 'score':
        model_name_in_file = parts[2]
        task = parts[3]
        source = parts[4]
        summary_length = parts[5]
        shot_count = parts[6]
        return model_name_in_file, task, source, summary_length, shot_count
    return None, None, None, None, None

def combine_scores():
    """
    Iterates through CSV files in a directory, combines the scores for a specific
    model, and saves the result to a new CSV file.
    """

    results_path = PROJECT_ROOT / RESULTS_FOLDER

    if not results_path.is_dir():
        print(f"Error: The directory '{results_path}' does not exist.")
        return

    all_scores = []

    print(f"Searching for results in: {results_path}")

    print(results_path.glob("*.csv"))

    for file_path in results_path.glob("*.csv"):
        model_in_file, task, source, summary_length, shot_count = parse_filename(file_path)

        if model_in_file == MODEL_NAME:
            print(f"Processing file: {file_path.name}")
            try:
                df = pd.read_csv(file_path)

                # Create the descriptive test name
                test_name = f"{MODEL_NAME}_{task}_{source}_{summary_length}_{shot_count}"
                if shot_count == "zero":
                    test_name = f"{MODEL_NAME}_{task}_{summary_length}_{shot_count}"

                df['test_name'] = test_name

                # Drop the original model_name column and reorder
                df = df.drop(columns=['model_name'])
                cols = ['test_name'] + [col for col in df.columns if col != 'test_name']
                df = df[cols]

                all_scores.append(df)
            except Exception as e:
                print(f"Could not process file {file_path.name}: {e}")

    if not all_scores:
        print(f"No CSV files found for the model '{MODEL_NAME}' in '{results_path}'.")
        return

    combined_df = pd.concat(all_scores, ignore_index=True)
    combined_df = combined_df.sort_values(by='test_name').reset_index(drop=True)

    output_path = PROJECT_ROOT / RESULTS_FOLDER / OUTPUT_FILENAME
    combined_df.to_csv(output_path, index=False)

    print("\n---")
    print(f"Successfully combined {len(all_scores)} result files.")
    print(f"Output saved to: {output_path}")
    print("\nCombined Data Head:")
    print(combined_df.head())
    print("---")


if __name__ == "__main__":
    combine_scores()