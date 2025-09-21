import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)

# --- Configuration ---
MODEL_NAME = "gemma-3-12b-it"
RESULTS_FOLDER = "results/evaluation/baseline"
OUTPUT_FILENAME = f"combined_scores_{MODEL_NAME}.csv"

# --- NEW: Set to True to include and average generation task results ---
RUN_GENERATION = True
# ---------------------

def parse_filename(filename):
    """
    Parses the filename to extract the test conditions.
    Expected format: corpus_score_{MODEL_NAME}_{TASK}_{source}_{summary_length}_{shot_count}.csv
    """
    parts = filename.stem.split('_')
    # Using a more robust check for model names that might contain underscores
    if len(parts) >= 6 and parts[0] == 'corpus' and parts[1] == 'score':
        shot_count = parts[-1]
        summary_length = parts[-2]
        source = parts[-3]
        task = parts[-4]
        model_name_in_file = "_".join(parts[2:-4])
        return model_name_in_file, task, source, summary_length, shot_count
    return None, None, None, None, None

def combine_scores():
    """
    Iterates through CSV files, combines scores for a specific model,
    averages scores for generation tasks if enabled, and saves the result.
    """
    results_path = PROJECT_ROOT / RESULTS_FOLDER
    if not results_path.is_dir():
        print(f"Error: The directory '{results_path}' does not exist.")
        return

    # --- MODIFIED: Separate lists for different task types ---
    summarization_scores = []
    generation_scores = []

    print(f"Searching for results in: {results_path}")

    for file_path in results_path.glob("*.csv"):
        # The parser is now more robust to handle different model name formats
        model_in_file, task, source, summary_length, shot_count = parse_filename(file_path)

        if model_in_file == MODEL_NAME:
            print(f"Processing file: {file_path.name} (Task: {task})")
            try:
                df = pd.read_csv(file_path)

                # --- FIX: Consistent test_name creation is crucial for grouping ---
                # This ensures that all 4 rows from a generation task CSV get the same identifier.
                test_name = f"{MODEL_NAME}_{task}_{source}_{summary_length}_{shot_count}"
                df['test_name'] = test_name

                # Drop original model_name and reorder (handles both task types)
                if 'model_name' in df.columns:
                    df = df.drop(columns=['model_name'])
                
                cols = ['test_name'] + [col for col in df.columns if col != 'test_name']
                df = df[cols]

                # --- MODIFIED: Route DataFrame to the correct list based on task ---
                if task == 'summarization':
                    summarization_scores.append(df)
                elif task == 'generation' and RUN_GENERATION:
                    # The generation df will have multiple rows (one per language)
                    generation_scores.append(df)
                else:
                    print(f"  -> Skipping file due to task type or RUN_GENERATION setting.")

            except Exception as e:
                print(f"Could not process file {file_path.name}: {e}")

    # --- NEW: Process and combine the collected dataframes ---
    final_dfs_to_combine = []

    if summarization_scores:
        # Summarization results are single-row, just concatenate them
        summarization_df = pd.concat(summarization_scores, ignore_index=True)
        final_dfs_to_combine.append(summarization_df)
        print(f"\nFound {len(summarization_scores)} summarization result file(s).")

    if generation_scores and RUN_GENERATION:
        # For generation, first concatenate all results into one big dataframe
        generation_df_raw = pd.concat(generation_scores, ignore_index=True)
        
        # Now, group by the unique test name and calculate the mean for all score columns.
        # This collapses the 4 language rows into a single, averaged row per test.
        print(f"\nFound {len(generation_scores)} generation result file(s), now averaging scores...")
        generation_df_averaged = generation_df_raw.groupby('test_name').mean().reset_index()
        final_dfs_to_combine.append(generation_df_averaged)

    if not final_dfs_to_combine:
        print(f"\nNo CSV files were processed for the model '{MODEL_NAME}' in '{results_path}'.")
        return

    # Combine the final, processed dataframes from all task types
    combined_df = pd.concat(final_dfs_to_combine, ignore_index=True)
    combined_df = combined_df.sort_values(by='test_name').reset_index(drop=True)

    output_path = PROJECT_ROOT / RESULTS_FOLDER / OUTPUT_FILENAME
    combined_df.to_csv(output_path, index=False)

    print("\n---")
    print(f"Successfully combined and processed results.")
    print(f"Output saved to: {output_path}")
    print("\nCombined Data Head:")
    print(combined_df.head())
    print("---")


if __name__ == "__main__":
    combine_scores()