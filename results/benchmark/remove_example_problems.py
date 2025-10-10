import sys
import os
import pandas as pd

RESULT_TYPE = "adapted" # "baseline" or "adapted"
CSV_FILE_PATH = f'./{RESULT_TYPE}/gemma-3-1b-it_summarization_xl_one_shot_short_results.csv'
FOLDER_PATH = f'./{RESULT_TYPE}/to_process/'

def get_problem_ids(shot_type):
    if shot_type not in ['one', 'three']:
        raise ValueError("Invalid shot type specified. Please use 'one' or 'three'.")

    file_path = f'examples_{shot_type}_shot.txt'

    with open(file_path, 'r') as f:
        try:
            problem_ids = {int(line.strip()) for line in f if line.strip()}
            return problem_ids
        except ValueError:
            raise TypeError(f"Error: All IDs in '{file_path}' must be integers.")

def filter_csv_file(csv_path, ids_to_remove):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise IOError(f"Error reading CSV file '{csv_path}': {e}")

    filtered_df = df[~df['id'].isin(ids_to_remove)]


    input_dir = os.path.dirname(csv_path)
    input_filename = os.path.basename(csv_path)
    output_dir = os.path.join(input_dir, 'processed_results')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, input_filename)
    
    try:
        filtered_df.to_csv(output_path, index=False)
        print(f"Successfully processed the input file: {csv_path}")
        print(f"Saved modified file to: {output_path}")
    except Exception as e:
        raise IOError(f"Error writing to CSV file '{output_path}': {e}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python filter_results.py <one|three> <file|folder>")
        sys.exit(1) # Exit with an error code

    shot_argument = sys.argv[1]
    process_mode = sys.argv[2]

    # Get the set of problem IDs to be removed
    problem_ids_to_remove = get_problem_ids(shot_argument)
    print(f"Loaded {len(problem_ids_to_remove)} problem IDs from 'examples_{shot_argument}_shot.txt'.")
    
    if process_mode == 'file':
        # Filter the specified CSV file
        filter_csv_file(CSV_FILE_PATH, problem_ids_to_remove)
    elif process_mode == 'folder':
        for filename in os.listdir(FOLDER_PATH):
            if filename.endswith('.csv'):
                file_path = os.path.join(FOLDER_PATH, filename)
                filter_csv_file(file_path, problem_ids_to_remove)
    else:
        print("Invalid argument. Please specify 'file' or 'folder'.")


if __name__ == '__main__':
    main()