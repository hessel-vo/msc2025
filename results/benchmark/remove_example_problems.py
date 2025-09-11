import sys
import os
import pandas as pd

CSV_FILE_PATH = './baseline/gemma-3-1b-it_summarization_xl_one_shot_short_results.csv'

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
    if len(sys.argv) != 2:
        print("Usage: python filter_results.py <one|three>")
        sys.exit(1) # Exit with an error code

    shot_argument = sys.argv[1]

    # Get the set of problem IDs to be removed
    problem_ids_to_remove = get_problem_ids(shot_argument)
    print(f"Loaded {len(problem_ids_to_remove)} problem IDs from 'examples_{shot_argument}_shot.txt'.")
    
    # Filter the specified CSV file
    filter_csv_file(CSV_FILE_PATH, problem_ids_to_remove)


if __name__ == '__main__':
    main()