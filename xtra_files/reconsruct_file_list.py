import json
import os

BASE_PATH = "/home/hessel/workspace/RUG/MSc2025/project/code/repositories/all_repos"

JSONL_FILE = "../scripts/data_preparation/03_deduplication/core/final_dataset_core.jsonl"

OUTPUT_FILE = "file_list_core.txt"


def generate_full_paths(base_path, jsonl_file):
    if not os.path.exists(jsonl_file):
        print(f"Error: Input file not found at '{jsonl_file}'")
        return []

    filepaths = []
    print(f"--> Reading data from '{jsonl_file}'...")

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if 'repo_id' in data and 'path_in_repo' in data:
                    repo_id = data['repo_id']
                    path_in_repo = data['path_in_repo']
                    
                    full_path = os.path.join(base_path, repo_id, path_in_repo)
                    filepaths.append(full_path)
                else:
                    print(f"Warning: Line {i+1} is missing 'repo_id' or 'path_in_repo' keys.")

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON on line {i+1}.")
            except Exception as e:
                print(f"An unexpected error occurred on line {i+1}: {e}")

    return filepaths

def write_paths_to_file(filepaths, output_filename):
    print(f"--> Writing {len(filepaths)} paths to '{output_filename}'...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            for path in filepaths:
                f.write(path + '\n')
        print(f"--> Successfully created '{output_filename}'.")
    except IOError as e:
        print(f"Error: Could not write to file '{output_filename}'. Reason: {e}")


if __name__ == "__main__":
    full_file_list = generate_full_paths(BASE_PATH, JSONL_FILE)

    if full_file_list:
        # Write the generated list directly to the output file
        write_paths_to_file(full_file_list, OUTPUT_FILE)
    else:
        print("\n--> No file paths were generated.")