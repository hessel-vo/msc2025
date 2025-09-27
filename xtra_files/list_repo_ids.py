import json
import os

def extract_unique_repo_ids(input_file_path, output_file_path):
    unique_repo_ids = set()
    try:
        with open(input_file_path, 'r') as infile:
            for line in infile:
                # Skip empty lines
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if 'repo_id' in data:
                        unique_repo_ids.add(data['repo_id'])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")

        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file_path, 'w') as outfile:
            for repo_id in sorted(list(unique_repo_ids)):
                outfile.write(repo_id + '\n')
        
        print(f"Successfully extracted {len(unique_repo_ids)} unique repository IDs to '{output_file_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    input_file = "../scripts/data_preparation/03_deduplication/target_only/final_dataset_target_only.jsonl"

    output_file = "repo_ids_to.txt"
    
    extract_unique_repo_ids(input_file, output_file)