import json

def analyze_file_content(input_path, output_path):
    """
    Parses a JSONL file to identify files with content less than 10 characters.

    Args:
        input_path (str): The file path for the input JSONL file.
        output_path (str): The destination file path for the output report.
    """
    try:
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                try:
                    record = json.loads(line)
                    content = record.get("content", "")
                    
                    if len(content) < 20:
                        repo_id = record.get("repo_id", "N/A")
                        path_in_repo = record.get("path_in_repo", "N/A")
                        reconstructed_path = f"{repo_id}/{path_in_repo}"
                        char_count = len(content)
                        outfile.write(f"Path: {reconstructed_path}, Characters: {char_count}\n")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: The input file was not found at {input_path}")


# --- Example of Usage ---

# 1. Create a sample JSONL file for demonstration purposes.
with open("sample_data.jsonl", "w") as f:
    f.write('{"repo_id": "example/repo-alpha", "path_in_repo": "src/component.js", "content": "init"}\n')
    f.write('{"repo_id": "example/repo-alpha", "path_in_repo": "README.md", "content": "This is a test repository with several files."}\n')
    f.write('{"repo_id": "example/repo-beta", "path_in_repo": "config.json", "content": "{}"}\n')
    f.write('{"repo_id": "example/repo-gamma", "path_in_repo": "main.py", "content": "import sys"}\n')
    f.write('{"repo_id": "example/repo-gamma", "path_in_repo": "data/values.csv", "content": "a,b,c,d,e,f,g,h,i,j"}\n')

# 2. Specify the input and output file paths.
input_file = "./processed_data.jsonl"
output_file = "./analysis_results.txt"

# 3. Execute the analysis function.
analyze_file_content(input_file, output_file)

# 4. (Optional) Display the generated output file.
with open(output_file, "r") as f:
    print("--- Analysis Results ---")
    print(f.read())