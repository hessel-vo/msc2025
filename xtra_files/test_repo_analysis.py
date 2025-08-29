"""
Note: This is a quick Gemini generated script to analyze cloc JSON output
for all repositories and summarize the top 10 files by lines of code for each repository.
Used in project as a rought guide to determine which repos to use for validation set during training.
"""

import json
import os
from collections import defaultdict

# --- Configuration ---

BASE_PATH = "/home/hessel/workspace/RUG/MSc2025/project/code/repositories/test_repos"

# The cloc JSON report file generated with the --by-file flag.
CLOC_JSON_FILE = "cloc_output_repo_specific.json"

# Define the specific languages you want to analyze.
# cloc language names are case-sensitive (e.g., "C++", not "c++").
LANGUAGES_TO_ANALYZE = {"Python", "Java", "C", "C++"}


def analyze_cloc_data_by_file(cloc_file, base_path, languages_to_include):
    """
    Parses a cloc JSON report and collects file-level code statistics for each repository.

    Args:
        cloc_file (str): The path to the cloc JSON output file.
        base_path (str): The base directory where all repositories are stored.
        languages_to_include (set): A set of language names to include in the analysis.

    Returns:
        dict: A dictionary structured as {repo_name: [(filepath, language, code_lines)]}.
              Returns None if the file cannot be processed.
    """
    if not os.path.exists(cloc_file):
        print(f"Error: cloc report not found at '{cloc_file}'")
        print("Please run 'cloc --list-file=... --by-file --json --out=cloc_output.json' first.")
        return None

    normalized_base_path = os.path.normpath(base_path) + os.sep
    repo_stats = defaultdict(list)
    
    print(f"--> Analyzing '{cloc_file}'...")
    print(f"--> Using repository base path: '{normalized_base_path}'")
    print(f"--> Filtering for languages: {', '.join(sorted(languages_to_include))}")

    with open(cloc_file, 'r', encoding='utf-8') as f:
        try:
            cloc_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON file. It might be corrupted. Details: {e}")
            return None

    file_count = len(cloc_data) - 1 if 'header' in cloc_data else len(cloc_data)
    if file_count <= 0:
        print("\nError: The JSON report contains no file data. It might be empty or only has a header.")
        return None
    
    print(f"--> Found {file_count} file entries in the report.")

    processed_files_count = 0
    path_mismatch_warning_shown = False

    for filepath, data in cloc_data.items():
        if filepath == 'header':
            continue

        language = data.get('language')
        code_lines = data.get('code', 0)

        if language not in languages_to_include or code_lines == 0:
            continue
        
        normalized_filepath = os.path.normpath(filepath)

        if not normalized_filepath.startswith(normalized_base_path):
            if not path_mismatch_warning_shown:
                print("\n" + "="*60)
                print("!!! CRITICAL WARNING: PATH MISMATCH !!!")
                print(f"  - Your BASE_PATH:     '{base_path}'")
                print(f"  - Example file path:  '{filepath}'")
                print("="*60 + "\n")
                path_mismatch_warning_shown = True
            continue

        try:
            relative_path = os.path.relpath(normalized_filepath, normalized_base_path)
            repo_name = relative_path.split(os.sep)[0]
            
            repo_stats[repo_name].append((filepath, language, code_lines))
            processed_files_count += 1
        except (ValueError, IndexError):
            print(f"Warning: Could not determine repository for path: {filepath}")

    print(f"--> Successfully processed {processed_files_count} files from the specified languages.")
    return repo_stats


def print_top_files_summary(repo_stats, top_n=20):
    """
    Prints a formatted summary of the top N files with the most lines of code for each repository.
    Also displays the percentage of the repository's total code that each file represents.

    Args:
        repo_stats (dict): A dictionary from analyze_cloc_data_by_file.
        top_n (int): The number of top files to display for each repository.
    """
    if not repo_stats:
        print("\n--> No data to display. Check if the specified languages exist in the cloc report.")
        return

    sorted_repos = sorted(repo_stats.keys())

    print("\n" + "="*70)
    print(f"      Top {top_n} Files with Most Lines of Code by Repository")
    print("="*70)

    for repo_name in sorted_repos:
        all_files_in_repo = repo_stats[repo_name]
        
        # MODIFICATION: Calculate the total lines of code for the repository to find percentages.
        total_repo_lines = sum(file_data[2] for file_data in all_files_in_repo)
        
        # Sort all files in the repository by lines of code, descending
        sorted_files = sorted(all_files_in_repo, key=lambda item: item[2], reverse=True)
        
        top_files = sorted_files[:top_n]

        print(f"\n--- Repository: {repo_name} (Total: {total_repo_lines:,} lines) ---")

        if not top_files:
            print("  No files found for the specified languages in this repository.")
            continue

        # Find the full path of the repository to create relative file paths for cleaner output
        repo_full_path = os.path.join(BASE_PATH, repo_name)
        
        for i, (filepath, language, lines) in enumerate(top_files):
            # MODIFICATION: Calculate the percentage of the repository's total code.
            percentage = (lines / total_repo_lines) * 100 if total_repo_lines > 0 else 0
            
            relative_filepath = os.path.relpath(filepath, repo_full_path)
            
            # MODIFICATION: Updated print format to include the percentage.
            print(f"  {i+1:>2}. {lines:>8,} lines ({percentage:5.1f}%)  ({language:<10})  ./{relative_filepath}")


if __name__ == "__main__":
    stats = analyze_cloc_data_by_file(CLOC_JSON_FILE, BASE_PATH, LANGUAGES_TO_ANALYZE)
    if stats:
        # MODIFICATION: Changed top_n from 10 to 20.
        print_top_files_summary(stats, top_n=20)