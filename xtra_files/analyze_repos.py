"""
Note: This is a quick Gemini generated script to analyze cloc JSON output
for all repositories and summarize lines of code by language and repository.
Used in project as a rought guide to determine which repos to use for validation set during training.
"""

import json
import os
from collections import defaultdict


BASE_PATH = "/home/hessel/workspace/RUG/MSc2025/project/code/repositories/all_repos"

# The cloc JSON report file
CLOC_JSON_FILE = "cloc_output_all.json"

LANGUAGES_TO_ANALYZE = {"Python", "Java", "C", "C++"}


def analyze_cloc_data(cloc_file, base_path, languages_to_include):
    if not os.path.exists(cloc_file):
        print(f"Error: cloc report not found at '{cloc_file}'")
        print("Please run 'cloc --list-file=... --by-file --json --out=cloc_output.json' first.")
        return None

    normalized_base_path = os.path.normpath(base_path) + os.sep
    language_stats = defaultdict(lambda: defaultdict(int))
    
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
            
            language_stats[language][repo_name] += code_lines
            processed_files_count += 1
        except (ValueError, IndexError):
            print(f"Warning: Could not determine repository for path: {filepath}")

    print(f"--> Successfully processed {processed_files_count} files from the specified languages.")
    return language_stats


def print_summary(language_stats):
    if not language_stats:
        print("\n--> No data to display. Check if the specified languages exist in the cloc report.")
        return

    sorted_languages = sorted(
        language_stats.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True
    )

    print("\n" + "="*60)
    print("      Lines of Code by Language and Repository")
    print("="*60)

    for language, repos in sorted_languages:
        total_lines_for_lang = sum(repos.values())
        print(f"\n--- {language} (Total: {total_lines_for_lang:,} lines) ---")

        sorted_repos = sorted(repos.items(), key=lambda item: item[1], reverse=True)

        for i, (repo_name, lines) in enumerate(sorted_repos):
            percentage = (lines / total_lines_for_lang) * 100
            print(f"  {i+1}. {repo_name:<40} {lines:>12,} lines ({percentage:.1f}%)")


if __name__ == "__main__":
    stats = analyze_cloc_data(CLOC_JSON_FILE, BASE_PATH, LANGUAGES_TO_ANALYZE)
    if stats:
        print_summary(stats)