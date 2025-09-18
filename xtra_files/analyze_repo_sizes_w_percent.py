import os
from collections import defaultdict
import numpy as np

FILEPATHS_FILE = "./file_list_target_only.txt"
PARENT_DIR_SEGMENT = "all_repos"
# Define the percentiles you want to calculate
PERCENTILES_TO_CALCULATE = [10, 25, 50, 75, 85, 90, 95, 99]


def analyze_repository_file_counts(filepath: str):
    """
    Analyzes the number of files per repository from a list of file paths
    and calculates specified percentiles for the file counts.
    """
    repo_counts = defaultdict(int)
    total_files_processed = 0

    print(f"--- Starting analysis of '{filepath}' ---")

    try:
        with open(filepath, 'r') as f:
            for line in f:
                full_path = line.strip()

                if not full_path:
                    continue

                try:
                    path_parts = full_path.split(os.sep)
                    parent_index = path_parts.index(PARENT_DIR_SEGMENT)

                    if parent_index + 1 < len(path_parts):
                        repo_name = path_parts[parent_index + 1]
                        repo_counts[repo_name] += 1
                        total_files_processed += 1
                except ValueError:
                    # This handles cases where PARENT_DIR_SEGMENT is not in the path
                    print(f"Skipping line due to format error: {line.strip()}")
                    continue

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    if not repo_counts:
        print("No repositories found or processed. Exiting.")
        return

    print(f"\nTotal files processed: {total_files_processed}")
    print(f"Total unique repositories found: {len(repo_counts)}")

    # --- Percentile Calculation ---
    file_counts = list(repo_counts.values())
    percentiles = np.percentile(file_counts, PERCENTILES_TO_CALCULATE)

    print("\n--- Percentiles of File Counts per Repository ---")
    for p, value in zip(PERCENTILES_TO_CALCULATE, percentiles):
        print(f"{p}th percentile: {int(round(value))} files")

    print("\n--- Top 10 Repositories by File Count ---")
    sorted_by_count = sorted(repo_counts.items(), key=lambda item: item[1], reverse=True)
    for repo_name, count in sorted_by_count[:10]:
        print(f"{repo_name}: {count} files")

    print("\n--- File Counts per Repository (Sorted by Name) ---")
    for repo_name in sorted(repo_counts.keys()):
        count = repo_counts[repo_name]
        print(f"{repo_name}: {count} files")


if __name__ == "__main__":
    analyze_repository_file_counts(FILEPATHS_FILE)