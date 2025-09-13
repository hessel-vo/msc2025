import os
from collections import defaultdict

FILEPATHS_FILE = "./file_list_target_only.txt"


PARENT_DIR_SEGMENT = "all_repos"


def analyze_repository_file_counts(filepath: str):
    repo_counts = defaultdict(int)
    total_files_processed = 0
    lines_with_errors = 0

    print(f"--- Starting analysis of '{filepath}' ---")

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            full_path = line.strip()

            if not full_path:
                continue

            path_parts = full_path.split(os.sep)

            parent_index = path_parts.index(PARENT_DIR_SEGMENT)

            if parent_index + 1 < len(path_parts):
                repo_name = path_parts[parent_index + 1]
                repo_counts[repo_name] += 1
                total_files_processed += 1

    print(f"Total files processed: {total_files_processed}")
    print(f"Total unique repositories found: {len(repo_counts)}")

    print("\n--- File Counts per Repository (Sorted by Name) ---")
    for repo_name in sorted(repo_counts.keys()):
        count = repo_counts[repo_name]
        print(f"{repo_name}: {count} files")
        
    print("\n--- Top 10 Repositories by File Count ---")
    sorted_by_count = sorted(repo_counts.items(), key=lambda item: item[1], reverse=True)
    for repo_name, count in sorted_by_count[:10]:
         print(f"{repo_name}: {count} files")


if __name__ == "__main__":
    analyze_repository_file_counts(FILEPATHS_FILE)