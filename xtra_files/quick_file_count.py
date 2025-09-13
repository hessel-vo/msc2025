import os

VALIDATION_REPOS_FILE = "validation_repos.txt"
FILE_LIST_FILE = "file_list.txt"

def count_repo_files():

    try:
        with open(VALIDATION_REPOS_FILE, 'r') as f:
            validation_repos = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"Error: The file '{VALIDATION_REPOS_FILE}' was not found.")
        return

    if not validation_repos:
        print("Warning: No repositories found in '{VALIDATION_REPOS_FILE}'.")
        return

    repo_counts = {repo: 0 for repo in validation_repos}

    try:
        with open(FILE_LIST_FILE, 'r') as f:
            for file_path in f:
                clean_path = file_path.strip()
                if not clean_path:
                    continue

                try:
                    if '/all_repos/' in clean_path:
                        path_after_base = clean_path.split('/all_repos/', 1)[1]
                        repo_name = path_after_base.split('/', 1)[0]

                        if repo_name in repo_counts:
                            repo_counts[repo_name] += 1
                except IndexError:
                    pass

    except FileNotFoundError:
        print(f"Error: The file '{FILE_LIST_FILE}' was not found.")
        return

    print("--- File Counts per Repository ---")
    for repo, count in repo_counts.items():
        print(f"{repo}: {count}")

    # Calculate and print the total
    total_files = sum(repo_counts.values())
    print("\n--- Total Validated Files ---")
    print(f"Total files from all listed repositories: {total_files}")


if __name__ == "__main__":
    count_repo_files()