import os
import csv
import subprocess

INPUT_CSV = 'unfiltered_repos.csv'
CLONE_BASE_DIR = 'all_repos'
OUTPUT_CSV = 'filtered_repos.csv'
FAILED_LOG = 'failed_clones.log'

def find_license(repo_path):

    common_license_files = ['license', 'license.md', 'license.txt', 'copying', 'copying.md', 'unlicense']
    
    try:
        license_filename = None
        for filename in os.listdir(repo_path):
            if filename.lower() in common_license_files:
                license_filename = filename
                break

        if not license_filename:
            return 'License file not found'

        license_file_path = os.path.join(repo_path, license_filename)
        with open(license_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    # Attempt to retrieve license name
                    if len(stripped_line) < 100:
                        if stripped_line.lower() == 'apache license':
                            stripped_line = 'Apache License Version 2.0'
                        return stripped_line
                    else:
                        return f"Found: {license_filename} (Content not parsed)"
            return f"Found: {license_filename} (File is empty)"
            
    except Exception as e:
        return f"Error reading license: {e}"

def get_commit_hash(repo_path):
    """Gets the latest commit hash from a local git repository."""
    try:
        result = subprocess.run(
            ['git', '-C', repo_path, 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True, encoding='utf-8'
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'N/A'

def clone_and_process_repositories():
    os.makedirs(CLONE_BASE_DIR, exist_ok=True)
    if not os.path.isfile(INPUT_CSV):
        print(f"Error: Input file '{INPUT_CSV}' not found.")
        return

    successful_repos = []
    failed_urls = []

    with open(INPUT_CSV, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        output_headers = reader.fieldnames + ['commit_hash', 'license']
        
        for row in reader:
            repo_url = row.get('repo link', '').strip()
            relevance_filter = row.get('relevance_filter', '').strip()

            if not repo_url or relevance_filter != '1':
                continue

            print(f"\nProcessing: {repo_url}")
            
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            clone_target_path = os.path.join(CLONE_BASE_DIR, repo_name)

            try:
                
                subprocess.run(
                    ['git', 'clone', '--depth', '1', repo_url, clone_target_path],
                    check=True, capture_output=True, text=True, encoding='utf-8'
                )
                

                commit_hash = get_commit_hash(clone_target_path)
                license_name = find_license(clone_target_path)
                
                row['commit_hash'] = commit_hash
                row['license'] = license_name
                successful_repos.append(row)

            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to clone {repo_url}.\nStderr: {e.stderr.strip()}")
                failed_urls.append(repo_url)
            except Exception as e:
                print(f"An unexpected error occurred processing {repo_url}: {e}")
                failed_urls.append(repo_url)

    if successful_repos:
        print(f"\nWriting details for {len(successful_repos)} repositories to '{OUTPUT_CSV}'...")
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_headers)
            writer.writeheader()
            writer.writerows(successful_repos)

    if failed_urls:
        print(f"Logging {len(failed_urls)} failed clone attempts to '{FAILED_LOG}'...")
        with open(FAILED_LOG, 'w', encoding='utf-8') as logfile:
            for url in failed_urls:
                logfile.write(url + '\n')
    
    print("\nScript finished.")


if __name__ == "__main__":
    # On Windows, you might need to enable long paths support if you encounter path length issues:
    # git config --global core.longpaths true
    clone_and_process_repositories()