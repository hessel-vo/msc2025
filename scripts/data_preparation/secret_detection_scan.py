import logging
from pathlib import Path
import time
import os
from dotenv import load_dotenv
from typing import Iterator
from collections import Counter

from detect_secrets import SecretsCollection
from detect_secrets.settings import default_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_file_paths_from_manifest(manifest_path: Path) -> Iterator[str]:
    logging.info(f"Reading file paths from manifest: '{manifest_path}'")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                yield stripped_line

def main():

    start_time = time.time()
    
    load_dotenv()
    project_root_str = os.getenv('PROJECT_ROOT')
    if not project_root_str:
        logging.error("'PROJECT_ROOT' environment variable not set.")
        return
        
    project_root = Path(project_root_str)
    data_prep_dir = project_root / "scripts" / "data_preparation"
    manifest_path = data_prep_dir / "01_filtering" / "filtered_file_paths_extended.txt"
    
    report_output_path = data_prep_dir / "secrets_report.txt"
    
    if not manifest_path.is_file():
        logging.error(f"Manifest file not found: '{manifest_path}'")
        return

    secret_type_counts = Counter()
    report_lines = []
    docs_scanned = 0
    
    logging.info("Starting secret analysis and report generation...")
    
    path_generator = load_file_paths_from_manifest(manifest_path)
    
    secrets = SecretsCollection()
    with default_settings():
        for file_path in path_generator:
            secrets.scan_file(file_path)
            docs_scanned += 1

            if docs_scanned % 2000 == 0:
                logging.info(f"Scanned {docs_scanned:,} documents...")

    results_dict = secrets.json()
    
    for filename, found_secrets_list in results_dict.items():
        relative_filename = str(Path(filename).relative_to(project_root))
        
        for secret in found_secrets_list:
            secret_type_counts[secret['type']] += 1
            
            report_line = (
                f"File: {relative_filename}\n"
                f"  - Type: {secret['type']}\n"
                f"  - Line: {secret['line_number']}\n"
            )
            report_lines.append(report_line)

    logging.info(f"Writing detailed secrets report to: {report_output_path}")
    with open(report_output_path, 'w', encoding='utf-8') as f:
        if not report_lines:
            f.write("No secrets were detected in the scanned files.\n")
        else:
            f.write("--- Detailed Secrets Report ---\n\n")
            for line in sorted(report_lines):
                f.write(line + "\n")
    logging.info("Report written successfully.")


    logging.info(f"--- Secret Analysis Complete ---")
    logging.info(f"Total documents scanned: {docs_scanned:,}")
    
    print("\n--- Found Secret Types (most common first) ---")
    if not secret_type_counts:
        print("No secrets found.")
    else:
        print(f"{'Secret Type':<40} {'Count'}")
        print(f"{'-'*40:<40} {'-'*10}")
        for secret_type, count in secret_type_counts.most_common():
            print(f"{secret_type:<40} {count:,}")
    
    end_time = time.time()
    logging.info(f"Analysis finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()