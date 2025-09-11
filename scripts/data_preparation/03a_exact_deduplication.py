import logging
from pathlib import Path
import time
import os
from dotenv import load_dotenv
import json
import sys
import csv
import hashlib
from tqdm import tqdm

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():

    start_time = time.time()

    # Path & Argument Setup
    load_dotenv()
    project_root_str = os.getenv('PROJECT_ROOT')
    if not project_root_str:
        logging.error("'PROJECT_ROOT' environment variable not set. Please check your .env file.")
        return
        
    project_root = Path(project_root_str)
    data_prep_dir = project_root / "scripts" / "data_preparation"

    if len(sys.argv) > 2:
        filter_level = sys.argv[1]
        all_or_repo = sys.argv[2]

        if all_or_repo == "all":
            processing_dir = data_prep_dir / "02_processing" / filter_level
            deduplication_dir = data_prep_dir / "03_deduplication" / filter_level
        elif all_or_repo == "repo":
            processing_dir = data_prep_dir / "02_processing" / "repository_specific" / filter_level
            deduplication_dir = data_prep_dir / "03_deduplication" / "repository_specific" / filter_level
    else:
        logging.error("Usage: python 03a_exact_deduplication.py <filter_level> <all|repo>")
        return
    
    logging.info(f"Using data filter level: '{filter_level}'")

    deduplication_dir.mkdir(parents=True, exist_ok=True)

    input_path = processing_dir / f"processed_data_{filter_level}.jsonl"
    output_path = deduplication_dir / f"exact_deduplicated_data_{filter_level}.jsonl"
    duplicate_log_path = deduplication_dir / f"exact_duplicate_log_{filter_level}.csv"
    
    if not input_path.is_file():
        logging.error(f"Input file not found: '{input_path}'")
        return

    # Load Data
    logging.info(f"Loading processed data from '{input_path}'...")
    with open(input_path, 'r', encoding='utf-8') as f:
        all_docs = [json.loads(line) for line in tqdm(f, desc="Loading documents")]
    total_docs_loaded = len(all_docs)
    logging.info(f"Loaded {total_docs_loaded:,} documents.")
    
    if not all_docs:
        logging.warning("Input file is empty. Nothing to process.")
        return

    # Deterministic Sorting
    logging.info("Sorting documents to ensure replicable selection of 'kept' files...")

    def get_sort_key(doc):
        metrics = doc.get('metrics', {})
        # Use negative values for descending order
        content_length = -metrics.get('content_length')
        alnum_ratio = -metrics.get('alnum_ratio')
        return (content_length, alnum_ratio)

    all_docs.sort(key=get_sort_key)
    logging.info("Sorting complete.")

    # Exact Deduplication using Hashing
    logging.info("Finding exact duplicates via SHA-256 hashing...")
    seen_hashes = set()
    docs_to_keep = []
    duplicate_log = []
    hash_to_kept_file = {}

    for doc in tqdm(all_docs, desc="Hashing documents"):
        content_bytes = doc['content'].encode('utf-8')
        doc_hash = hashlib.sha256(content_bytes).hexdigest()
        
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            docs_to_keep.append(doc)
            kept_file_path = f"{doc['repo_id']}/{doc['path_in_repo']}"
            hash_to_kept_file[doc_hash] = kept_file_path
        else:
            kept_file = hash_to_kept_file[doc_hash]
            duplicate_file = f"{doc['repo_id']}/{doc['path_in_repo']}"
            duplicate_log.append((kept_file, duplicate_file))

    # Save Dataset and Log
    logging.info(f"Saving {len(docs_to_keep):,} unique documents to '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for doc in docs_to_keep:
            f_out.write(json.dumps(doc) + '\n')

    logging.info(f"Writing exact duplicate log with {len(duplicate_log):,} entries...")
    with open(duplicate_log_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['kept_file', 'exact_duplicate_of_kept_file'])
        for kept_file, duplicate_file in sorted(duplicate_log):
            writer.writerow([kept_file, duplicate_file])

    # Final Summary
    num_final_docs = len(docs_to_keep)
    num_duplicates = total_docs_loaded - num_final_docs
    logging.info("--- Exact Deduplication Summary ---")
    logging.info(f"Initial document count: {total_docs_loaded:,}")
    logging.info(f"Exact duplicates removed: {num_duplicates:,}")
    logging.info(f"Final unique document count: {num_final_docs:,}")
    logging.info(f"Exact-deduplicated dataset saved to: {output_path}")
    logging.info(f"Exact duplicate log saved to: {duplicate_log_path}")
    
    end_time = time.time()
    logging.info(f"Script execution finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()