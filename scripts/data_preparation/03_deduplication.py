import logging
from pathlib import Path
import time
import os
from dotenv import load_dotenv
import json
import sys
from collections import defaultdict
import csv

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Deduplication Parameter Sets ---
DEDUPE_CONFIGS = {
    "stack": {
        "description": "Parameters from 'The Stack' paper (optimized for full files).",
        "threshold": 0.85,
        "ngram_size": 13,
        "num_perm": 256,
    },
    "santacoder": {
        "description": "Parameters from 'SantaCoder' paper (optimized for smaller snippets).",
        "threshold": 0.7,
        "ngram_size": 5,
        "num_perm": 256,
    }
}

def create_minhash(text: str, num_perm: int, ngram_size: int) -> MinHash:
    """
    Creates a MinHash object for a given text.
    The text is split into n-grams of UTF-8 bytes.
    """
    minhash = MinHash(num_perm=num_perm)
    # Split text into n-grams of characters, then encode to bytes
    ngrams = ["".join(i) for i in zip(*[text[i:] for i in range(ngram_size)])]
    for gram in ngrams:
        minhash.update(gram.encode('utf-8'))
    return minhash

def main():
    """
    Main execution function for near-deduplication using datasketch.
    """
    start_time = time.time()

    # --- Path & Argument Setup ---
    load_dotenv()
    project_root_str = os.getenv('PROJECT_ROOT')
    if not project_root_str:
        logging.error("'PROJECT_ROOT' environment variable not set. Please check your .env file.")
        return
        
    project_root = Path(project_root_str)
    data_prep_dir = project_root / "scripts" / "data_preparation"

    if len(sys.argv) < 3:
        logging.error("Usage: python 03_deduplication.py <data_suffix> <config_name>")
        logging.error(f"Available configs: {list(DEDUPE_CONFIGS.keys())}")
        return
    
    filter_level = sys.argv[1]
    config_name = sys.argv[2]

    if config_name not in DEDUPE_CONFIGS:
        logging.error(f"Invalid config name '{config_name}'. Available configs: {list(DEDUPE_CONFIGS.keys())}")
        return
    
    config = DEDUPE_CONFIGS[config_name]
    logging.info(f"Using data suffix: '{filter_level}'")
    logging.info(f"Using deduplication config '{config_name}': {config['description']}")

    processing_dir = data_prep_dir / "02_processing" / filter_level
    deduplication_dir = data_prep_dir / "03_deduplication" / filter_level
    deduplication_dir.mkdir(parents=True, exist_ok=True)

    input_path = processing_dir / f"processed_data_{filter_level}.jsonl"
    output_path = deduplication_dir / f"final_dataset_{filter_level}_{config_name}.jsonl"
    duplicate_log_path = deduplication_dir / f"duplicate_log_{filter_level}_{config_name}.csv"
    
    if not input_path.is_file():
        logging.error(f"Input file not found: '{input_path}'")
        return

    # --- 1. Load Data ---
    logging.info(f"Loading processed data from '{input_path}'...")
    with open(input_path, 'r', encoding='utf-8') as f:
        docs_to_process = [json.loads(line) for line in tqdm(f, desc="Loading documents")]
    total_docs_loaded = len(docs_to_process)
    logging.info(f"Loaded {total_docs_loaded:,} documents.")
    
    if not docs_to_process:
        logging.warning("Input file is empty. Nothing to process.")
        return

    # --- 2. Deterministic Sorting ---
    logging.info("Sorting documents to ensure replicable deduplication...")
    docs_to_process.sort(
        key=lambda x: (-x['metrics']['content_length'], -x['metrics']['alnum_ratio'], x['path_in_repo'])
    )
    logging.info("Sorting complete.")

    # --- 3. Create MinHashes for all documents ---
    logging.info("Creating MinHash fingerprints for all documents...")
    minhashes = []
    for doc in tqdm(docs_to_process, desc="Fingerprinting"):
        minhashes.append(create_minhash(doc['content'], config['num_perm'], config['ngram_size']))

    # --- 4. Index MinHashes in LSH ---
    logging.info("Indexing MinHash fingerprints in LSH...")
    lsh = MinHashLSH(threshold=config['threshold'], num_perm=config['num_perm'])
    for i, minhash in enumerate(tqdm(minhashes, desc="Indexing")):
        # Use the index `i` as the key for each document
        lsh.insert(i, minhash)

    # --- 5. Identify Duplicate Clusters ---
    logging.info("Querying LSH to identify duplicate clusters...")
    processed_indices = set()
    duplicate_clusters = []
    for i, minhash in enumerate(tqdm(minhashes, desc="Clustering")):
        if i in processed_indices:
            continue
        
        # Find all document indices that are similar to the current one
        cluster = lsh.query(minhash)
        
        # The first document in a cluster (due to our sort) is the one we keep.
        # All other documents in the cluster are duplicates.
        duplicate_clusters.append(cluster)
        
        # Mark all documents in this cluster as processed
        processed_indices.update(cluster)
        
    # --- 6. Filter, Log, and Save ---
    logging.info("Filtering unique documents and preparing logs...")
    indices_to_keep = set()
    duplicate_log = []

    for cluster in duplicate_clusters:
        # Because the original list was sorted, the document with the smallest index
        # in the cluster is the one we designated as the "best" one to keep.
        doc_to_keep_idx = min(cluster)
        indices_to_keep.add(doc_to_keep_idx)
        
        # Log the relationship for all other documents in the cluster
        kept_file_path = f"{docs_to_process[doc_to_keep_idx]['repo_id']}/{docs_to_process[doc_to_keep_idx]['path_in_repo']}"
        for dup_idx in cluster:
            if dup_idx != doc_to_keep_idx:
                dup_file_path = f"{docs_to_process[dup_idx]['repo_id']}/{docs_to_process[dup_idx]['path_in_repo']}"
                duplicate_log.append((kept_file_path, dup_file_path))

    # Save the final unique documents
    logging.info(f"Saving {len(indices_to_keep):,} unique documents to '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # Iterate in sorted order of indices to maintain some consistency
        for i in sorted(list(indices_to_keep)):
            f_out.write(json.dumps(docs_to_process[i]) + '\n')

    # Save the detailed duplicate log
    logging.info(f"Writing duplicate log with {len(duplicate_log):,} entries...")
    with open(duplicate_log_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['kept_file', 'duplicate_of_kept_file'])
        for kept_file, duplicate_file in sorted(duplicate_log):
            writer.writerow([kept_file, duplicate_file])

    # --- 7. Final Summary ---
    num_final_docs = len(indices_to_keep)
    num_duplicates = total_docs_loaded - num_final_docs
    logging.info("--- Deduplication Summary ---")
    logging.info(f"Configuration used: '{config_name}'")
    logging.info(f"Initial document count: {total_docs_loaded:,}")
    logging.info(f"Duplicates removed: {num_duplicates:,}")
    logging.info(f"Final unique document count: {num_final_docs:,}")
    logging.info(f"✅ Final dataset saved successfully to: {output_path}")
    logging.info(f"✅ Duplicate log saved successfully to: {duplicate_log_path}")
    
    end_time = time.time()
    logging.info(f"Script execution finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()