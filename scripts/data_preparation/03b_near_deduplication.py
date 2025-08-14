import logging
from pathlib import Path
import time
import os
from dotenv import load_dotenv
import json
import sys
from collections import defaultdict
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

from datasketch import MinHash
from datasketch.lsh import MinHashLSH
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Deduplication Constants ---
DEDUPE_CONFIG = {
    "threshold": 0.9,
    "num_perm": 256,
    "ngram_size": 5,
}
NON_ALPHA = re.compile(r'[^a-zA-Z0-9]')
SEED = 42

"""
This script performs near-deduplication using datasketch library (MinHash and LSH).
Based on BigCode Project data processing pipeline.
"""

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if root_i < root_j:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j

def _create_token_ngram_minhash(doc_tuple):
    index, content = doc_tuple
    num_perm = DEDUPE_CONFIG['num_perm']
    ngram_size = DEDUPE_CONFIG['ngram_size']
    
    minhash = MinHash(num_perm=num_perm, seed=SEED)
    tokens = [t for t in NON_ALPHA.split(content) if t]

    if len(tokens) < ngram_size:
        return index, None

    ngrams = (" ".join(t) for t in zip(*[tokens[i:] for i in range(ngram_size)]))
    for gram in ngrams:
        minhash.update(gram.encode('utf-8'))
        
    return index, minhash

def main():
    """
    Main execution function for token-based near-deduplication.
    """
    start_time = time.time()

    # --- Path and Argument Setup ---
    load_dotenv()
    project_root_str = os.getenv('PROJECT_ROOT')
    if not project_root_str:
        logging.error("'PROJECT_ROOT' environment variable not set. Please check your .env file.")
        return
        
    project_root = Path(project_root_str)
    data_prep_dir = project_root / "scripts" / "data_preparation"

    if len(sys.argv) < 2:
        logging.error("Usage: python 03b_near_deduplication.py <data_suffix>")
        return
    
    filter_level = sys.argv[1]
    logging.info(f"Using data suffix: '{filter_level}'")

    deduplication_dir = data_prep_dir / "03_deduplication" / filter_level
    deduplication_dir.mkdir(parents=True, exist_ok=True)

    input_path = deduplication_dir / f"exact_deduplicated_data_{filter_level}.jsonl"
    output_path = deduplication_dir / f"final_dataset_{filter_level}.jsonl"
    duplicate_log_path = deduplication_dir / f"near_duplicate_log_{filter_level}.csv"
    
    if not input_path.is_file():
        logging.error(f"Input file not found: '{input_path}'")
        logging.error("Please run '03a_exact_deduplication.py' first.")
        return

    # --- 1. Load Data ---
    logging.info(f"Loading data from '{input_path}'...")
    with open(input_path, 'r', encoding='utf-8') as f:
        docs_to_process = [json.loads(line) for line in tqdm(f, desc="Loading documents")]
    total_docs_loaded = len(docs_to_process)
    logging.info(f"Loaded {total_docs_loaded:,} documents for near-duplication.")
    
    if not docs_to_process:
        logging.warning("Input file is empty. Nothing to process.")
        sys.exit(1)

    # --- 2. Deterministic Sorting ---
    logging.info("Sorting documents to ensure replicable near-deduplication...")
    def get_sort_key(doc):
        metrics = doc.get('metrics', {})
        content_length = -metrics.get('content_length')
        alnum_ratio = -metrics.get('alnum_ratio')
        return (content_length, alnum_ratio)
    
    docs_to_process.sort(key=get_sort_key)
    logging.info("Sorting complete.")

    # --- 3. Create Token N-gram MinHash ---
    logging.info("Creating TOKEN n-gram MinHash fingerprints in parallel...")
    minhashes = [None] * total_docs_loaded
    with ProcessPoolExecutor() as executor:
        doc_tuples = list(enumerate(doc['content'] for doc in docs_to_process))
        future_to_index = {executor.submit(_create_token_ngram_minhash, dt): dt[0] for dt in doc_tuples}
        for future in tqdm(as_completed(future_to_index), total=total_docs_loaded, desc="Fingerprinting"):
            index, minhash = future.result()
            minhashes[index] = minhash

    # --- 4. Index MinHashes in LSH ---
    logging.info("Indexing fingerprints in LSH...")
    lsh = MinHashLSH(threshold=DEDUPE_CONFIG['threshold'], num_perm=DEDUPE_CONFIG['num_perm'])
    for i, minhash in enumerate(tqdm(minhashes, desc="Indexing")):
        if minhash:
            lsh.insert(i, minhash)

    # --- 5. Identify and Merge Overlapping Clusters (with Union-Find) ---
    logging.info("Querying LSH and merging overlapping clusters using Union-Find...")
    uf = UnionFind(total_docs_loaded)
    for i, minhash in enumerate(tqdm(minhashes, desc="Clustering")):
        if not minhash:
            continue
        cluster_indices = lsh.query(minhash)
        for member_idx in cluster_indices:
            uf.union(i, member_idx)

    # --- 6. Finalize Clusters, Filter, and Log ---
    logging.info("Finalizing clusters and preparing logs...")
    indices_to_keep = set()
    duplicate_log = []
    
    final_clusters = defaultdict(list)
    for i in range(total_docs_loaded):
        root = uf.find(i)
        final_clusters[root].append(i)

    for root_idx, cluster_indices in final_clusters.items():
        indices_to_keep.add(root_idx)
        if len(cluster_indices) > 1:
            kept_doc = docs_to_process[root_idx]
            kept_file_path = f"{kept_doc['repo_id']}/{kept_doc['path_in_repo']}"
            for dup_idx in cluster_indices:
                if dup_idx != root_idx:
                    dup_doc = docs_to_process[dup_idx]
                    dup_file_path = f"{dup_doc['repo_id']}/{dup_doc['path_in_repo']}"
                    duplicate_log.append((kept_file_path, dup_file_path))

    # --- 7. Save Final Dataset and Logs ---
    logging.info(f"Saving {len(indices_to_keep):,} unique documents to '{output_path}'...")
    final_docs = [docs_to_process[i] for i in sorted(list(indices_to_keep))]

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for doc in final_docs:
            f_out.write(json.dumps(doc) + '\n')
    
    logging.info(f"Writing near-duplicate log with {len(duplicate_log):,} entries...")
    with open(duplicate_log_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['kept_file', 'near_duplicate_of_kept_file'])
        for kept_file, duplicate_file in sorted(duplicate_log):
            writer.writerow([kept_file, duplicate_file])

    # --- 8. Final Summary ---
    num_final_docs = len(indices_to_keep)
    num_duplicates = total_docs_loaded - num_final_docs
    logging.info("--- Near-Deduplication Summary ---")
    logging.info(f"Initial document count (after exact dedupe): {total_docs_loaded:,}")
    logging.info(f"Near-duplicates removed: {num_duplicates:,}")
    logging.info(f"Final unique document count: {num_final_docs:,}")
    logging.info(f"Final dataset saved successfully to: {output_path}")
    logging.info(f"Near-duplicate log saved successfully to: {duplicate_log_path}")
    
    end_time = time.time()
    logging.info(f"Script execution finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()