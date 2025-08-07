import logging
from pathlib import Path
import time
import os
from dotenv import load_dotenv
import json
import sys
from collections import defaultdict
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasketch import MinHash
from datasketch.lsh import MinHashLSH
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Deduplication Constants (Aligned with The Stack) ---
DEDUPE_CONFIG = {
    "description": "Hybrid approach: Tokenization for candidates, N-grams for verification.",
    "threshold": 0.90,
    "num_perm": 256,
    "ngram_size": 13, # For the verification stage
}

# --- Tokenization Setup (from BigCode) ---
NON_ALPHA = re.compile("[^A-Za-z_0-9]")

def _tokenize(text: str):
    """Splits a string into a set of unique tokens."""
    return set(t for t in NON_ALPHA.split(text) if len(t.strip()) > 0)

def _create_token_minhash(doc_tuple):
    """WORKER 1: Creates a MinHash based on the SET OF TOKENS."""
    index, doc = doc_tuple
    num_perm = DEDUPE_CONFIG['num_perm']
    
    minhash = MinHash(num_perm=num_perm)
    tokens = _tokenize(doc['content'])
    for token in tokens:
        minhash.update(token.encode('utf-8'))
    return index, minhash

def _create_ngram_minhash(doc_tuple):
    """WORKER 2: Creates a MinHash based on CHARACTER N-GRAMS."""
    index, doc = doc_tuple
    num_perm = DEDUPE_CONFIG['num_perm']
    ngram_size = DEDUPE_CONFIG['ngram_size']
    
    minhash = MinHash(num_perm=num_perm)
    content = doc['content']
    ngrams = ["".join(i) for i in zip(*[content[i:] for i in range(ngram_size)])]
    for gram in ngrams:
        minhash.update(gram.encode('utf-8'))
    return index, minhash

def main():
    """
    Main execution function for two-stage hybrid near-deduplication.
    """
    start_time = time.time()

    # --- Path & Argument Setup ---
    # ... (This section remains unchanged)
    load_dotenv()
    project_root_str = os.getenv('PROJECT_ROOT')
    if not project_root_str:
        logging.error("'PROJECT_ROOT' environment variable not set. Please check your .env file.")
        return
        
    project_root = Path(project_root_str)
    data_prep_dir = project_root / "scripts" / "data_preparation"

    if len(sys.argv) < 2:
        logging.error("Usage: python 03_deduplication.py <data_suffix>")
        return
    
    filter_level = sys.argv[1]
    logging.info(f"Using data suffix: '{filter_level}'")
    
    processing_dir = data_prep_dir / "02_processing" / filter_level
    deduplication_dir = data_prep_dir / "03_deduplication" / filter_level
    deduplication_dir.mkdir(parents=True, exist_ok=True)

    input_path = processing_dir / f"processed_data_{filter_level}.jsonl"
    output_path = deduplication_dir / f"final_dataset_{filter_level}.jsonl"
    duplicate_log_path = deduplication_dir / f"duplicate_log_{filter_level}.csv"
    
    if not input_path.is_file():
        logging.error(f"Input file not found: '{input_path}'")
        return


    # --- 1. Load Data ---
    # ... (This section remains unchanged)
    logging.info(f"Loading processed data from '{input_path}'...")
    with open(input_path, 'r', encoding='utf-8') as f:
        docs_to_process = [json.loads(line) for line in tqdm(f, desc="Loading documents")]
    total_docs_loaded = len(docs_to_process)
    logging.info(f"Loaded {total_docs_loaded:,} documents.")

    if not docs_to_process:
        logging.warning("Input file is empty. Nothing to process.")
        return


    # --- 2. Deterministic Sorting ---
    # ... (This section remains unchanged)
    logging.info("Sorting documents to ensure replicable deduplication...")
    docs_to_process.sort(
        key=lambda x: (-x['metrics']['content_length'], -x['metrics']['alnum_ratio'], x['path_in_repo'])
    )
    logging.info("Sorting complete.")


    # --- STAGE 1: Candidate Generation with Token MinHash ---
    # ... (Steps 3, 4, and 5 for candidate generation remain unchanged)
    logging.info("--- STAGE 1: Finding candidate duplicate clusters (with Tokenization) ---")
    logging.info("Creating token-based MinHash fingerprints in parallel...")
    token_minhashes = [None] * total_docs_loaded
    with ProcessPoolExecutor() as executor:
        doc_tuples = list(enumerate(docs_to_process))
        future_to_index = {executor.submit(_create_token_minhash, dt): dt[0] for dt in doc_tuples}
        for future in tqdm(as_completed(future_to_index), total=total_docs_loaded, desc="Fingerprinting (Tokens)"):
            index, minhash = future.result()
            token_minhashes[index] = minhash

    logging.info("Indexing token-based fingerprints in LSH...")
    lsh = MinHashLSH(threshold=DEDUPE_CONFIG['threshold'], num_perm=DEDUPE_CONFIG['num_perm'])
    for i, minhash in enumerate(tqdm(token_minhashes, desc="Indexing (Tokens)")):
        if minhash is not None:
            lsh.insert(i, minhash)

    logging.info("Querying LSH to identify candidate clusters...")
    processed_indices = set()
    candidate_clusters = []
    for i, minhash in enumerate(tqdm(token_minhashes, desc="Clustering (Tokens)")):
        if i in processed_indices or minhash is None:
            continue
        cluster = lsh.query(minhash)
        candidate_clusters.append(cluster)
        processed_indices.update(cluster)


    # --- STAGE 2: Verification and FINAL Logging ---
    logging.info(f"--- STAGE 2: Verifying {len(candidate_clusters):,} candidate clusters (with N-grams) ---")
    
    # MODIFIED: Logic is now based on a Union-Find data structure for correctness
    parent = list(range(total_docs_loaded))
    def find_set(v):
        if v == parent[v]:
            return v
        parent[v] = find_set(parent[v])
        return parent[v]
    def unite_sets(a, b):
        a = find_set(a)
        b = find_set(b)
        if a != b:
            # Union by rank/size could be an optimization, but min() is fine for correctness
            parent[max(a, b)] = min(a, b)

    # --- 6. Verify each cluster and build the true, disjoint clusters ---
    with ProcessPoolExecutor() as executor:
        for cluster in tqdm(candidate_clusters, desc="Verifying Clusters"):
            if len(cluster) <= 1:
                continue
            
            cluster_docs = [(i, docs_to_process[i]) for i in cluster]
            ngram_minhashes = {index: minhash for index, minhash in executor.map(_create_ngram_minhash, cluster_docs)}

            # Perform pairwise verification within the candidate cluster
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    idx1 = cluster[i]
                    idx2 = cluster[j]
                    
                    minhash1 = ngram_minhashes.get(idx1)
                    minhash2 = ngram_minhashes.get(idx2)

                    if minhash1 and minhash2 and minhash1.jaccard(minhash2) >= DEDUPE_CONFIG['threshold']:
                        unite_sets(idx1, idx2)

    # --- 7. Finalize Clusters, Filter, and Log ---
    logging.info("Finalizing clusters and preparing logs...")
    final_clusters = defaultdict(list)
    for i in range(total_docs_loaded):
        final_clusters[find_set(i)].append(i)

    indices_to_keep = set()
    duplicate_log = []

    for root_idx, cluster_indices in final_clusters.items():
        # The root_idx is always the smallest due to our sort and the unite_sets logic
        indices_to_keep.add(root_idx)
        
        kept_file_path = f"{docs_to_process[root_idx]['repo_id']}/{docs_to_process[root_idx]['path_in_repo']}"
        for dup_idx in cluster_indices:
            if dup_idx != root_idx:
                dup_file_path = f"{docs_to_process[dup_idx]['repo_id']}/{docs_to_process[dup_idx]['path_in_repo']}"
                duplicate_log.append((kept_file_path, dup_file_path))

    # --- 8. Save Final Dataset and Logs ---
    logging.info(f"Saving {len(indices_to_keep):,} unique documents to '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i in sorted(list(indices_to_keep)):
            f_out.write(json.dumps(docs_to_process[i]) + '\n')

    logging.info(f"Writing duplicate log with {len(duplicate_log):,} entries...")
    with open(duplicate_log_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['kept_file', 'duplicate_of_kept_file'])
        for kept_file, duplicate_file in sorted(duplicate_log):
            writer.writerow([kept_file, duplicate_file])

    # --- 9. Final Summary ---
    num_final_docs = len(indices_to_keep)
    num_duplicates = total_docs_loaded - num_final_docs # This count is now consistent with the log
    logging.info("--- Deduplication Summary ---")
    logging.info(f"Initial document count: {total_docs_loaded:,}")
    logging.info(f"Duplicates removed (after verification): {num_duplicates:,}")
    logging.info(f"Final unique document count: {num_final_docs:,}")
    logging.info(f"✅ Final dataset saved successfully to: {output_path}")
    logging.info(f"✅ Duplicate log saved successfully to: {duplicate_log_path}")
    
    end_time = time.time()
    logging.info(f"Script execution finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()