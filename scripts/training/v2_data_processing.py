import config
import datasets
from transformers import AutoTokenizer
from collections import defaultdict
import pandas as pd
import random

def preprocess_and_chunk_all_data(config, tokenizer):
    """
    Loads the entire dataset, processes it, and returns an in-memory dictionary
    of all chunks grouped by repository ID.
    """
    print("--- [Step 1] Loading and Pre-processing All Data into Master Pool ---")
    full_dataset = datasets.load_dataset(
        'json', data_files=str(config.DATASET_PATH), split='train'
    )
    files_by_repo = defaultdict(list)
    for example in full_dataset:
        files_by_repo[example['repo_id']].append(example)
    print(f"Found {len(files_by_repo)} unique repositories.")

    all_chunks_by_repo = defaultdict(list)
    REPO_NAME_TOKEN = "<repo_name>"
    FILE_SEP_TOKEN = "<file_sep>"
    END_OF_TEXT_TOKEN = "<endoftext>"
    
    print("\n--- Tokenizing and chunking all repositories... ---")
    include_metadata = True 
    for repo_id, repo_files in files_by_repo.items():
        repo_content_parts = []
        if include_metadata:
            repo_header = f"{REPO_NAME_TOKEN}{repo_id}"
            for file_example in repo_files:
                file_str = f"{FILE_SEP_TOKEN}{file_example['path_in_repo']}\n{file_example['content']}"
                repo_content_parts.append(file_str)
            repo_full_content = repo_header + "".join(repo_content_parts)
        else:
            for file_example in repo_files:
                file_str = f"{FILE_SEP_TOKEN}{file_example['content']}"
                repo_content_parts.append(file_str)
            repo_full_content = "".join(repo_content_parts)
        repo_full_content += END_OF_TEXT_TOKEN
        
        token_ids = tokenizer(repo_full_content, truncation=False, padding=False)['input_ids']

        for i in range(0, len(token_ids), config.MAX_SEQ_LENGTH):
            chunk = token_ids[i : i + config.MAX_SEQ_LENGTH]
            all_chunks_by_repo[repo_id].append(chunk)

    print("--- Pre-processing and chunking complete ---")
    return all_chunks_by_repo

def apply_max_chunk_sampling(all_chunks_by_repo, config):
    """
    Samples chunks from the master pool to simulate a single training epoch,
    respecting the MAX_CHUNKS_PER_REPO limit.
    """
    print(f"\n--- [Step 2] Applying Max Chunks Per Repo Sampling (Cap = {config.MAX_CHUNKS_PER_REPO}) ---")
    
    sampled_chunks_for_epoch = defaultdict(list)
    
    for repo_id, chunks in all_chunks_by_repo.items():
        if len(chunks) > config.MAX_CHUNKS_PER_REPO:
            # Randomly sample to get a different set each time this would be run
            sampled_chunks_for_epoch[repo_id] = random.sample(chunks, config.MAX_CHUNKS_PER_REPO)
        else:
            # If the repo is smaller than the cap, take all its chunks
            sampled_chunks_for_epoch[repo_id] = chunks
            
    print("--- Sampling complete ---")
    return sampled_chunks_for_epoch

def analyze_chunk_distribution(sampled_chunks_by_repo):
    """
    Analyzes and prints a report on the distribution of a sampled set of chunks.
    """
    print("\n--- [Step 3] Analyzing Sampled Epoch Distribution ---")
    
    repo_stats = []
    total_chunks = 0
    for repo_id, chunks in sampled_chunks_by_repo.items():
        num_chunks = len(chunks)
        repo_stats.append({'repo_id': repo_id, 'num_chunks': num_chunks})
        total_chunks += num_chunks
        
    if total_chunks == 0:
        print("No chunks were created. Please check the dataset and processing logic.")
        return

    df = pd.DataFrame(repo_stats)
    # Filter out repos that have zero chunks in this sample (though our current logic includes all)
    df = df[df['num_chunks'] > 0]
    df['percentage'] = (df['num_chunks'] / total_chunks * 100).round(2)
    df = df.sort_values(by='num_chunks', ascending=False)
    
    print(f"Total number of chunks in this simulated epoch: {total_chunks}")
    print("Distribution of chunks per repository for this epoch:")
    
    pd.set_option('display.max_rows', None)
    print(df)
    pd.reset_option('display.max_rows')

    print("\n--- Statistical Summary of Chunks per Repository (Post-Sampling) ---")
    print(df['num_chunks'].describe())
    
    print("\n--- Analysis Complete ---")


# --- Testing Block ---
if __name__ == "__main__":
    print("Running data_processing.py in standalone mode for analysis.")
    
    # Set a seed for reproducibility of the random sampling in this test run
    random.seed(config.SEED)
    
    print("\n--- Initializing tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, token=config.HF_TOKEN)
    special_tokens_dict = {'additional_special_tokens': ['<repo_name>', '<file_sep>', '<endoftext>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    print("Tokenizer initialized with special tokens.")

    # Step 1: Create the master pool of all chunks
    all_chunks = preprocess_and_chunk_all_data(config, tokenizer)

    print(f"\nTotal repositories in master pool: {len(all_chunks)}")
    total_chunks_in_pool = sum(len(chunks) for chunks in all_chunks.values())
    print(f"Total chunks in master pool: {total_chunks_in_pool}")
    
    # Step 2: Simulate a single epoch by sampling from the master pool
    sampled_epoch_chunks = apply_max_chunk_sampling(all_chunks, config)
    
    # Step 3: Analyze the result of the sampling
    analyze_chunk_distribution(sampled_epoch_chunks)