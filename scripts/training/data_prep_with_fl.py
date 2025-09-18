"""
This script is a specialized tool for analyzing the file distribution of the
ENTIRE corpus used in a training run (both training and validation data).

It performs the following steps:
1.  Loads the full dataset and splits it into raw training and validation sets.
2.  Seeds the random number generator for a single, reproducible run.
3.  Applies the `apply_dynamic_sampling` function to the raw training data to
    generate ONE representative epoch's worth of training files.
4.  Iterates through this sampled training set and reconstructs the full,
    absolute file paths for each file.
5.  Iterates through the ENTIRE raw validation set and reconstructs the full,
    absolute file paths for each file.
6.  Combines both lists of paths.
7.  Writes the final, comprehensive list to a text file (`full_corpus_file_list.txt`).

This output file can then be used directly with tools like 'cloc' to
analyze the lines-of-code distribution of the entire corpus.

Example usage with cloc:
cloc --list-file=full_corpus_file_list.txt --by-file --json --out=cloc_results.json
"""
import datasets
import config
from tqdm import tqdm
import numpy as np

# This is the base path where your repositories are cloned.
REPOSITORIES_BASE_PATH = config.PROJECT_ROOT / "repositories" / "all_repos"

def load_and_split_data(config):
    """
    Loads the dataset from the specified JSONL file and splits it into
    training and evaluation sets based on repository IDs.
    (Copied verbatim from data_processing.py)
    """
    print("--- Loading and splitting data ---")
    full_dataset = datasets.load_dataset(
        'json', data_files=str(config.DATASET_PATH), split='train'
    )
    print(f"Total examples loaded: {len(full_dataset)}")

    raw_train_dataset = full_dataset.filter(
        lambda example: example['repo_id'] not in config.VALIDATION_REPO_IDS
    )
    raw_eval_dataset = full_dataset.filter(
        lambda example: example['repo_id'] in config.VALIDATION_REPO_IDS
    )

    print(f"Raw training set size: {len(raw_train_dataset)} examples")
    print(f"Raw evaluation set size: {len(raw_eval_dataset)} examples")
    print("--- Data loading and splitting complete ---")
    return raw_train_dataset, raw_eval_dataset

def apply_dynamic_sampling(raw_train_dataset, config):
    """
    Balances and randomizes the training data for a single epoch by sampling
    from large repositories and shuffling the file order for all.
    (Copied verbatim from data_processing.py)
    """
    print("\n--- Applying dynamic sampling and shuffling for one epoch ---")
    unique_repo_ids = sorted(list(set(raw_train_dataset['repo_id'])))
    print(f"Found {len(unique_repo_ids)} unique repositories in the training set.")

    sampled_datasets = []
    for repo_id in tqdm(unique_repo_ids, desc="Sampling and shuffling repositories"):
        repo_dataset = raw_train_dataset.filter(lambda x: x['repo_id'] == repo_id)
        shuffled_repo_dataset = repo_dataset.shuffle()

        if len(shuffled_repo_dataset) > config.MAX_FILES_PER_REPO:
            sampled_repo_dataset = shuffled_repo_dataset.select(range(config.MAX_FILES_PER_REPO))
        else:
            sampled_repo_dataset = shuffled_repo_dataset
        
        sampled_datasets.append(sampled_repo_dataset)

    final_epoch_dataset = datasets.concatenate_datasets(sampled_datasets)
    
    print(f"Total examples in this epoch's dataset: {len(final_epoch_dataset)}")
    print("--- Dynamic sampling and shuffling complete ---")
    return final_epoch_dataset


# --- Main execution block for analysis ---
if __name__ == "__main__":
    print("Running analysis script to generate a file list for the full training corpus.")
    
    # Seed NumPy to ensure the sampling part of this analysis is reproducible.
    print(f"\n--- Seeding NumPy for reproducible sampling with SEED={config.SEED} ---")
    np.random.seed(config.SEED)

    # 1. Load BOTH the raw training and validation data
    raw_train, raw_eval = load_and_split_data(config)

    # 2. Apply dynamic sampling to the training data
    sampled_train_dataset = apply_dynamic_sampling(raw_train, config)
    
    # 3. Reconstruct full file paths from the SAMPLED training set
    print("\n--- Reconstructing file paths from the sampled training set ---")
    train_file_paths = []
    for example in tqdm(sampled_train_dataset, desc="Extracting train paths"):
        full_path = REPOSITORIES_BASE_PATH / example['repo_id'] / example['path_in_repo']
        train_file_paths.append(str(full_path))

    # 4. Reconstruct full file paths from the FULL validation set
    print("\n--- Reconstructing file paths from the validation set ---")
    eval_file_paths = []
    for example in tqdm(raw_eval, desc="Extracting eval paths"):
        full_path = REPOSITORIES_BASE_PATH / example['repo_id'] / example['path_in_repo']
        eval_file_paths.append(str(full_path))

    # 5. Combine the lists and write to a single file
    all_paths_to_analyze = train_file_paths + eval_file_paths
    output_filename = "full_corpus_file_list.txt"
    
    print(f"\n--- Writing {len(all_paths_to_analyze)} total file paths to '{output_filename}' ---")
    print(f"  ({len(train_file_paths)} from the sampled train set)")
    print(f"  ({len(eval_file_paths)} from the validation set)")
    
    with open(output_filename, "w") as f:
        for path in all_paths_to_analyze:
            f.write(f"{path}\n")
            
    print("\n--- Analysis script complete ---")
    print(f"Successfully created '{output_filename}'.")
    print("You can now use this file with cloc to analyze the distribution of the entire corpus.")
    print("\nExample cloc command:")
    print(f"cloc --list-file={output_filename}")