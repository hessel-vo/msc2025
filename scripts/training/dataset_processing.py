import datasets
import config
import random

def load_and_split_data(config):
    """
    Loads the dataset from the specified JSONL file and splits it into
    training and evaluation sets based on repository IDs.

    Args:
        config: A configuration object containing DATASET_PATH and
                VALIDATION_REPO_IDS.

    Returns:
        A tuple containing two datasets: (raw_train_dataset, raw_eval_dataset).
    """
    print("--- Loading and splitting data ---")
    full_dataset = datasets.load_dataset(
        'json',
        data_files=str(config.DATASET_PATH),
        split='train'
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
    Applies dynamic sampling to the raw training dataset for one epoch.

    It groups the dataset by repository ID and samples up to a maximum
    number of files from each repository. This creates a balanced dataset
    for a single training epoch.

    Args:
        raw_train_dataset: The full, unsampled training dataset.
        config: A configuration object containing MAX_FILES_PER_REPO.

    Returns:
        A new, sampled datasets.Dataset object for the current epoch.
    """
    print(f"\n--- Applying dynamic sampling (max {config.MAX_FILES_PER_REPO} files per repo) ---")
    
    # Group the dataset by the 'repo_id' column.
    # This allows us to process each repository's files as a distinct group.
    grouped_by_repo = raw_train_dataset.group_by('repo_id')

    def sample_repo_files(repo_group):
        """
        A function to be mapped over each group (repository). It samples
        files if the group is larger than the configured maximum.
        """
        num_files = len(repo_group['content'])
        
        # If the number of files is within the limit, return them all.
        if num_files <= config.MAX_FILES_PER_REPO:
            return repo_group

        # If the repository is too large, randomly sample indices.
        # This is more memory-efficient than shuffling the entire group data.
        indices = list(range(num_files))
        random.shuffle(indices)
        sampled_indices = indices[:config.MAX_FILES_PER_REPO]
        
        # Create a new dictionary containing only the sampled data.
        sampled_batch = {key: [values[i] for i in sampled_indices] for key, values in repo_group.items()}
        return sampled_batch

    # Use the .map() function to apply our sampling logic to each group.
    # The `batched=True` and `batch_size=-1` arguments ensure that our
    # `sample_repo_files` function receives each entire repository group as a single batch.
    sampled_dataset = grouped_by_repo.map(
        sample_repo_files,
        batched=True,
        batch_size=-1, # Process one full group at a time
        desc="Sampling files per repository"
    )

    print(f"Total files before sampling: {len(raw_train_dataset)}")
    print(f"Total files after sampling for this epoch: {len(sampled_dataset)}")
    print("--- Dynamic sampling complete ---")
    
    return sampled_dataset


# --- Testing Block ---
if __name__ == "__main__":
    print("Running data_processing.py in standalone mode for testing.")
    raw_train_dataset, raw_eval_dataset = load_and_split_data(config)

    # --- Test Step 2: Dynamic Sampling ---
    sampled_epoch_dataset = apply_dynamic_sampling(raw_train_dataset, config)

    # Verification: Group the *sampled* data and check group sizes.
    print("\nVerifying sampled dataset integrity...")
    sampled_groups = sampled_epoch_dataset.group_by('repo_id')
    
    max_files_found = 0
    all_repos_within_limit = True

    for repo_id in sampled_groups.keys():
        group_size = len(sampled_groups[repo_id])
        if group_size > max_files_found:
            max_files_found = group_size
        
        if group_size > config.MAX_FILES_PER_REPO:
            all_repos_within_limit = False
            print(f"Error: Repo '{repo_id}' has {group_size} files, which exceeds the limit of {config.MAX_FILES_PER_REPO}.")

    assert all_repos_within_limit, "Verification failed: One or more repos exceed the max file limit after sampling."
    
    print(f"Verification successful: All repositories in the sampled dataset have <= {config.MAX_FILES_PER_REPO} files.")
    print(f"The largest repository in the sampled set has {max_files_found} files.")