import datasets
import config
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_split_data(cfg):
    """
    Loads the dataset from the JSONL file and splits it into raw training
    and evaluation sets based on predefined repository IDs.

    Args:
        cfg: The configuration module containing all settings.

    Returns:
        A tuple containing two datasets.hf_hub_download
        (raw_train_dataset, raw_eval_dataset)
    """
    logging.info(f"Loading dataset from: {cfg.DATASET_PATH}")
    
    # Load the entire dataset from the specified JSONL file.
    # The `Dataset` object is memory-mapped, making it efficient for large files.
    full_dataset = datasets.load_dataset('json', data_files=str(cfg.DATASET_PATH), split='train')
    logging.info(f"Successfully loaded {len(full_dataset)} total files.")

    # Flatten the list of validation repository IDs from the mapping.
    # The mapping is structured by language, so we collect all IDs into a single set.
    validation_repo_ids = set()
    for lang_repos in cfg.VALIDATION_SET_MAPPING.values():
        validation_repo_ids.update(lang_repos)
    
    logging.info(f"Using these repository IDs for validation: {sorted(list(validation_repo_ids))}")

    # Use the `.filter()` method to split the data.
    # The filter function checks if a file's 'repo_id' is in our validation set.
    
    # The evaluation set contains only the files from the validation repositories.
    raw_eval_dataset = full_dataset.filter(
        lambda example: example['repo_id'] in validation_repo_ids
    )

    # The training set contains all files *not* in the validation repositories.
    raw_train_dataset = full_dataset.filter(
        lambda example: example['repo_id'] not in validation_repo_ids
    )
    
    logging.info(f"Split dataset: {len(raw_train_dataset)} training files, {len(raw_eval_dataset)} evaluation files.")

    return raw_train_dataset, raw_eval_dataset

# This block allows us to run and test this script directly.
if __name__ == "__main__":
    logging.info("--- Running Data Processing Script in Standalone Mode for Testing ---")
    
    # Call the function to perform the loading and splitting.
    raw_train_data, raw_eval_data = load_and_split_data(config)

    # Print a summary to verify the results.
    print("\n--- Verification ---")
    print(f"Total training files loaded: {len(raw_train_data)}")
    print(f"Total evaluation files loaded: {len(raw_eval_data)}")

    print(raw_train_data)
    print(raw_eval_data)
    
    # Optionally, inspect a single example from each set to ensure correctness.
    if len(raw_train_data) > 0:
        print("\nExample training file:")
        print(f"  Repo ID: {raw_train_data[0]['repo_id']}")
    
    if len(raw_eval_data) > 0:
        print("\nExample evaluation file:")
        print(f"  Repo ID: {raw_eval_data[0]['repo_id']}")

    print("\n--- Initial data loading and splitting test complete. ---")