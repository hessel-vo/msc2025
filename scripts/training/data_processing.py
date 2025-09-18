import datasets
import config
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import sys

def load_and_split_data(config):
    """
    Loads the dataset from the specified JSONL file and splits it into
    training and evaluation sets based on repository IDs.
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
    This function relies on a pre-seeded NumPy global random state for reproducibility.
    """
    print("\n--- Applying dynamic sampling and shuffling for new epoch ---")
    unique_repo_ids = sorted(list(set(raw_train_dataset['repo_id'])))
    print(f"Found {len(unique_repo_ids)} unique repositories in the training set.")

    sampled_datasets = []
    for repo_id in unique_repo_ids:
        repo_dataset = raw_train_dataset.filter(lambda x: x['repo_id'] == repo_id)
        
        # Relies on the global NumPy random state, which should be seeded once
        # at the beginning of the main training script.
        shuffled_repo_dataset = repo_dataset.shuffle()

        if len(shuffled_repo_dataset) > config.MAX_FILES_PER_REPO:
            sampled_repo_dataset = shuffled_repo_dataset.select(range(config.MAX_FILES_PER_REPO))
        else:
            sampled_repo_dataset = shuffled_repo_dataset
        
        sampled_datasets.append(sampled_repo_dataset)

    final_epoch_dataset = datasets.concatenate_datasets(sampled_datasets)
    # --- MODIFIED FOR REPRODUCIBILITY ---
    final_epoch_dataset = final_epoch_dataset.shuffle()

    print(f"Total examples in this epoch's dataset: {len(final_epoch_dataset)}")
    print("--- Dynamic sampling and shuffling complete ---")
    return final_epoch_dataset

def process_dataset_for_training(dataset, tokenizer, config):
    """
    Processes a dataset using the StarCoder2 methodology:
    - Groups files by repository.
    - With 50% probability, prepends repository metadata.
    - Concatenates files with a <file_sep> token.
    - Appends an <endoftext> token after each repository's content.
    - Chunks the result, keeping all data (including partial final chunks).
    This function relies on a pre-seeded NumPy global random state for reproducibility.
    """
    print("\n--- Processing dataset for training (StarCoder2 format) ---")
    
    all_chunks = {'input_ids': []}
    
    REPO_NAME_TOKEN = "<repo_name>"
    FILE_SEP_TOKEN = "<file_sep>"
    END_OF_TEXT_TOKEN = "<endoftext>"
    
    unique_repo_ids = sorted(list(set(dataset['repo_id'])))
    
    for repo_id in unique_repo_ids:
        repo_files = dataset.filter(lambda x: x['repo_id'] == repo_id)
        
        repo_content_parts = []
        
        # StarCoder2: 50% chance to include repository metadata, using NumPy's RNG.
        include_metadata = np.random.rand() < 0.5
        
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
        
        token_ids = tokenizer(
            repo_full_content,
            truncation=False,
            padding=False,
            return_attention_mask=False
        )['input_ids']

        total_length = len(token_ids)

        if total_length == 0:
            continue

        for i in range(0, total_length, config.MAX_SEQ_LENGTH):
            end = i + config.MAX_SEQ_LENGTH
            chunk_ids = token_ids[i:end]
            
            all_chunks['input_ids'].append(chunk_ids)
    
    processed_dataset = datasets.Dataset.from_dict(all_chunks)
    
    print(f"Total training chunks created: {len(processed_dataset)} (including partial chunks)")
    print("--- Dataset processing complete ---")


    return processed_dataset


# --- Testing Block ---
if __name__ == "__main__":
    print("Running data_processing.py in standalone mode for testing.")
    
    # --- ADDED FOR STANDALONE REPRODUCIBILITY ---
    # To make this testing block reproducible, we mimic what train.py will do.
    # In the real run, these lines will be in the main training script.
    print(f"\n--- Seeding NumPy for reproducible testing with SEED={config.SEED} ---")
    np.random.seed(config.SEED)

    raw_train, raw_eval = load_and_split_data(config)

    # print("\n--- RUN 1: Generating first epoch dataset ---")
    # epoch_dataset_1 = apply_dynamic_sampling(raw_train, config)

    
    # print("\n--- RUN 2: Generating second epoch dataset ---")
    # epoch_dataset_2 = apply_dynamic_sampling(raw_train, config)

    # # --- Verification of epoch-to-epoch randomness ---
    # order_is_different = epoch_dataset_1[0]['content'] != epoch_dataset_2[0]['content']
    # print(f"\nIs the order of examples different between epochs? {'Yes' if order_is_different else 'No'}")
    # assert order_is_different, "Epochs are not being shuffled differently! Check seeding."
    # print("Validation successful: Per-epoch data is correctly randomized.")
    
    print("\n--- Initializing tokenizer for testing ---")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, token=config.HF_TOKEN)
    
    special_tokens_dict = {
        'additional_special_tokens': ['<repo_name>', '<file_sep>', '<endoftext>']
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    print("Tokenizer initialized with StarCoder2 special tokens.")

    print("\n--- Processing evaluation dataset for verification ---")
    processed_eval_dataset = process_dataset_for_training(raw_eval, tokenizer, config)

    sys.exit(1)  

    # Process one of the sampled datasets for a final check
    processed_train_dataset = process_dataset_for_training(epoch_dataset_1, tokenizer, config)
    
    print("\n--- Verifying processed dataset ---")
    if len(processed_train_dataset) > 0:
        first_chunk = processed_train_dataset[0]
        last_chunk = processed_train_dataset[-1]
        
        print(f"Number of chunks: {len(processed_train_dataset)}")
        print(f"Keys in a chunk: {list(first_chunk.keys())}")
        print(f"Length of input_ids in first chunk: {len(first_chunk['input_ids'])}")
        print(f"Length of input_ids in last chunk: {len(last_chunk['input_ids'])} (can be partial)")
        
        print("\nDecoding the first 50 tokens of first chunk to check format:")
        decoded_tokens = tokenizer.decode(first_chunk['input_ids'][:50])
        print(decoded_tokens)

        print("\nDecoding the last 50 tokens of last chunk to check format:")
        decoded_tokens = tokenizer.decode(last_chunk['input_ids'][-50:])
        print(decoded_tokens)
        
        print("\nValidation successful: Processed dataset has the correct structure.")
    else:
        print("Warning: No training chunks were created.")

    print("\n--- data_processing.py all tests complete ---")