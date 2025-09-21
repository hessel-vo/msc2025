import config 

import numpy as np
import datasets
from collections import defaultdict
from transformers import TrainerCallback, AutoTokenizer


# --- Step 1: Upfront Data Preparation ---

def load_and_preprocess_data(config, tokenizer):
    """
    Main data loading and preparation function.
    - Loads raw files from JSONL.
    - Processes them into a master pool of chunks, grouped by repository.
    - Splits this master pool into training and validation sets.
    - Returns the training pool (Python dict) and a ready-to-use HF Dataset for validation.
    """
    print("--- Loading and Pre-processing All Data into Master Pool ---")
    all_chunks_by_repo = _preprocess_and_chunk_all_data(config, tokenizer)
    
    print(f"validation repo ids: {config.VALIDATION_REPO_IDS}")
    print("\n--- Splitting chunks into Training and Validation sets ---")
    train_chunks_by_repo = {
        repo_id: chunks for repo_id, chunks in all_chunks_by_repo.items()
        if repo_id not in config.VALIDATION_REPO_IDS
    }
    eval_chunks_by_repo = {
        repo_id: chunks for repo_id, chunks in all_chunks_by_repo.items()
        if repo_id in config.VALIDATION_REPO_IDS
    }

    print(f"Training repositories: {len(train_chunks_by_repo)}")
    print(f"Validation repositories: {len(eval_chunks_by_repo)}")

    eval_chunks_flat = [chunk for chunks in eval_chunks_by_repo.values() for chunk in chunks]
    eval_dataset = datasets.Dataset.from_dict({"input_ids": eval_chunks_flat})
    
    print(f"Total training repositories: {len(train_chunks_by_repo)}")
    print(f"Created a static validation dataset with {len(eval_dataset)} chunks.")
    print("--- Data preparation complete ---")
    
    return train_chunks_by_repo, eval_dataset

def _preprocess_and_chunk_all_data(config, tokenizer):
    """Internal helper to load raw data and convert it into a dictionary of chunks."""
    full_dataset = datasets.load_dataset('json', data_files=str(config.DATASET_PATH), split='train', cache_dir=config.HF_CACHE_DIR)
    files_by_repo = defaultdict(list)
    for example in full_dataset:
        files_by_repo[example['repo_id']].append(example)

    all_chunks_by_repo = defaultdict(list)
    REPO_NAME_TOKEN = "<repo_name>"
    FILE_SEP_TOKEN = "<file_sep>"
    END_OF_TEXT_TOKEN = "<endoftext>"

    for repo_id, repo_files in files_by_repo.items():
        repo_content_parts = []
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
        token_ids = tokenizer(repo_full_content, truncation=False, padding=False)['input_ids']

        for i in range(0, len(token_ids), config.MAX_SEQ_LENGTH):
            chunk = token_ids[i : i + config.MAX_SEQ_LENGTH]
            all_chunks_by_repo[repo_id].append(chunk)
            
    return all_chunks_by_repo


# --- Step 2: Epoch Dataset Creation and Callback ---

def create_sampled_epoch_dataset(train_chunks_pool, config):
    """
    A stateless function that takes the master pool of training chunks and
    returns a complete, sampled Hugging Face Dataset for one epoch.
    """
    sampled_chunks = []
    repo_names = list(train_chunks_pool.keys())
    print(repo_names)
    
    for repo_id in repo_names:
        repo_chunks = train_chunks_pool[repo_id]
        if len(repo_chunks) > config.MAX_CHUNKS_PER_REPO:
            indices = np.random.choice(len(repo_chunks), config.MAX_CHUNKS_PER_REPO, replace=False)
            sampled_chunks.extend([repo_chunks[i] for i in indices])
        else:
            sampled_chunks.extend(repo_chunks)
    
    # Create the final Hugging Face Dataset object
    epoch_dataset = datasets.Dataset.from_dict({"input_ids": sampled_chunks})
    return epoch_dataset


class ResamplingCallback(TrainerCallback):
    """
    A TrainerCallback that resamples and replaces the training dataset at the
    beginning of each epoch.
    """
    def __init__(self, train_chunks_pool, config):
        self.train_chunks_pool = train_chunks_pool
        self.config = config

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n--- Resampling data for Epoch {int(state.epoch + 1)} ---")
        if 'trainer' in kwargs:
            trainer = kwargs['trainer']
            new_epoch_dataset = create_sampled_epoch_dataset(self.train_chunks_pool, self.config)
            trainer.train_dataset = new_epoch_dataset
            print(f"New epoch dataset created with {len(new_epoch_dataset)} examples.")
        else:
            print("Warning: Trainer instance not found in callback. Cannot resample dataset.")


# --- Standalone Testing Block ---
if __name__ == "__main__":
    print("="*80)
    print("--- Running data_processing.py in standalone testing mode ---")
    print("="*80)
    
    # 1. Initialization
    print("\n--- [Test 1] Initializing components ---")
    # Seed numpy for reproducible random sampling during this test
    np.random.seed(500)
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, token=config.HF_TOKEN)
    special_tokens_dict = {'additional_special_tokens': ['<repo_name>', '<file_sep>', '<endoftext>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    print("Tokenizer initialized successfully.")

    # 2. Test the main data loading and splitting function
    print("\n--- [Test 2] Running main data loading and preprocessing ---")
    train_chunks_pool, eval_dataset = load_and_preprocess_data(config, tokenizer)
    
    print("\n--- Verification of loaded data ---")
    print(f"Type of train_chunks_pool: {type(train_chunks_pool)}")
    print(f"Number of repositories in training pool: {len(train_chunks_pool)}")
    print(f"Type of eval_dataset: {type(eval_dataset)}")
    print(f"Features in eval_dataset: {eval_dataset.features}")

    # 3. Test the epoch sampling function and its dynamic nature
    print("\n--- [Test 3] Simulating dynamic epoch creation ---")
    epoch_1_dataset = create_sampled_epoch_dataset(train_chunks_pool, config)
    epoch_2_dataset = create_sampled_epoch_dataset(train_chunks_pool, config)
    epoch_3_dataset = create_sampled_epoch_dataset(train_chunks_pool, config)
    epoch_4_dataset = create_sampled_epoch_dataset(train_chunks_pool, config)
    epoch_5_dataset = create_sampled_epoch_dataset(train_chunks_pool, config)

    print("\n--- Verification of epoch datasets ---")
    print(f"Size of Epoch 1 dataset: {len(epoch_1_dataset)}")
    print(f"Size of Epoch 2 dataset: {len(epoch_2_dataset)}")
    print(f"Size of Epoch 3 dataset: {len(epoch_3_dataset)}")
    print(f"Size of Epoch 4 dataset: {len(epoch_4_dataset)}")
    print(f"Size of Epoch 5 dataset: {len(epoch_5_dataset)}")
    
    # Verify that the two epochs have different content due to random sampling
    # We compare the token IDs of the first example in each dataset.
    epoch_1_first_item = epoch_1_dataset[250]['input_ids']
    epoch_2_first_item = epoch_2_dataset[250]['input_ids']
    
    are_different = (epoch_1_first_item != epoch_2_first_item)
    print(f"Are the first examples of Epoch 1 and Epoch 2 different? {'Yes' if are_different else 'No'}")
    assert are_different, "Epoch sampling is not random! Check np.random seeding and logic."
    print("Dynamic sampling confirmed: Subsequent epochs generate different data.")

    # 4. Show a sample of the processed data
    print("\n--- [Test 4] Inspecting sample data outputs ---")
    
    print("\n--- Sample from Evaluation Dataset ---")
    print(f"Total chunks: {len(eval_dataset)}")
    sample_eval_chunk = eval_dataset[0]['input_ids']
    print(f"Length of first chunk: {len(sample_eval_chunk)}")
    print("Decoded first 50 tokens:")
    print(f"'{tokenizer.decode(sample_eval_chunk[:50])}'")

    print("\n--- Sample from a Training Epoch Dataset ---")
    print(f"Total chunks: {len(epoch_1_dataset)}")
    sample_train_chunk = epoch_1_dataset[0]['input_ids']
    print(f"Length of first chunk: {len(sample_train_chunk)}")
    print("Decoded first 50 tokens:")
    print(f"'{tokenizer.decode(sample_train_chunk[:50])}'")

    print("\n" + "="*80)
    print("--- Standalone testing complete. All checks passed. ---")
    print("="*80)