import config

import numpy as np
import datasets
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import TrainerCallback, AutoTokenizer


# Extra function for analyzing repo sizes
import numpy as np

import pandas as pd
import numpy as np


def _sanity_check_splits(train_chunks_by_repo, eval_chunks_by_repo, validation_repo_ids):
    train_repos = set(train_chunks_by_repo.keys())
    eval_repos = set(eval_chunks_by_repo.keys())

    overlap = train_repos & eval_repos
    assert not overlap, f"Train/Eval repo overlap detected: {sorted(list(overlap))[:10]}"

    missing = [rid for rid in validation_repo_ids if rid not in eval_repos]
    if missing:
        print(f"[WARN] Some intended validation repos not found in dataset: {missing}")

    n_train_chunks = sum(len(v) for v in train_chunks_by_repo.values())
    n_eval_chunks = sum(len(v) for v in eval_chunks_by_repo.values())
    total = n_train_chunks + n_eval_chunks
    share = (n_eval_chunks / max(1, total)) * 100.0
    print(f"[SPLIT] train repos={len(train_repos)} eval repos={len(eval_repos)}")
    print(f"[SPLIT] train_chunks={n_train_chunks} eval_chunks={n_eval_chunks} (eval={share:.2f}% of total)")

    top_eval = sorted(((rid, len(v)) for rid, v in eval_chunks_by_repo.items()), key=lambda x: -x[1])[:5]
    print("[SPLIT] Top-5 eval repos by chunks:", top_eval)


def analyze_repo_distribution(train_chunks_by_repo, eval_chunks_by_repo,
                              cap = None,
                              percentiles=(50, 75, 90, 95, 99)):
    # Combine train + validation counts per repo
    combined_counts: dict[str, int] = {}
    for rid, chunks in train_chunks_by_repo.items():
        combined_counts[rid] = combined_counts.get(rid, 0) + len(chunks)
    for rid, chunks in eval_chunks_by_repo.items():
        combined_counts[rid] = combined_counts.get(rid, 0) + len(chunks)

    # Apply cap
    if cap is not None:
        combined_counts = {rid: min(c, int(cap)) for rid, c in combined_counts.items()}

    rows = [(rid, c) for rid, c in combined_counts.items()]
    df = pd.DataFrame(rows, columns=["repo_id", "num_chunks"])

    print("\n=== Repository Distribution (train+valid combined)"
          + (f" — AFTER CAP={cap}" if cap is not None else " — BEFORE CAP")
          + " ===")

    if df.empty:
        print("  (No repositories found.)")
        return

    df = df.sort_values(by=["num_chunks", "repo_id"], ascending=[False, True]).reset_index(drop=True)
    total_chunks = int(df["num_chunks"].sum())
    df["percentage"] = (df["num_chunks"] / total_chunks * 100.0).round(2)

    # Print dataframe chunk overview
    pd.set_option("display.max_rows", None)
    print(f"  repos={len(df)}  total_chunks={total_chunks}")
    print("\n  Per-repo breakdown:")
    print(df)

    # Percentiles overview
    counts = df["num_chunks"].to_numpy(dtype=np.int64)
    print("\nPercentiles over num_chunks:")
    pct_vals = np.percentile(counts, percentiles, method="linear")
    for p, v in zip(percentiles, pct_vals):
        p_str = f"{str(p)}"
        print(f"    {p_str}= {v}")


def load_and_preprocess_data(config, tokenizer):
    print("--- Loading and Pre-processing All Data into Training Pool ---")
    all_chunks_by_repo = _preprocess_and_chunk_all_data(config, tokenizer)

    print("\n--- Splitting chunks into Training and Validation sets ---")
    train_chunks_by_repo = {
        repo_id: chunks for repo_id, chunks in all_chunks_by_repo.items()
        if repo_id not in config.VALIDATION_REPO_IDS
    }
    eval_chunks_by_repo = {
        repo_id: chunks for repo_id, chunks in all_chunks_by_repo.items()
        if repo_id in config.VALIDATION_REPO_IDS
    }

    analyze_repo_distribution(train_chunks_by_repo, eval_chunks_by_repo)
    analyze_repo_distribution(train_chunks_by_repo, eval_chunks_by_repo, cap=config.MAX_CHUNKS_PER_REPO)

    print(f"Training repositories: {len(train_chunks_by_repo)}")
    print(f"Validation repositories: {len(eval_chunks_by_repo)}")

    eval_chunks_flat = [chunk for chunks in eval_chunks_by_repo.values() for chunk in chunks]
    eval_dataset = datasets.Dataset.from_dict({"input_ids": eval_chunks_flat})

    print(f"Total training repositories: {len(train_chunks_by_repo)}")
    print(f"Created a static validation dataset with {len(eval_dataset)} chunks.")
    print("--- Data preparation complete ---")

    _sanity_check_splits(train_chunks_by_repo, eval_chunks_by_repo, config.VALIDATION_REPO_IDS)

    return train_chunks_by_repo, eval_dataset


def _preprocess_and_chunk_all_data(config, tokenizer):
    # Helper function to preprocess data

    full_dataset = datasets.load_dataset(
        'json',
        data_files=str(config.DATASET_PATH),
        split='train',
        cache_dir=config.HF_CACHE_DIR
    )
    files_grouped_by_repo = defaultdict(list)
    for example in full_dataset:
        files_grouped_by_repo[example['repo_id']].append(example)

    chunks_grouped_by_repo = defaultdict(list)
    REPO_NAME_TOKEN = "<repo_name>"
    FILE_SEP_TOKEN = "<file_sep>"
    END_OF_TEXT_TOKEN = "<endoftext>"
    
    eval_metadata_count = 0
    for repo_id, repo_files in files_grouped_by_repo.items():
        repo_content_parts = []

        # Applying StarCoder2 format, 50% chance to include repository metadata
        include_metadata = np.random.rand() < 0.5
        if include_metadata:
            if repo_id in config.VALIDATION_REPO_IDS:
                eval_metadata_count += 1
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
            chunk = token_ids[i: i + config.MAX_SEQ_LENGTH]
            chunks_grouped_by_repo[repo_id].append(chunk)
    print(f"Total metadata-included validation repos: {eval_metadata_count}")
    return chunks_grouped_by_repo


class BalancedRollingRepoDataset(Dataset):
    # Custom dataset to regularize/balance large repository contributions to training data

    def __init__(self, chunks_by_repo, max_chunks_per_repo, base_seed=42):
        self.repo_to_chunks = chunks_by_repo
        self.per_repo_quota_cap = max_chunks_per_repo
        self.base_seed = base_seed

        self.ordered_repo_ids = list(self.repo_to_chunks.keys())

        # Preprocessing: determine repo sizes, quotas, permutation
        rng = np.random.default_rng(self.base_seed)
        self.repo_size_map = {rid: len(self.repo_to_chunks[rid]) for rid in self.ordered_repo_ids}
        self.per_repo_quota = {rid: min(self.repo_size_map[rid], self.per_repo_quota_cap)
                               for rid in self.ordered_repo_ids}
        self.per_repo_perm = {
            rid: rng.permutation(self.repo_size_map[rid]).tolist()
            for rid in self.ordered_repo_ids
        }

        # Set dataset size (stays constant across epochs)
        self._fixed_epoch_length = sum(self.per_repo_quota[rid] for rid in self.ordered_repo_ids)

        self._current_epoch_index_map = []
        # Apply initial sampling
        self.resample(epoch_index=0)

    def __len__(self):
        return self._fixed_epoch_length

    def __getitem__(self, flat_index):
        repo_id, repo_local_idx = self._current_epoch_index_map[flat_index]
        return {"input_ids": self.repo_to_chunks[repo_id][repo_local_idx]}

    def _per_repo_window(self, repo_id, epoch_index):
        # Helper function to map epoch to new window to ensure we iterate over larger repos without repetition
        size = self.repo_size_map[repo_id]
        quota = self.per_repo_quota[repo_id]
        perm = self.per_repo_perm[repo_id]

        if size <= quota:
            # Small repos contribute all items every epoch
            return perm

        start = (epoch_index * quota) % size
        end = start + quota
        if end <= size:
            return perm[start:end]
        else:
            wrap = end - size
            return perm[start:] + perm[:wrap]

    def resample(self, epoch_index: int):
        # Reconstruct the actual dataset for a new epoch
        flat_index_map = []
        for rid in self.ordered_repo_ids:
            repo_indices_for_epoch = self._per_repo_window(rid, epoch_index)
            flat_index_map.extend((rid, int(j)) for j in repo_indices_for_epoch)

        # Safety check, length should remain constant
        assert len(flat_index_map) == self._fixed_epoch_length, "Epoch length changed unexpectedly."
        self._current_epoch_index_map = flat_index_map


class AdvanceEpochWindows(TrainerCallback):
    # Custom callback to use per epoch
    def __init__(self, rolling_dataset: BalancedRollingRepoDataset):
        self.rolling_dataset = rolling_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch_idx = int(state.epoch) if state.epoch is not None else 0
        print(f"\n--- Advancing rolling windows for Epoch {epoch_idx + 1} ---")
        self.rolling_dataset.resample(epoch_index=epoch_idx)


def create_balanced_rolling_dataset(train_chunks_by_repo, config, base_seed=42):
    return BalancedRollingRepoDataset(
        chunks_by_repo=train_chunks_by_repo,
        max_chunks_per_repo=config.MAX_CHUNKS_PER_REPO,
        base_seed=base_seed,
    )


if __name__ == "__main__":
    # The entire main function is used for testing, not used in training process
    print("=" * 80)
    print("--- Testing data_processing.py functionality ---")
    print("=" * 80)

    print("\n--- [Test 1] Initializing components ---")
    np.random.seed(config.SEED)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, token=config.HF_TOKEN)
    special_tokens_dict = {'additional_special_tokens': ['<repo_name>', '<file_sep>', '<endoftext>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    print("Tokenizer initialized successfully.")

    print("\n--- [Test 2] Running main data loading and preprocessing ---")
    train_chunks_by_repo, eval_dataset = load_and_preprocess_data(config, tokenizer)

    print("\n--- Verification of loaded data ---")
    print(f"Type of train_chunks_by_repo: {type(train_chunks_by_repo)}")
    print(f"Number of repositories in training pool: {len(train_chunks_by_repo)}")
    print(f"Type of eval_dataset: {type(eval_dataset)}")
    print(f"Features in eval_dataset: {eval_dataset.features}")

    print("\n--- [Test 3] Building rolling dataset and simulating epochs ---")
    rolling_train_ds = create_balanced_rolling_dataset(train_chunks_by_repo, config, base_seed=1234)
    print(f"Fixed training epoch length: {len(rolling_train_ds)} "
          f"(sum of per-repo min(size, MAX_CHUNKS_PER_REPO))")

    sample_positions = [0, min(10, len(rolling_train_ds) - 1), max(0, len(rolling_train_ds)//2)]
    saved_samples = []

    for epoch_idx in range(5):
        rolling_train_ds.resample(epoch_index=epoch_idx)
        snapshot = [rolling_train_ds[pos]["input_ids"] for pos in sample_positions]
        saved_samples.append(snapshot)
        print(f"Epoch {epoch_idx + 1} sample snapshot collected.")

    # Check that samples differ across epochs
    any_diff = any(saved_samples[e] != saved_samples[0] for e in range(1, len(saved_samples)))
    print(f"Do later epochs differ from Epoch 1 at inspected positions? {'Yes' if any_diff else 'No'}")
    if not any_diff:
        raise AssertionError("Rolling windows did not change content across epochs—check implementation.")

    print("\n--- [Test 4] Inspecting sample data outputs ---")

    print("\n--- Sample from Evaluation Dataset ---")
    print(f"Total eval chunks: {len(eval_dataset)}")
    sample_eval_chunk = eval_dataset[0]['input_ids']
    print(f"Length of first eval chunk: {len(sample_eval_chunk)}")
    print("Decoded first 50 tokens of eval chunk:")
    print(f"'{tokenizer.decode(sample_eval_chunk[:50])}'")

    print("\n--- Sample from Training Rolling Dataset (Epoch 1) ---")
    rolling_train_ds.resample(epoch_index=0)
    sample_train_chunk = rolling_train_ds[0]['input_ids']
    print(f"Length of first train chunk: {len(sample_train_chunk)}")
    print("Decoded first 50 tokens of first train chunk:")
    print(f"'{tokenizer.decode(sample_train_chunk[:50])}'")

    print("\n" + "=" * 80)
    print("--- Standalone testing complete. All checks passed. ---")
    print("=" * 80)
