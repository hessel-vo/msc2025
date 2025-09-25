# main_train.py

import os
import sys
import numpy as np
import torch

import config
import data_processing as dp

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig


# -----------------------
# Step 0: Canary helpers
# -----------------------

def token_accuracy_from_logits(logits, labels, ignore_index=-100):
    """
    Correct next-token accuracy:
      - shift predictions by 1 (compare logits[:, :-1] to labels[:, 1:])
      - ignore positions where labels == ignore_index
    """
    import torch
    with torch.no_grad():
        pred = logits[:, :-1, :].argmax(-1)   # (B, T-1)
        ref  = labels[:, 1:]                  # (B, T-1)
        mask = ref != ignore_index
        if mask.any():
            return (pred[mask] == ref[mask]).float().mean().item()
        return float("nan")


def token_accuracy_unmasked(logits, labels):
    """
    Unmasked (incorrect-by-design) accuracy for A/B comparison.
    If this is much higher than masked accuracy, pads are probably leaking.
    """
    import torch
    with torch.no_grad():
        pred = logits[:, :-1, :].argmax(-1)
        ref  = labels[:, 1:]
        return (pred == ref).float().mean().item()


def token_accuracy_unshifted(logits, labels, ignore_index=-100):
    """
    Accuracy without the one-step shift (incorrect-by-design).
    If this beats the masked, shifted accuracy by a lot, comparisons are misaligned elsewhere.
    """
    import torch
    with torch.no_grad():
        pred = logits.argmax(-1)       # (B, T)
        ref  = labels
        mask = ref != ignore_index
        if mask.any():
            return (pred[mask] == ref[mask]).float().mean().item()
        return float("nan")


def label_pad_sanity(labels, pad_id, ignore_index=-100):
    """
    Quick counts:
      - pad_in_labels: any pad IDs present in labels tensor at all
      - unmasked_pad: pad IDs that are NOT set to ignore_index (-100)
    We want both to be zero when using CLM with padding masked out.
    """
    import torch
    labels = torch.as_tensor(labels)
    pad_in_labels = (labels == pad_id).sum().item()
    unmasked_pad  = ((labels == pad_id) & (labels != ignore_index)).sum().item()
    return pad_in_labels, unmasked_pad


def repetition_rate(labels_tensor, ignore_index=-100):
    """
    Fraction of positions where labels[t+1] == labels[t], ignoring -100.
    High repetition is common in code and helps explain high token accuracy.
    """
    import torch
    labels = labels_tensor
    # valid when both current and next positions are not ignored
    valid = (labels[:, :-1] != ignore_index) & (labels[:, 1:] != ignore_index)
    if not valid.any():
        return float("nan")
    reps = (labels[:, 1:][valid] == labels[:, :-1][valid]).float().mean().item()
    return float(reps)


def _stats(x):
    import statistics as stats
    if not x:
        return {}
    return {
        "mean": float(stats.mean(x)),
        "min": float(min(x)),
        "max": float(max(x)),
        "n":   int(len(x)),
    }


def run_canary_over_k_batches_spread(trainer, pad_id, k=8, ignore_index=-100):
    """
    Runs forward passes on ~K eval batches, spaced across the eval dataloader.
    Prints aggregated stats (+ simple warnings) for:
      - masked_acc, unmasked_acc, unshifted_acc
      - repetition_rate
      - total pad presence in labels
    """
    import torch

    model = trainer.model
    model.eval()
    dl = trainer.get_eval_dataloader()

    # If the dataloader length is known, choose spread indices; else fall back to first K
    try:
        total_batches = len(dl)
    except TypeError:
        total_batches = None

    if total_batches and total_batches > 0:
        k_eff = min(k, total_batches)
        # even spread: 0, step, 2*step, ...
        step = max(1, total_batches // k_eff)
        chosen_idxs = sorted(set(min(i * step, total_batches - 1) for i in range(k_eff)))
    else:
        chosen_idxs = list(range(k))  # best-effort

    masked, unmasked, unshifted, reps = [], [], [], []
    total_pad_in, total_unmasked_pad = 0, 0

    # Iterate once and evaluate only the selected batch indices (spread)
    for bidx, batch in enumerate(dl):
        if bidx not in chosen_idxs:
            continue
        with torch.no_grad():
            batch_on_device = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(**batch_on_device)
            logits = out.logits.detach().cpu()
            labels = batch["labels"].detach().cpu()

        masked.append(token_accuracy_from_logits(logits, labels, ignore_index))
        unmasked.append(token_accuracy_unmasked(logits, labels))
        unshifted.append(token_accuracy_unshifted(logits, labels, ignore_index))
        reps.append(repetition_rate(labels, ignore_index))
        pi, up = label_pad_sanity(labels, pad_id, ignore_index)
        total_pad_in += pi
        total_unmasked_pad += up

    report = {
        "masked_acc_stats": _stats(masked),
        "unmasked_acc_stats": _stats(unmasked),
        "unshifted_acc_stats": _stats(unshifted),
        "repetition_rate_stats": _stats(reps),
        "total_pad_in_labels": int(total_pad_in),
        "total_unmasked_pad": int(total_unmasked_pad),
        "sampled_batch_indices": chosen_idxs,
    }

    print("\n[Canary] Spread multi-batch eval diagnostics:")
    print(report)

    # Heuristic warnings
    flagged = False
    if total_unmasked_pad > 0:
        flagged = True
        print("  ⚠️  Detected PAD tokens in labels that are not ignored (padding leak).")
    # Compare means if available
    m_masked = report["masked_acc_stats"].get("mean", None)
    m_unmasked = report["unmasked_acc_stats"].get("mean", None)
    m_unshifted = report["unshifted_acc_stats"].get("mean", None)
    if m_masked is not None and m_unmasked is not None and m_unmasked > max(m_masked + 0.10, m_masked * 1.25):
        flagged = True
        print("  ⚠️  Unmasked accuracy >> masked accuracy on average (likely counting pads).")
    if m_masked is not None and m_unshifted is not None and m_unshifted > m_masked + 0.05:
        flagged = True
        print("  ⚠️  Unshifted accuracy > masked accuracy (possible shift/misalignment bug).")
    if not flagged:
        print("  ✅  Canary looks good across spread batches.")


# -----------------------
# Step 1: Environment
# -----------------------

def setup_environment():
    """Sets random seeds for reproducibility and ensures output dir exists."""
    print("--- [Step 1] Initializing Setup ---")
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
        # Optional: speed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Random seed set to: {config.SEED}")
    print(f"Output directory set to: {config.OUTPUT_DIR}")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("--- Setup complete ---")


# -----------------------
# Step 2: Model + Tokenizer
# -----------------------

def load_model_and_tokenizer():
    """Loads base model & tokenizer, registers special tokens, resizes embeddings."""
    print("\n--- [Step 2] Loading Tokenizer & Model ---")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        token=config.HF_TOKEN,
    )

    # Add task/domain markers as additional special tokens
    # NOTE: Avoid adding a duplicate EOS like "<endoftext>" as an *additional* token.
    special_tokens_dict = {"additional_special_tokens": ["<repo_name>", "<file_sep>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} new special tokens.")

    # Ensure we have a pad token (common for causal LMs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Using `eos_token` as `pad_token`.")
    tokenizer.padding_side = "right"

    # Base model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        token=config.HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Make checkpointing compatible
    model.config.use_cache = False
    # Ensure model knows the pad id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Resize embeddings to include newly added tokens
    model.resize_token_embeddings(len(tokenizer))
    print(f"Base model '{config.MODEL_ID}' loaded. Token embeddings resized to {len(tokenizer)}.")
    print("--- Model and tokenizer loading complete ---")
    return model, tokenizer


# -----------------------
# Step 3: LoRA
# -----------------------

def apply_lora_to_model(model):
    """Applies LoRA configuration to the model."""
    print("\n--- [Step 3] Configuring and Applying LoRA ---")
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    print("LoRA configuration applied to the model.")
    model.print_trainable_parameters()
    print("--- LoRA setup complete ---")
    return model


# -----------------------
# Step 4: Data
# -----------------------

def prepare_datasets(tokenizer):
    """
    Uses the rolling, in-place dataset from data_processing.py
    and returns (train_dataset, eval_dataset, advance_callback).
    """
    print("\n--- [Step 4] Preparing Datasets ---")
    train_chunks_by_repo, eval_dataset = dp.load_and_preprocess_data(config, tokenizer)

    # Rolling train dataset (fixed length per epoch, per-repo window advances each epoch)
    rolling_train_ds = dp.BalancedRollingRepoDataset(
        chunks_by_repo=train_chunks_by_repo,
        max_chunks_per_repo=config.MAX_CHUNKS_PER_REPO,
        base_seed=config.SEED,
    )

    # Callback that advances the dataset window each epoch (no replacement of dataset object)
    advance_callback = dp.AdvanceEpochWindows(rolling_train_ds)

    # Eval dataset is a static HF Dataset built in load_and_preprocess_data
    print("--- Data preparation complete ---")
    return rolling_train_ds, eval_dataset, advance_callback


# -----------------------
# Step 5: Trainer
# -----------------------

def build_trainer(model, tokenizer, train_dataset, eval_dataset, advance_callback):
    """Configures the HF Trainer with rolling dataset + LoRA + early stopping."""
    print("\n--- [Step 5] Configuring Hugging Face Trainer ---")

    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        logging_steps=config.LOGGING_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        evaluation_strategy="steps",        # <-- correct key
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.EVAL_STEPS,
        save_total_limit=4,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        bf16=True,
        # If you later compute logits via Trainer APIs, set:
        # prediction_loss_only=False,
        # eval_accumulation_steps=2,
        report_to="none",  # set to "tensorboard"/"wandb"/"mlflow" if desired
    )

    # Collator for causal LM (pads per batch; labels from input_ids)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback, advance_callback],
    )
    return trainer


# -----------------------
# Step 6: Orchestration
# -----------------------

def main():
    """
    Orchestrates setup, data, trainer, and training loop.
    """
    setup_environment()
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora_to_model(model)

    print(model.config)
    print(getattr(model.config, "attn_implementation", None))

    train_dataset, eval_dataset, advance_callback = prepare_datasets(tokenizer)
    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset, advance_callback)

    # --- Canary: run a spread multi-batch eval sanity check BEFORE training ---
    print("\n--- [Canary] Spread multi-batch evaluation sanity check ---")
    if eval_dataset is not None and len(eval_dataset) > 0:
        K = getattr(config, "CANARY_EVAL_BATCHES", 8)  # tweakable from config
        run_canary_over_k_batches_spread(trainer, pad_id=tokenizer.pad_token_id, k=K, ignore_index=-100)
    else:
        print("[Canary] No eval dataset available; skipping canary.")

    print("\n--- [Step 6] Training ---")
    train_output = trainer.train()
    print("Training complete.")
    print(f"Final training loss: {train_output.training_loss:.6f}")

    print("\n--- [Step 7] Saving adapter + tokenizer ---")
    final_model_dir = config.OUTPUT_DIR / "final_adapter"
    final_tokenizer_dir = config.OUTPUT_DIR / "final_tokenizer"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    final_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_tokenizer_dir))
    print(f"Saved adapter and tokenizer to: {config.OUTPUT_DIR}")

    # Final metrics snapshot (loss, runtime, etc.) saved to disk
    final_metrics = trainer.evaluate(metric_key_prefix="final")
    trainer.log_metrics("final", final_metrics)
    trainer.save_metrics("final", final_metrics)
    trainer.save_state()

    # merged = trainer.model.merge_and_unload()
    # merged.save_pretrained(str(config.OUTPUT_DIR / "merged"))
    # tokenizer.save_pretrained(str(final_tokenizer_dir / "merged"))


if __name__ == "__main__":
    main()
