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

# Step 1: Environment
def setup_environment():
    """Sets random seeds for reproducibility and ensures output dir exists."""
    print("--- [Step 1] Initializing Setup ---")
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    print(f"Random seed set to: {config.SEED}")
    print(f"Output directory set to: {config.OUTPUT_DIR}")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("--- Setup complete ---")


# Step 2: Model + Tokenizer
def load_model_and_tokenizer():
    """Loads base model & tokenizer, registers special tokens, resizes embeddings."""
    print("\n--- [Step 2] Loading Tokenizer & Model ---")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        token=config.HF_TOKEN,
    )

    # Add task/domain markers as additional special tokens (note: removed for v2 training run)
    # special_tokens_dict = {"additional_special_tokens": ["<repo_name>", "<file_sep>", "<endoftext>"]}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print(f"Added {num_added_toks} new special tokens.")

    # Ensure we have a pad token (common for causal LMs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Using `eos_token` as `pad_token`.")

    # Base model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        token=config.HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Resize embeddings to include newly added tokens
    # model.resize_token_embeddings(len(tokenizer))
    print(f"Base model '{config.MODEL_ID}' loaded. Token embeddings resized to {len(tokenizer)}.")
    print("--- Model and tokenizer loading complete ---")
    return model, tokenizer


# Step 3: LoRA
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


# Step 4: Data
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

    # Callback that advances the dataset window each epoch
    advance_callback = dp.AdvanceEpochWindows(rolling_train_ds)

    print("--- Data preparation complete ---")
    return rolling_train_ds, eval_dataset, advance_callback


# Step 5: Trainer
def build_trainer(model, tokenizer, train_dataset, eval_dataset, advance_callback):
    """Configures the HF Trainer with rolling dataset + LoRA + early stopping."""
    print("\n--- [Step 5] Configuring Hugging Face Trainer ---")

    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=0.08,
        weight_decay=0.01,
        logging_steps=config.LOGGING_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.EVAL_STEPS,
        save_total_limit=4,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        bf16=True,
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


# Step 6: Main script
def main():
    """
    Orchestrates setup, data, trainer, and training loop.
    """
    setup_environment()
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora_to_model(model)

    print(model.config)
    print(model.config._attn_implementation)

    train_dataset, eval_dataset, advance_callback = prepare_datasets(tokenizer)
    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset, advance_callback)

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

    final_metrics = trainer.evaluate(metric_key_prefix="final")
    trainer.log_metrics("final", final_metrics)
    trainer.save_metrics("final", final_metrics)

    # merged = trainer.model.merge_and_unload()
    # merged.save_pretrained(str(config.OUTPUT_DIR / "merged"))
    # tokenizer.save_pretrained(str(config.OUTPUT_DIR / "merged"))


if __name__ == "__main__":
    main()
