import config
import data_processing as dp

import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig
import datasets
import sys


def setup_environment():
    """Sets random seeds for reproducibility and creates the output directory."""
    print("--- [Step 1] Initializing Setup ---")
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    print(f"Random seed set to: {config.SEED}")
    print(f"Output directory set to: {config.OUTPUT_DIR}")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("--- Setup complete ---")

def load_model_and_tokenizer():
    """Loads the base model and tokenizer, adding special tokens."""
    print("\n--- [Step 2] Loading Tokenizer & Model ---")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        token=config.HF_TOKEN,
    )

    special_tokens_dict = {'additional_special_tokens': ['<repo_name>', '<file_sep>', '<endoftext>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} new special tokens.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Using `eos_token` as `pad_token`.")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        token=config.HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation='eager'
    )
    
    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))
    print(f"Base model '{config.MODEL_ID}' loaded. Token embeddings resized to {len(tokenizer)}.")
    print("--- Model and tokenizer loading complete ---")
    return model, tokenizer

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
    print("LoRA configuration applied to the model.")
    model.print_trainable_parameters()
    print("--- LoRA setup complete ---")
    return model

def main():
    """
    Main function to orchestrate the entire training and evaluation process.
    """
    # --- Part 1: Setup ---
    setup_environment()
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora_to_model(model)

    # --- Part 2: Data Preparation ---
    print("\n--- [Step 4] Preparing Datasets ---")
    raw_train_dataset, raw_eval_dataset = dp.load_and_split_data(config)
    
    # The evaluation dataset is static, so we can process it once upfront.
    print("\n--- Processing static evaluation dataset ---")
    print(raw_eval_dataset)
    processed_eval_dataset = dp.process_dataset_for_training(raw_eval_dataset, tokenizer, config)
    print("--- Evaluation dataset processing complete ---")

    # --- Part 3: Trainer Setup ---
    print("\n--- [Step 5] Configuring Hugging Face Trainer ---")
    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        logging_steps=config.LOGGING_STEPS,
        num_train_epochs=config.NUM_EPOCHS, # This is a placeholder; the loop controls epochs.
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.EVAL_STEPS,
        load_best_model_at_end=True,
        bf16=True,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=processed_eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[early_stopping_callback],
    )
    print("--- Trainer configured ---")

    # --- Part 4: Dynamic Training Loop ---
    print("\n--- [Step 6] Starting Dynamic Training Loop ---")
    last_checkpoint = None
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n" + "="*50)
        print(f"--- Starting Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")
        print("="*50)
        
        # Resample and process data for the current epoch
        epoch_train_dataset_sampled = dp.apply_dynamic_sampling(raw_train_dataset, config)
        processed_train_dataset = dp.process_dataset_for_training(epoch_train_dataset_sampled, tokenizer, config)
        
        # Update the trainer with the new dataset for this epoch
        trainer.train_dataset = processed_train_dataset
        
        # The trainer internally handles the number of steps for one epoch.
        # We resume from the last checkpoint to continue training state (optimizer, etc.).
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # After each epoch, we find the latest checkpoint to resume from in the next iteration.
        # This ensures we don't restart training from scratch every epoch.
        last_checkpoint = training_args.output_dir + f"/checkpoint-{trainer.state.global_step}"
        
        # Check if early stopping has been triggered
        if trainer.state.is_early_stopping:
            print(f"\nEarly stopping triggered after epoch {epoch + 1}. Training is stopping.")
            break
            
    print("\n--- Training loop finished ---")

    # --- Part 5: Save Final Model ---
    print("\n--- [Step 7] Saving Final Model ---")
    # The trainer already loaded the best model at the end of training
    # because `load_best_model_at_end=True`. We just need to save it.
    final_save_path = os.path.join(config.OUTPUT_DIR, "final_best_model")
    trainer.save_model(final_save_path)
    print(f"Best model saved to {final_save_path}")
    print("\n--- All steps complete! ---")

if __name__ == "__main__":
    main()