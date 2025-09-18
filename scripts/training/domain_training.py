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


def main():
    """
    Main function to orchestrate the training process.
    """
    print("--- [Step 1] Initializing Setup ---")

    # --- Seed for reproducibility ---
    # Set the seed for numpy and torch to ensure that all random operations
    # (like model weight initialization and data shuffling) are deterministic.
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    print(f"Random seed set to: {config.SEED}")
    print(f"Output directory set to: {config.OUTPUT_DIR}")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    print("\n--- [Step 2] Loading Tokenizer ---")
    
    # Load the tokenizer specified in the config file.
    # The 'trust_remote_code=True' flag is sometimes necessary for custom model architectures.
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        token=config.HF_TOKEN,
        trust_remote_code=True
    )

    # --- Add special tokens based on data_processing.py ---
    # We define the special tokens that will be used to structure our repository-level context.
    # These tokens help the model distinguish between file metadata and file content.
    special_tokens_dict = {
        'additional_special_tokens': ['<repo_name>', '<file_sep>', '<endoftext>']
    }
    
    # Add the new tokens to the tokenizer's vocabulary.
    print(f"Tokenizer size: {len(tokenizer)}")
    print(tokenizer.vocab_size)
    print(tokenizer.special_tokens_map)


    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} new special tokens to the tokenizer.")

    # Set the padding token. For causal language models, it's common practice to
    # use the end-of-sentence token as the padding token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("`pad_token` was not set, using `eos_token` as `pad_token`.")

    print("Tokenizer loaded and configured successfully.")


    print("\n--- [Step 3] Loading Base Model ---")
    
    # Load the base model for causal language modeling.
    # We use bfloat16 for memory efficiency without a significant loss in precision.
    # `device_map="auto"` automatically distributes model layers across available GPUs.
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        token=config.HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"Base model '{config.MODEL_ID}' loaded successfully.")
    
    # --- Resize model embeddings ---
    # After adding new tokens to the tokenizer, we need to resize the model's
    # token embedding matrix to accommodate them. This adds new, randomly initialized
    # embedding vectors for our special tokens, which will be learned during training.
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model token embeddings resized to: {len(tokenizer)}")
    print(f"Model vocab size: {tokenizer.vocab_size}")
    print(tokenizer.special_tokens_map)


    print("\n--- [Step 4] Configuring LoRA ---")

    # Configure the LoRA parameters using the settings from config.py.
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none" # Typically, bias parameters are not trained in LoRA.
    )

    # --- Apply LoRA to the model ---
    # `get_peft_model` wraps the base model with the LoRA configuration.
    # This freezes the original model weights and inserts the small, trainable
    # LoRA adapter layers into the specified target modules.
    model = get_peft_model(model, lora_config)
    print("LoRA configuration applied to the model.")

    # Print the number of trainable parameters to verify that LoRA is set up correctly.
    # We expect this number to be a small fraction of the total model parameters.
    model.print_trainable_parameters()


if __name__ == "__main__":
    main()