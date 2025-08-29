import os
import pandas as pd
import torch
from pathlib import Path
from dotenv import load_dotenv
import sys

load_dotenv()

project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)

HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

# Set the HF_HOME environment variable before importing transformers
os.environ['HF_HOME'] = str(HF_CACHE_DIR)
HF_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN')

print("--- Project Setup Confirmation ---")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Hugging Face Cache Directory (HF_HOME) set to: {os.environ['HF_HOME']}")
print("---------------------------------")

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

# --- 1. Configuration ---

MODEL_ID = "google/gemma-3-4b-it"


# --- 2. Setup Device (GPU or CPU) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using CUDA.")
    print("Current device:", device)
    print("Device name:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU.")
    print("Current device:", device)

# --- 3. Load Model and Processor ---
print(f"Loading model: {MODEL_ID}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN
).eval()
print("Model and processor loaded successfully.")


# Access the config object attached to the model
# config = model.config

# print("All model config:")
# print(config)

# print("max_position_embeddings:", config.text_config.max_position_embeddings)


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

print("--- Tokenizer Special Tokens Map ---")
print(tokenizer.special_tokens_map)
# Example output: {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}

print("\n--- Full List of Special Tokens ---")
print(tokenizer.all_special_tokens)

print("\n--- Added Tokens (Often includes special ones) ---")
# This shows tokens added after initial pre-training
# print(tokenizer.added_tokens_decoder) 
