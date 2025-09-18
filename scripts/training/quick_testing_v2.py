import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# --- 1. Environment and Path Setup ---
load_dotenv()

project_root_str = os.getenv("PROJECT_ROOT")
if not project_root_str:
    print("Warning: PROJECT_ROOT environment variable not set. Using default Hugging Face cache.")
    # Set a default cache directory if PROJECT_ROOT is not defined
    HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"
else:
    PROJECT_ROOT = Path(project_root_str)
    HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

HF_CACHE_DIR.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(HF_CACHE_DIR)
HF_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN')

print("--- Project Setup ---")
print(f"Hugging Face Cache Directory (HF_HOME) set to: {os.environ['HF_HOME']}")
print("---------------------\n")

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer



# --- 2. Configuration ---
# NOTE: Using the model from your latest script
MODEL_ID = "google/gemma-3-1b-it"


# --- 3. Setup Device (GPU or CPU) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using CUDA.")
    print(f"Device Name: {torch.cuda.get_device_name(device)}\n")
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU.\n")


# --- 4. Load Model and Extract Information ---
print(f"Loading model '{MODEL_ID}' to gather information...")

try:
    # Load the model onto the CPU to inspect it without using GPU VRAM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN
    )
    print("Model loaded successfully.\n")

    # --- 5. Print Detailed Model Information ---

    # --- Parameter Count ---
    print("--- Parameter Count ---")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} (~{total_params / 1e9:.2f} Billion)")
    print("-----------------------\n")

    # --- Key Configuration Values ---
    config = model.config
    print("--- Key Configuration Values ---")
    print(f"Hidden Size (d_model): {config.hidden_size}")
    print(f"Number of Hidden Layers: {config.num_hidden_layers}")
    print(f"Intermediate Size (FFN): {config.intermediate_size}")
    print(f"Number of Attention Heads: {config.num_attention_heads}")
    print(f"Number of Key/Value Heads: {config.num_key_value_heads}")
    print(f"Max Position Embeddings: {config.max_position_embeddings}")
    print("------------------------------\n")

    # --- Model Architecture ---
    # This will show the layer structure and names (e.g., 'q_proj', 'v_proj')
    print("--- Full Model Architecture ---")
    print(model)
    print("-------------------------------\n")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you are logged into Hugging Face CLI (`huggingface-cli login`) and that the HF_TOKEN is correct.")