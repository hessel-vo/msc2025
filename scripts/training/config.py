import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROJECT_ROOT_STR = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(PROJECT_ROOT_STR)
HF_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN')
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
HF_CACHE_DIR.mkdir(exist_ok=True)

os.environ['HF_HOME'] = str(HF_CACHE_DIR)

SEED=42

# --- Model Parameters ---
MODEL_ID = "google/gemma-3-12b-it"

MAX_SEQ_LENGTH = 2048

# The maximum number of chunks to sample from any single repository per epoch.
# This helps prevent dominant repositories from biasing the training.
MAX_CHUNKS_PER_REPO = 260


# --- Dataset & Paths ---
DATASET_TYPE = "target_only"
# The path to the input JSONL file containing the processed source code.
DATASET_PATH = PROJECT_ROOT / "scripts/training/datasets" / f"final_dataset_{DATASET_TYPE}.jsonl"

# A list of repository IDs to use for the validation set.
VALIDATION_REPO_IDS = [
    'cdsp',
    'wayland-ivi-extension',
    'ramses-citymodel-demo',
    'pybip',
    'barefoot',
    'roadC',
    's2dm',
    'MoCOCrW',
    'paho.mqtt.java',
]

# Directory to save the trained LoRA adapter and any training checkpoints.
OUTPUT_DIR = PROJECT_ROOT / "scripts/training/trained_models"


# --- Training Hyperparameters ---
# The number of training examples per device (e.g., per GPU).
# Adjust based on your GPU's VRAM.
BATCH_SIZE = 1

# The initial learning rate for the AdamW optimizer. 2e-4 is a common
# starting point for LoRA fine-tuning.
LEARNING_RATE = 1.5e-4

# The total number of training epochs to perform.
NUM_EPOCHS = 10

# How often to log training metrics (e.g., loss) to the console.
LOGGING_STEPS = 40

# How often to run evaluation on the validation set.
EVAL_STEPS = 120

# The number of evaluation steps to wait for improvement before stopping
# training early. This helps prevent overfitting.
EARLY_STOPPING_PATIENCE = 12


# --- LoRA (Low-Rank Adaptation) Configuration ---
# The rank of the update matrices. A lower rank means fewer trainable parameters.
# Common values are 8, 16, 32.
LORA_R = 8

# The alpha parameter for LoRA scaling. A common practice is to set alpha
# to be twice the rank (r).
LORA_ALPHA = 16

# Dropout probability for the LoRA layers.
LORA_DROPOUT = 0.1

# A list of model module names to apply LoRA to. This is model-specific.
# For Llama-based models, targeting query and value projection matrices is standard.
# You might need to inspect the model architecture to find the correct module names.
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj"
]