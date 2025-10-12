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

# Training setup
MODEL_ID = "google/gemma-3-12b-it"

MAX_SEQ_LENGTH = 2048

# Limits the number of chunks per repository to avoid over-representation
MAX_CHUNKS_PER_REPO = 260


# Dataset & Paths
DATASET_TYPE = "extended"
# Path to JSONL dataset file
DATASET_PATH = PROJECT_ROOT / "scripts/training/datasets" / f"final_dataset_{DATASET_TYPE}.jsonl"

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

# Directory to save the trained LoRA adapter and training checkpoints
OUTPUT_DIR = PROJECT_ROOT / "scripts/training/trained_models"


# Hyperparameters

# Low batch size to fit in GPU memory, gradient accumulated used (simulates larger batches)
BATCH_SIZE = 1

LEARNING_RATE = 1e-4

NUM_EPOCHS = 10

# Steps per logging and validation even (training is done on step basis)
LOGGING_STEPS = 25

EVAL_STEPS = 100

# Validation patience (number of eval steps)
EARLY_STOPPING_PATIENCE = 12


# LoRA Configuration
LORA_R = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.1

LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj"
]