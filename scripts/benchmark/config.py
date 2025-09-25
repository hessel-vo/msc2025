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
MODEL_ID = "google/gemma-3-1b-it"