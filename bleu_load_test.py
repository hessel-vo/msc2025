# scripts/evaluation/test_metric_loading_enhanced.py

import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

# --- Step 0: Enhanced Configuration & Verbosity ---
print("--- Starting Enhanced Metric Loading Test ---")

# Set verbosity for huggingface_hub to see detailed logs of network activity
os.environ['HF_HUB_VERBOSITY'] = 'debug'

# Load environment variables
load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
if not project_root_str:
    raise ValueError("PROJECT_ROOT not found in .env file.")

PROJECT_ROOT = Path(project_root_str)
# HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
# os.environ['HF_HOME'] = str(HF_CACHE_DIR)

# print(f"Using Hugging Face cache directory: {HF_CACHE_DIR}")

# Create the cache directory if it doesn't exist to avoid errors
# HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Import libraries AFTER setting environment variables
try:
    import evaluate
    import datasets
    import huggingface_hub
    import requests
    print(f"Python version: {sys.version.split()[0]}")
    print(f"evaluate version: {evaluate.__version__}")
    print(f"datasets version: {datasets.__version__}")
    print(f"huggingface_hub version: {huggingface_hub.__version__}")
except ImportError as e:
    print(f"\n❌ FATAL: A required library is not installed. Please run 'pip install {e.name}'.")
    sys.exit(1)
# --- End Configuration ---


# --- Step 1: Check Write Permissions ---
# print("\n--- [Step 1: Checking Directory Permissions] ---")
# if os.access(HF_CACHE_DIR, os.W_OK):
#     print(f"✅ SUCCESS: Write permissions are confirmed for {HF_CACHE_DIR}")
# else:
#     print(f"❌ FAILURE: No write permissions for the cache directory: {HF_CACHE_DIR}")
#     print("Please check the folder permissions. Test finished.")
#     sys.exit(1)


# --- Step 2: Clear Old Cache Entry ---
# print("\n--- [Step 2: Clearing Existing BLEU Cache] ---")
# bleu_cache_path = HF_CACHE_DIR / "metrics" / "bleu"
# if bleu_cache_path.exists():
#     try:
#         shutil.rmtree(bleu_cache_path)
#         print(f"✅ SUCCESS: Removed existing cache at '{bleu_cache_path}' to ensure a fresh download.")
#     except OSError as e:
#         print(f"⚠️ WARNING: Could not remove existing cache directory. This might cause issues. Error: {e}")
# else:
#     print("INFO: No previous BLEU cache found in the local directory to clear.")


# --- Step 3: Direct Network Connectivity Test ---
print("\n--- [Step 3: Testing Direct Network Connection] ---")
hf_url = "https://huggingface.co"
try:
    response = requests.get(hf_url, timeout=10)
    response.raise_for_status()  # This will raise an exception for 4xx or 5xx status codes
    print(f"✅ SUCCESS: Successfully connected to {hf_url} (Status code: {response.status_code})")
except requests.exceptions.RequestException as e:
    print(f"\n❌ FAILURE: Could not connect to {hf_url}.")
    print("This almost always indicates a network problem (no internet, firewall, or proxy).")
    print(f"Error details: {e}")
    print("Test finished.")
    sys.exit(1)


# --- Step 4: Final Loading Attempt ---
print("\n--- [Step 4: Attempting to Load Metric with evaluate.load] ---")
try:
    bleu_metric = evaluate.load("bleu")

    if bleu_metric:
        print("\n✅ SUCCESS: The 'bleu' metric was loaded successfully!")
        print(f"Loaded object: {bleu_metric}")
    else:
        print("\n❌ FAILURE: `evaluate.load('bleu')` returned a 'falsy' value (e.g., None).")
        print("Since previous checks passed, this may be a subtle library bug or version conflict.")

except Exception as e:
    print(f"\n❌ FAILURE: An exception occurred while trying to load the metric.")
    print(f"Error details: {e}")

print("\n--- Test Finished ---")