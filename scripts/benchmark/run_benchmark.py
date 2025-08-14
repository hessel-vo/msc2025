import os
import pandas as pd
import torch
from pathlib import Path
from dotenv import load_dotenv

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

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# --- 1. Configuration ---

MODEL_ID = "google/gemma-3-4b-it"
INPUT_CSV_PATH = PROJECT_ROOT / "benchmark_dataset" / "dataset.csv"
PROMPTS_DIR = PROJECT_ROOT / "benchmark_dataset" / "prompts" / "generated_prompts"
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline"
OUTPUT_FILENAME = OUTPUT_DIR / "gemma-3-4b-it_baseline_results.csv"

# 3. Ensure output directory exists/is made
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Check configuration
print(f"Input CSV: {INPUT_CSV_PATH}")
print(f"Prompts Directory: {PROMPTS_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Output Filename: {OUTPUT_FILENAME}")

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

processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN
).eval()
print("Model and processor loaded successfully.")

# --- 4. Main Benchmarking Logic ---
def run_benchmark():
    try:
        dataset_df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input CSV not found at '{INPUT_CSV_PATH}'. Make sure the file exists.")
        return

    results = []
    print(f"\nStarting benchmark on {len(dataset_df)} problems...")

    for index, row in dataset_df.iterrows():
        problem_id = row['id']
        original_summary = row['summary']
        prompt_filepath = os.path.join(PROMPTS_DIR, f"{problem_id}.txt")

        print(f"  Processing problem ID: {problem_id}...")

        try:
            with open(prompt_filepath, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
        except FileNotFoundError:
            print(f"    - Warning: Prompt file not found for ID '{problem_id}'. Skipping.")
            continue

        # 1. Format the input using chat template
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]
        
        # 2. Apply the template and tokenize the input
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(model.device)

        # Store the length of the input to remove from output
        input_len = inputs.shape[-1]

        # 3. Generate the response
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=100,
                do_sample=False
            )
        
        # 4. Decode only the newly generated tokens
        generated_ids = outputs[0][input_len:]
        generated_summary = processor.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"    - Generated Summary: {generated_summary}...")
        results.append({
            "id": problem_id,
            "summary": original_summary,
            "generated_summary": generated_summary
        })

    print("Benchmark run complete.")

    # --- 5. Save Results ---
    if not results:
        print("No results were generated. Exiting without saving.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    results_df.to_csv(output_path, index=False)
    
    print(f"Results saved successfully to '{output_path}'")

if __name__ == "__main__":
    run_benchmark()