import os
import pandas as pd
import torch
from pathlib import Path
from dotenv import load_dotenv
import sys

load_dotenv()

project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)
HF_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN')
MODEL_ID = "google/gemma-3-4b-it"
MODEL_NAME = MODEL_ID.split("/")[-1]

HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

# Set the HF_HOME environment variable before importing transformers
os.environ['HF_HOME'] = str(HF_CACHE_DIR)

print("--- Project Setup Confirmation ---")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Hugging Face Cache Directory (HF_HOME) set to: {os.environ['HF_HOME']}")
print("---------------------------------")

from transformers import AutoProcessor, AutoModelForCausalLM



if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using CUDA.")
    print("Current device:", device)
    print("Device name:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU.")
    print("Current device:", device)

print(f"Loading model: {MODEL_ID}...")

processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN
).eval()
print("Model and processor loaded successfully.")


def run_benchmark():

    # Configure input and output paths from arguments
    if len(sys.argv) != 4:
        print("Usage: python run_benchmark.py <'generation' or 'summarization'> <'one' or 'three'> <'short' or 'long'>")
        return

    task_type = sys.argv[1]
    shot_count = sys.argv[2]
    short_or_long = sys.argv[3]  

    if task_type not in ["generation", "summarization"]:
        print(f"Error: Invalid task type '{task_type}'. Use 'generation' or 'summarization'.")
        return
    if shot_count not in ["one", "three"]:
        print(f"Error: Invalid shot count '{shot_count}'. Use 'one' or 'three'.")
        return
    if short_or_long not in ["short", "long"]:
        print(f"Error: Invalid summary length '{short_or_long}'. Use 'short' or 'long'.")
        return

    shot_folder_name = "one_shot" if shot_count == "one" else "three_shot"
    
    INPUT_CSV_PATH = PROJECT_ROOT / "benchmark_dataset" / "benchmark_dataset.csv"
    PROMPTS_DIR = PROJECT_ROOT / "benchmark_dataset" / "prompts" / f"prompts_{task_type}_{short_or_long}" / shot_folder_name
    OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark" / "baseline" # Swich "baseline" to "adapted" for final eval
    OUTPUT_FILENAME = OUTPUT_DIR / f"{MODEL_NAME}_{task_type}_{shot_folder_name}_{short_or_long}_results.csv"


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check configurations
    print("\n--- Running Benchmark with Configuration ---")
    print(f"Task Type: {task_type}")
    print(f"Shot Count: {shot_count}")
    print(f"Input CSV: {INPUT_CSV_PATH}")
    print(f"Prompts Directory: {PROMPTS_DIR}")
    print(f"Output Filename: {OUTPUT_FILENAME}")
    print("------------------------------------------\n")


    try:
        dataset_df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input CSV not found at '{INPUT_CSV_PATH}'. Make sure the file exists.")
        return

    results = []
    print(f"Starting benchmark on {len(dataset_df)} problems...")


    summary_type = f'summary_{short_or_long}'

    for index, row in dataset_df.iterrows():
        if row['id'] > 3:
            break
        problem_id = row['id']
        original_summary = row[summary_type]
        prompt_filepath = PROMPTS_DIR / f"{problem_id}.txt"

        print(f"  Processing problem ID: {problem_id}...")

        try:
            with open(prompt_filepath, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
        except FileNotFoundError:
            print(f"    - Warning: Prompt file not found for ID '{problem_id}' at '{prompt_filepath}'. Skipping.")
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
        
        # 4. Decode new tokens
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
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILENAME, index=False)
    
    print(f"Results saved successfully to '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    # The sys.exit() from the original script is removed to allow the script to run.
    # If it was for debugging, it's no longer needed here.
    run_benchmark()