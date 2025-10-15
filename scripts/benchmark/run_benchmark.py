import os
import pandas as pd
import torch
from pathlib import Path
from dotenv import load_dotenv
import sys
import re
import rcimsg

print(torch._dynamo.config.cache_size_limit)
torch._dynamo.config.cache_size_limit = 256

load_dotenv()

project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)
HF_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN')
MODEL_ID = "google/gemma-3-12b-it"
RESULT_TYPE = "adapted" # "baseline" or "adapted" for model eval
DATASET_TYPE = "core"

MODEL_SIZE = MODEL_ID.split("-")[2]
ADAPTER_ID = PROJECT_ROOT / "scripts" / "training" / f"new_{MODEL_SIZE}_{DATASET_TYPE}_trained_models" / "final_adapter"

if RESULT_TYPE == "adapted":
    MODEL_NAME = f"adapted_{MODEL_SIZE}_{DATASET_TYPE}"
else:
    MODEL_NAME = MODEL_ID.split("/")[-1]

HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

# Set the HF_HOME environment variable before importing transformers
os.environ['HF_HOME'] = str(HF_CACHE_DIR)

print("--- Project Setup Confirmation ---")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Hugging Face Cache Directory (HF_HOME) set to: {os.environ['HF_HOME']}")
print("---------------------------------")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def apply_peft_adapter(model):
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    model.eval()
    return model

def load_model_and_tokenizer():
    print("\n--- [Step 2] Loading Tokenizer & Model ---")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
    )

    if RESULT_TYPE == "baseline":

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        return model, tokenizer
    else:

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, str(ADAPTER_ID))
        model.eval()

        return model, tokenizer

def remove_markdown_wrapping(code_string):
    pattern = r"^\s*```(?:\w+)?\n(.*?)\n```\s*$"
    match = re.search(pattern, code_string, re.DOTALL)
    if match:
        return match.group(1)
    return code_string

def generate_with_chat(model, tokenizer, messages, max_new_tokens=800):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs.shape[-1]
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    new_token_ids = outputs[0][input_len:]
    decoded = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    return decoded

def run_benchmark():

    # Configure input and output paths from arguments
    if len(sys.argv) < 5:
        print("Usage: python run_benchmark.py <generation|summarization> <xl|auto> <short|long> <zero|one|three> [subset]")
        return

    task_type = sys.argv[1]
    source = sys.argv[2]
    short_or_long = sys.argv[3]
    shot_count = sys.argv[4]
    subset = sys.argv[5] if len(sys.argv) == 6 else None

    if task_type not in ["generation", "summarization"]:
        print(f"Error: Invalid task type '{task_type}'. Use 'generation' or 'summarization'.")
        return
    if source not in ["xl", "auto"]:
        print(f"Error: Invalid task type '{source}'. Use 'xl' or 'auto'.")
        return
    if short_or_long not in ["short", "long"]:
        print(f"Error: Invalid summary length '{short_or_long}'. Use 'short' or 'long'.")
        return
    if shot_count not in ["zero", "one", "three"]:
        print(f"Error: Invalid shot count '{shot_count}'. Use 'zero', 'one', or 'three'.")
        return

    print(f"Running-- task: {task_type}, Source: {source}, Summary Length: {short_or_long}, Shots: {shot_count}, Subset: {subset}")

    source_folder = 'xlcost' if source == 'xl' else 'automotive'
    summary_type = f'summary_{short_or_long}'
    rci_message = rcimsg.RCI_GEN if task_type == "generation" else rcimsg.RCI_SUM

    INPUT_CSV_PATH = PROJECT_ROOT / "benchmark_dataset" / "benchmark_dataset.csv"

    if shot_count != "zero":
        PROMPTS_DIR = PROJECT_ROOT / "benchmark_dataset" / "prompts" / "created_prompts" / task_type / f"{shot_count}_shot" / short_or_long / source_folder
    else:
        PROMPTS_DIR = PROJECT_ROOT / "benchmark_dataset" / "prompts" / "created_prompts" / task_type / "zero_shot" / short_or_long
    
    OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark" / RESULT_TYPE / "habrok"
    OUTPUT_FILENAME = OUTPUT_DIR / f"{MODEL_NAME}_{task_type}_{source}_{short_or_long}_{shot_count}_results.csv"

    if subset == "subset":
        summary_type = "summary_long"
        INPUT_CSV_PATH = PROJECT_ROOT / "benchmark_dataset" / "benchmark_dataset_subset.csv"
        PROMPTS_DIR = PROJECT_ROOT / "benchmark_dataset" / "prompts" / "created_prompts" / task_type / "subset_context"
        OUTPUT_FILENAME = OUTPUT_DIR / f"{MODEL_NAME}_{task_type}_subset_results.csv"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and tokenizer")
    # Load and prepare model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using CUDA.")
        print("Current device:", device)
        print("Device name:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU.")
        print("Current device:", device)

    print(f"Loading model: {MODEL_ID}")

    model, tokenizer = load_model_and_tokenizer()

    # Load dataset
    try:
        dataset_df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input CSV not found at '{INPUT_CSV_PATH}'. Make sure the file exists.")
        return

    results = []
    print(f"Starting benchmark on {len(dataset_df)} problems...")

    count = 0
    for index, row in dataset_df.iterrows():
        
        problem_id = row['id']
        language = row['language']

        # # TESTING/DEBUGGING, REMOVE FOR FULL RUN
        # if language != "python":
        #     continue
        # else:
        #     count += 1
        # if count > 3:
        #     break

        if task_type == "generation":
            reference = row['code']
        else:
            reference = row[summary_type]
        prompt_filepath = PROMPTS_DIR / f"{problem_id}.txt"

        print(f"  Processing problem ID: {problem_id}")

        try:
            with open(prompt_filepath, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
        except FileNotFoundError:
            print(f"    - Warning: Prompt file not found for ID '{problem_id}' at '{prompt_filepath}'. Skipping.")
            continue

        # First-pass generation
        messages_first = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]
        generated_output = generate_with_chat(
            model=model,
            tokenizer=tokenizer,
            messages=messages_first,
            max_new_tokens=800
        )
        print(f"    - Generated (first pass): {generated_output}")

        # RCI step
        # original instructions, first output, RCI instructions
        messages_rci = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": generated_output}]},
            {"role": "user", "content": [{"type": "text", "text": rci_message}]}
        ]
        generated_output_rci = generate_with_chat(
            model=model,
            tokenizer=tokenizer,
            messages=messages_rci,
            max_new_tokens=800
        )
        print(f"    - Generated (RCI): {generated_output_rci}")

        # Collect results
        results.append({
            "id": problem_id,
            "language": language,
            "reference": reference,
            "generated": remove_markdown_wrapping(generated_output),
            "generated_rci": remove_markdown_wrapping(generated_output_rci),
        })

    print("Benchmark run complete.")

    if not results:
        print("No results were generated. Exiting without saving.")
        return
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILENAME, index=False)
    
    print(f"Results saved successfully to '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    run_benchmark()
