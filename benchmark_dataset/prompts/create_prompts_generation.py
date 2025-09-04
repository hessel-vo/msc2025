import os
import csv
import sys

# --- Configuration ---
TEMPLATE_FILE_PATH = "prompt_template_generation.txt"
TEMPLATE_SUBSET = "prompt_template_generation_context.txt"
CSV_FILE_PATH = "benchmark_dataset_subset.csv"
BASE_OUTPUT_DIR = "prompts_generation"
BASE_EXAMPLES_DIR = "examples/generation"
ADDITIONAL_CONTEXT_DIR = "additional_context"

def load_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found at '{filepath}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading '{filepath}': {e}")
        sys.exit(1)

def create_prompts(sum_length, num_examples, repo=None):

    output_dir = f"{BASE_OUTPUT_DIR}_{sum_length}/{num_examples}_shot"
    examples_dir = f"{BASE_EXAMPLES_DIR}/{sum_length}_summ"

    # 1. Prompt template
    print(f"Loading prompt template from '{TEMPLATE_FILE_PATH}'...")
    prompt_template = load_text_file(TEMPLATE_FILE_PATH)

    if repo:
        prompt_template = load_text_file(TEMPLATE_SUBSET)

    if num_examples == "three":
        prompt_template = prompt_template.replace("[Example]", "[Examples]")

    # 2. Output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to the '{output_dir}' directory.")

    # 3. Store examples
    example_cache = {}

    # 4. Process CSV file.
    print(f"Reading data from '{CSV_FILE_PATH}'...")
    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            
            for i, row in enumerate(reader):
                try:
                    # Get data from the current row
                    file_id = row['id']
                    language = row['language']
                    
                    summary_column = f"summary_{sum_length}"
                    target_summary = row[summary_column]
                    function_signature = row['function_signature']
                    
                    lang_key = language.lower()

                    # 5. Load the language-specific example (use cache if available)
                    if lang_key not in example_cache:
                        example_path = os.path.join(examples_dir, f"{num_examples}_shot_example_{lang_key}.txt")
                        if os.path.exists(example_path):
                            loaded_text = load_text_file(example_path)
                            example_cache[lang_key] = loaded_text
                            print(f"Loaded example for '{language}' from '{example_path}'.")
                        else:
                            print(f"Example file missing for '{language}' at '{example_path}'.")
                            sys.exit(1)

                    full_example_text = example_cache[lang_key]

                    # 6. Fill template with data
                    filled_prompt = prompt_template.replace("<generation example>", full_example_text)
                    filled_prompt = filled_prompt.replace("<code language>", lang_key.capitalize())
                    filled_prompt = filled_prompt.replace("<target language>", lang_key)
                    filled_prompt = filled_prompt.replace("<target summary>", target_summary)
                    filled_prompt = filled_prompt.replace("<function signature>", function_signature)

                    if repo:
                        additional_context = load_text_file(os.path.join(ADDITIONAL_CONTEXT_DIR, f"{file_id}.txt"))
                        filled_prompt = filled_prompt.replace("<additional_context>", additional_context)
                    
                    # 7. Save the final prompt to a new file using name='id'
                    output_filename = f"{file_id}.txt"
                    if repo:
                        output_filepath = os.path.join(output_dir, repo, output_filename)
                    else:
                        output_filepath = os.path.join(output_dir, output_filename)
                    
                    with open(output_filepath, 'w', encoding='utf-8') as out_file:
                        out_file.write(filled_prompt)
                        
                except KeyError as e:
                    print(f"  -> ERROR: Row {i+2} is missing a required column: {e}. Skipping row.")
                except Exception as e:
                    print(f"  -> ERROR: An unexpected error occurred on row {i+2}: {e}. Skipping row.")

    except FileNotFoundError:
        print(f"ERROR: CSV file not found '{CSV_FILE_PATH}'.")
        sys.exit(1)

    print("\nPrompt generation complete.")

def main():
    
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("ERROR: Incorrect number of arguments provided.")
        print(f"Usage: python {sys.argv[0]} <num_examples> <summarization_length> <repo (optional)>")
        sys.exit(1)

    num_examples = sys.argv[1]
    sum_length = sys.argv[2]

    repo = None
    
    if len(sys.argv) == 4:
        repo = sys.argv[3]
        

    create_prompts(
        sum_length=sum_length,
        num_examples=num_examples,
        repo=repo
    )

if __name__ == "__main__":
    main()