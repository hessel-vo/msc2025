import os
import csv
import sys

# --- Configuration ---
TEMPLATE_FILE_PATH = "prompt_template_generation.txt"
TEMPLATE_SUBSET = "prompt_template_generation.txt" # Set to "prompt_template_generation_subset.txt" for benchmark subset
CSV_FILE_PATH = "benchmark_dataset.csv" # Set to "benchmark_dataset_subset.csv" for benchmark subset
BASE_OUTPUT_DIR = "created_prompts/generation"
BASE_EXAMPLES_DIR = "examples/generation"
ADDITIONAL_CONTEXT_DIR = "additional_context"

def load_text_file(filepath):
    """Loads a text file and handles potential errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found at '{filepath}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading '{filepath}': {e}")
        sys.exit(1)

def create_prompts(source, num_examples, sum_length, repo=None):
    """Generates prompt files based on a template and a CSV dataset."""

    source_folder = 'xlcost' if source == 'xl' else 'automotive'

    # --- MODIFIED SECTION: New Directory Structure ---
    # Construct the output directory path with the new structure.
    if num_examples == "zero":
        # For zero-shot, source is not relevant: zero_shot/long/
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"{num_examples}_shot", sum_length)
    else:
        # For one/three-shot, the structure is: one_shot/long/xlcost/
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"{num_examples}_shot", sum_length, source_folder)
    # --- END MODIFIED SECTION ---

    if repo:
        output_dir = os.path.join(BASE_OUTPUT_DIR, repo)

    # Construct the path for few-shot examples (this remains unchanged)
    examples_dir = os.path.join(BASE_EXAMPLES_DIR, source_folder, f"{sum_length}_summ")

    # Prompt template
    template_path = TEMPLATE_SUBSET if repo else TEMPLATE_FILE_PATH
    print(f"Loading prompt template from '{template_path}'...")
    prompt_template = load_text_file(template_path)

    if num_examples == "zero":
        # For zero-shot, remove the example intro and the entire example section
        parts = prompt_template.split('---')
        if len(parts) == 3:
            header = parts[0]
            header_lines = [line for line in header.splitlines() if "Below is an example" not in line]
            prompt_template = "\n".join(header_lines).strip() + "\n\n" + parts[2].strip()
        else:
            print("ERROR: Template format is not suitable for zero-shot modification (expected '---' separators).")
            sys.exit(1)

    elif num_examples == "three":
        prompt_template = prompt_template.replace("[Example]", "[Examples]")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to the '{output_dir}' directory.")

    # Store examples to avoid reloading
    example_cache = {}

    # Process input CSV file.
    print(f"Reading data from '{CSV_FILE_PATH}'...")
    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            
            for i, row in enumerate(reader):
                try:
                    file_id = row['id']
                    language = row['language']
                    
                    summary_column = f"summary_{sum_length}"
                    target_summary = row[summary_column]
                    function_signature = row['function_signature']
                    
                    lang_key = language.lower()
                    
                    filled_prompt = prompt_template

                    if num_examples in ["one", "three"]:
                        if lang_key not in example_cache:
                            example_path = os.path.join(examples_dir, f"{num_examples}_shot_{lang_key}.txt")
                            if os.path.exists(example_path):
                                example_cache[lang_key] = load_text_file(example_path)
                                print(f"Loaded example for '{language}' from '{example_path}'.")
                            else:
                                print(f"ERROR: Example file missing for '{language}' at '{example_path}'.")
                                sys.exit(1)

                        full_example_text = example_cache[lang_key]
                        filled_prompt = filled_prompt.replace("<generation example>", full_example_text)

                    # Fill template with data
                    filled_prompt = filled_prompt.replace("<code language>", lang_key.capitalize())
                    filled_prompt = filled_prompt.replace("<target language>", lang_key)
                    filled_prompt = filled_prompt.replace("<target summary>", target_summary)
                    filled_prompt = filled_prompt.replace("<function signature>", function_signature)

                    if repo:
                        additional_context = load_text_file(os.path.join(ADDITIONAL_CONTEXT_DIR, f"{file_id}.txt"))
                        filled_prompt = filled_prompt.replace("<additional_context>", additional_context)
                    
                    # Save the final prompt using name='id'
                    output_filename = f"{file_id}.txt"
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

    print(f"\nPrompt generation complete. Prompts are located in '{output_dir}'.")

def main():
    """Main function to parse arguments and run the prompt creation."""
    
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("ERROR: Incorrect number of arguments provided.")
        print(f"Usage: python {sys.argv[0]} <xl|auto> <zero|one|three> <short|long> [repo]")
        sys.exit(1)

    source = sys.argv[1]
    sum_length = sys.argv[2]
    num_examples = sys.argv[3]
    repo = sys.argv[4] if len(sys.argv) == 5 else None

    # Validate arguments
    if source not in ['xl', 'auto']:
        print("Error: First argument must be 'xl' or 'auto'.")
        sys.exit(1)
    if sum_length not in ['short', 'long']:
        print("Error: Second argument must be 'short' or 'long'.")
        sys.exit(1)
    if num_examples not in ['zero', 'one', 'three']:
        print("Error: Third argument must be 'zero', 'one', or 'three'.")
        sys.exit(1)


    create_prompts(
        source=source,
        num_examples=num_examples,
        sum_length=sum_length,
        repo=repo
    )

if __name__ == "__main__":
    main()