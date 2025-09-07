import os
import csv
import sys

# --- Configuration ---
# Adjusted base directories to match the project structure
TEMPLATE_FILE_PATH = "prompt_template_summarization.txt"
TEMPLATE_SUBSET = "prompt_template_summarization.txt"
CSV_FILE_PATH = "benchmark_dataset.csv" # Set to "benchmark_dataset_subset.csv" for benchmark subset
BASE_OUTPUT_DIR = "created_prompts/summarization"
BASE_EXAMPLES_DIR = "examples/summarization"

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

def create_prompts(source, sum_length, num_examples, subset=None):
    """Generates prompt files based on a template and a CSV dataset."""

    source_folder = 'xlcost' if source == 'xl' else 'automotive'

    # --- MODIFIED SECTION: New Directory Structure ---
    # Construct the output directory path with the new structure.
    # The number of shots is now the base directory.
    if num_examples == "zero":
        # For zero-shot, source is not relevant, so the path is: zero_shot/long/
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"{num_examples}_shot", sum_length)
    else:
        # For one/three-shot, the new structure is: one_shot/long/xlcost/
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"{num_examples}_shot", sum_length, source_folder)
    # --- END MODIFIED SECTION ---

    if subset:
        output_dir = os.path.join(BASE_OUTPUT_DIR, subset)

    # Construct the path to few-shot examples (this remains unchanged as it's the source)
    examples_dir = os.path.join(BASE_EXAMPLES_DIR, source_folder, f"{sum_length}_summ")

    # Prompt template
    print(f"Loading prompt template from '{TEMPLATE_FILE_PATH}'...")
    prompt_template = load_text_file(TEMPLATE_FILE_PATH)

    # Modify template based on arguments
    if num_examples == "zero":
        template_parts = prompt_template.split('---')
        if len(template_parts) == 3:
            prompt_template = template_parts[0].strip() + "\n\n" + template_parts[2].strip()
        else:
            print("ERROR: Template format is not suitable for zero-shot modification (expected '---' separators).")
            sys.exit(1)
    elif num_examples == "three":
        prompt_template = prompt_template.replace("[Example]", "[Examples]")

    if sum_length == "long":
        prompt_template = prompt_template.replace("Provide a concise", "Provide a detailed")

    # Output directory
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
                    
                    code_from_csv = row['code']
                    code = '\n'.join(code_from_csv.splitlines())
                    
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
                        filled_prompt = filled_prompt.replace("<summary example>", full_example_text)

                    # Fill template with data
                    filled_prompt = filled_prompt.replace("<code language>", lang_key.capitalize())
                    filled_prompt = filled_prompt.replace("<target language>", lang_key)
                    filled_prompt = filled_prompt.replace("<target code>", code)
                    
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
    
    if len(sys.argv) < 4:
        print("ERROR: Incorrect number of arguments provided.")
        print(f"Usage: python {sys.argv[0]} <xl|auto> <short|long> <zero|one|three> [subset]")
        sys.exit(1)

    source = sys.argv[1]
    sum_length = sys.argv[2]
    num_examples = sys.argv[3]
    subset = sys.argv[4] if len(sys.argv) == 5 else None


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
        sum_length=sum_length,
        num_examples=num_examples,
        subset=subset
    )

if __name__ == "__main__":
    main()