import os
import csv
import sys

# --- Configuration ---
TEMPLATE_FILE_PATH = "prompt_template.txt"
CSV_FILE_PATH = "benchmark_dataset.csv"
OUTPUT_DIR = "generated_prompts"
EXAMPLES_DIR = "examples"

def load_text_file(filepath):
    """Loads content from a text file, exiting if the file is not found."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"FATAL ERROR: A required file could not be found at '{filepath}'.")
        print("Please make sure the file exists and the path is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading '{filepath}': {e}")
        sys.exit(1)

def create_prompts():
    """
    Reads a CSV, combines data with a template and examples, and saves new prompt files.
    """
    # 1. Load the main prompt template once.
    print(f"Loading prompt template from '{TEMPLATE_FILE_PATH}'...")
    prompt_template = load_text_file(TEMPLATE_FILE_PATH)

    # 2. Ensure the output directory exists.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to the '{OUTPUT_DIR}' directory.")

    # 3. Cache for loaded examples to avoid reading the same file multiple times.
    example_cache = {}

    # 4. Process the CSV file.
    print(f"Reading data from '{CSV_FILE_PATH}'...")
    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            
            for i, row in enumerate(reader):
                try:
                    # Get required data from the current row
                    file_id = row['id']
                    language = row['language']
                    
                    # Normalize newlines in the code to fix potential formatting issues
                    code_from_csv = row['code']
                    code = '\n'.join(code_from_csv.splitlines())
                    
                    # --- NEW: Create different versions of the language string ---
                    # For display text (e.g., "Python", "Java")
                    capitalized_language = language.capitalize()
                    # For file lookups and code blocks (e.g., "python", "java")
                    lang_key = language.lower()
                    # --- END OF NEW ---

                    # 5. Load the language-specific example (use cache if available)
                    if lang_key not in example_cache:
                        example_path = os.path.join(EXAMPLES_DIR, f"example_{lang_key}.txt")
                        if os.path.exists(example_path):
                            example_cache[lang_key] = load_text_file(example_path)
                        else:
                            print(f"  -> WARNING: No example file found for '{language}' at '{example_path}'.")
                            example_cache[lang_key] = "[Example not available]"
                    
                    full_example_text = example_cache[lang_key]

                    # 6. Fill in the template with data
                    # --- MODIFIED: Use the correct language string for each placeholder ---
                    filled_prompt = prompt_template.replace("<summary example>", full_example_text)
                    filled_prompt = filled_prompt.replace("<code language>", capitalized_language) # For display
                    filled_prompt = filled_prompt.replace("<target language>", lang_key)          # For code block
                    filled_prompt = filled_prompt.replace("<target code>", code)
                    # --- END OF MODIFICATION ---
                    
                    # 7. Save the final prompt to a new file named after the 'id'
                    output_filename = f"{file_id}.txt"
                    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                    
                    with open(output_filepath, 'w', encoding='utf-8') as out_file:
                        out_file.write(filled_prompt)
                        
                    print(f"  -> Successfully created prompt: {output_filename}")

                except KeyError as e:
                    print(f"  -> ERROR: Row {i+2} is missing a required column: {e}. Skipping row.")
                except Exception as e:
                    print(f"  -> ERROR: An unexpected error occurred on row {i+2}: {e}. Skipping row.")

    except FileNotFoundError:
        print(f"FATAL ERROR: The CSV file was not found at '{CSV_FILE_PATH}'.")
        sys.exit(1)

    print("\nPrompt generation complete.")

if __name__ == "__main__":
    create_prompts()