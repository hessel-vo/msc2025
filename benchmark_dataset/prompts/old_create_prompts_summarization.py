import os
import csv
import sys
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
TEMPLATE_FILE_PATH = "prompt_template_summarization.txt"
CSV_FILE_PATH = "benchmark_dataset.csv"
OUTPUT_DIR = "prompts_summarization"
EXAMPLES_DIR = "examples/summarization"

def load_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found found at '{filepath}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading '{filepath}': {e}")
        sys.exit(1)

def create_prompts():
    # 1. Prompt template
    print(f"Loading prompt template from '{TEMPLATE_FILE_PATH}'...")
    prompt_template = load_text_file(TEMPLATE_FILE_PATH)

    # 2. Output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to the '{OUTPUT_DIR}' directory.")

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
                    
                    code_from_csv = row['code']
                    code = '\n'.join(code_from_csv.splitlines())
                    
                    # For display text (e.g., "Python", "Java")
                    capitalized_language = language.capitalize()
                    # For file lookups and code blocks (e.g., "python", "java")
                    lang_key = language.lower()

                    # 5. Load the language-specific example (use cache if available)
                    if lang_key not in example_cache:
                        example_path = os.path.join(EXAMPLES_DIR, f"summarization/{num_examples}_shot_example_{lang_key}.txt")
                        if os.path.exists(example_path):
                            example_cache[lang_key] = load_text_file(example_path)
                            print(f"Loaded example for '{language}' from '{example_path}'.")
                            print(example_cache[lang_key])
                        else:
                            print(f"Example file missing '{language}' at '{example_path}'.")
                            sys.exit(1)

                    full_example_text = example_cache[lang_key]

                    # 6. Fill template with data
                    filled_prompt = prompt_template.replace("<summary example>", full_example_text)
                    filled_prompt = filled_prompt.replace("<code language>", capitalized_language) # For display
                    filled_prompt = filled_prompt.replace("<target language>", lang_key)          # For code block
                    filled_prompt = filled_prompt.replace("<target code>", code)
                    
                    # 7. Save the final prompt to a new file using name='id'
                    output_filename = f"{file_id}.txt"
                    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                    
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

if __name__ == "__main__":
    create_prompts()