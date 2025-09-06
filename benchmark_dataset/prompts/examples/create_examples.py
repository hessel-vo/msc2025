import sys
import os
import csv

# Define constants for file paths/directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XL_CSV_PATH = os.path.join(BASE_DIR, 'examples_xl.csv')
AUTO_CSV_PATH = os.path.join(BASE_DIR, 'examples_auto.csv')

GENERATION_TEMPLATE_ONE_PATH = os.path.join(BASE_DIR, 'examples_template_generation_one.txt')
GENERATION_TEMPLATE_THREE_PATH = os.path.join(BASE_DIR, 'examples_template_generation_three.txt')
SUMMARIZATION_TEMPLATE_ONE_PATH = os.path.join(BASE_DIR, 'examples_template_summarization_one.txt')
SUMMARIZATION_TEMPLATE_THREE_PATH = os.path.join(BASE_DIR, 'examples_template_summarization_three.txt')

GENERATION_DIR = os.path.join(BASE_DIR, 'generation')
SUMMARIZATION_DIR = os.path.join(BASE_DIR, 'summarization')

def get_output_dir(task, source, summary_len):
    source_folder = 'xlcost' if source == 'xl' else 'automotive'
    len_folder = f"{summary_len}_summ"
    if task == 'generation':
        return os.path.join(GENERATION_DIR, source_folder, len_folder)
    else: # summarization
        return os.path.join(SUMMARIZATION_DIR, source_folder, len_folder)

def read_template(task, num_shots):
    if task == 'generation':
        if num_shots == 1:
            template_path = GENERATION_TEMPLATE_ONE_PATH
        else:
            template_path = GENERATION_TEMPLATE_THREE_PATH
    else: # summarization
        if num_shots == 1:
            template_path = SUMMARIZATION_TEMPLATE_ONE_PATH
        else:
            template_path = SUMMARIZATION_TEMPLATE_THREE_PATH

    with open(template_path, 'r') as f:
        return f.read()

def read_csv_data(source):
    csv_path = XL_CSV_PATH if source == 'xl' else AUTO_CSV_PATH
    data_by_lang = {}
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lang = row['language']
            if lang not in data_by_lang:
                data_by_lang[lang] = []
            data_by_lang[lang].append(row)
    return data_by_lang

def create_one_shot_example(template, language_data, summary_len):
    example_data = language_data[0]
    summary_key = f'summary_{summary_len}'
    summary = example_data[summary_key]
    code = example_data['code']
    language = example_data['language']

    return template.replace('<summary>', summary).replace('<code>', code).replace('<language>', language)

def create_three_shot_example(template, language_data, summary_len):
    filled_template = template
    summary_key = f'summary_{summary_len}'
    for i in range(3):
        example_data = language_data[i]
        summary = example_data[summary_key]
        code = example_data['code']
        language = example_data['language']

        filled_template = filled_template.replace(f'<summary{i+1}>', summary)
        filled_template = filled_template.replace(f'<code{i+1}>', code)
        if i == 0:
            filled_template = filled_template.replace('<language>', language)

    return filled_template


def main():
    if len(sys.argv) != 5:
        print("Usage: python create_examples.py <generation|summarization> <xl|auto> <1|3> <short|long>")
        sys.exit(1)

    task, source, num_shots_str, summary_len = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    num_shots = int(num_shots_str)

    # Validate arguments
    if task not in ['generation', 'summarization']:
        print("Error: First argument must be 'generation' or 'summarization'.")
        sys.exit(1)
    if source not in ['xl', 'auto']:
        print("Error: Second argument must be 'xl' or 'auto'.")
        sys.exit(1)
    if num_shots not in [1, 3]:
        print("Error: Third argument must be '1' or '3'.")
        sys.exit(1)
    if summary_len not in ['short', 'long']:
        print("Error: Fourth argument must be 'short' or 'long'.")
        sys.exit(1)


    output_dir = get_output_dir(task, source, summary_len)
    os.makedirs(output_dir, exist_ok=True)

    template = read_template(task, num_shots)
    data_by_lang = read_csv_data(source)

    shot_name = "one" if num_shots == 1 else "three"

    for lang, lang_data in data_by_lang.items():
        if num_shots == 1:
            if len(lang_data) >= 1:
                example_content = create_one_shot_example(template, lang_data, summary_len)
            else:
                print(f"Warning: Not enough data for a one-shot example for language '{lang}'.")
                continue
        else: # num_shots == 3
            if len(lang_data) >= 3:
                example_content = create_three_shot_example(template, lang_data, summary_len)
            else:
                print(f"Warning: Not enough data for a three-shot example for language '{lang}'.")
                continue

        output_filename = f"{shot_name}_shot_{lang}.txt"
        output_filepath = os.path.join(output_dir, output_filename)
        with open(output_filepath, 'w') as f:
            f.write(example_content)
        print(f"Generated example for '{lang}' at: {output_filepath}")


if __name__ == "__main__":
    main()