import os
from pathlib import Path
from collections import Counter

def get_extension_counts_from_file(input_file, output_file):
    """
    Reads filepaths from a text file, counts each extension's occurrences,
    and writes the results to a CSV-formatted output file.

    The output format is: extension,count,common (yes/no if count >= 4)

    Args:
        input_file (str or Path): The path to the input .txt file containing filepaths.
        output_file (str or Path): The path for the output .txt file.
    """
    extension_counter = Counter()
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            filepaths = f.read().splitlines()
            for line in filepaths:
                extension = Path(line).suffix
                if extension:  # Only count non-empty extensions
                    extension_counter.update([extension])
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
        return

    # Write the results to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write the header for the CSV file
        f.write("extension,count,common\n")
        
        # Sort extensions alphabetically for consistent output
        for extension, count in sorted(extension_counter.items()):
            common_status = "yes" if count >= 4 else "no"
            f.write(f"{extension},{count},{common_status}\n")
            
    print(f"Extension counts from '{input_file}' have been written to '{output_file}'")

def get_extension_counts_from_folder(folder_path, output_file):
    """
    Recursively finds and counts all file extensions in a folder,
    and writes the results to a CSV-formatted output file.

    The output format is: extension,count,common (yes/no if count >= 4)

    Args:
        folder_path (str or Path): The path to the folder to search.
        output_file (str or Path): The path for the output .txt file.
    """
    repos_dir = Path(folder_path)
    if not repos_dir.is_dir():
        print(f"Error: The folder '{folder_path}' does not exist or is not a directory.")
        return

    extension_counter = Counter()
    
    # Use os.walk for efficient directory traversal
    for dirpath, _, filenames in os.walk(repos_dir):
        for filename in filenames:
            extension = Path(filename).suffix
            if extension:  # Only count non-empty extensions
                extension_counter.update([extension])

    # Write the results to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write the header for the CSV file
        f.write("extension,count,common\n")

        # Sort extensions alphabetically for consistent output
        for extension, count in sorted(extension_counter.items()):
            common_status = "yes" if count >= 4 else "no"
            f.write(f"{extension},{count},{common_status}\n")
            
    print(f"All extension counts from '{folder_path}' have been written to '{output_file}'")


if __name__ == '__main__':
    # --- Part 1: Process filepaths from a .txt file ---
    # TO USE:
    # 1. Replace with the actual path to your input file.
    #    Examples: 'C:/Users/YourUser/Desktop/files.txt' or '/home/user/files.txt'
    filepaths_input_file = './scripts/data_preparation/01_filtering/filtered_file_paths.txt'
    filtered_extensions_output_file = 'filtered_extensions.txt'

    print("--- Starting Part 1: Processing extensions from input file ---")
    get_extension_counts_from_file(filepaths_input_file, filtered_extensions_output_file)
    print("-" * 60)


    # --- Part 2: Process files from a folder of repositories ---
    # TO USE:
    # 1. Replace with the path to your folder containing all the repositories.
    #    Examples: 'C:/Users/YourUser/Desktop/github' or '/home/user/github_repos'
    repos_folder_path = './repositories/all_repos'
    all_extensions_output_file = 'all_extensions.txt'

    print("--- Starting Part 2: Processing extensions from repositories folder ---")
    get_extension_counts_from_folder(repos_folder_path, all_extensions_output_file)
    print("-" * 60)