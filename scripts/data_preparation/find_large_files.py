import os

def find_large_files(input_filepath, output_filepath, size_limit_mb=450):
    """
    Reads a list of filepaths from a text file, checks if each file's size
    exceeds a given limit, and writes the paths of large files to a new text file.

    Args:
        input_filepath (str): The path to the input .txt file containing filepaths.
        output_filepath (str): The path to the output .txt file to store large filepaths.
        size_limit_mb (int): The size limit in megabytes.
    """
    # Convert size limit from megabytes to bytes
    size_limit_bytes = size_limit_mb * 1024 * 1024
    large_files = []

    try:
        with open(input_filepath, 'r') as infile:
            for line in infile:
                filepath = line.strip()
                if os.path.exists(filepath):
                    try:
                        file_size = os.path.getsize(filepath)
                        if file_size > size_limit_bytes:
                            large_files.append(filepath)
                            print(f"Found large file: {filepath} ({file_size / (1024*1024):.2f} MB)")
                    except OSError as e:
                        print(f"Could not access {filepath}: {e}")
                else:
                    print(f"File not found: {filepath}")
    except FileNotFoundError:
        print(f"Error: The input file '{input_filepath}' was not found.")
        return

    try:
        with open(output_filepath, 'w') as outfile:
            for filepath in large_files:
                outfile.write(filepath + '\n')
        print(f"\nSuccessfully wrote {len(large_files)} large file(s) to '{output_filepath}'")
    except IOError as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # Set the path to your input text file with the list of filepaths.
    # Make sure each filepath is on a new line.
    input_file = "./scripts/data_preparation/01_filtering/filtered_file_paths_extensive.txt"

    # Set the desired name for the output text file.
    output_file = "large_files_list.txt"

    # Set the size limit in Megabytes (MB).
    size_limit = 0.45
    # -------------------

    find_large_files(input_file, output_file, size_limit)