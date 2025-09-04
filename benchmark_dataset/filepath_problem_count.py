import pandas as pd

# Define the constant for the input CSV file path
CSV_FILE_PATH = 'benchmark_dataset.csv'

def analyze_filepath_problems(file_path):
    """
    Analyzes a CSV file to count the number of problems for each filepath.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        pandas.Series: A Series containing the counts of problems for each filepath.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Count the occurrences of each unique value in the 'filepath' column
        filepath_counts = df['filepath'].value_counts()

        return filepath_counts

    except FileNotFoundError:
        return f"Error: The file '{file_path}' was not found."
    except KeyError:
        return "Error: The CSV file must contain a 'filepath' column."

if __name__ == "__main__":
    # Analyze the CSV file and get the problem counts for each filepath
    problem_counts = analyze_filepath_problems(CSV_FILE_PATH)

    # Print the results
    if isinstance(problem_counts, pd.Series):
        print("Number of problems per filepath:")
        print(problem_counts)
    else:
        print(problem_counts)