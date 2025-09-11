import pandas as pd

CSV_FILE_PATH = 'benchmark_dataset.csv'

def analyze_filepath_problems(file_path):
    df = pd.read_csv(file_path)

    filepath_counts = df['filepath'].value_counts()

    return filepath_counts

if __name__ == "__main__":
    problem_counts = analyze_filepath_problems(CSV_FILE_PATH)

    if isinstance(problem_counts, pd.Series):
        print("Number of problems per filepath:")
        print(problem_counts)
    else:
        print(problem_counts)