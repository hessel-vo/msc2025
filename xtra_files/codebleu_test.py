import evaluate

print("Attempting to load the community-hosted CodeBLEU metric...")

try:
    # Load the community metric from the Hugging Face Hub.
    # The path "dvitel/codebleu" points to a community space, not a canonical metric.
    codebleu_metric = evaluate.load("dvitel/codebleu")
    
    print("Successfully loaded the CodeBLEU metric from 'dvitel/codebleu'.")

    # Example data to test the metric
    prediction = "def add(a, b):\n return a + b"
    # Note: References must be a list of lists
    reference = [["def sum(first, second):\n return second + first"]]

    # Compute the metric
    results = codebleu_metric.compute(
        predictions=[prediction],
        references=reference,
        lang="python"
    )

    print("\nMetric computed successfully!")
    print("Results:", results)

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("This could mean the metric is not available or a dependency is missing.")
    print("Please ensure you have run 'pip install evaluate codebleu tree-sitter-python'.")