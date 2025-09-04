import evaluate
import math

# 1. Load the sacrebleu metric from the Hugging Face library
print("Loading 'sacrebleu' metric from Hugging Face evaluate...")
sacrebleu_metric = evaluate.load("sacrebleu")
print("Metric loaded successfully.")

# 2. Define the problematic sentence pair with 0-count high-order n-grams
prediction = "Checks if a value is a valid xintx integer within specified signedness and bit constraints."
reference = "Checks if an integer value fits within the range of a specified bit-width and signedness."

print("\n" + "="*60)
print("              Test Case: Single Sentence Pair")
print("="*60)
print(f"Prediction: {prediction}")
print(f"Reference:  {reference}\n")

# 3. --- Run the computation with `use_effective_order=True` ---
# We only need to run this once, as we've established the score is the same
# whether the flag is True or False.
use_e_order = True
print("\n" + "="*60)
print(f"  Running Computation with `use_effective_order={use_e_order}`")
print("="*60)

results = sacrebleu_metric.compute(
    predictions=[prediction],
    references=[[reference]],
    smooth_method='none',
    use_effective_order=use_e_order
)

print("\n--- Library Output ---")
print(f"Raw Counts:      {results['counts']}")
print(f"Precisions Array:  {[round(p, 2) for p in results['precisions']]}")
print(f"Brevity Penalty:   {results['bp']}")
print(f"Library Score:     {results['score']}")

