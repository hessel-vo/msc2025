import numpy as np
from dotenv import load_dotenv
import sys
import os
from pathlib import Path

load_dotenv()

project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)

HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

import evaluate


def run_rouge_tests():
    """
    Performs a series of tests on the Hugging Face ROUGE metric
    to empirically verify its behavior.
    """
    print("--- Loading ROUGE metric from Hugging Face evaluate ---")
    try:
        rouge = evaluate.load('rouge')
    except Exception as e:
        print(f"Failed to load the metric. Please ensure you have the necessary libraries installed:")
        print("pip install evaluate rouge_score")
        print(f"Error: {e}")
        return

    # --- Test Case 1: Batch vs. Looping (for single-sentence data) ---
    print("\n" + "="*50)
    print("Test Case 1: Batch vs. Manual Loop/Average")
    print("="*50)
    print("We will compare scores from processing a list of sentences all at once")
    print("versus looping through each pair and averaging the scores manually.\n")

    # Sample data with multiple single-sentence pairs
    predictions_single = [
        "the cat sat on the mat",
        "the dog played in the garden"
    ]
    references_single = [
        "the cat was on the mat",
        "a dog was playing in a garden"
    ]

    # Method 1: Batch processing (the standard way)
    print("--- Method 1: Batch Processing ---")
    batch_results = rouge.compute(predictions=predictions_single, references=references_single)
    print(batch_results)

    # Method 2: Manual loop and average
    print("\n--- Method 2: Manual Loop and Average ---")
    loop_scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}

    batch_results = rouge.compute(predictions=predictions_single, references=references_single, use_aggregator=False)
    print(batch_results)

    # for pred, ref in zip(predictions_single, references_single):
    #     score = rouge.compute(predictions=[pred], references=[ref], use_aggregator=False)
    #     for key in loop_scores:
    #         loop_scores[key].append(score[key])

    # # Calculate the average for each ROUGE type
    # averaged_results = {key: np.mean(values) for key, values in loop_scores.items()}
    # print(averaged_results)

    # print("\n*Conclusion for Test 1:*")
    # print("The results are identical. This shows that the batch calculation is equivalent")
    # print("to averaging the scores of individual prediction-reference pairs.")


    # # --- Test Case 2: rougeL vs. rougeLsum ---
    # print("\n" + "="*50)
    # print("Test Case 2: rougeL vs. rougeLsum Comparison")
    # print("="*50)
    # print("We will now see how these two metrics differ when handling")
    # print("single-sentence vs. multi-sentence inputs.\n")

    # # Sub-test 2a: Single-sentence (using results from above)
    # print("--- Sub-test 2a: Single-Sentence Input ---")
    # print("Using the batch results from Test Case 1:")
    # print(f"  rougeL:  {batch_results['rougeL']}")
    # print(f"  rougeLsum: {batch_results['rougeLsum']}")
    # print("\n*Conclusion for 2a:*")
    # print("With single-sentence inputs, rougeL and rougeLsum produce identical scores.")

    # # Sub-test 2b: Multi-sentence input with reordered sentences
    # print("\n--- Sub-test 2b: Multi-Sentence Input ---")
    # print("Here, the prediction has the same content as the reference, but the sentences are swapped.")
    # print("Sentences are separated by a newline character '\\n'.\n")

    # predictions_multi = ["A new policy was announced. The president gave a speech."]
    # references_multi = ["The president gave a speech. A new policy was announced."]
    
    # # Note: For the library to correctly process this as multi-sentence for rougeLsum,
    # # we need to join with \n. If they were already formatted like that, it would also work.
    # predictions_multi_formatted = ["\n".join(predictions_multi[0].split(". "))]
    # references_multi_formatted = ["\n".join(references_multi[0].split(". "))]

    # print(f"Prediction: {predictions_multi_formatted[0].replace(chr(10), ' ')}")
    # print(f"Reference:  {references_multi_formatted[0].replace(chr(10), ' ')}\n")

    # multi_sentence_results = rouge.compute(
    #     predictions=predictions_multi_formatted,
    #     references=references_multi_formatted
    # )
    # print(multi_sentence_results)

    # print("\n*Conclusion for 2b:*")
    # print(f"  - rougeL ({multi_sentence_results['rougeL']:.4f}) is low because it compares sentence by sentence and finds a mismatch.")
    # print(f"  - rougeLsum ({multi_sentence_results['rougeLsum']:.4f}) is high because it treats the entire text as one unit and finds that all the content is present.")

if __name__ == '__main__':
    run_rouge_tests()