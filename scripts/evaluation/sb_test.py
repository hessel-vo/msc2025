# You would need to: pip install sacrebleu
import sacrebleu

prediction = "Checks if a value is a valid xintx integer within specified signedness and bit constraints."
reference = "Checks if an integer value fits within the range of a specified bit-width and signedness."

# Using the original library's sentence_bleu function
# This is what use_effective_order=True is supposed to do
results = sacrebleu.sentence_bleu(
    prediction,
    [reference],
    smooth_method='exp', # The default smoothing in HF
    use_effective_order=True
)

# sentence_bleu with smoothing turned off also works as expected
results_no_smooth = sacrebleu.sentence_bleu(
    prediction,
    [reference],
    smooth_method='none',
    use_effective_order=True
)

print(f"Direct Sacrebleu (exp smooth): {results.score}")
# Expected output: A score like 7.31...

print(f"Direct Sacrebleu (no smooth): {results_no_smooth.score}")
# Expected output: A non-zero score, likely around 20.4,
# calculated from sqrt(62.5 * 6.67)