import re
from collections import defaultdict
from safetensors import safe_open

# Define the path to the safetensors file as a constant
SAFETENSORS_FILE_PATH = "1_adapter_model.safetensors"

def inspect_unique_tensor_names(file_path):
  """
  Inspects a .safetensors file to find and count unique tensor name patterns.

  Args:
    file_path: The path to the .safetensors file.
  """
  # Use defaultdict to easily count occurrences of each unique name pattern
  unique_name_counts = defaultdict(int)

  try:
    with safe_open(file_path, framework="pt", device="cpu") as f:
      all_keys = f.keys()
      if not all_keys:
        print("No tensors found in this file.")
        return

      for key in all_keys:
        # Create a generic pattern by replacing numerical layer indices with a wildcard '*'.
        # This helps group tensors that are the same type but for different layers.
        # For example, '...layers.0.self_attn...' and '...layers.1.self_attn...'
        # will both map to the pattern '...layers.*.self_attn...'
        pattern = re.sub(r'\.\d+\.', '.*.', key)
        unique_name_counts[pattern] += 1

    print(f"Inspection Summary for: {file_path}")
    print("=" * 50)
    
    if not unique_name_counts:
      # This case is handled above but included for robustness
      print("No unique tensor name patterns found.")
    else:
      print("Found the following unique tensor name patterns and their counts:")
      for pattern, count in sorted(unique_name_counts.items()):
        print(f"\n -> Pattern: {pattern}")
        print(f"      Count: {count}")
            
    print("\n" + "=" * 50)

  except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
  except Exception as e:
    print(f"An error occurred: {e}")

if __name__ == "__main__":
  inspect_unique_tensor_names(SAFETENSORS_FILE_PATH)