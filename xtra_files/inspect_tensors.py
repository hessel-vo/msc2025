from collections import defaultdict
from safetensors import safe_open

# Define the path to the safetensors file as a constant
SAFETENSORS_FILE_PATH = "1_adapter_model.safetensors"
SAMPLE_SIZE = 5

def inspect_tensor_types(file_path):
  """
  Inspects a .safetensors file to count unique tensor types and display a sample.

  Args:
    file_path: The path to the .safetensors file.
  """
  tensor_type_info = defaultdict(lambda: {"count": 0, "samples": []})

  try:
    with safe_open(file_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        tensor_info = f.get_tensor(key)
        dtype = tensor_info.dtype
        
        # Add to samples if the list is not full
        if len(tensor_type_info[dtype]["samples"]) < SAMPLE_SIZE:
          tensor_type_info[dtype]["samples"].append(key)
        
        # Increment the count for this data type
        tensor_type_info[dtype]["count"] += 1

    print(f"Inspection Summary for: {file_path}")
    print("=" * 50)
    
    if not tensor_type_info:
        print("No tensors found in this file.")
    else:
        for dtype, info in tensor_type_info.items():
          print(f"\nData Type: {dtype}")
          print(f"  - Total Count: {info['count']}")
          print(f"  - Sample Tensors (up to {SAMPLE_SIZE}):")
          for sample in info['samples']:
            print(f"    - {sample}")
            
    print("\n" + "=" * 50)

  except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
  except Exception as e:
    print(f"An error occurred: {e}")

if __name__ == "__main__":
  inspect_tensor_types(SAFETENSORS_FILE_PATH)