import torch
import os
from transformers import TrainingArguments # It's good practice to import the class

def inspect_trainer_args(file_path):
  """
  Loads and prints the contents of a trainer_args.bin file.

  Args:
    file_path: The path to the trainer_args.bin file.
  """
  if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    return

  try:
    # The fix is to add `weights_only=False`.
    # This is safe because you are loading a file you generated yourself.
    training_args = torch.load(file_path, weights_only=False)
    
    print("Successfully loaded trainer_args.bin")
    print("-" * 30)
    print("Training Arguments:")
    
    # vars() converts the object's attributes to a dictionary for easy printing
    for arg, value in sorted(vars(training_args).items()):
      print(f"{arg}: {value}")
      
    print("-" * 30)
  except Exception as e:
    print(f"An error occurred while loading the file: {e}")

if __name__ == "__main__":
  # IMPORTANT: Replace with the actual path to your trainer_args.bin file
  # For example: "./my_model_checkpoint/checkpoint-500/trainer_args.bin"
  file_to_inspect = "./training_args.bin"
  inspect_trainer_args(file_to_inspect)