import os

# --- Configuration ---
# Set to False to see what files will be renamed without actually renaming them.
# Set to True to perform the renaming.
PERFORM_RENAME = True
# ---------------------

# Get the directory where the script is located.
current_dir = os.getcwd() 
print(f"Scanning for files in: {current_dir}\n")

# A counter for renamed files
files_renamed_count = 0

# Iterate through each file in the current directory
for filename in os.listdir(current_dir):
    # We only care about files that match the old structure
    if '_shot_' in filename and filename.endswith('_results.csv'):
        try:
            # 1. PARSE the old filename
            # Remove the suffix to make splitting easier
            base_name = filename.removesuffix('_results.csv')
            parts = base_name.split('_')

            # The old structure is ..._{shot_count}_shot_{short_or_long}
            # We can identify it by finding the literal 'shot'
            shot_literal_index = parts.index('shot')
            
            # Extract the relevant pieces based on the position of 'shot'
            shot_count = parts[shot_literal_index - 1]
            short_or_long = parts[shot_literal_index + 1]
            
            # The prefix is everything before the shot_count
            prefix = '_'.join(parts[:shot_literal_index - 1])

            # 2. CONSTRUCT the new filename
            # New structure: {prefix}_{short_or_long}_{shot_count}_results.csv
            new_filename = f"{prefix}_{short_or_long}_{shot_count}_results.csv"

            # 3. RENAME the file
            # It's good practice to print the change before making it
            print(f"  -> Found: '{filename}'")
            print(f"     Rename to: '{new_filename}'")
            
            if PERFORM_RENAME:
                # Create full paths for the rename operation
                old_path = os.path.join(current_dir, filename)
                new_path = os.path.join(current_dir, new_filename)
                
                # Perform the actual rename
                os.rename(old_path, new_path)
                print("     Status: RENAMED")
            else:
                print("     Status: SKIPPED (Dry Run)")
            
            print("-" * 20)
            files_renamed_count += 1

        except (ValueError, IndexError):
            # This will catch files that contain '_shot_' but don't match the
            # expected structure, preventing the script from crashing.
            print(f"  -> Skipping '{filename}': does not match expected pattern.\n")
            continue

if files_renamed_count == 0:
    print("No files matching the pattern were found to rename.")
else:
    mode = "Renamed" if PERFORM_RENAME else "Identified"
    print(f"\nFinished. {mode} {files_renamed_count} file(s).")