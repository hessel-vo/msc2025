import csv
import os
from urllib.parse import urlparse

# Define the input CSV file path
# You can change this to the actual path of your input file
INPUT_CSV_FILE = 'repos_relevance.csv'

# Define the output CSV file path
OUTPUT_CSV_FILE = 'processed_repos.csv'

def process_csv():
    """
    Reads repository data from the input CSV, processes it, 
    and writes the formatted data to the output CSV.
    """
    # Define the header for the output CSV file
    output_header = ["Organization", "Repository name", "Code language", "Relevance"]

    # Check if the input file exists before proceeding
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"Error: The file '{INPUT_CSV_FILE}' was not found.")
        return

    try:
        with open(INPUT_CSV_FILE, mode='r', encoding='utf-8') as infile, \
             open(OUTPUT_CSV_FILE, mode='w', encoding='utf-8', newline='') as outfile:

            # Create a CSV reader and writer
            reader = csv.DictReader(infile)
            writer = csv.writer(outfile)

            # Write the header to the new CSV file
            writer.writerow(output_header)

            # Process each row from the input CSV
            for row in reader:
                repo_link = row.get('repo link', '')
                language = row.get('language', '')
                relevance_filter = row.get('relevance_filter', '0')

                # Skip rows with missing repository links
                if not repo_link:
                    continue
                
                # Parse the repository link to extract the organization and repository name
                try:
                    parsed_url = urlparse(repo_link)
                    path_parts = parsed_url.path.strip('/').split('/')
                    
                    if len(path_parts) >= 2:
                        organization = path_parts[0]
                        repository_name = path_parts[1]
                    else:
                        organization, repository_name = "Unknown", "Unknown"
                
                except Exception as e:
                    print(f"Could not parse URL {repo_link}: {e}")
                    organization, repository_name = "Unknown", "Unknown"
                
                language = "C++" if language.lower() == "cpp" else language.capitalize()
                
                # Write the processed data to the new CSV file
                writer.writerow([organization, repository_name, language, relevance_filter])

        print(f"Successfully created '{OUTPUT_CSV_FILE}'")

    except FileNotFoundError:
        print(f"Error: Could not find the input file '{INPUT_CSV_FILE}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
      
    # Run the processing function
    process_csv()