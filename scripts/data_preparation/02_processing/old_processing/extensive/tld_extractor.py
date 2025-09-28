import csv
import os

if __name__ == "__main__":
    csv_file_path = "./scripts/data_preparation/02_processing/extensive/pii_detections_log_extensive.csv"
    output_dir = os.path.dirname(csv_file_path)
    output_file_path = os.path.join(output_dir, "extracted_tlds.txt")
    
    EMAIL_PII_TYPE = '[EMAIL_ADDRESS]'
    
    found_tlds = set()

    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['pii_type'] == EMAIL_PII_TYPE:
                email_string = row['matched_value']
                domain_part = email_string.split('@')[-1]
                tld = domain_part.split('.')[-1]
                
                cleaned_tld = ''.join(filter(str.isalpha, tld))
                
                if cleaned_tld:
                    found_tlds.add(cleaned_tld.lower())

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for tld in sorted(list(found_tlds)):
            f_out.write(f"{tld}\n")