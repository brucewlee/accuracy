import os
import json
from tqdm import tqdm

def update_records_with_additional_keys(base_directory):
    # Walk through each subdirectory in the base directory
    for root, dirs, files in os.walk(base_directory):
        for dir_name in dirs:
            task_dir = os.path.join(root, dir_name)
            beta_files = [f for f in os.listdir(task_dir) if 'records-beta-' in f and f.endswith('.jsonl')]
            regular_files = [f for f in os.listdir(task_dir) if 'records.jsonl' in f]

            if not beta_files or not regular_files:
                print(f"Skipping {task_dir}, missing required files.")
                continue

            # Assuming there's only one of each file type per task directory
            beta_file_path = os.path.join(task_dir, beta_files[0])
            regular_file_path = os.path.join(task_dir, regular_files[0])
            
            # Extract beta value from the beta file name for naming the new alpha file
            beta_value = beta_files[0].split('-')[-1].replace('.jsonl', '')
            
            # Construct the new alpha file name based on the beta file's unique value
            new_file_name = f"gpt-3.5-turbo-1106-{dir_name}-records-alpha-{beta_value}.jsonl"
            new_file_path = os.path.join(task_dir, new_file_name)

            # Read and merge the records
            with open(beta_file_path, 'r', encoding='utf-8') as beta_file, open(regular_file_path, 'r', encoding='utf-8') as regular_file:
                beta_records = [json.loads(line) for line in beta_file]
                regular_records = [json.loads(line) for line in regular_file]

                updated_records = []
                for regular_record in regular_records:
                    updated_record = regular_record.copy()  # Make a copy to update
                    for beta_record in beta_records:
                        # Merge missing keys/values from beta record to regular record
                        for key, value in beta_record.items():
                            if key not in updated_record:
                                if key == "response_evaluator_engine":
                                    value = "mcq-engine-alpha"
                                updated_record[key] = value
                    updated_records.append(updated_record)

                # Write the updated records to the new alpha file
                with open(new_file_path, 'w', encoding='utf-8') as new_file:
                    for record in tqdm(updated_records):
                        json.dump(record, new_file)
                        new_file.write('\n')
                
            print(f"Updated records saved to {new_file_path}")

# Example usage
base_directory = 'ckpt/gpt-3.5-turbo-1106'
update_records_with_additional_keys(base_directory)


