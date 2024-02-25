import os
import glob
import json
import csv

def calculate_accuracy(records):
    """Calculate accuracy from records."""
    correct_responses = sum(record.get("response_correct", False) for record in records)
    total_responses = len(records)
    accuracy_percentage = (correct_responses / total_responses * 100) if total_responses else 0
    return round(accuracy_percentage, 2)

def process_task_directories(base_directory, model_name):
    """Process task directories and files based on the given model name."""
    accuracy_data = {}

    model_base_path = os.path.join(base_directory, model_name)
    if not os.path.exists(model_base_path):
        print(f"Model directory {model_base_path} does not exist.")
        return accuracy_data

    for task_dir in os.listdir(model_base_path):
        task_path = os.path.join(model_base_path, task_dir)
        if os.path.isdir(task_path):
            task_accuracy_data = {}
            for file in glob.glob(os.path.join(task_path, f"{model_name}-{task_dir}-records-*-*.jsonl")):
                strategy = file.split('-')[-2]
                with open(file, 'r') as f:
                    records = [json.loads(line.strip()) for line in f]
                task_accuracy_data[strategy] = calculate_accuracy(records)
            accuracy_data[task_dir] = task_accuracy_data

    return accuracy_data

def save_to_csv(accuracy_data, filename):
    """Dynamically save accuracy data to a CSV file based on available strategies."""
    # Dynamically determine fieldnames based on encountered strategies
    all_strategies = set(strategy for accuracies in accuracy_data.values() for strategy in accuracies)
    fieldnames = ['Task'] + [f"{strategy} Accuracy (%)" for strategy in sorted(all_strategies)]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for task, accuracies in accuracy_data.items():
            row = {'Task': task}
            # Update row with available accuracies for each strategy
            for strategy, acc in accuracies.items():
                row[f"{strategy} Accuracy (%)"] = acc
            writer.writerow(row)

def save_to_jsonl(accuracy_data, filename):
    """Save accuracy data to a JSONL file."""
    with open(filename, 'w') as jsonlfile:
        for task, accuracies in accuracy_data.items():
            data = {'Task': task}
            data.update(accuracies)
            jsonlfile.write(json.dumps(data) + '\n')

# Example usage
base_directory = "ckpt"
model_name = "gpt-3.5-turbo-1106"
accuracy_data = process_task_directories(base_directory, model_name)

# Save to CSV and JSONL
csv_filename = f"accuracy_data_{model_name}.csv"
jsonl_filename = f"accuracy_data_{model_name}.jsonl"
save_to_csv(accuracy_data, csv_filename)
save_to_jsonl(accuracy_data, jsonl_filename)

print(f"Accuracy data saved to {csv_filename} and {jsonl_filename}.")
