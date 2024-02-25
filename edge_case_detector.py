import os
import glob
import json

def find_edge_cases(base_directory, model_name, engine):
    edge_cases = []
    pattern = os.path.join(base_directory, model_name, "*", f"{model_name}-*-records-{engine}-*.jsonl")

    for file_path in glob.glob(pattern):
        with open(file_path, 'r') as file:
            for line in file:
                record = json.loads(line)
                
                # Check if evaluated by beta
                correct_options = record.get("correct_options", [])
                logprobs = record.get("model_response_logprobs", [])
                
                # The chosen response is the first token text of the first entry in the first logprobs array
                chosen_response = logprobs[0][0][0]
                # Check if chosen response is among correct options and response is marked incorrect
                print(chosen_response)
                print(correct_options)
                if chosen_response in correct_options and not record.get("response_correct", True):
                    print("edge")
                    edge_cases.append(record)
    
    return edge_cases

def save_edge_cases_to_jsonl(edge_cases, output_filename):
    with open(output_filename, 'w') as file:
        for case in edge_cases:
            # Extracting the first logprob value (assuming structure is as expected)
            first_logprob = case.get("model_response_logprobs", [[]])
            first_logprob = first_logprob[0][0]

            
            filtered_case = {
                "config.task_name": case.get("config.task_name"),
                "config.few_shot": case.get("config.few_shot"),
                "centerpiece": case.get("centerpiece"),
                "options": case.get("options"),
                "correct_options": case.get("correct_options"),
                "model_response": case.get("model_response"),
                "first_logprob": first_logprob,
                "response_correct": case.get("response_correct"),
                "response_evaluator_engine": case.get("response_evaluator_engine"),
            }
            file.write(json.dumps(filtered_case) + '\n')

# Example usage
base_directory = "ckpt"
model_name = "gpt-3.5-turbo-1106"
edge_cases = find_edge_cases(base_directory, model_name, 'alpha')

# Specify the output filename for the edge cases
output_filename = "edge_cases_evaluation.jsonl"
save_edge_cases_to_jsonl(edge_cases, output_filename)

print(f"Edge cases saved to {output_filename}.")
