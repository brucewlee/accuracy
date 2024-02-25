from nutcracker.data import Task, Pile
task_object = Task.load_from_file(
    load_path = '/Users/bruce/Research&Project/WalnutResearch/accuracy/ckpt/gpt-3.5-turbo-1106/mmlu-virology/gpt-3.5-turbo-1106-mmlu-virology-ckpt-1.pkl'
)
for instance in task_object:
    print(instance.model_response)