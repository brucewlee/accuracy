import os, logging, sys
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.CRITICAL)
from nutcracker.data import Task, Pile
from nutcracker.runs import Schema
from nutcracker.evaluator import generate_report, AutoEvaluator, MCQEvaluator, FRQEvaluator

from models import ChatGPT





"""
CRITICAL:::

experiment hyperparameters start
"""

models_to_run = ['gpt-3.5-turbo-1106']

"""
^^^^^^^^^^
experiment hyperparameters end
"""





def run(models_dict, task_dict):
    for task_name, task_object in task_dict.items():
        for model_to_run in models_to_run:
            print(f'Running ... {task_name} on {model_to_run}')

            # create directory
            ckpt_dir = os.path.join('.', 'ckpt')
            model_dir = os.path.join(ckpt_dir, model_to_run)
            task_dir = os.path.join(model_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            print(f"Directory '{task_dir}' is ready for use.")

            # running this experiment updates each instance's model_response property in data data object with ChatGPT responses
            experiment = Schema(
                model = models_dict[model_to_run], 
                data = task_object
            )
            #experiment.run()

            # save checkpoint with retrieved responses
            save_path = f'{task_dir}/{model_to_run}-{task_name}-ckpt-1.pkl'
            #task_object.save_to_file(
            #    save_path = save_path
            #)
            task_object = task_object.load_from_file(
                load_path = save_path
            )

            # running this evaluation updates each instance's response_correct property in data data object with evaluations
            if task_object.config['construction']['class'] == 'mcq':
                """MCQ: engine alpha"""
                #evaluation = MCQEvaluator(
                #    data = task_object,
                #    engine = 'alpha',
                #    )
                #evaluation.run()

                # save checkpoint with retrieved responses + evaluation results
                #save_path = f'{task_dir}/{model_to_run}-{task_name}-ckpt-2-alpha-#{task_object[0].config["few_shot"]}.pkl'
                #task_object.save_to_file(
                #    save_path = save_path
                #)
                #task_object = task_object.load_from_file(
                #    load_path = save_path
                #)

                # save human-readable results for research dissemination and analysis
                #task_object.save_records(
                #    f'{task_dir}/{model_to_run}-{task_name}-records-alpha-{task_object[0].config["few_shot"]}.jsonl', 
                #    keys=[
                #        "config.task_name", 
                #        "config.construction.class", 
                #        "config.few_shot", 
                #        "config.task_name", 
                #        "centerpiece", 
                #        "user_prompt", 
                #        "options", 
                #        "correct_options", 
                #        "model_response", 
                #        "model_response_logprobs",
                #        "response_correct",
                #        "response_evaluator_engine"
                #    ]
                #)
                #generate_report(
                #    task_object, 
                #    save_path = f'{task_dir}/{model_to_run}-{task_name}-report-alpha-{task_object[0].config["few_shot"]}.txt'
                #)



                """MCQ: engine beta"""
                evaluation = MCQEvaluator(
                    data = task_object,
                    engine = 'beta',
                    )
                evaluation.run()
                # save checkpoint with retrieved responses + evaluation results
                save_path = f'{task_dir}/{model_to_run}-{task_name}-ckpt-2-beta-{task_object[0].config["few_shot"]}.pkl'
                task_object.save_to_file(
                    save_path = save_path
                )
                task_object = task_object.load_from_file(
                    load_path = save_path
                )

                # save human-readable results for research dissemination and analysis
                task_object.save_records(
                    f'{task_dir}/{model_to_run}-{task_name}-records-beta-{task_object[0].config["few_shot"]}.jsonl', 
                    keys=[
                        "config.task_name", 
                        "config.construction.class", 
                        "config.few_shot", 
                        "config.task_name", 
                        "centerpiece", 
                        "user_prompt", 
                        "options", 
                        "correct_options", 
                        "model_response", 
                        "model_response_logprobs",
                        "response_correct",
                        "response_evaluator_engine"
                    ]
                )
                generate_report(
                    task_object, 
                    save_path = f'{task_dir}/{model_to_run}-{task_name}-report-beta-{task_object[0].config["few_shot"]}.txt'
                )



                """MCQ: engine gamma"""
                evaluation = MCQEvaluator(
                    data = task_object,
                    engine = 'gamma',
                    )
                evaluation.run()
                # save checkpoint with retrieved responses + evaluation results
                save_path = f'{task_dir}/{model_to_run}-{task_name}-ckpt-2-gamma-{task_object[0].config["few_shot"]}.pkl'
                task_object.save_to_file(
                    save_path = save_path
                )
                task_object = task_object.load_from_file(
                    load_path = save_path
                )

                # save human-readable results for research dissemination and analysis
                task_object.save_records(
                    f'{task_dir}/{model_to_run}-{task_name}-records-gamma-{task_object[0].config["few_shot"]}.jsonl', 
                    keys=[
                        "config.task_name", 
                        "config.construction.class", 
                        "config.few_shot", 
                        "config.task_name", 
                        "centerpiece", 
                        "user_prompt", 
                        "options", 
                        "correct_options", 
                        "model_response", 
                        "model_response_logprobs",
                        "response_correct",
                        "response_evaluator_engine"
                    ]
                )
                generate_report(
                    task_object, 
                    save_path = f'{task_dir}/{model_to_run}-{task_name}-report-gamma-{task_object[0].config["few_shot"]}.txt'
                )



            elif task_object.config['construction']['class'] == 'frq':
                """FRQ: engine alpha"""
                #evaluation = FRQEvaluator(
                #    data = task_object,
                #    engine = 'alpha',
                #    )
                #evaluation.run()
                # save checkpoint with retrieved responses + evaluation results
                #save_path = f'{task_dir}/{model_to_run}-{task_name}-ckpt-2-alpha.pkl'
                #task_object.save_to_file(
                #    save_path = save_path
                #)
                #task_object = task_object.load_from_file(
                #    load_path = save_path
                #)

                # save human-readable results for research dissemination and analysis
                #task_object.save_records(
                #    f'{task_dir}/{model_to_run}-{task_name}-records-alpha-{task_object[0].config["few_shot"]}.jsonl', 
                #    keys=[
                #        "config.task_name", 
                #        "config.construction.class", 
                #        "config.few_shot", 
                #        "config.task_name", 
                #        "centerpiece", 
                #        "user_prompt", 
                #        "options", 
                #        "correct_options", 
                #        "model_response", 
                #        "model_response_logprobs",
                #        "response_correct",
                #        "response_evaluator_engine"
                #    ]
                #)
                #generate_report(
                #    task_object, 
                #    save_path = f'{task_dir}/{model_to_run}-{task_name}-report-alpha-{task_object[0].config["few_shot"]}.txt'
                #)



if __name__ == "__main__":
    # make a dictionary of all tasks available in Nutcracker (version 0.01a7)
    task_dict = {}
    print(Task.list_all()) # task_list
    task_list = [
        'mmlu-high-school-european-history', 'math-algebra', 
        'mmlu-high-school-us-history', 'hhh-alignment-harmless', 'mmlu-high-school-government-and-politics', 'htest-repeated-word', 
        'mmlu-elementary-mathematics', 'mmlu-human-aging', 
        'mmlu-high-school-mathematics', 'mmlu-business-ethics', 
        'socialiqa', 'mmlu-security-studies', 
        'openbookqa', 'mmlu-moral-scenarios', 
        'htest-spelled-number', 'mmlu-high-school-computer-science', 
        'mmlu-miscellaneous', 'arc-challenge', 
        'hhh-alignment-other', 'mmlu-international-law', 
        'mmlu-electrical-engineering', 'mmlu-college-mathematics', 
        'mmlu-professional-medicine', 'mmlu-global-facts', 
        'htest-end-punctuation', 'hhh-alignment-honest', 
        'mmlu-formal-logic', 'mmlu-professional-accounting', 
        'htest-start-vowel', 'mmlu-virology', 
        'arc-easy', 'mmlu-college-biology', 
        'mmlu-college-physics', 'mmlu-anatomy', 
        'mmlu-machine-learning', 'winogrande', 
        'mmlu-high-school-microeconomics', 'htest-hyphenated-word', 
        'mmlu-college-chemistry', 'mmlu-public-relations', 
        'mmlu-astronomy', 'mmlu-high-school-geography', 
        'htest-rhyme', 'htest-uppercase', 
        'mmlu-clinical-knowledge', 'mmlu-abstract-algebra', 
        'htest-end-ly', 'mmlu-college-computer-science', 
        'mmlu-human-sexuality', 'mmlu-high-school-chemistry', 
        'mmlu-high-school-world-history', 'math-prealgebra', 
        'mmlu-high-school-statistics', 'mmlu-management', 
        'aqua-rat', 'mmlu-prehistory', 
        'math-precalculus', 'mmlu-world-religions', 
        'mmlu-sociology', 'medqa-usmle', 
        'mmlu-college-medicine', 'mmlu-jurisprudence', 
        'mmlu-high-school-macroeconomics', 'piqa', 
        'htest-palindrome', 'gsm8k', 
        'mmlu-professional-law', 'math-geometry', 
        'mmlu-us-foreign-policy', 'htest-spelled-math', 
        'math-number-theory', 'mmlu-medical-genetics', 
        'mmlu-conceptual-physics', 'math-counting-and-probability', 
        'truthfulqa-mc1', 'mmlu-professional-psychology', 
        'mmlu-moral-disputes', 'mmlu-high-school-biology', 
        'mmlu-high-school-physics', 'math-intermediate-algebra', 
        'hhh-alignment-helpful', 'hellaswag', 
        'mmlu-nutrition', 'commonsenseqa', 
        'mmlu-marketing', 'mmlu-econometrics', 
        'mmlu-philosophy', 'mmlu-computer-security', 
        'mmlu-logical-fallacies', 'mmlu-high-school-psychology'
    ]
    for task_name in task_list:
        task_object = Task.load_from_db(
            task_name = task_name, 
            db_directory = 'nutcracker-db/db'
            )
        #task_object.sample(5, seed = 1, in_place = True)
        task_dict[f'{task_name}'] = task_object
    # 90 tasks
    print(f'Number of tasks: {len(task_dict)}')
    # 45566 instances
    print(f'Number of instances: {sum([len(data) for data in task_dict.values()])}')

    # models dictionary
    models_dict = {
        'gpt-3.5-turbo-1106': ChatGPT()
    }

    # run
    run(models_dict, task_dict)
