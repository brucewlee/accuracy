from openai import OpenAI
import os, sys
import numpy as np
os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()

class ChatGPT:
    def __init__(self):
        self.model = "gpt-3.5-turbo-1106"

    def log_probs_preprocess(self, logprobs_raw):
        """
        Given a logprobs_raw like this
        [
            ChatCompletionTokenLogprob(
                token='A', bytes=[65], logprob=-0.059714206, top_logprobs=[TopLogprob(token='A', bytes=[65], logprob=-0.059714206), TopLogprob(token='C', bytes=[67], logprob=-3.2466688), TopLogprob(token=' A', bytes=[32, 65], logprob=-4.2426405), TopLogprob(token='B', bytes=[66], logprob=-5.7856627), TopLogprob(token='D', bytes=[68], logprob=-7.175002)]
            ), 
            ChatCompletionTokenLogprob(
                token='.', bytes=[46], logprob=-0.05477338, top_logprobs=[TopLogprob(token='.', bytes=[46], logprob=-0.05477338), TopLogprob(token='<|end|>', bytes=None, logprob=-3.189314), TopLogprob(token=' \n', bytes=[32, 10], logprob=-5.8193235), TopLogprob(token=' (', bytes=[32, 40], logprob=-6.292186), TopLogprob(token='\n\n', bytes=[10, 10], logprob=-6.382016)]
            ), ...
        ]

        Parses to
        [
            [
                ('A', -0.059714206, '94.2%'), ('C', -3.2466688, '3.89%'), (' A', -4.2426405, '1.44%'), ('B', -5.7856627, '0.31%'), ('D', -7.175002, '0.08%')
            ], 
            [
                ('.', -0.05477338, '94.67%'), ('<|end|>', -3.189314, '4.12%'), (' \n', -5.8193235, '0.3%'), (' (', -6.292186, '0.19%'), ('\n\n', -6.382016, '0.17%')
            ], ...
        ]
        """

        # preprocess for nutcracker
        logprobs = []
        for position in logprobs_raw:
            logprobs_at_one_position = []

            for option_at_one_position in position.top_logprobs:
                option = (
                    # A
                    option_at_one_position.token,
                    # -2.45..
                    option_at_one_position.logprob,
                    # optional linear transformation (Nutcracker ignores this third item)
                    f'{np.round(np.exp(option_at_one_position.logprob)*100,2)}%' 
                )
                logprobs_at_one_position.append(option)
            logprobs.append(logprobs_at_one_position)

        return logprobs

    def respond(self, user_prompt):
        response = None
        while response is None:
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": f"{user_prompt}"}
                    ],
                    timeout=15,
                    logprobs=True,
                    top_logprobs=5,
                    seed=1
                )

                # passing in logprobs to Nutcracker is optional
                logprobs_raw = completion.choices[0].logprobs.content
                logprobs = self.log_probs_preprocess(logprobs_raw)
                response = completion.choices[0].message.content
                break
            except KeyboardInterrupt:
                sys.exit()
            except Exception as error:
                print(error)
        # nutcracker accepts either a string raw response
        # or a tuple of (string raw response, list of list of tuple logprobs)
        return response, logprobs
    

class Gemma7BIt:
    def __init__(self):
        self.API_URL = ""

    def query(self, payload):
        headers = {
            "Accept" : "application/json",
            "Content-Type": "application/json" 
        }
        response = requests.post(self.API_URL, headers=headers, json=payload)
        return response.json()

    def respond(self, user_prompt):
        output = self.query({
            "inputs": f"<s>[INST] <<SYS>> You are a helpful assistant. You keep your answers short. <</SYS>> {user_prompt}",
        })
        return output[0]['generated_text']