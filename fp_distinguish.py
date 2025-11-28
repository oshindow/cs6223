import json
import re 

prompt = """
    # OBJECTIVE #
    Extract the main objects and the descriptive attributes from the following question.
    Return JSON with fields "objects" and "attributes".
    
    # OBJECTIVE #
    1. only output JSON format.

    ###
    Here are some <<EXAMPLES>>:
    Question: "Does the utensil on top of the table look clean and black?"
    Answer:  {
        "objects": ["utensil", "table"],
        "attributes": ["clean", "black"]
    }
    ###

    # Instruction #
    Generate the the JSON for the following query:
    <<Query>>: INSET_QUERY_HERE
    """



"""
Adding a new functionality is easy. Just implement your new model as a subclass of BaseModel.
The code will make the rest: it will make it available for the processes to call by using
process(name, *args, **kwargs), where *args and **kwargs are the arguments of the models process() method.
"""
 
import openai
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
 
import warnings
from PIL import Image
from collections import Counter
from contextlib import redirect_stdout
from functools import partial
from itertools import chain
from joblib import Memory
from rich.console import Console
from torch import hub
from torch.nn import functional as F
from torchvision import transforms
from typing import List, Union
import requests
import io
import time


from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

from configs import config
from utils import HiddenPrints, format_and_check_string
import multiprocessing as mp
from openai import OpenAI

class CodexModel():
    name = 'codex'
    requires_gpu = False
    max_batch_size = 5

    # Not batched, but every call will probably be a batch (coming from the same process)

    def __init__(self):
        super().__init__()
        with open("llm.prompt") as f:
            self.base_prompt = f.read().strip()
        self.fixed_code = None
         
        model_id = f"meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f'Loaded the model: {model_id}')


    def count_returns(self, input_string):
        # Split the input string into lines
        lines = input_string.split('\n')

        # Initialize a counter for "return" occurrences not part of comments
        count = 0

        # Iterate over each line
        for line in lines:
            # Check if "return" is in the line and "#" is not before the "return"
            if "return" in line and not "#" in line.split("return")[0]:
                count += 1

        return count

    def post_process(self, output):
        text = output[0] if isinstance(output, list) else output
        match = re.search(r'\{[\s\S]*\}', text)  # 提取第一个 JSON 块
        if not match:
            raise ValueError("No JSON found in text")
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # 尝试修复常见错误
            fixed = json_str.replace("'", '"')
            return json.loads(fixed)
    
    def llama3_codeGen(self, extended_prompt, test_case=False):
        # assert 0 <= config.codex.temperature <= 1
        # assert 1 <= config.codex.best_of <= 20

        # if test_case == False:
            # system = "Only answer with a function starting def execute_command."
        # else:
            # system = "You are an expert programming assistant. Only answer with a function starting with def execute_test."
        system = "You are an expert language analysis assistant"
        
        gen_code = []
        for p in extended_prompt:
            try:
                prompt = [{"role": "system", "content": system},
                         {"role": "user", "content": p},
                        ]
                inputs = self.tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to("cuda")
                terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                output = self.model.generate(inputs, max_new_tokens=256, eos_token_id=terminators, do_sample=False,)
                out = output[0][inputs.shape[-1]:]
                final_out = self.tokenizer.decode(out, skip_special_tokens=True)
                #print(f"final_out: {final_out}")
                gen_code.append(self.post_process(final_out))
                # gen_code.append([self.post_process(final_out, test_case)])
            except Exception as e:
                print(f"Error: {str(e)}")
                gen_code.append("[")
        return gen_code

    def forward(self, query):
 
        extended_prompt = [prompt.replace("INSET_QUERY_HERE", query)]
        result = self.forward_(extended_prompt)
         
        return result

    def forward_(self, extended_prompt):
        try:
            
            response = self.llama3_codeGen(extended_prompt)
            
        except openai.error.RateLimitError as e:
            print("Retrying Codex, splitting batch")
            if len(extended_prompt) == 1:
                warnings.warn("This is taking too long, maybe OpenAI is down? (status.openai.com/)")
            # Will only be here after the number of retries in the backoff decorator.
            # It probably means a single batch takes up the entire rate limit.
            sub_batch_1 = extended_prompt[:len(extended_prompt) // 2]
            sub_batch_2 = extended_prompt[len(extended_prompt) // 2:]
            if len(sub_batch_1) > 0:
                response_1 = self.forward_(sub_batch_1, test_case)
            else:
                response_1 = []
            if len(sub_batch_2) > 0:
                response_2 = self.forward_(sub_batch_2, test_case)
            else:
                response_2 = []
            response = response_1 + response_2
        except Exception as e:
            # Some other error like an internal OpenAI error
            print("Retrying Codex")
            print(e)
            response = ["[" for p in extended_prompt]

        return response



model = CodexModel()
result = model.forward("Does the utensil on top of the table look clean and black?")
print(result[0])

result = model.forward("Is the ground blue or brown?")
print(result[0])