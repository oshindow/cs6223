"""
Adding a new functionality is easy. Just implement your new model as a subclass of BaseModel.
The code will make the rest: it will make it available for the processes to call by using
process(name, *args, **kwargs), where *args and **kwargs are the arguments of the models process() method.
"""
 
import openai
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import re

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
        with open("llm_mutant.prompt") as f:
            self.base_prompt = f.read().strip()
        self.fixed_code = None
         
        model_id = f"meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:6"},
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

    def post_process(self, gen_code, test_case=False):
        
        if test_case == False:
            print(f"Originally gen_code: \n {gen_code}")
            gen_code = gen_code.replace("execute_command(image)", "execute_command(image, my_fig, time_wait_between_lines, syntax)")
            gen_code = gen_code.replace(" -> ImagePatch", "")
        else:
            # Postprocess for test code
            print(f"Originally gen_code: \n {gen_code}")
            gen_code = gen_code.replace("solve_query(image)", "execute_command(image, my_fig, time_wait_between_lines, syntax)")

        # [Non-test case Code] : See if the generated code starts with def and ends with return
        if "while" in gen_code:
            gen_code = gen_code.replace("while", "if")

        if len(gen_code.split("```")) >= 2:
            gen_code = gen_code.split("```")[1]

        if gen_code[:1] == '\n':
            gen_code = gen_code[1:]

        lines = gen_code.split('\n')
        if 'def execute_command' not in lines[0] and gen_code.count("execute_command") == 0:
            # Add the function declaration at the beginning
            modified_string = " def execute_command(image):\n"
            # Split the string by newline and add the necessary indentation
            modified_string += "\n".join(["    " + line for line in gen_code[1:].split("\n")])
            gen_code = modified_string
            lines = gen_code.split('\n')

        # For non-test code
        if ('def' not in lines[0] or 'return' not in lines[-1] or gen_code.count("return") > 1):
            # If there is one or more def execute_command, only extract the first one.
            if gen_code.count("def execute_command") > 1:
                first_occurrence_index = gen_code.find("def execute_command")
                second_occurrence_index = gen_code.find("def execute_command",
                                                        first_occurrence_index + len("def execute_command"))
                if second_occurrence_index != -1:
                    gen_code = gen_code[:second_occurrence_index]
                    lines = gen_code.split('\n')
            # Find the index of the line that starts with 'def '
            start_index = next((i for i, line in enumerate(lines) if line.strip().startswith('def ')), None)
            # Find the index of the line that starts with 'return ' after the start_index and the 'return' is the end of the function definition
            end_index = None
            for i in range(start_index, len(lines) - 1, 1):  # -1 to avoid index out of range on next line check
                if ("return" in lines[i] and len(lines[i + 1]) > 1 and lines[i + 1][1] != ' ' and end_index is None) or \
                        lines[i + 1] == '```':  # when next line is not empty
                    end_index = i
                    break
                elif "return" in lines[i] and lines[i + 1] == '' and end_index is None and len(lines) < i+4:
                    end_index = i
                    break
                elif "return" in lines[i + 1] and i == len(lines) - 2 and end_index is None and lines[i+1] == ' ':
                    end_index = i + 1
                    break
                elif "    return" in lines[i] and not any("    return" in s for s in lines[i+1:]):
                    end_index = i
                    break
                elif "    return" in lines[i] and any(s and s[0].isalpha() for s in lines[i+1:]) and not any("    return" in s for s in lines[i+1:]):
                    end_index = i
                    break

            if start_index is not None and end_index is not None:
                # Get the lines from start_index to end_index inclusive
                selected_lines = lines[start_index:end_index + 1]

                # Join the selected lines with '\n' to get the final string
                final_string = '\n'.join(selected_lines)
                final_string = final_string.replace(" -> ImagePatch", "")
                final_string = final_string.replace("->str", "")
                final_string = final_string.replace("execute_command(image)",
                                            "execute_command(image, my_fig, time_wait_between_lines, syntax)")
                if final_string[:1] == '\n':
                    final_string = final_string[1:]
                return final_string
            else:
                print('The required lines were not found in the input string.')
                print(f'{gen_code}')
                return gen_code
        else:
            gen_code = gen_code.replace(" -> ImagePatch", "")
            gen_code = gen_code.replace("->str", "")
            gen_code = gen_code.replace("execute_command(image)",
                                        "execute_command(image, my_fig, time_wait_between_lines, syntax)")
            if gen_code[:1] == '\n':
                gen_code = gen_code[1:]
            return gen_code

    def llama3_codeGen(self, extended_prompt, test_case=False):
        system = "You are an expert property test assistant"
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
                ).to("cuda:6")
                terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                output = self.model.generate(inputs, max_new_tokens=256, eos_token_id=terminators, do_sample=False,)
                out = output[0][inputs.shape[-1]:]
                final_out = self.tokenizer.decode(out, skip_special_tokens=True)
                
                gen_code.append(final_out)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                gen_code.append("[")
        return gen_code

    def forward(self, result, answer, query, base_prompt=None):
        
        with open("llm_mutant.prompt") as f:
            base_prompt = f.read().strip()
        
        extended_prompt = [base_prompt.replace("INSERT_ANSWER_HERE", answer).
                               replace('INSERT_QUERY_HERE', query)]

        if answer == 'yes' or answer == 'no':
            result = ['yes', 'no']
        else:
            result = self.forward_(extended_prompt)

            if '\n\n<<Answer>>' in result[0]:
                result[0] = re.sub(r"\n\n<<Answer>>.*", "", result[0], flags=re.DOTALL)

            if '\n\nMutants: ' in result[0]:
                result = result[0].split('\n\nMutants: ')[1].split(', ')
            elif '\nMutants: ' in result[0]:
                result = result[0].split('\nMutants: ')[1].split(', ')
                # print(result[0])
            
            
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



