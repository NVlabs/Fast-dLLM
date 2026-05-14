# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# Boo
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream

import torch
from transformers import AutoModel, AutoTokenizer
import time
from model.modeling_dream import DreamModel

import types
# Load model and tokenizer

# Read use_cache from command line
use_cache = True if input("Use cache? (y/n): ").lower() == 'y' else False

if use_cache:
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to("cuda").eval()

    from model.generation_utils_block import DreamGenerationMixin
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)
else:
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to("cuda").eval()


# Initialize conversation history
messages = []

print("Multi-turn conversation with Dream-v0-Instruct-7B")
print("Type 'exit' to end the conversation")
print("----------------------------------------------")

while True:
    # Get user input
    user_input = input("You: ")

    # Check if user wants to exit
    if user_input.lower() == 'exit':
        print("Conversation ended.")
        break

    # Add user message to conversation history
    messages.append({"role": "user", "content": user_input})

    # Format input with chat template
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(device="cuda")
    attention_mask = inputs.attention_mask.to(device="cuda")

    def generation_tokens_hook_func(step, x, logits):
        print(f"############ Step {step} ############")
        # print(tokenizer.decode(h[0].tolist()))
        print(tokenizer.decode(x[0].tolist()).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, " "), end="\r")
        time.sleep(0.01)
        return x
    
    # Generate response
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        output_history=True,
        return_dict_in_generate=True,
        steps=128,
        temperature=0.,
        top_p=None,
        alg="entropy",
        alg_temp=0.1,
        top_k=None,
        block_length=32,
        # generation_tokens_hook_func=generation_tokens_hook_func
    )

    # Process response
    generation = tokenizer.decode(output.sequences[0][len(input_ids[0]):].tolist())
    generation = generation.split(tokenizer.eos_token)[0].strip()

    # Print response
    print("Model:", generation)

    # Add model response to conversation history
    messages.append({"role": "assistant", "content": generation})


'''An example conversation (maybe different due to randomness)
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|im_end|>
<|im_start|>assistant
Janet sells 16 - 3 - 4 = 9 eggs per day.
She makes 9 * $2 = $18 per day.<|im_end|>
<|im_start|>user
what if her duck lay three more eggs<|im_end|>
<|im_start|>assistant
If Janet's ducks lay three more eggs per day, she would have 16 + 3 = 19 eggs per day.<|im_end|>
<|im_start|>user
yes, so how many dollars she make<|im_end|>
<|im_start|>assistant
Janet sells 19 - 3 - 4 = 12 eggs per day.
She makes 12 * $2 = $24 per day.
'''
