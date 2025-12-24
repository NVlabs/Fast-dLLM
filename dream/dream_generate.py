import torch
from transformers import AutoTokenizer
from model.modeling_dream import DreamModel

if __name__ == "__main__":

    model_path = "Dream-org/Dream-v0-Instruct-7B"
    model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to("cuda").eval()

    question_1 = "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
    question_2 = 'Write a story that ends with "Finally, Joey and Rachel get married."'
    question_3 = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"   
    question_4 = "Give me a short introduction to large language model?"
    question_5 = "Can you introduce something about Paris?"
    question_6 = "Write a code for quicksort. "

    messages = [
        [{"role": "user", "content": "Answer the question step by step and put the answer in \\boxed\{\}: " + question_1}], 
        [{"role": "user", "content": question_2}], 
        [{"role": "user", "content": "Answer the question step by step and put the answer in \\boxed\{\}: " + question_3}], 
        [{"role": "user", "content": question_4}], 
        [{"role": "user", "content": question_5}], 
        [{"role": "user", "content": question_6}]
    ]

    prompts = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = prompt_ids.input_ids.to(device="cuda")
    attention_mask = prompt_ids.attention_mask.to(device="cuda")
   
    output = model.diffusion_generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        output_history=True,
        return_dict_in_generate=True,
        steps=128,
        temperature=0.0,
        top_p=0.95,
        alg="entropy",
        threshold=0.9,
        block_length=32,
        dual_cache=True,
    )
    
    for b in range(len(messages)):
        print()
        print(f"----Question {b+1}: {messages[b][0]['content']}")
        sequence = output.sequences[b]
        print(tokenizer.decode(sequence[len(input_ids[0]):]).split('<|endoftext|>')[0])
    
        