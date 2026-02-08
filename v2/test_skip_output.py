from transformers import AutoModelForCausalLM, AutoTokenizer
from patch_dllm.monkey_patch_layer_skip import patch_layer_skip, unpatch_layer_skip
from patch_dllm.layer_skip_policy import LayerSkipPolicy
from patch_dllm.monkey_patch_token_skip import patch_token_skip, unpatch_token_skip
from patch_dllm.token_skip_policy import TokenSkipPolicy
from patch_dllm.utils import fix_seed

model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

patch_token_skip(model)
skip_policy = TokenSkipPolicy(threshold=0.995)
# patch_layer_skip(model)
# skip_policy = LayerSkipPolicy(threshold=0.9)

fix_seed(42)

user_input = "Write a short story about a robot learning to paint."
messages = [{"role": "user", "content": user_input}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    model_inputs["input_ids"],
    tokenizer=tokenizer,
    block_size=32,
    max_new_tokens=256,
    small_block_size=8,
    threshold=0.9,
    use_block_cache=True,
    skip_policy=skip_policy,
)
response = tokenizer.decode(generated_ids[0][model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)

unpatch_token_skip()
# unpatch_layer_skip()
