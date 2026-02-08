from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from patch_dllm.layer_skip_policy import LayerSkipPolicy
from patch_dllm.monkey_patch_layer_skip import patch_layer_skip, unpatch_layer_skip
from patch_dllm.utils import fix_seed, token_overlap, bert_score_f1, ComputeCounter

model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

user_input = "Write a short story about a robot learning to paint."
messages = [{"role": "user", "content": user_input}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

gen_kwargs = dict(
    tokenizer=tokenizer,
    block_size=32,
    max_new_tokens=256,
    small_block_size=8,
    threshold=0.9,
    use_block_cache=True,
)

print("=== BASELINE (no skip) ===")
fix_seed(42)
t0 = time.time()
baseline_ids = model.generate(model_inputs["input_ids"], **gen_kwargs)
baseline_time = time.time() - t0
baseline_toks = baseline_ids[0][model_inputs["input_ids"].shape[1]:]
baseline_text = tokenizer.decode(baseline_toks, skip_special_tokens=True)
baseline_mpt = baseline_time / len(baseline_toks) * 1000
print(baseline_text)
print(f"\n{baseline_mpt:.2f} ms/tok ({len(baseline_toks)} tokens)")

# sweep
patch_layer_skip(model)

baseline_counter = ComputeCounter()
fix_seed(42)
model.generate(model_inputs["input_ids"], skip_policy=LayerSkipPolicy(threshold=1.0), compute_counter=baseline_counter, **gen_kwargs)
baseline_processed_tokens = baseline_counter.processed_tokens

for thresh in [0.95, 0.9, 0.85, 0.8]:
    print(f"\n=== LAYER SKIP (threshold={thresh}) ===")
    policy = LayerSkipPolicy(threshold=thresh)
    counter = ComputeCounter()
    fix_seed(42)
    t0 = time.time()
    skip_ids = model.generate(model_inputs["input_ids"], skip_policy=policy, compute_counter=counter, **gen_kwargs)
    skip_time = time.time() - t0
    skip_toks = skip_ids[0][model_inputs["input_ids"].shape[1]:]
    skip_text = tokenizer.decode(skip_toks, skip_special_tokens=True)
    skip_mpt = skip_time / len(skip_toks) * 1000
    print(skip_text)

    overlap = token_overlap(baseline_toks, skip_toks)
    bert_f1 = bert_score_f1([skip_text], [baseline_text])
    speedup = baseline_mpt / skip_mpt
    flops_red = 1 - counter.processed_tokens / baseline_processed_tokens
    print(f"\n{skip_mpt:.2f} ms/tok ({len(skip_toks)} tok) | Speedup: {speedup:.2f}x | Token overlap: {overlap:.3f} | BERTScore F1: {bert_f1:.3f} | FLOPs red: {flops_red:.1%}")
unpatch_layer_skip()
