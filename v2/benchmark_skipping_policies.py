from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import matplotlib.pyplot as plt
import os

from patch_dllm.token_skip_policy import TokenSkipPolicy
from patch_dllm.layer_skip_policy import LayerSkipPolicy
from patch_dllm.monkey_patch_token_skip import patch_token_skip, unpatch_token_skip
from patch_dllm.monkey_patch_layer_skip import patch_layer_skip, unpatch_layer_skip
from patch_dllm.utils import fix_seed, token_overlap, bert_score_f1, ComputeCounter

model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cuda:0", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompts = [
    "Write a short story about a robot learning to paint."
]

REAL_STOP_TOKEN = 151645

gen_kwargs = dict(
    tokenizer=tokenizer, block_size=32, max_new_tokens=256,
    small_block_size=8, threshold=0.9, use_block_cache=True,
    stop_token=-1,
)

def truncate_at_stop(toks):
    positions = (toks == REAL_STOP_TOKEN).nonzero(as_tuple=True)[0]
    return toks[:positions[0]] if len(positions) > 0 else toks

def prepare_inputs():
    inputs = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs.append(tokenizer([text], return_tensors="pt").to(model.device))
    return inputs

def make_plots(results):
    from adjustText import adjust_text
    os.makedirs("skipping_policies_results", exist_ok=True)
    for policy_name, title in [("token_skip", "Token Skip"), ("layer_skip", "Layer Skip")]:
        data = [r for r in results if r["policy"] == policy_name]
        if not data:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        x = [r["flops_reduction"] * 100 for r in data]
        y_overlap = [r["overlap"] * 100 for r in data]
        y_bert = [r.get("bert_f1", 0) for r in data]

        ax1.scatter(x, y_overlap, s=60, zorder=5)
        texts1 = [ax1.text(xi, yi, f't={r["threshold"]}', fontsize=7)
                   for r, xi, yi in zip(data, x, y_overlap)]
        adjust_text(texts1, ax=ax1)
        ax1.set_xlabel("FLOPs Reduction (%)")
        ax1.set_ylabel("Token Overlap (%)")
        ax1.set_title(f"{title}: Token Overlap vs FLOPs Reduction")
        ax1.grid(True, alpha=0.3)

        ax2.scatter(x, y_bert, s=60, zorder=5)
        texts2 = [ax2.text(xi, yi, f't={r["threshold"]}', fontsize=7)
                   for r, xi, yi in zip(data, x, y_bert)]
        adjust_text(texts2, ax=ax2)
        ax2.set_xlabel("FLOPs Reduction (%)")
        ax2.set_ylabel("BERTScore F1")
        ax2.set_title(f"{title}: BERTScore F1 vs FLOPs Reduction")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"skipping_policies_results/{policy_name}_plots.png", dpi=150)
        print(f"Saved skipping_policies_results/{policy_name}_plots.png")
        plt.close()

inputs = prepare_inputs()

# warmup
fix_seed(0)
model.generate(inputs[0]["input_ids"], **gen_kwargs)

# baselines (truncated at real stop token for quality)
print("Running baselines...")
baselines = []
for i, inp in enumerate(inputs):
    fix_seed(42)
    ids = model.generate(inp["input_ids"], **gen_kwargs)
    toks = ids[0][inp["input_ids"].shape[1]:]
    qtoks = truncate_at_stop(toks)
    qtext = tokenizer.decode(qtoks, skip_special_tokens=True)
    baselines.append({"toks": toks, "qtoks": qtoks, "qtext": qtext})
    print(f"  prompt {i}: {len(toks)} total, {len(qtoks)} until stop token")

def run_sweep(policy_cls, thresholds, patch_fn, unpatch_fn, policy_name):
    results = []
    patch_fn(model)

    # baseline at t=1.0
    baseline_procs = []
    for i, inp in enumerate(inputs):
        counter = ComputeCounter()
        fix_seed(42)
        model.generate(inp["input_ids"], skip_policy=policy_cls(threshold=1.0), compute_counter=counter, **gen_kwargs)
        baseline_procs.append(counter.processed_tokens)
    print(f"  baseline processed: {baseline_procs}")

    for thresh in thresholds:
        all_reds = []
        all_overlaps = []
        skip_texts = []
        baseline_texts = []
        total_proc = 0
        total_diter = 0
        total_unmasked = 0
        total_tokskip = 0
        total_fullblk = 0

        for i, inp in enumerate(inputs):
            policy = policy_cls(threshold=thresh)
            counter = ComputeCounter()
            fix_seed(42)
            ids = model.generate(inp["input_ids"], skip_policy=policy, compute_counter=counter, **gen_kwargs)
            out_toks = ids[0][inp["input_ids"].shape[1]:]
            qtoks = truncate_at_stop(out_toks)
            qtext = tokenizer.decode(qtoks, skip_special_tokens=True)

            red = 1 - counter.processed_tokens / baseline_procs[i]
            all_reds.append(red)
            all_overlaps.append(token_overlap(baselines[i]["qtoks"], qtoks))
            skip_texts.append(qtext)
            baseline_texts.append(baselines[i]["qtext"])
            total_proc += counter.processed_tokens
            total_diter += counter.denoising_iters
            total_unmasked += counter.tokens_unmasked
            total_tokskip += counter.tokens_skipped
            total_fullblk += counter.full_block_fwds

        flops_red = round(float(np.mean(all_reds)), 4)
        avg_overlap = round(float(np.mean(all_overlaps)), 3)
        bert_f1 = bert_score_f1(skip_texts, baseline_texts)

        r = {
            "policy": policy_name,
            "threshold": thresh,
            "flops_reduction": flops_red,
            "overlap": avg_overlap,
            "processed_tokens": total_proc,
            "denoising_iters": total_diter,
            "tokens_unmasked": total_unmasked,
            "tokens_skipped": total_tokskip,
            "full_block_fwds": total_fullblk,
        }
        if bert_f1 is not None:
            r["bert_f1"] = round(bert_f1, 3)
        results.append(r)

        bert_str = f", BERT F1={bert_f1:.3f}" if bert_f1 is not None else ""
        print(f"  t={thresh}: FLOPs red={flops_red:.1%}, overlap={avg_overlap:.3f}{bert_str}")

    unpatch_fn()
    return results

# ---- token skip sweep ----
print("\nToken skip sweep...")
token_skip_results = run_sweep(
    TokenSkipPolicy, [1.0, 0.995, 0.99, 0.98, 0.95, 0.90, 0.85, 0.80],
    patch_token_skip, unpatch_token_skip, "token_skip"
)

# ---- layer skip sweep ----
print("\nLayer skip sweep...")
layer_skip_results = run_sweep(
    LayerSkipPolicy, [1.0, 0.95, 0.92, 0.90, 0.85, 0.82, 0.80, 0.78, 0.75],
    patch_layer_skip, unpatch_layer_skip, "layer_skip"
)

all_results = token_skip_results + layer_skip_results

# ---- summary table ----
print(f"\n{'='*70}")
print("Accuracy vs FLOPs Reduction")
print(f"{'='*70}")
print(f"{'Policy':12s} | {'Threshold':>9s} | {'FLOPs Red.':>10s} | {'Overlap':>7s} | {'BERT F1':>7s}")
print("-" * 70)
for r in all_results:
    bert = f"{r['bert_f1']:.3f}" if r.get("bert_f1") is not None else "n/a"
    print(f"{r['policy']:12s} | {r['threshold']:>9.3f} | {r['flops_reduction']:>9.1%} | {r['overlap']:>7.3f} | {bert:>7s}")

# ---- detailed tables (for non-monotonic analysis) ----
print(f"\n{'='*100}")
print("Detailed Diagnostics")
print(f"{'='*100}")

print("\nToken Skip:")
print(f"{'Threshold':>9s} | {'FLOPs Red.':>10s} | {'Tokens Processed':>16s} | {'Denoising Iters':>15s} | "
      f"{'Full-Block Fwds':>15s} | {'Tokens Skipped':>14s} | {'Unmasked/Iter':>13s}")
print("-" * 110)
for r in [x for x in all_results if x["policy"] == "token_skip"]:
    upi = r["tokens_unmasked"] / r["denoising_iters"] if r["denoising_iters"] > 0 else 0
    print(f"{r['threshold']:>9.3f} | {r['flops_reduction']:>9.1%} | {r['processed_tokens']:>16d} | {r['denoising_iters']:>15d} | "
          f"{r['full_block_fwds']:>15d} | {r['tokens_skipped']:>14d} | {upi:>13.2f}")

print("\nLayer Skip:")
print(f"{'Threshold':>9s} | {'FLOPs Red.':>10s} | {'Tokens Processed':>16s} | {'Denoising Iters':>15s} | "
      f"{'Full-Block Fwds':>15s} | {'Layers Skipped':>14s} | {'Unmasked/Iter':>13s}")
print("-" * 110)
for r in [x for x in all_results if x["policy"] == "layer_skip"]:
    upi = r["tokens_unmasked"] / r["denoising_iters"] if r["denoising_iters"] > 0 else 0
    print(f"{r['threshold']:>9.3f} | {r['flops_reduction']:>9.1%} | {r['processed_tokens']:>16d} | {r['denoising_iters']:>15d} | "
          f"{r['full_block_fwds']:>15d} | {r['tokens_skipped']:>14d} | {upi:>13.2f}")

# save
os.makedirs("skipping_policies_results", exist_ok=True)
with open("skipping_policies_results/results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nResults saved to skipping_policies_results/")

make_plots(all_results)
