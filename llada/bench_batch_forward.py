"""
Benchmark batched candidate evaluation during dual-cache generation.

This script keeps the decoding procedure identical to `generate_with_dual_cache`
but, at every denoising step, it calls `benchmark_candidate_batching` to measure how
long it takes to evaluate a random masked position with different beam sizes.  The
results are plotted as a scatter chart (beam size vs. latency) and written to disk.
"""

from __future__ import annotations

import argparse
import math
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer

from generate import get_num_transfer_tokens, get_transfer_index
from model.modeling_llada import LLaDAModelLM
from batched_dual_cache import benchmark_candidate_batching


def _select_first_mask_position(mask_row: torch.Tensor) -> int | None:
    """
    Deterministically pick the first masked position so measurements are comparable.
    """
    idx = mask_row.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return None
    return int(idx[0])


def benchmark_during_generation(
    model,
    prompt: torch.Tensor,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str,
    mask_id: int,
    threshold: float | None,
    beam_sizes: Iterable[int],
    repeats: int,
    warmup_repeats: int = 2,
    maximum_test_num_each_block: int = 4,
) -> List[Dict[str, Any]]:
    """
    Clone of `generate_with_dual_cache` that simply inserts a benchmarking call before
    each confidence-based update. Returns a list of timing dicts for each tested beam.
    """
    assert prompt.shape[0] == 1, "benchmark currently assumes batch size 1"
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by number of blocks")
    steps_per_block = steps // num_blocks

    seq = torch.full(
        (1, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=prompt.device,
    )
    seq[:, : prompt.shape[1]] = prompt

    timing_records: List[Dict[str, Any]] = []

    for block_idx in range(num_blocks):
        s = prompt.shape[1] + block_idx * block_length
        e = s + block_length

        block_mask_index = (seq[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        out_full = model(seq, use_cache=True)
        past_key_values = out_full.past_key_values

        replace_position = torch.zeros_like(seq, dtype=torch.bool)
        replace_position[:, s:e] = True

        mask_global = (seq == mask_id)
        mask_global[:, e:] = False

        quota0 = None if threshold is not None else num_transfer_tokens[:, 0]
        x0, transfer_index = get_transfer_index(
            out_full.logits,
            temperature,
            remasking,
            mask_global,
            seq,
            quota0,
            threshold,
        )
        seq = torch.where(transfer_index, x0, seq)
        test_num=0

        for step in tqdm(range(1, steps_per_block), desc=f"Block {block_idx+1}/{num_blocks}"):
            logits_blk = model(
                seq[:, s:e],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
            ).logits

            mask_blk = (seq[:, s:e] == mask_id)
            if not mask_blk.any():
                break

            candidate_pos = _select_first_mask_position(mask_blk[0])
            if candidate_pos is not None and test_num < maximum_test_num_each_block:
                timing_entries = benchmark_candidate_batching(
                    model=model,
                    base_block=seq[:, s:e].detach(),
                    past_key_values=past_key_values,
                    logits=logits_blk.detach(),
                    candidate_position=candidate_pos,
                    beam_sizes=beam_sizes,
                    repeats=repeats,
                    warmup_repeats=warmup_repeats,
                )
                timing_records.extend(timing_entries)
                test_num += 1
            quota = None if threshold is not None else num_transfer_tokens[:, step]
            x0_blk, transfer_idx_blk = get_transfer_index(
                logits_blk, temperature, remasking, mask_blk, seq[:, s:e], quota, threshold
            )
            blk_old = seq[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            seq = torch.cat([seq[:, :s], blk_new, seq[:, e:]], dim=1)

    return timing_records


def plot_timings(beam_to_times: Dict[int, List[float]], output_path: Path) -> None:
    """
    Plot beam size vs. latency with mean and 95% CI error bars,
    overlaid with jittered raw measurements, and save to `output_path`.
    """
    if not beam_to_times:
        raise ValueError("No timing data collected; cannot plot.")

    beams_sorted = sorted(beam_to_times.keys())
    means: List[float] = []
    ci95: List[float] = []
    for b in beams_sorted:
        arr = np.array(beam_to_times[b], dtype=np.float64)
        n = max(1, arr.size)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
        se = sd / math.sqrt(n) if n > 0 else 0.0
        means.append(mu)
        ci95.append(1.96 * se)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(7, 4.5))

    plt.errorbar(
        beams_sorted,
        means,
        yerr=ci95,
        fmt="o-",
        color="C0",
        ecolor="C0",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        markersize=5.5,
        linewidth=1.5,
        label="Mean ± 95% CI",
    )

    rng = np.random.default_rng(12345)
    for i, b in enumerate(beams_sorted):
        xs = np.full(len(beam_to_times[b]), b, dtype=np.float64)
        x_jitter = (rng.random(len(xs)) - 0.5) * 0.35
        plt.scatter(
            xs + x_jitter,
            beam_to_times[b],
            color="C0",
            alpha=0.35,
            s=18,
            linewidths=0,
            label="Raw repeats" if i == 0 else "_nolegend_",
        )

    plt.xlabel("Beam size")
    plt.ylabel("Batched forward time (s)")
    plt.title("KV-cache batched forward latency")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark batched KV-cache forwards.")
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompt", default="Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?")
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Path to JSONL file containing entries with type=='prediction' and a 'prompt' field.")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--beam-max", type=int, default=16)
    parser.add_argument("--beam-min", type=int, default=1)
    parser.add_argument("--beam-repeats", type=int, default=5, help="Average over this many calls per beam size.")
    parser.add_argument("--warmup-repeats", type=int, default=2, help="Number of initial iterations to warm up (not recorded) per beam.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-plot", type=Path, default=Path("beam_vs_time.png"))
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Optional path to write JSONL mapping of input profiles to wallclock times.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = (
        LLaDAModelLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        .to(device)
        .eval()
    )

    def load_prompts_from_jsonl(path: Path) -> List[Dict[str, Any]]:
        prompts: List[Dict[str, Any]] = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "prediction":
                    continue
                prompt_text = obj.get("prompt")
                if not isinstance(prompt_text, str):
                    continue
                prompts.append({
                    "idx": obj.get("idx"),
                    "prompt": prompt_text,
                })
        return prompts

    # Collect prompts
    prompt_entries: List[Dict[str, Any]]
    if args.input_jsonl is not None:
        prompt_entries = load_prompts_from_jsonl(args.input_jsonl)
        if not prompt_entries:
            raise ValueError("No prediction entries with 'prompt' found in input JSONL.")
    else:
        # Fallback to single provided prompt, wrapped via chat template for consistency
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_entries = [{"idx": None, "prompt": chat_prompt}]

    beam_sizes = range(args.beam_min, args.beam_max + 1)

    # Aggregate timing data across prompts (collect all repeats per beam)
    all_timings: Dict[int, List[float]] = defaultdict(list)
    jsonl_records: List[Dict[str, Any]] = []

    for entry in tqdm(prompt_entries, desc="Benchmarking prompts"):
        raw_prompt: str = entry["prompt"]
        # The provided JSONL 'prompt' is already in chat-formatted text with special tokens.
        input_ids = tokenizer(raw_prompt)["input_ids"]
        prompt_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

        timing_entries = benchmark_during_generation(
            model=model,
            prompt=prompt_tensor,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            remasking=args.remasking,
            mask_id=args.mask_id,
            threshold=args.threshold,
            beam_sizes=beam_sizes,
            repeats=args.beam_repeats,
            warmup_repeats=args.warmup_repeats,
        )
        for rec in timing_entries:
            b = int(rec["beam"])
            all_timings[b].extend(list(rec.get("times", [])))

        # Prepare JSONL records mapping input profiles to wallclock times
        for rec in timing_entries:
            # Per-repeat records
            for t in rec.get("times", []):
                jsonl_records.append({
                    "type": "timing",
                    "idx": entry.get("idx"),
                    "model": args.model_name,
                    "steps": args.steps,
                    "gen_length": args.gen_length,
                    "block_length": args.block_length,
                    "beam_size": int(rec["beam"]),
                    "repeats": args.beam_repeats,
                    "prompt_len_tokens": len(input_ids),
                    "time_s": float(t),
                })
            # Summary record per (prompt, beam)
            jsonl_records.append({
                "type": "timing_summary",
                "idx": entry.get("idx"),
                "model": args.model_name,
                "steps": args.steps,
                "gen_length": args.gen_length,
                "block_length": args.block_length,
                "beam_size": int(rec["beam"]),
                "repeats": int(rec.get("num_repeats", args.beam_repeats)),
                "prompt_len_tokens": len(input_ids),
                "avg_time_s": float(rec.get("avg_time_s", 0.0)),
                "std_time_s": float(rec.get("std_time_s", 0.0)),
            })

    

    plot_timings(all_timings, args.output_plot)
    # Optional: write JSONL mapping of input profiles to wallclock times
    if args.output_jsonl is not None:
        with args.output_jsonl.open("w") as f:
            for rec in jsonl_records:
                f.write(json.dumps(rec) + "\n")
    print(f"Saved scatter plot to {args.output_plot.resolve()}")


if __name__ == "__main__":
    main()

