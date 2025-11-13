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
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
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
    maximum_test_num_each_block: int = 4,
) -> List[Tuple[int, float]]:
    """
    Clone of `generate_with_dual_cache` that simply inserts a benchmarking call before
    each confidence-based update. Returns flattened `(beam_size, latency)` data.
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

    scatter_records: List[Tuple[int, float]] = []

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
                timing_pairs = benchmark_candidate_batching(
                    model=model,
                    base_block=seq[:, s:e].detach(),
                    past_key_values=past_key_values,
                    logits=logits_blk.detach(),
                    candidate_position=candidate_pos,
                    beam_sizes=beam_sizes,
                    repeats=repeats,
                )
                scatter_records.extend(timing_pairs)
                test_num += 1
            quota = None if threshold is not None else num_transfer_tokens[:, step]
            x0_blk, transfer_idx_blk = get_transfer_index(
                logits_blk, temperature, remasking, mask_blk, seq[:, s:e], quota, threshold
            )
            blk_old = seq[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            seq = torch.cat([seq[:, :s], blk_new, seq[:, e:]], dim=1)

    return scatter_records


def plot_timings(data: List[Tuple[int, float]], output_path: Path) -> None:
    """
    Draw a scatter plot (beam size vs. latency) and save it to `output_path`.
    """
    if not data:
        raise ValueError("No timing data collected; cannot plot.")
    beams, times = zip(*data)

    plt.figure(figsize=(6, 4))
    plt.scatter(beams, times, alpha=0.7)
    plt.xlabel("Beam size")
    plt.ylabel("Average batched forward time (s)")
    plt.title("KV-cache batched forward latency")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark batched KV-cache forwards.")
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompt", default="Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--beam-max", type=int, default=16)
    parser.add_argument("--beam-min", type=int, default=1)
    parser.add_argument("--beam-repeats", type=int, default=5, help="Average over this many calls per beam size.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-plot", type=Path, default=Path("beam_vs_time.png"))
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

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer(chat_prompt)["input_ids"]
    prompt_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    beam_sizes = range(args.beam_min, args.beam_max + 1)

    data = benchmark_during_generation(
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
    )

    

    plot_timings(data, args.output_plot)
    #save the data as json
    with open("beam_timings.json", "w") as f:
        
        json.dump(data, f)
    print(f"Saved scatter plot to {args.output_plot.resolve()}")


if __name__ == "__main__":
    main()
