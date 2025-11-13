"""
Utilities for experimenting with batched candidate evaluation when using the dual-cache
generation routine.  This module mirrors `generate_with_dual_cache` but, at each
denoising step, it also samples one low-confidence position, enumerates its top-k
realisations, and evaluates them in a single forward pass that reuses the KV cache.

The `select_candidate_hook` function is intentionally left as a stub. Plug in your own
decision rule to pick which candidate realisation to keep after inspecting the batched
logits.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from generate import get_num_transfer_tokens, get_transfer_index


def tile_past_key_values(
    past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    repeats: int,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Duplicate cached keys/values along the batch dimension so every candidate in a
    beam can share the same prefix cache.
    """
    if past_key_values is None or repeats == 1:
        return past_key_values

    tiled: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for key, value in past_key_values:
        tiled.append(
            (
                key.repeat_interleave(repeats, dim=0),
                value.repeat_interleave(repeats, dim=0),
            )
        )
    return tiled


def sample_random_mask_position(
    mask_row: torch.Tensor, generator: Optional[torch.Generator] = None
) -> Optional[int]:
    """
    Uniformly sample one of the still-masked positions.
    """
    candidates = mask_row.nonzero(as_tuple=False).flatten()
    if candidates.numel() == 0:
        return None
    choice = torch.randint(
        low=0,
        high=candidates.numel(),
        size=(1,),
        generator=generator,
        device=mask_row.device,
    )
    return int(candidates[choice])


def batched_candidate_forward(
    model,
    base_block: torch.Tensor,
    past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    candidate_position: int,
    candidate_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a beam from `base_block` by substituting `candidate_position` with every
    token in `candidate_ids`, run a single forward pass, and return both the mutated
    blocks and their logits.
    """
    beam_width = candidate_ids.size(0)
    assert beam_width > 0, "candidate_ids must be non-empty"

    block_batch = base_block.repeat(beam_width, 1)
    block_batch[:, candidate_position] = candidate_ids

    tiled_cache = tile_past_key_values(past_key_values, beam_width)

    with torch.inference_mode():
        logits = model(
            block_batch,
            past_key_values=tiled_cache,
            use_cache=False,
            replace_position=None,
        ).logits

    return block_batch, logits


def select_candidate_hook(
    logits_batch: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_position: int,
) -> int:
    """
    Decide which candidate realisation to keep.

    Override this hook with your own scoring logic. It must return the integer index
    (0 <= idx < candidate_blocks.size(0)) of the candidate to keep.
    """
    raise NotImplementedError("Customize `select_candidate_hook` to pick candidates.")


def generate_with_dual_cache_batched(
    model,
    prompt: torch.Tensor,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    extra_beam_width: int = 4,
    rng: Optional[torch.Generator] = None,
    candidate_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], int] = select_candidate_hook,
) -> torch.Tensor:
    """
    Variant of `generate_with_dual_cache` that, after the usual confidence-based update,
    randomly selects one remaining masked position and evaluates `extra_beam_width`
    candidate tokens for it in a batched forward pass.

    Note: this prototype assumes batch size 1 for the outer generation loop.
    """
    assert prompt.dim() == 2, "prompt must be 2D (batch, seq_len)"
    assert prompt.size(0) == 1, "batched exploration currently assumes batch size 1"
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"

    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by num_blocks")
    steps_per_block = steps // num_blocks

    device = prompt.device
    full_length = prompt.size(1) + gen_length
    x = torch.full(
        (1, full_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, : prompt.size(1)] = prompt

    for block_idx in range(num_blocks):
        s = prompt.size(1) + block_idx * block_length
        e = s + block_length

        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True

        global_mask_index = (x == mask_id)
        global_mask_index[:, e:] = False

        quota0 = None if threshold is not None else num_transfer_tokens[:, 0]
        x0, transfer_index = get_transfer_index(
            out_full.logits,
            temperature,
            remasking,
            global_mask_index,
            x,
            quota0,
            threshold,
        )
        x = torch.where(transfer_index, x0, x)

        for step in tqdm(range(1, steps_per_block), desc=f"Block {block_idx+1}/{num_blocks}"):
            logits_blk = model(
                x[:, s:e],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
            ).logits

            mask_blk = (x[:, s:e] == mask_id)
            if mask_blk.sum() == 0:
                break

            quota = None if threshold is not None else num_transfer_tokens[:, step]
            x0_blk, transfer_idx_blk = get_transfer_index(
                logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota, threshold
            )

            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)

            if extra_beam_width > 0:
                residual_mask = (blk_new == mask_id)
                pos = sample_random_mask_position(residual_mask[0], generator=rng)
                if pos is not None:
                    topk = torch.topk(logits_blk[0, pos], k=extra_beam_width)
                    candidates, logits_batch = batched_candidate_forward(
                        model,
                        blk_new,
                        past_key_values,
                        pos,
                        topk.indices,
                    )
                    keep_idx = candidate_hook(
                        logits_batch,
                        candidates,
                        topk.indices,
                        pos,
                    )
                    if not 0 <= keep_idx < candidates.size(0):
                        raise ValueError("candidate_hook returned an invalid index")
                    blk_new[:, pos] = candidates[keep_idx, pos]

            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

    return x


def benchmark_candidate_batching(
    model,
    base_block: torch.Tensor,
    past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    logits: torch.Tensor,
    candidate_position: int,
    beam_sizes: Iterable[int],
    repeats: int = 10,
) -> List[Tuple[int, float]]:
    """
    Measure average latency (seconds) for running `batched_candidate_forward` with
    varying beam sizes. Returns a list of `(beam_size, avg_time)` tuples.
    """
    timings: List[Tuple[int, float]] = []
    base_block = base_block.detach()

    for beam in beam_sizes:
        if beam <= 0:
            continue
        candidate_ids = torch.topk(logits[0, candidate_position], k=beam).indices
        start = time.perf_counter()
        for _ in range(repeats):
            batched_candidate_forward(
                model,
                base_block,
                past_key_values,
                candidate_position,
                candidate_ids,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / repeats
        timings.append((beam, elapsed))

    return timings

