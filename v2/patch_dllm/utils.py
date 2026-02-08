import torch
import numpy as np
import random
from bert_score import score as compute_bert_score


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def token_overlap(base_ids, skip_ids):
    b, s = base_ids.tolist(), skip_ids.tolist()
    m = min(len(b), len(s))
    match = sum(1 for a, c in zip(b[:m], s[:m]) if a == c)
    return match / max(len(b), len(s)) if max(len(b), len(s)) > 0 else 1.0


def bert_score_f1(candidates, references):
    _, _, F1 = compute_bert_score(candidates, references, lang="en", verbose=False)
    return F1.mean().item()


class ComputeCounter:
    """Counts tokens processed at each decoder layer during generation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.processed_tokens = 0
        self.denoising_iters = 0
        self.tokens_unmasked = 0
        self.tokens_skipped = 0
        self.full_block_fwds = 0

    def log_tokens(self, num_tokens):
        self.processed_tokens += num_tokens

    def log_denoising_iter(self):
        self.denoising_iters += 1

    def log_unmasked(self, count):
        self.tokens_unmasked += count

    def log_full_block(self):
        self.full_block_fwds += 1

    def log_layer_skip(self, batch_seq):
        self.tokens_skipped += batch_seq

    def log_token_skip(self, count):
        self.tokens_skipped += count
