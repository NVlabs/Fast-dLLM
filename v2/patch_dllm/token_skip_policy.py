import torch
import torch.nn.functional as F


class TokenSkipPolicy:
    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self._reset()

    def _reset(self):
        self.past_hidden_states = (None, None)
        self.skip_mask = None
        self.keep_pos = None

    def reset_state(self):
        self._reset()

    def update_hidden_states(self, hidden_states):
        prev, curr = self.past_hidden_states
        if prev is None:
            self.past_hidden_states = (hidden_states, None)
        else:
            self.past_hidden_states = (curr if curr is not None else prev, hidden_states)

    def check_token_skip(self):
        prev, curr = self.past_hidden_states
        if curr is None:
            return False

        cos_sim = F.cosine_similarity(prev, curr, dim=-1).clamp(max=1.0)
        skip_mask = cos_sim > self.threshold

        B = skip_mask.shape[0]
        if B > 1:
            skip_mask = self._match_skips_across_batch(skip_mask, cos_sim)

        if skip_mask.sum() == 0:
            return False

        self.skip_mask = skip_mask
        self.keep_pos = (~skip_mask).nonzero(as_tuple=True)[1].view(B, -1)
        return True

    def filter_embeds(self, embeds):
        B, N, D = embeds.shape
        return embeds.gather(1, self.keep_pos.unsqueeze(-1).expand(-1, -1, D))

    def filter_positions(self, position_ids):
        B = self.keep_pos.shape[0]
        return position_ids.expand(B, -1).gather(1, self.keep_pos)

    def update_block_cache(self, block_cache, new_states, replace_position):
        L = self.skip_mask.shape[1]
        cache_slice = block_cache[:, :, replace_position:replace_position + L, :]
        B, H, _, d = cache_slice.shape
        idx = self.keep_pos[:, None, :, None].expand(B, H, -1, d)
        cache_slice.scatter_(dim=2, index=idx, src=new_states)

    def reconstruct_hidden_states(self, hidden_states):
        full = self.past_hidden_states[1].clone()
        B, M, D = hidden_states.shape
        idx = self.keep_pos[:, :, None].expand(B, -1, D)
        full.scatter_(dim=1, index=idx, src=hidden_states)
        return full

    def _match_skips_across_batch(self, mask, cos_sim):
        m = int(mask.sum(dim=1).min().item())
        if m == 0:
            return torch.zeros_like(mask, dtype=torch.bool)
        scores = cos_sim.masked_fill(~mask, -float('inf'))
        _, idx = scores.topk(m, dim=1, largest=True)
        aligned = torch.zeros_like(mask, dtype=torch.bool)
        aligned.scatter_(1, idx, True)
        return aligned
