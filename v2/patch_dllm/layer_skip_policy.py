import torch
import torch.nn.functional as F


class LayerSkipPolicy:
    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.past_hidden_states = (None, None)

    def reset_state(self):
        self.past_hidden_states = (None, None)

    def update_hidden_states(self, hidden_states):
        prev, curr = self.past_hidden_states
        if prev is None:
            self.past_hidden_states = (hidden_states, None)
        else:
            self.past_hidden_states = (curr if curr is not None else prev, hidden_states)

    def check_layer_skip(self):
        prev, curr = self.past_hidden_states
        if curr is None:
            return False

        B = curr.shape[0]
        cos_sim = F.cosine_similarity(
            prev.view(B, -1),
            curr.view(B, -1), dim=-1
        ).clamp(max=1.0).mean()
        skip = cos_sim > self.threshold
        if skip:
            self.reset_state()
        return skip
