# utils/similarity.py
import torch

def pool_hidden(h: torch.Tensor, how: str = "last") -> torch.Tensor:
    """
    h: [B, T, H] hidden states on GPU/CPU.
    Return a pooled 1D vector [H] on CPU to keep memory small.
    """
    with torch.no_grad():
        if isinstance(h, (tuple, list)):
            h = h[0]
        if how == "last":
            v = h[0, -1, :]
        else:
            v = h.mean(dim=1)[0]
        return v.detach().float().cpu()

def cosine_matrix(X: torch.Tensor) -> torch.Tensor:
    """X: [L, H] -> [L, L] cosine sim matrix."""
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    return Xn @ Xn.T

def cosine_diag(A_prev: torch.Tensor, A_curr: torch.Tensor) -> torch.Tensor:
    """Diagonal cosine sim between prev and curr per layer. Both [L, H] -> [L]."""
    a = A_prev / (A_prev.norm(dim=1, keepdim=True) + 1e-8)
    b = A_curr / (A_curr.norm(dim=1, keepdim=True) + 1e-8)
    return (a * b).sum(dim=1)

class LayerTap:
    """
    Simple forward-hook manager. Register on decoder layers in order.
    After a forward pass (one diffusion step), call stacked() to get [L, H].
    """
    def __init__(self, modules, pool: str = "last"):
        self.pool = pool
        self.vecs = []
        self.handles = [m.register_forward_hook(self._hook) for m in modules]

    def _hook(self, module, inputs, output):
        try:
            self.vecs.append(pool_hidden(output, self.pool))
        except Exception:
            # Be permissive; skip weird outputs
            pass

    def clear(self):
        self.vecs.clear()

    def stacked(self) -> torch.Tensor:
        return torch.stack(self.vecs, dim=0)  # [L, H]

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()
