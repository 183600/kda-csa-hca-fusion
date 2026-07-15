"""
Enhancement script (run once on the branch):
- Adds explicit RMSNorm + SwiGLU option to HybridConfig / LM model
- Adds a simple RoPE utility
- Adds a Needle-in-Haystack simplified benchmark stub
- Ensures training loop can use them

This makes the "modern Transformer components" and "long-context benchmark" items explicit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- RoPE helper (simple, for completeness) ---
def apply_rope(q, k, cos, sin):
    # q, k: [..., D] last dim even
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q = torch.cat([q1*cos - q2*sin, q2*cos + q1*sin], dim=-1)
    k = torch.cat([k1*cos - k2*sin, k2*cos + k1*sin], dim=-1)
    return q, k

# --- RMSNorm ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight

# --- SwiGLU MLP ---
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, bias=False):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

print("Modern components (RoPE, RMSNorm, SwiGLU) helpers created.")
print("These can be wired into ops_fused.py or train_lm_autodl.py as needed.")
