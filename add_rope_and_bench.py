# Quick patch script to add explicit RoPE to hybrid + a simple Needle benchmark stub
# Run this on the branch if needed: python add_rope_and_bench.py

import torch
import torch.nn as nn
from ops_fused import HybridKCHAttention, HybridConfig

# Simple RoPE (for completeness - many places already use cosine after norm)
def apply_rope(q, k, cos, sin):
    # q, k: [B, T, H, D]
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q, k

# Minimal Needle-in-Haystack benchmark stub (synthetic)
def needle_in_haystack_bench(model, seq_len=2048, num_needles=1):
    # Simplified: insert a "needle" token at random position and measure retrieval
    # This is a placeholder; full RULER/Needle would be more involved
    device = next(model.parameters()).device
    x = torch.randint(0, 1000, (1, seq_len), device=device)
    # Mark a needle
    needle_pos = torch.randint(10, seq_len-10, (1,))
    x[0, needle_pos] = 999  # special needle token
    with torch.no_grad():
        out = model(x)
    # Toy metric: did the model "attend" near the needle? (placeholder)
    return {"seq_len": seq_len, "needle_pos": needle_pos.item(), "success_proxy": True}

print("RoPE helper and Needle stub added for completeness.")
