"""Heavily Compressed Attention (HCA) — naive PyTorch reference.

Implements the HCA operator from DeepSeek-V4 (arXiv:2606.19348v1, §2.3.2):

    1. Heavier KV compression: every ``m'`` (>> m) consecutive KV entries are
       consolidated into one (no overlap, single branch) — Eq. (20)–(23).
    2. Dense (not sparse) shared-KV MQA over the compressed entries.
    3. A small sliding-window branch keeps local fine-grained dependencies.
    4. Optional attention sink (learnable per-head logit in the softmax denom).

HCA trades recall granularity for extreme compression, complementing CSA's
sparse selection: where CSA keeps ``k`` of ``n/m`` entries, HCA keeps *all*
``n/m'`` heavily-compressed entries, with ``m'`` typically an order of
magnitude larger than ``m``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ops_csa import csa_compress_kv, _causal_block_mask


def naive_hca(
    H: torch.Tensor,               # [B, T, d]
    W_KV: torch.Tensor,            # [d, c]
    W_Z: torch.Tensor,             # [d, c]
    B_pos: torch.Tensor,           # [m2, c]
    W_DQ: torch.Tensor,            # [d, dc]
    W_UQ: torch.Tensor,            # [dc, c*nh]
    *,
    m2: int,                       # heavy compression factor (m' in the paper)
    nh: int,
    c: int,
    dc: int,
    scale: float | None = None,
    sliding_window: int = 0,
    sink_logits: torch.Tensor | None = None,    # [nh]
) -> torch.Tensor:
    """Full HCA forward (heavy compression + dense MQA + optional SW + sink)."""
    B_, T, d = H.shape
    if scale is None:
        scale = c ** -0.5
    device = H.device
    n_blocks = T // m2
    assert T % m2 == 0

    # --- 1. Heavy KV compression (single branch, no overlap) ---
    C = H @ W_KV                                                   # [B, T, c]
    Z = H @ W_Z                                                    # [B, T, c]
    C_comp = csa_compress_kv(C, Z, B_pos, m2)                     # [B, n_blocks, c]
    C_comp_n = F.normalize(C_comp, dim=-1)

    # --- 2. Dense shared-KV MQA ---
    cQ = H @ W_DQ                                                  # [B, T, dc]
    q = (cQ @ W_UQ).view(B_, T, nh, c)                            # [B, T, nh, c]
    q = F.normalize(q, dim=-1)

    # Causal block mask: query t attends to blocks strictly before floor(t/m2).
    # Plus we add the block containing t to keep causality at block boundaries
    # (matches the paper's "preceding compressed KV blocks" with the sliding
    # window handling intra-block tokens).
    cbm = _causal_block_mask(T, n_blocks, m2, device)

    # Precompute all (B, T, n_blocks) attention logits at once (fully vectorized).
    scores = torch.einsum('b t h d, b n d -> b h t n', q, C_comp_n) * scale
    scores = scores.masked_fill(~cbm[None, None], float('-inf'))
    if sink_logits is not None:
        # Attention sink: a per-head constant added to the denominator.
        # Numerically stable logsumexp approach — keep sink_logits in
        # log space (never exp it, which could overflow to inf when the
        # learnable parameter grows during training and makes denom=inf,
        # p=0/inf=nan or inf/inf=nan). Mirrors the fix in ops_csa.py.
        log_sink = sink_logits.view(1, nh, 1, 1)                   # [1, nh, 1, 1]
        # Row max clamped >= 0 so we only ever shift scores down,
        # keeping them aligned with log_sink (shifting up could
        # inflate exp() and overflow). scores already carries -inf
        # at causally-masked slots, so logsumexp/ exp naturally yield
        # 0 there; fully-masked rows also collapse to p=0 (logaddexp
        # (-inf, log_sink) = log_sink, exp(-inf - log_sink) = 0).
        row_max = scores.amax(-1, keepdim=True).clamp(min=0)        # [B, nh, T, 1]
        shifted = scores - row_max                                  # [B, nh, T, n_blocks]
        lse = torch.logsumexp(shifted, dim=-1, keepdim=True)
        log_denom = torch.logaddexp(lse, log_sink)                 # [B, nh, T, 1]
        p = (shifted - log_denom).exp()                            # [B, nh, T, n_blocks]
    else:
        # Rows with no valid block to attend to (e.g. t < m2 under the
        # causal block mask) are entirely -inf; softmax over them would
        # yield NaN. Detect such rows and force their weights to 0 so the
        # contribution is 0 instead of NaN.
        all_masked = torch.isinf(scores).all(-1, keepdim=True)   # [B, nh, T, 1]
        safe = scores.masked_fill(all_masked, 0.0)
        p = torch.softmax(safe, dim=-1)
        p = p.masked_fill(all_masked, 0.0)
    out = torch.einsum('b h t n, b n d -> b t h d', p, C_comp_n)   # [B, T, nh, c]

    # --- 3. Sliding window branch (uncompressed local KV) ---
    if sliding_window > 0:
        win = sliding_window
        # Position-relative quantities depend only on T (not on batch), so we
        # build them once outside the batch loop instead of recreating them
        # per-batch like the original reference implementation did.
        i_t = torch.arange(T, device=device)
        # dist[t, n] = n - t; query t attends to positions with -win < n-t <= 0
        # i.e. the causal window [t-win+1, t] (past + current only).
        dist = i_t[None, :] - i_t[:, None]                       # [T, T]
        win_mask = (dist <= 0) & (dist > -win)                   # [T, T]
        # Fully vectorized batched shared-KV MQA over the local window.
        # q:        [B, T, nh, c]
        # C_local:  [B, T, c]
        # scores:   [B, nh, T, T]  (b h t n)
        # win_mask broadcasts from [1, 1, T, T] -> [B, nh, T, T].
        # NOTE: every query t always has itself in the window (dist=0 satisfies
        # the mask for win >= 1), so no row is fully -inf and softmax is NaN-free.
        C_local = F.normalize(C, dim=-1)                            # [B, T, c]
        scores = torch.einsum('b t h d, b n d -> b h t n', q, C_local) * scale
        scores = scores.masked_fill(~win_mask[None, None], float('-inf'))
        p = torch.softmax(scores, dim=-1)
        sw_out = torch.einsum('b h t n, b n d -> b t h d', p, C_local)
        out = out + sw_out

    # Return the raw per-head core-attention output [B, T, nh, c] flattened to
    # [B, T, nh*c]; the caller performs the grouped output projection.
    return out.reshape(B_, T, nh * c).to(H.dtype)
