"""Compressed Sparse Attention (CSA) — naive PyTorch reference.

Implements the CSA operator from DeepSeek-V4 (arXiv:2606.19348v1, §2.3.1):

    1. KV compression: every ``m`` consecutive KV entries are consolidated into
       one via a softmax-weighted combination (overlapped compression with two
       branches C^a, C^b is supported, matching the paper).
    2. Lightning indexer: low-rank per-head queries score the compressed
       entries; the top-k entries per query token are retained (DeepSeek Sparse
       Attention selection, ReLU-based head-wise aggregation).
    3. Shared-KV MQA core attention over the selected compressed entries.

This is a faithful, readable CPU implementation intended for correctness checks
and small-scale experiments; it is NOT the production kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _causal_block_mask(T: int, n_blocks: int, m: int, device) -> torch.Tensor:
    """Return ``[T, n_blocks]`` mask: query t can attend to compressed block b
    only if ``b < t // m`` (strictly preceding blocks)."""
    i_t = torch.arange(T, device=device)
    i_b = torch.arange(n_blocks, device=device)
    return i_t[:, None] // m > i_b[None, :]


def csa_compress_kv(
    C: torch.Tensor,
    Z: torch.Tensor,
    B_pos: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """Compress ``[B, T, c]`` KV entries ``C`` into ``[B, T//m, c]``.

    Implements Eq. (22)/(23) of the DeepSeek-V4 paper for a single branch (HCA
    style) and the two-branch overlapped variant for CSA via the public
    ``csa_compress_kv_overlapped`` wrapper.

    Args:
        C: KV entries ``[B, T, c]``
        Z: compression weights ``[B, T, c]``
        B_pos: learnable positional bias ``[m, c]``
        m: compression factor (T must be divisible by m).
    """
    B_, T, c = C.shape
    assert T % m == 0, f"T={T} must be divisible by m={m}"
    n_blocks = T // m
    # Preserve float64 for high-precision tests; default to float32 otherwise.
    compute_dtype = torch.float64 if C.dtype == torch.float64 else torch.float
    C = C.to(compute_dtype).view(B_, n_blocks, m, c)
    Z = Z.to(compute_dtype).view(B_, n_blocks, m, c)
    logits = Z + B_pos[None, None, :, :].to(Z)
    S = torch.softmax(logits, dim=2)                      # [B, n_blocks, m, c]
    return (S * C).sum(dim=2)                             # [B, n_blocks, c]


def csa_compress_kv_overlapped(
    Ca: torch.Tensor,
    Cb: torch.Tensor,
    Za: torch.Tensor,
    Zb: torch.Tensor,
    Ba: torch.Tensor,
    Bb: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """Two-branch overlapped compression (CSA, Eq. 11–12).

    Each compressed entry fuses ``m`` entries from ``Ca`` at the current block
    and ``m`` entries from ``Cb`` of the *previous* block, so consecutive
    compressed entries share half of their source tokens.
    """
    B_, T, c = Ca.shape
    assert T % m == 0
    n_blocks = T // m
    compute_dtype = torch.float64 if Ca.dtype == torch.float64 else torch.float
    Ca = Ca.to(compute_dtype).view(B_, n_blocks, m, c)
    Cb = Cb.to(compute_dtype).view(B_, n_blocks, m, c)
    Za = Za.to(compute_dtype).view(B_, n_blocks, m, c)
    Zb = Zb.to(compute_dtype).view(B_, n_blocks, m, c)

    # Vectorized two-branch overlapped compression.
    # a-branch logits for ALL blocks: [B, n_blocks, m, c]
    A_logits = Za + Ba[None, None, :, :].to(Za)
    # b-branch logits shifted by one so block i uses block i-1.
    # First block has no previous -> pad with -inf so softmax assigns 0 weight.
    neg_inf_pad = torch.full((B_, 1, m, c), float('-inf'),
                             device=Ca.device, dtype=compute_dtype)
    Zb_prev = torch.cat([neg_inf_pad, Zb[:, :-1]], dim=1)         # [B, n_blocks, m, c]
    Bb_logits = Zb_prev + Bb[None, None, :, :].to(Zb_prev)        # [B, n_blocks, m, c]

    # cat along the m axis -> [B, n_blocks, 2m, c]
    all_logits = torch.cat([A_logits, Bb_logits], dim=2)
    S = torch.softmax(all_logits, dim=2)                          # [B, n_blocks, 2m, c]
    Sa = S[:, :, :m]                                              # [B, n_blocks, m, c]
    Sb = S[:, :, m:]                                              # [B, n_blocks, m, c]

    # Cb shifted the same way: first block has no previous -> zero contribution.
    zero_pad = torch.zeros((B_, 1, m, c),
                           device=Ca.device, dtype=compute_dtype)
    Cb_prev = torch.cat([zero_pad, Cb[:, :-1]], dim=1)            # [B, n_blocks, m, c]

    out = (Sa * Ca).sum(2) + (Sb * Cb_prev).sum(2)                # [B, n_blocks, c]
    return out


def csa_lightning_indexer(
    q_idx: torch.Tensor,            # [B, T, HI, DI]
    k_idx: torch.Tensor,            # [B, n_blocks, DI]
    w_idx: torch.Tensor | None,     # [B, T, HI]
    topk: int,
    scale: float | None = None,
    causal_block_mask: torch.Tensor | None = None,   # [T, n_blocks]
) -> torch.Tensor:
    """Top-k selection over compressed indexer keys (Eq. 13–17).

    Returns indices of shape ``[B, T, topk]`` (padded with -1).
    """
    if scale is None:
        scale = q_idx.shape[-1] ** -0.5
    B_, T, HI, DI = q_idx.shape
    n_blocks = k_idx.shape[1]
    compute_dtype = torch.float64 if q_idx.dtype == torch.float64 else torch.float
    q_idx = q_idx.to(compute_dtype)
    k_idx = k_idx.to(compute_dtype)
    if w_idx is not None:
        w_idx = w_idx.to(compute_dtype)

    # Vectorized batched scoring (no per-batch loop).
    # head-wise similarities [B, HI, T, n_blocks]
    score = torch.einsum('b t h d, b n d -> b h t n', q_idx, k_idx) * scale
    score = F.relu(score)
    if w_idx is None:
        logits = score.sum(1)                                          # [B, T, n_blocks]
    else:
        logits = torch.einsum('b h t n, b t h -> b t n', score, w_idx)  # [B, T, n_blocks]
    if causal_block_mask is not None:
        logits = logits.masked_fill(~causal_block_mask, float('-inf'))
    S = min(topk, n_blocks)
    values, idx = torch.topk(logits, S, dim=-1)                        # [B, T, S]
    idx = idx.masked_fill(torch.isinf(values), -1)
    if topk > S:
        idx = torch.cat([idx, idx.new_full((B_, T, topk - S), -1)], dim=-1)
    return idx


def naive_csa(
    H: torch.Tensor,               # [B, T, d]   input hidden states
    W_aKV: torch.Tensor,           # [d, c]
    W_bKV: torch.Tensor,           # [d, c]
    W_aZ: torch.Tensor,            # [d, c]
    W_bZ: torch.Tensor,            # [d, c]
    Ba: torch.Tensor,              # [m, c]
    Bb: torch.Tensor,              # [m, c]
    W_DQ: torch.Tensor,            # [d, dc]
    W_UQ: torch.Tensor,            # [dc, c*nh]
    W_IUQ: torch.Tensor,           # [dc, c_I*nIh]
    W_w: torch.Tensor,             # [d, nIh]
    W_KV_idx: torch.Tensor,        # [d, c_I]   for indexer key compression
    W_Z_idx: torch.Tensor,         # [d, c_I]
    B_idx: torch.Tensor,           # [m, c_I]
    *,
    m: int,
    topk: int,
    nh: int,
    nIh: int,
    c: int,
    c_I: int,
    dc: int,
    scale: float | None = None,
    sliding_window: int = 0,
    sink_logits: torch.Tensor | None = None,    # [nh]
) -> torch.Tensor:
    """Full CSA forward (compression + indexer + sparse MQA core attention).

    Returns output ``[B, T, d]`` (after a simple grouped-output projection is
    elided here for clarity; we project ``[B, T, c*nh] -> d`` with one matrix).
    """
    B_, T, d = H.shape
    if scale is None:
        scale = c ** -0.5
    device = H.device
    n_blocks = T // m
    assert T % m == 0

    # --- 1. Compress KV (two-branch overlapped) ---
    Ca = H @ W_aKV
    Cb = H @ W_bKV
    Za = H @ W_aZ
    Zb = H @ W_bZ
    C_comp = csa_compress_kv_overlapped(Ca, Cb, Za, Zb, Ba, Bb, m)   # [B, n_blocks, c]

    # --- 2. Lightning indexer ---
    # compressed indexer keys via the same compression (single-branch here for simplicity)
    K_idx_raw = H @ W_KV_idx
    Z_idx = H @ W_Z_idx
    K_IComp = csa_compress_kv(K_idx_raw, Z_idx, B_idx, m)            # [B, n_blocks, c_I]
    # indexer queries (low-rank)
    cQ = H @ W_DQ                                                   # [B, T, dc]
    q_idx = (cQ @ W_IUQ).view(B_, T, nIh, c_I)                     # [B, T, nIh, c_I]
    w_idx = H @ W_w                                                # [B, T, nIh]
    cbm = _causal_block_mask(T, n_blocks, m, device)
    indices = csa_lightning_indexer(q_idx, K_IComp, w_idx, topk, scale=scale,
                                    causal_block_mask=cbm)          # [B, T, topk]

    # --- 3. Shared-KV MQA core attention ---
    # attention queries (low-rank up-projection)
    q = (cQ @ W_UQ).view(B_, T, nh, c)                             # [B, T, nh, c]
    q = F.normalize(q, dim=-1)
    C_comp_n = F.normalize(C_comp, dim=-1)                         # head-wise RMSNorm simplified

    # --- Vectorized sparse MQA core attention ---
    # Gather selected compressed KV entries for every (b, t) in one shot.
    # indices: [B, T, topk], padded with -1 for invalid slots.
    valid_mask = indices >= 0                                        # [B, T, topk]
    idx_safe = indices.clamp(min=0)                                  # [B, T, topk]
    batch_idx = torch.arange(B_, device=device).view(B_, 1, 1)      # [B, 1, 1]
    kv = C_comp_n[batch_idx, idx_safe]                               # [B, T, topk, c]

    # Per-head attention scores over the topk selected blocks.
    scores = torch.einsum('b t h d, b t k d -> b t h k', q, kv) * scale  # [B, T, nh, topk]
    # Mask -1 padding so those slots get zero weight after softmax.
    scores = scores.masked_fill(~valid_mask[:, :, None, :], float('-inf'))

    if sink_logits is not None:
        # Attention sink: a per-head constant added to the denominator.
        # Numerically stable logsumexp approach — keep sink_logits in
        # log space (never exp it, which could overflow to inf when the
        # learnable parameter grows during training and makes denom=inf,
        # p=0/inf=nan or inf/inf=nan).
        log_sink = sink_logits.view(1, 1, nh, 1)                    # [1, 1, nh, 1]
        vmask = valid_mask[:, :, None, :].to(scores.dtype)          # [B, T, 1, topk]
        # scores already carries -inf at invalid slots (masked above),
        # i.e. it is the "safe scores". Row max clamped >= 0 so we only
        # ever shift scores down, keeping them aligned with log_sink
        # which is already in log space (shifting up would inflate
        # exp() and could overflow).
        m = scores.amax(-1, keepdim=True).clamp(min=0)              # [B, T, nh, 1]
        shifted = scores - m                                        # [B, T, nh, topk]
        log_sum_exp = torch.logsumexp(shifted, dim=-1, keepdim=True)
        log_denom = torch.logaddexp(log_sum_exp, log_sink)          # [B, T, nh, 1]
        p = ((shifted - log_denom).exp() * vmask)                   # [B, T, nh, topk]
    else:
        # NaN-safe softmax: rows that are entirely -inf yield all-zero p.
        row_max = scores.amax(-1, keepdim=True)                     # [B, T, nh, 1]
        row_max = torch.where(torch.isinf(row_max),
                              torch.zeros_like(row_max), row_max)
        vmask = valid_mask[:, :, None, :].to(scores.dtype)
        exp_scores = (scores - row_max).exp() * vmask               # 0 at invalid slots
        denom = exp_scores.sum(-1, keepdim=True).clamp(min=1e-20)
        p = exp_scores / denom                                       # [B, T, nh, topk]

    out = torch.einsum('b t h k, b t k d -> b t h d', p, kv)        # [B, T, nh, c]
    out = out.to(H.dtype)

    # optional sliding window branch (local uncompressed KV)
    if sliding_window > 0:
        win = sliding_window
        # Precompute H @ W_aKV once (reuse Ca from §1) instead of redoing the
        # matmul per (b, t).
        H_proj = Ca  # [B, T, c], already == H @ W_aKV
        # Position-relative quantities depend only on T (not on batch), so we
        # build them once; query t attends to positions with -win < n-t <= 0
        # i.e. the causal window [t-win+1, t] (past + current only).
        i_t = torch.arange(T, device=device)
        dist = i_t[None, :] - i_t[:, None]                       # [T, T]
        win_mask = (dist <= 0) & (dist > -win)                   # [T, T]
        # Fully vectorized batched shared-KV MQA over the local window.
        # q:        [B, T, nh, c]
        # C_local:  [B, T, c]
        # scores:   [B, nh, T, T]  (b h t n)
        # win_mask broadcasts from [1, 1, T, T] -> [B, nh, T, T].
        # NOTE: every query t always has itself in the window (dist=0 satisfies
        # the mask for win >= 1), so no row is fully -inf and softmax is NaN-free.
        C_local = F.normalize(H_proj, dim=-1)                    # [B, T, c]
        q_local = F.normalize(q, dim=-1)                         # [B, T, nh, c]
        scores = torch.einsum('b t h d, b n d -> b h t n', q_local, C_local) * scale
        scores = scores.masked_fill(~win_mask[None, None], float('-inf'))
        p = torch.softmax(scores, dim=-1)
        sw_out = torch.einsum('b h t n, b n d -> b t h d', p, C_local)
        out = out + sw_out

    # Return the raw per-head core-attention output [B, T, nh, c] flattened to
    # [B, T, nh*c]; the caller performs the grouped output projection.
    return out.reshape(B_, T, nh * c).to(H.dtype)
