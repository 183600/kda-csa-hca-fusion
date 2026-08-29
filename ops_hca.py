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

from ops_csa import (
    csa_compress_kv,
    _causal_block_mask,
    _sliding_window_scores,
    _nan_safe_softmax,
    _qk_normalize,
)
from ops_rope import (
    effective_rope_dim,
    partial_rope,
    token_positions,
    block_positions,
)


def naive_hca(
    H: torch.Tensor,               # [B, T, d]
    W_KV: torch.Tensor,            # [c, d]   (nn.Linear.weight layout: [out, in])
    W_Z: torch.Tensor,             # [c, d]
    B_pos: torch.Tensor,           # [m2, c]
    W_DQ: torch.Tensor,            # [dc, d]
    W_UQ: torch.Tensor,             # [c*nh, dc]
    *,
    m2: int,                       # heavy compression factor (m' in the paper)
    nh: int,
    c: int,
    dc: int,
    scale: float | None = None,
    sliding_window: int = 0,
    sink_logits: torch.Tensor | None = None,    # [nh]
    return_projections: bool = False,
    W_KV_local: torch.Tensor | None = None,     # [c, d] local SW key/value projection
    rope_dim: int | None = None,
    rope_base: float = 10000.0,
    qk_norm_mode: str = 'l2',
):
    """Full HCA forward (heavy compression + dense MQA + optional SW + sink).

    When ``return_projections=True``, returns ``(output, projections)`` where
    ``projections`` is a tuple ``(C, Z)`` of the 2 per-token KV compression
    projections (each ``[B, original_T, c]``, trimmed to the input's original
    T before any right-padding). This lets incremental-decoding callers (e.g.
    ``run_decoding.HCAAttnDecoding``) populate an
    :class:`ops_decoding_cache.HCADecodingCache` WITHOUT recomputing the 2
    projections a second time — eliminating a redundant matmul that previously
    inflated HCA/hybrid prefill latency relative to softmax/KDA.

    **Weight layout** (P0 API fix): all ``W_*`` tensors follow the
    ``nn.Linear.weight`` convention — shape ``[out_features, in_features]``.
    Internally we use ``F.linear(x, W)`` (which computes ``x @ W.T``) instead
    of the previous ``x @ W`` form that required callers to pass
    ``self.W_KV.weight.T``. Callers now pass ``self.W_KV.weight`` directly.

    ``T`` does NOT need to be divisible by ``m2``: the function right-pads
    the sequence with zeros up to the next multiple of ``m2`` and trims the
    output back to the original length, mirroring the contract of
    ``naive_chunk_kda`` and ``naive_csa``. Real tokens keep their original
    positions; only the last partial block contains padding zeros, and the
    causal block mask ensures no real token attends to it.
    """
    B_, T, d = H.shape
    if m2 < 1:
        raise ValueError(f"heavy compression factor m2={m2} must be >= 1")
    if nh < 1:
        raise ValueError(f"nh={nh} must be >= 1")
    if c < 1:
        raise ValueError(f"c={c} must be >= 1")
    if dc < 1:
        raise ValueError(f"dc={dc} must be >= 1")
    if sliding_window < 0:
        raise ValueError(
            f"sliding_window={sliding_window} must be >= 0 "
            f"(0 disables the branch)")
    if not torch.is_floating_point(H):
        raise TypeError(
            f"naive_hca: H must be a floating-point tensor, got dtype={H.dtype}")
    if W_KV.shape != (c, d):
        raise ValueError(
            f"naive_hca: W_KV.shape={tuple(W_KV.shape)} must equal (c, d)="
            f"({c}, {d})")
    if W_Z.shape != (c, d):
        raise ValueError(
            f"naive_hca: W_Z.shape={tuple(W_Z.shape)} must equal (c, d)="
            f"({c}, {d})")
    if W_DQ.shape != (dc, d):
        raise ValueError(
            f"naive_hca: W_DQ.shape={tuple(W_DQ.shape)} must equal (dc, d)="
            f"({dc}, {d})")
    if W_UQ.shape != (c * nh, dc):
        raise ValueError(
            f"naive_hca: W_UQ.shape={tuple(W_UQ.shape)} must equal "
            f"(c*nh, dc)=({c*nh}, {dc})")
    if B_pos.shape != (m2, c):
        raise ValueError(
            f"naive_hca: B_pos.shape={tuple(B_pos.shape)} must equal "
            f"(m2, c)=({m2}, {c})")
    if sink_logits is not None and sink_logits.shape != (nh,):
        raise ValueError(
            f"naive_hca: sink_logits.shape={tuple(sink_logits.shape)} must "
            f"equal (nh,)=({nh},)")
    if W_KV_local is not None and W_KV_local.shape != (c, d):
        raise ValueError(
            f"naive_hca: W_KV_local.shape={tuple(W_KV_local.shape)} must "
            f"equal (c, d)=({c}, {d})")
    if qk_norm_mode not in ('l2', 'rms'):
        raise ValueError(
            f"naive_hca: qk_norm_mode={qk_norm_mode!r} must be 'l2' (cosine, "
            f"historical) or 'rms' (paper §2.3.3 RMSNorm form)")
    # Partial RoPE (DeepSeek-V4 §2.3.3): ``rope_dim=None`` (default)
    # disables it and preserves the historical behaviour; the model-level
    # wrappers default to the paper's 64-dim partial RoPE (clamped to the
    # head dim for small geometries).
    rope_dim_eff = effective_rope_dim(rope_dim, c)
    if rope_base <= 0:
        raise ValueError(
            f"naive_hca: rope_base={rope_base!r} must be positive")
    # Cosine-attention scale: when both ``q`` and ``C_comp`` are L2-normalized
    # (see ``F.normalize`` calls below), their dot product is already a cosine
    # similarity in ``[-1, 1]``. The previous default ``scale = c ** -0.5``
    # further shrinks the scores into a narrow band, making softmax over the
    # compressed blocks nearly uniform — effectively turning dense attention
    # into average pooling. Standard cosine-attention uses ``τ = 1``. The extra
    # ``1/sqrt(c)`` was a leftover from un-normalized softmax-attention.
    # H7 fix: scale defaults to 1.0 in the signature; no None sentinel needed.
    device = H.device
    if T == 0:
        out_empty = torch.zeros(B_, 0, nh * c, dtype=H.dtype, device=device)
        if return_projections:
            C_empty = torch.zeros(B_, 0, c, dtype=H.dtype, device=device)
            Z_empty = torch.zeros(B_, 0, c, dtype=H.dtype, device=device)
            return out_empty, (C_empty, Z_empty)
        return out_empty
    original_T = T
    pad = (-T) % m2
    if pad:
        H = F.pad(H, (0, 0, 0, pad))
        T = T + pad
    n_blocks = T // m2

    # --- 1. Heavy KV compression (single branch, no overlap) ---
    C = F.linear(H, W_KV)                                          # [B, T, c]
    Z = F.linear(H, W_Z)                                           # [B, T, c]
    C_comp = csa_compress_kv(C, Z, B_pos, m2)                     # [B, n_blocks, c] in compute_dtype
    # Partial RoPE on the compressed KV entries (DeepSeek-V4 §2.3.3): the
    # rotated entry serves as BOTH the attention key and the attention
    # value (shared-KV MQA). The anchor position of heavy-compressed entry
    # ``s`` is ``s * m2`` (block start). Rotation happens BEFORE
    # normalization — ``ops_decoding_cache.HCADecodingCache`` applies the
    # same order (rotate at append time, normalize at read time) so the
    # incremental decode path stays bit-identical to this path.
    if rope_dim_eff > 0:
        C_comp = partial_rope(
            C_comp, block_positions(n_blocks, m2, device),
            rope_dim_eff, rope_base, dim=1)
    C_comp_n = _qk_normalize(C_comp, qk_norm_mode)

    # --- 2. Dense shared-KV MQA ---
    compute_dtype = torch.float64 if H.dtype == torch.float64 else torch.float
    cQ = F.linear(H, W_DQ)                                         # [B, T, dc]
    q = F.linear(cQ, W_UQ).view(B_, T, nh, c).to(compute_dtype)    # [B, T, nh, c]
    q = _qk_normalize(q, qk_norm_mode)
    # Partial RoPE on the core-attention queries (§2.3.3): rotate each
    # query at its own token position AFTER normalization (the decoding
    # cache applies the same order: the caller normalizes ``q``, the cache
    # rotates it).
    if rope_dim_eff > 0:
        q = partial_rope(
            q, token_positions(T, device), rope_dim_eff, rope_base, dim=1)
    # Core-attention scale auto-selection: ``scale=None`` (the default)
    # resolves to 1.0 for the historical cosine 'l2' mode (identical to the
    # previous hard-coded default of 1.0) and to the standard 1/sqrt(c)
    # temperature for the paper-form 'rms' mode. An explicit ``scale=`` is
    # always honoured.
    if scale is None:
        scale = c ** -0.5 if qk_norm_mode == 'rms' else 1.0

    cbm = _causal_block_mask(T, n_blocks, m2, device)

    scores = torch.einsum('b t h d, b n d -> b h t n', q, C_comp_n) * scale
    scores = scores.masked_fill(~cbm[None, None], float('-inf'))
        # --- 3. Sliding-window branch (uncompressed local KV) ---
    if sliding_window > 0:
        win = sliding_window
        if W_KV_local is not None:
            C_local_raw = F.linear(H, W_KV_local)                   # [B, T, c]
        else:
            C_local_raw = C
        C_local = _qk_normalize(C_local_raw.to(compute_dtype), qk_norm_mode)  # [B, T, c]
        # Partial RoPE on the sliding-window keys (§2.3.3): each key
        # rotates at its own token position; the decoding cache's ring
        # buffer applies the same normalize-then-rotate order at append
        # time.
        if rope_dim_eff > 0:
            C_local = partial_rope(
                C_local, token_positions(T, device), rope_dim_eff,
                rope_base, dim=1)
        # The third return value (per-window validity mask) is not needed
        # here: unlike naive_csa, HCA's softmax consumes ``scores_w`` alone
        # (already ``-inf``-masked for the left padding inside
        # ``_sliding_window_scores``), so the mask is intentionally dropped.
        scores_w, C_windows, _ = _sliding_window_scores(
            q, C_local, win, scale, device)
        scores_w = scores_w.permute(0, 2, 1, 3)                     # [B, nh, T, win]
    else:
        scores_w = torch.empty(B_, nh, T, 0, dtype=compute_dtype, device=device)
        C_windows = torch.empty(B_, T, 0, c, dtype=compute_dtype, device=device)

    # --- 4. JOINT softmax over the union of compressed + window entries ---
    # One softmax over the concatenated scores (DeepSeek-V4 §2.3.1 Eq. 27),
    # with the attention sink in the single shared denominator. The sink is a
    # *virtual* entry that appears ONLY in the denominator,
    # ``denom = sum_i exp(s_i) + exp(sink)``. It does NOT contribute a value
    # to the output: the expected KV of a virtual entry with no source tokens
    # is zero, not a constant 1 broadcast across the head dim.
    cat = torch.cat([scores, scores_w], dim=-1)                    # [B, nh, T, n_blocks+win]
    if cat.shape[-1] == 0:
        out = torch.zeros(B_, T, nh, c, dtype=compute_dtype, device=device)
    elif sink_logits is not None:
        log_sink = sink_logits.view(1, nh, 1, 1).to(cat)           # [1, nh, 1, 1]
        row_max = cat.amax(-1, keepdim=True)                       # [B, nh, T, 1]
        all_masked = torch.isneginf(row_max)                       # [B, nh, T, 1]
        row_max_safe = torch.where(all_masked, torch.zeros_like(row_max), row_max)
        shifted = cat - row_max_safe
        shifted_sink = log_sink - row_max_safe
        lse = torch.logsumexp(shifted, dim=-1, keepdim=True)
        log_denom = torch.logaddexp(lse, shifted_sink)             # [B, nh, T, 1]
        p = (shifted - log_denom).exp()                            # [B, nh, T, n_blocks+win]
        p = p.masked_fill(all_masked, 0.0)
    else:
        all_masked = cat.isinf().all(-1, keepdim=True)
        p = _nan_safe_softmax(cat, dim=-1, all_masked_mask=all_masked)

    n_blk = scores.shape[-1]
    p_dense = p[..., :n_blk]
    p_w = p[..., n_blk:]
    out = (torch.einsum('b h t n, b n d -> b t h d', p_dense, C_comp_n)   # [B, T, nh, c]
           + torch.einsum('b h t w, b t w d -> b t h d', p_w, C_windows))

    # Partial RoPE output countermeasure (§2.3.3): because the compressed
    # KV entries serve as both keys AND values, the naive output carries
    # absolute-position rotations; inverse-rotating at the NEGATED query
    # position makes each entry's contribution depend on the relative
    # (entry − query) distance only.
    if rope_dim_eff > 0:
        out = partial_rope(
            out, -token_positions(T, device), rope_dim_eff, rope_base, dim=1)

    out_final = out.reshape(B_, T, nh * c).to(H.dtype)[:, :original_T]
    if return_projections:
        projections = (
            C[:, :original_T],   # [B, original_T, c]
            Z[:, :original_T],   # [B, original_T, c]
        )
        return out_final, projections
    return out_final
