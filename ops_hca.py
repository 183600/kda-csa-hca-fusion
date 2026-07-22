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
    _sliding_window_attention,
    _nan_safe_softmax,
    CHUNKED_SW_THRESHOLD,
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
    scale: float = 1.0,
    sliding_window: int = 0,
    sink_logits: torch.Tensor | None = None,    # [nh]
    return_projections: bool = False,
    W_KV_local: torch.Tensor | None = None,     # [c, d] local SW key/value projection
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
    C_comp_n = F.normalize(C_comp, dim=-1)

    # --- 2. Dense shared-KV MQA ---
    compute_dtype = torch.float64 if H.dtype == torch.float64 else torch.float
    cQ = F.linear(H, W_DQ)                                         # [B, T, dc]
    q = F.linear(cQ, W_UQ).view(B_, T, nh, c).to(compute_dtype)    # [B, T, nh, c]
    q = F.normalize(q, dim=-1)

    cbm = _causal_block_mask(T, n_blocks, m2, device)

    scores = torch.einsum('b t h d, b n d -> b h t n', q, C_comp_n) * scale
    scores = scores.masked_fill(~cbm[None, None], float('-inf'))
    if sink_logits is not None:
        log_sink = sink_logits.view(1, nh, 1, 1).to(scores)  # [1, nh, 1, 1]
        row_max = scores.amax(-1, keepdim=True)                    # [B, nh, T, 1]
        all_masked = torch.isneginf(row_max)                     # [B, nh, T, 1]
        row_max_safe = torch.where(all_masked, torch.zeros_like(row_max), row_max)
        shifted = scores - row_max_safe                            # [B, nh, T, n_blocks]
        shifted_sink = log_sink - row_max_safe                      # [B, nh, T, 1]
        lse = torch.logsumexp(shifted, dim=-1, keepdim=True)
        log_denom = torch.logaddexp(lse, shifted_sink)              # [B, nh, T, 1]
        p = (shifted - log_denom).exp()                            # [B, nh, T, n_blocks]
        p = p.masked_fill(all_masked, 0.0)
        p_sink = (shifted_sink - log_denom).exp()                 # [B, nh, T, 1]
        p_sink = p_sink.masked_fill(all_masked, 0.0)
    else:
        p = _nan_safe_softmax(scores, dim=-1)
    out = torch.einsum('b h t n, b n d -> b t h d', p, C_comp_n)   # [B, T, nh, c]
    if sink_logits is not None:
        out = out + p_sink

    # --- 3. Sliding window branch (uncompressed local KV) ---
    if sliding_window > 0:
        win = sliding_window
        if W_KV_local is not None:
            C_local_raw = F.linear(H, W_KV_local)                   # [B, T, c]
        else:
            C_local_raw = C
        C_local = F.normalize(C_local_raw.to(compute_dtype), dim=-1)  # [B, T, c]
        C_local = C_local.unsqueeze(2).expand(-1, -1, nh, -1)        # [B, T, nh, c]
        sw_out = _sliding_window_attention(q, C_local, win, scale, device)
        out = out + sw_out

    out_final = out.reshape(B_, T, nh * c).to(H.dtype)[:, :original_T]
    if return_projections:
        projections = (
            C[:, :original_T],   # [B, original_T, c]
            Z[:, :original_T],   # [B, original_T, c]
        )
        return out_final, projections
    return out_final
