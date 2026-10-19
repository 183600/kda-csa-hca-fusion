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
    _causal_block_mask_invalid,
    _sliding_window_chunk_windows,
    _sliding_window_output,
    _sliding_window_scores,
    _nan_safe_softmax,
    _qk_normalize,
    _qk_normalize_,
    CHUNKED_SW_THRESHOLD,
)
from ops_rope import (
    effective_rope_dim,
    partial_rope,
    token_positions,
    block_positions,
)


# ---------------------------------------------------------------------------
# bfloat16 numerical stability helpers (mirrors ops_kda_autograd.py / ops_csa.py)
# ---------------------------------------------------------------------------
def _bf16_safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Safe softmax for bfloat16: clamp logits before softmax to prevent Inf."""
    if logits.dtype == torch.bfloat16:
        logits = logits.clamp(min=-50.0, max=50.0)
    return torch.softmax(logits, dim=dim)


def _bf16_safe_exp(x: torch.Tensor, clamp_max: float = 50.0) -> torch.Tensor:
    """Safe exp for bfloat16: clamp before exp to prevent overflow."""
    if x.dtype == torch.bfloat16:
        x = x.clamp(max=clamp_max)
    return x.exp()


def _bf16_scale_if_needed(
    x: torch.Tensor, max_norm: float = 1e4
) -> torch.Tensor:
    """Scale tensor down if its norm exceeds max_norm (bf16 only)."""
    if x.dtype != torch.bfloat16:
        return x
    norms = x.norm(p=2, dim=-1, keepdim=True)
    scale = (max_norm / (norms + 1e-8)).clamp(max=1.0)
    return x * scale


# Query-axis tile budget for the dense score / joint-softmax / value
# pipeline (elements of the ``[B, nh, T_tile, n_blocks]`` score tensor).
# Below this size the single-shot path runs unchanged.
_HCA_SCORE_TILE_ELEMS = 1 << 22


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
    :class:`ops_decoding_cache.HCADecodingCache` directly from this call.

    **Weight layout**: all ``W_*`` tensors follow the
    ``nn.Linear.weight`` convention — shape ``[out_features, in_features]``.
    Internally we use ``F.linear(x, W)`` (which computes ``x @ W.T``).
    Callers pass ``self.W_KV.weight`` directly.

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
    # disables it; the model-level wrappers default to the paper's 64-dim
    # partial RoPE (clamped to the head dim for small geometries).
    rope_dim_eff = effective_rope_dim(rope_dim, c)
    if rope_base <= 0:
        raise ValueError(
            f"naive_hca: rope_base={rope_base!r} must be positive")
    # Cosine-attention scale: when both ``q`` and ``C_comp`` are L2-normalized
    # (see ``F.normalize`` calls below), their dot product is already a cosine
    # similarity in ``[-1, 1]``. A ``1/sqrt(c)`` scale would further shrink the
    # scores into a narrow band, making softmax over the compressed blocks
    # nearly uniform — effectively turning dense attention into average
    # pooling. Standard cosine attention uses ``τ = 1`` (the default).
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
    # ``C_comp`` is a fresh compression output (optionally re-bound by
    # ``partial_rope``) with no other readers left, so with autograd
    # disabled the normalization runs in place (identical elementwise
    # chain, one fewer block-sized temporary).
    C_comp_n = (_qk_normalize(C_comp, qk_norm_mode)
                if torch.is_grad_enabled()
                else _qk_normalize_(C_comp, qk_norm_mode))

    # --- 2. Dense shared-KV MQA ---
    compute_dtype = torch.float64 if H.dtype == torch.float64 else torch.float
    cQ = F.linear(H, W_DQ)                                         # [B, T, dc]
    q = F.linear(cQ, W_UQ).view(B_, T, nh, c).to(compute_dtype)    # [B, T, nh, c]
    # Same reasoning as above: ``q`` is this call's private tensor.
    q = (_qk_normalize(q, qk_norm_mode) if torch.is_grad_enabled()
         else _qk_normalize_(q, qk_norm_mode))
    # Partial RoPE on the core-attention queries (§2.3.3): rotate each
    # query at its own token position AFTER normalization (the decoding
    # cache applies the same order: the caller normalizes ``q``, the cache
    # rotates it). One position vector is shared by every RoPE call below
    # (query, sliding-window keys, negated output rotation).
    tpos = token_positions(T, device) if rope_dim_eff > 0 else None
    if rope_dim_eff > 0:
        q = partial_rope(q, tpos, rope_dim_eff, rope_base, dim=1)
    # Core-attention scale auto-selection: ``scale=None`` (the default)
    # resolves to 1.0 for the cosine 'l2' mode and to the standard 1/sqrt(c)
    # temperature for the paper-form 'rms' mode. An explicit ``scale=`` is
    # always honoured.
    if scale is None:
        scale = c ** -0.5 if qk_norm_mode == 'rms' else 1.0

    # Query-axis tiling for the dense score / joint-softmax / value
    # pipeline. Every operation below is row-wise along the query axis, so
    # evaluating query tiles is value-identical per row while the peak score
    # tensor shrinks from ``B*nh*T*n_blocks`` to
    # ``B*nh*T_tile*n_blocks``. Small shapes (including every bit-exact test
    # geometry) keep the single-shot path below.
    score_elems = B_ * nh * T * n_blocks
    tile_T = T
    if T > 1 and n_blocks > 0 and score_elems > _HCA_SCORE_TILE_ELEMS:
        # Floor the tile so the per-tile call always stays on the same
        # batched-GEMM path as the single-shot call (single/small-M
        # dispatch can select a different reduction kernel).
        tile_T = max(8, _HCA_SCORE_TILE_ELEMS // max(1, B_ * nh * n_blocks))
    use_tiles = tile_T < T
    # When tiling, the blocked mask is rebuilt per query tile
    # (O(T_tile*n_blocks)) instead of materializing and caching the full
    # [T, n_blocks] table; the untiled path keeps using the cached negation
    # (same values for every row).
    cbm_invalid = (None if use_tiles else
                   _causal_block_mask_invalid(T, n_blocks, m2, device))

    # --- 3. Sliding-window branch (uncompressed local KV) ---
    if sliding_window > 0:
        win = sliding_window
        if W_KV_local is not None:
            C_local_raw = F.linear(H, W_KV_local)                   # [B, T, c]
        else:
            C_local_raw = C
        C_local_src = C_local_raw.to(compute_dtype)
        # In-place only when the cast produced a fresh copy or ``C`` is not
        # returned (otherwise the projection storage would be mutated).
        if (not torch.is_grad_enabled()) and (
                C_local_src is not C or not return_projections):
            C_local = _qk_normalize_(C_local_src, qk_norm_mode)     # [B, T, c]
        else:
            C_local = _qk_normalize(C_local_src, qk_norm_mode)      # [B, T, c]
        # Partial RoPE on the sliding-window keys (§2.3.3): each key
        # rotates at its own token position; the decoding cache's ring
        # buffer applies the same normalize-then-rotate order at append
        # time.
        if rope_dim_eff > 0:
            C_local = partial_rope(
                C_local, tpos, rope_dim_eff, rope_base, dim=1)
        # The third return value (per-window validity mask) is not needed
        # here: unlike naive_csa, HCA's softmax consumes ``scores_w`` alone
        # (already ``-inf``-masked for the left padding inside
        # ``_sliding_window_scores``), so the mask is intentionally dropped.
        # Inference fast path: materialize the full [B, T, win, c] window
        # tensor ONCE when it fits the same per-tile element budget the
        # tiled path already uses; the window value output then reuses it
        # instead of re-deriving it tile-by-tile. Same per-row contraction
        # either way (forward values bit-identical); only one densification
        # copy is dropped. GATED ON no_grad so backward keeps the original
        # per-tile graph exactly (see ``ops_csa.naive_csa``). Larger
        # sequences keep the bounded-memory tiled path.
        use_full_windows = ((not torch.is_grad_enabled())
                            and (B_ * T * win * c) <= CHUNKED_SW_THRESHOLD)
        scores_w, C_windows_full, _ = _sliding_window_scores(
            q, C_local, win, scale, device,
            return_windows=use_full_windows)
        scores_w = scores_w.permute(0, 2, 1, 3)                     # [B, nh, T, win]
    else:
        scores_w = torch.empty(B_, nh, T, 0, dtype=compute_dtype, device=device)
        C_windows_full = None

    # --- 4. JOINT softmax over the union of compressed + window entries ---
    # One softmax over the concatenated scores (DeepSeek-V4 §2.3.1 Eq. 27),
    # with the attention sink in the single shared denominator. The sink is a
    # *virtual* entry that appears ONLY in the denominator,
    # ``denom = sum_i exp(s_i) + exp(sink)``. It does NOT contribute a value
    # to the output: the expected KV of a virtual entry with no source tokens
    # is zero, not a constant 1 broadcast across the head dim.
    # ``torch.cat`` with a zero-width side is a pure copy of the other
    # side; the non-empty tensor is used directly instead.
    #
    # Every row below is independent, so the whole pipeline is evaluated per
    # query tile with the identical expressions (bit-identical per row).
    def _joint_p(scores_t, sw_t):
        cat_fresh = False
        if scores_t.shape[-1] and sw_t.shape[-1]:
            cat_t = torch.cat([scores_t, sw_t], dim=-1)
            cat_fresh = True
        elif sw_t.shape[-1]:
            cat_t = sw_t
        else:
            cat_t = scores_t
        if cat_t.shape[-1] == 0:
            return None
        if sink_logits is None:
            all_masked_t = cat_t.isinf().all(-1, keepdim=True)
            return _nan_safe_softmax(
                cat_t, dim=-1, all_masked_mask=all_masked_t)
        log_sink_t = sink_logits.view(1, nh, 1, 1).to(cat_t)
        row_max_t = cat_t.amax(-1, keepdim=True)
        all_masked_t = torch.isneginf(row_max_t)
        row_max_safe_t = torch.where(
            all_masked_t, torch.zeros_like(row_max_t),
            row_max_t.clamp(min=0))
        if torch.is_grad_enabled() or not cat_fresh:
            shifted_t = cat_t - row_max_safe_t
        else:
            # ``cat_t`` is the freshly concatenated joint buffer with no
            # other readers, so the shift runs in place: the same
            # elementwise subtraction with one fewer full-size temporary.
            shifted_t = cat_t.sub_(row_max_safe_t)
        shifted_sink_t = log_sink_t - row_max_safe_t
        # Numerically stable logsumexp approach — keep sink_logits in log
        # space (never exp it) and shift it by -row_max alongside every
        # score so the largest exp() argument is 0. An all-``-inf`` row
        # yields ``-inf`` here (and ``-inf - log_sink`` for ``p``, i.e. 0
        # before the ``masked_fill`` below).
        lse_t = torch.logsumexp(shifted_t, dim=-1, keepdim=True)
        log_denom_t = torch.logaddexp(lse_t, shifted_sink_t)
        if torch.is_grad_enabled():
            # Out-of-place: autograd's Sub/Exp backward save this chain's
            # operands/outputs, so in-place writes here would trip the
            # version-counter check.
            p_t = (shifted_t - log_denom_t).exp()
            p_t = p_t.masked_fill(all_masked_t, 0.0)
        else:
            # ``shifted`` has no readers left (``lse`` is computed above),
            # and its ``- log_denom`` / ``exp`` / fill are the same
            # elementwise ops as the out-of-place chain (bit-identical).
            p_t = shifted_t.sub_(log_denom_t).exp_()
            p_t.masked_fill_(all_masked_t, 0.0)
        return p_t

    n_blk = n_blocks
    if use_tiles:
        C_padded_sw = (F.pad(C_local, (0, 0, win - 1, 0))
                       if sliding_window > 0 else None)
        tile_outs: list[torch.Tensor] = []
        out = (torch.empty(B_, T, nh, c, dtype=compute_dtype, device=device)
               if not torch.is_grad_enabled() else None)
        for lo in range(0, T, tile_T):
            hi = min(T, lo + tile_T)
            scores_t = torch.einsum(
                'b t h d, b n d -> b h t n', q[:, lo:hi], C_comp_n)
            if scale != 1.0:
                # Multiplying by exactly 1.0 is an IEEE no-op; skipping the
                # multiply keeps values/gradients bit-for-bit identical.
                if torch.is_grad_enabled():
                    scores_t = scores_t * scale
                else:
                    scores_t.mul_(scale)
            # The einsum output is freshly allocated and only consumed by
            # the joint softmax below, so the mask is written in place.
            # Same blocked-causal table as ``_causal_block_mask`` (strictly
            # preceding blocks visible), built for this tile only.
            invalid_t = (
                torch.arange(lo, hi, device=device)[:, None] // m2
                <= torch.arange(n_blocks, device=device)[None, :])
            scores_t.masked_fill_(invalid_t[None, None], float('-inf'))
            p_t = _joint_p(scores_t, scores_w[:, :, lo:hi])
            p_dense_t = p_t[..., :n_blk]
            p_w_t = p_t[..., n_blk:]
            # Accumulate the window contribution IN PLACE into the dense
            # einsum output (same values as ``a + b``; see ``naive_csa``
            # for the gradient-safety note).
            out_t = torch.einsum(
                'b h t n, b n d -> b t h d', p_dense_t, C_comp_n)
            if sliding_window > 0:
                p_w_perm = p_w_t.permute(0, 2, 1, 3)
                if C_windows_full is not None:
                    windows_t = C_windows_full[:, lo:hi]
                else:
                    windows_t = _sliding_window_chunk_windows(
                        C_padded_sw, lo, hi, win)
                out_t.add_(torch.einsum(
                    'b t h w, b t w d -> b t h d', p_w_perm, windows_t))
            if out is None:
                tile_outs.append(out_t)
            else:
                out[:, lo:hi] = out_t
        if out is None:
            out = torch.cat(tile_outs, dim=1)
    else:
        scores = torch.einsum('b t h d, b n d -> b h t n', q, C_comp_n)
        if scale != 1.0:
            # Multiplying by exactly 1.0 is an IEEE no-op; skipping the
            # multiply keeps values/gradients bit-for-bit identical. The
            # einsum output is fresh, so the in-place multiply is the same
            # elementwise op with autograd disabled.
            if torch.is_grad_enabled():
                scores = scores * scale
            else:
                scores.mul_(scale)
        # The einsum output (or its scaled copy) is freshly allocated and
        # only consumed by the joint softmax below, so the mask can be
        # written in place.
        scores.masked_fill_(cbm_invalid[None, None], float('-inf'))
        if scores.shape[-1] + scores_w.shape[-1] == 0:
            out = torch.zeros(
                B_, T, nh, c, dtype=compute_dtype, device=device)
        else:
            p = _joint_p(scores, scores_w)
            p_dense = p[..., :n_blk]
            p_w = p[..., n_blk:]
            # Accumulate the window contribution IN PLACE into the dense
            # einsum output (same values as ``a + b``).
            out = torch.einsum(
                'b h t n, b n d -> b t h d', p_dense, C_comp_n)
            if sliding_window > 0:
                # The joint-softmax weights are [B, nh, T, win]; move the
                # query axis back to dim 1 for the value contraction (a pure
                # permute).
                p_w_t = p_w.permute(0, 2, 1, 3)
                if C_windows_full is not None:
                    # Reuse the window slab materialized by the scoring pass
                    # above (same per-row contraction as the tiled rebuild).
                    out.add_(torch.einsum(
                        'b t h w, b t w d -> b t h d', p_w_t,
                        C_windows_full))
                else:
                    out.add_(
                        _sliding_window_output(p_w_t, C_local, win, device))

    # Partial RoPE output countermeasure (§2.3.3): because the compressed
    # KV entries serve as both keys AND values, the naive output carries
    # absolute-position rotations; inverse-rotating at the NEGATED query
    # position makes each entry's contribution depend on the relative
    # (entry − query) distance only.
    if rope_dim_eff > 0:
        out = partial_rope(
            out, -tpos, rope_dim_eff, rope_base, dim=1)

    # Trim the padded SUFFIX before the dtype cast (elementwise cast,
    # identical values on the kept rows; a padded call only casts the real
    # rows).
    if original_T != T:
        out_final = out[:, :original_T].reshape(
            B_, original_T, nh * c).to(H.dtype)
    else:
        out_final = out.reshape(B_, T, nh * c).to(H.dtype)
    if return_projections:
        projections = (
            C[:, :original_T],   # [B, original_T, c]
            Z[:, :original_T],   # [B, original_T, c]
        )
        return out_final, projections
    return out_final
