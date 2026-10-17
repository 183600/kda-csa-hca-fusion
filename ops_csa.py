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

from ops_rope import (
    effective_rope_dim,
    partial_rope,
    token_positions,
    block_positions,
)


# =============================================================================
# Chunked sliding-window attention helper
# -----------------------------------------------------------------------------
# A plain ``C_padded.unfold(1, win, 1)`` implementation materializes a
# ``[B, T, win, c]`` tensor in a single allocation.
#
# This helper processes the window axis in chunks of ``chunk_t`` query
# positions at a time. The default threshold ``CHUNKED_SW_THRESHOLD``
# auto-engages the chunked path when ``T * win * c`` exceeds 8M elements;
# the ``unfold`` path is used otherwise.
#
# Numerical contract: IDENTICAL to the ``unfold``-based path. Softmax over
# the ``win`` non-masked entries of a row is the same operation whether the
# row was extracted as a slice of a ``[T, win]`` unfold view or as one chunk
# of ``[chunk_t, win]`` slices.
# ============================================================================

# Auto-engage the chunked path when the materialized unfold tensor would
# exceed 8M elements.
CHUNKED_SW_THRESHOLD = 8_000_000

# Shape-keyed cache for the static ``[T, win]`` sliding-window validity mask
# (True = real, non-padding position). The mask depends only on
# (T, win, device) and never on the data. Every consumer treats the returned
# tensor as immutable: callers only negate it, ``.expand`` it, or ``cat`` it
# into a larger mask — none of which mutate the cached tensor.
_SLIDING_WINDOW_MASK_CACHE: dict = {}


def _sliding_window_valid_mask(T: int, win: int, device) -> torch.Tensor:
    """Cached ``[T, win]`` boolean mask (True = real position)."""
    key = (T, win, str(device))
    mask = _SLIDING_WINDOW_MASK_CACHE.get(key)
    if mask is None:
        _j = torch.arange(win, device=device)
        _t = torch.arange(T, device=device)
        mask = _j[None, :] >= (win - 1 - _t[:, None])
        if len(_SLIDING_WINDOW_MASK_CACHE) >= 64:
            _SLIDING_WINDOW_MASK_CACHE.clear()
        _SLIDING_WINDOW_MASK_CACHE[key] = mask
    return mask


def _sliding_window_attention(
    q: torch.Tensor,           # [B, T, nh, c]   (L2-normalized)
    C_local: torch.Tensor,     # [B, T, c]       (L2-normalized)
    win: int,
    scale: float,
    device,
    *,
    chunk_t: int | None = None,
) -> torch.Tensor:
    """Chunked sliding-window attention.

    Returns ``[B, T, nh, c]`` — the local-window contribution to the per-head
    output. Adds nothing to the global state; the caller adds it to the
    sparse-attention output.

    Args:
        q: per-head queries, shape ``[B, T, nh, c]``, already L2-normalized.
        C_local: per-position keys, shape ``[B, T, c]``, already L2-normalized.
        win: sliding window size (causal: query ``t`` attends to positions
            ``[t-win+1, t]``).
        scale: scalar multiplied to scores before softmax (cosine-attention
            scale; typically 1.0 for normalized q/k).
        device: torch.device for any intermediates.
        chunk_t: number of query positions to process per chunk. ``None``
            auto-picks ``min(T, max(64, ceil(T * win / (CHUNKED_SW_THRESHOLD /
            win))))``.

    Numerical contract:
        Identical to the equivalent ``unfold``-based implementation, up to
        floating-point reassociation in the per-chunk softmax. Differences
        are well below the fp32 tolerance enforced by ``run_correctness.py``.
    """
    B, T, nh, c = q.shape
    # Static validity mask (shape [T, win]). The mask depends only on
    # (T, win, device) and is cached across calls (read-only contract).
    valid_mask = _sliding_window_valid_mask(T, win, device)  # True=real

    # Auto-pick chunk_t if not given, targeting
    # CHUNKED_SW_THRESHOLD / (win * c) positions per chunk; rounded up to a
    # multiple of 64.
    if chunk_t is None:
        # Target: chunk_t * win * c <= CHUNKED_SW_THRESHOLD
        # => chunk_t <= CHUNKED_SW_THRESHOLD / (win * c)
        target = max(64, CHUNKED_SW_THRESHOLD // max(1, win * c))
        chunk_t = min(T, ((target + 63) // 64) * 64)
    chunk_t = max(1, chunk_t)

    # Left-pad C_local so we can extract the window for query ``t`` as a
    # contiguous slice of length ``win`` starting at padded position ``t``.
    C_padded = F.pad(C_local, (0, 0, win - 1, 0))          # [B, T+win-1, c]

    out = q.new_zeros(B, T, nh, c)
    # Process query positions [t_lo, t_hi) per chunk. For each chunk we
    # extract the per-query windows via unfold on a small slice of C_padded,
    # compute scores/softmax/output, and accumulate into ``out``. The
    # per-chunk intermediate is ``[B, chunk_t, win, c]``.
    for t_lo in range(0, T, chunk_t):
        t_hi = min(T, t_lo + chunk_t)
        ct = t_hi - t_lo
        # Extract windows for this chunk: window for query ``t_lo + i`` starts
        # at padded position ``t_lo + i``, ends at ``t_lo + i + win - 1``.
        # Equivalent to slicing C_padded[:, t_lo : t_hi + win - 1] and then
        # unfolding along dim=1 with size=win, step=1. We use unfold here
        # too — but on the small ``ct + win - 1`` slice, not on the full
        # ``T + win - 1`` tensor.
        C_slice = C_padded[:, t_lo : t_hi + win - 1]       # [B, ct+win-1, c]
        # When ct < win, the slice length (ct+win-1) is less than win,
        # and ``unfold(size=win, step=1)`` returns an empty tensor
        # (dimension 1 becomes 0).  Guard against that by padding the
        # slice on the left so its length is at least win.  The extra
        # positions are all zeros (from the original F.pad) and are
        # masked out by ``valid_mask``, so they contribute nothing to
        # the softmax output — the numerical result is identical to the
        # unfold-on-full-T path for the same query positions.
        if ct < win:
            C_slice = F.pad(C_slice, (0, 0, win - ct, 0))   # [B, win+win-1, c]
        C_windows = C_slice.unfold(1, win, 1).permute(0, 1, 3, 2)  # [B, max(ct,win), win, c]
        # When ct < win the left-padded slice unfolds into ``win`` windows;
        # only the last ``ct`` of them are aligned with the ``ct`` query
        # positions in this chunk (window ``j`` = padded position
        # ``t_lo + j - (win - ct)``, so query ``t_lo + i`` needs window
        # ``j = win - ct + i``).  Discard the leading ``win - ct``
        # all-zero windows so the window axis has length ``ct`` and matches
        # ``q_chunk`` in the einsum below.
        C_windows = C_windows[:, -ct:]                       # [B, ct, win, c]
        q_chunk = q[:, t_lo:t_hi]                            # [B, ct, nh, c]
        mask_chunk = valid_mask[t_lo:t_hi]                   # [ct, win]
        scores = torch.einsum('b t h d, b t w d -> b t h w', q_chunk, C_windows)
        if scale != 1.0:
            # Multiplying by exactly 1.0 is an IEEE no-op; skipping the
            # multiply keeps values/gradients bit-for-bit identical.
            scores = scores * scale
        # ``scores`` is a fresh einsum (or freshly scaled) tensor that no
        # other consumer reads, so the mask can be written in place.
        scores.masked_fill_(~mask_chunk[None, :, None, :], float('-inf'))
        p = torch.softmax(scores, dim=-1)                   # [B, ct, nh, win]
        out[:, t_lo:t_hi] = torch.einsum('b t h w, b t w d -> b t h d', p, C_windows)
    return out


def _sliding_window_scores(
    q: torch.Tensor,           # [B, T, nh, c]   (L2-normalized)
    C_local: torch.Tensor,     # [B, T, c]       (L2-normalized)
    win: int,
    scale: float,
    device,
) -> tuple:
    """Compute per-query sliding-window attention SCORES for the joint softmax.

    Returns ``(scores_w, C_windows, win_valid)`` where
    ``scores_w`` is ``[B, T, nh, win]`` (already ``-inf``-masked for the
    out-of-sequence left-padding), ``C_windows`` is ``[B, T, win, c]`` (the
    per-query window keys), and ``win_valid`` is ``[T, win]`` (boolean, True =
    real position). The caller concatenates ``scores_w`` with the sparse-block
    scores and applies ONE softmax over the union (DeepSeek-V4 §2.3.1 Eq. 27),
    so the window entries share the compressed entries' denominator and sink.
    """
    B, T, nh, c = q.shape
    C_padded = F.pad(C_local, (0, 0, win - 1, 0))          # [B, T+win-1, c]
    C_windows = C_padded.unfold(1, win, 1).permute(0, 1, 3, 2).contiguous()  # [B, T, win, c]
    # Cached [T, win] mask (read-only contract, see the helper).
    win_valid = _sliding_window_valid_mask(T, win, device)  # True=real
    scores_w = torch.einsum('b t h d, b t w d -> b t h w', q, C_windows)
    if scale != 1.0:
        # Multiplying by exactly 1.0 is an IEEE no-op; skipping the
        # multiply keeps values/gradients bit-for-bit identical.
        scores_w = scores_w * scale
    # ``scores_w`` is a fresh einsum (or freshly scaled) tensor that no
    # other consumer reads, so the mask can be written in place.
    scores_w.masked_fill_(~win_valid[None, :, None, :], float('-inf'))
    return scores_w, C_windows, win_valid


# Shape-keyed cache for ``_causal_block_mask``, which depends only on
# (T, n_blocks, m, device) and never on the data. The returned tensor is
# treated as immutable by every caller (callers only negate / broadcast it).
_CAUSAL_BLOCK_MASK_CACHE: dict = {}


def _causal_block_mask(T: int, n_blocks: int, m: int, device) -> torch.Tensor:
    """Return the DeepSeek-V4 block-causal mask ``[T, n_blocks]``.

    Paper-exact contract (arXiv:2606.19348v1, Eq. 16 and §2.3.3): a query
    token ``t`` attends only to STRICTLY PRECEDING compressed blocks,
    i.e. block ``s`` is visible iff ``s < floor(t / m)``. The paper
    motivates this explicitly: "each query attends to only preceding
    compressed KV blocks. Consequently, a query cannot access information
    from other tokens within its own compressed block" — the local
    context inside the query's own block is instead served by the
    sliding-window branch (§2.3.3 "Additional Branch of Sliding Window
    Attention").

    For example, with ``m=8`` the block covering positions ``0..7`` first
    becomes visible to query ``t=8`` (the first token of the NEXT block);
    queries ``t=0..7`` cannot see it through the sparse branch.

    .. note:: The query's own block is never visible through the sparse
        branch, whether it is still partial or already complete; any
        in-block context is served by the sliding-window branch instead.
        This strictly-preceding rule is applied consistently (naive
        operators, decoding caches, and the FLOPs accounting in
        ``run_kv_cache``).
    """
    if not isinstance(T, int) or T < 0:
        raise ValueError(f"T must be a non-negative int, got {T!r}")
    if not isinstance(n_blocks, int) or n_blocks < 0:
        raise ValueError(f"n_blocks must be a non-negative int, got {n_blocks!r}")
    if not isinstance(m, int) or m < 1:
        raise ValueError(f"m must be a positive int, got {m!r}")
    key = (T, n_blocks, m, str(device))
    cached = _CAUSAL_BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    i_t = torch.arange(T, device=device)
    i_b = torch.arange(n_blocks, device=device)
    mask = i_t[:, None] // m > i_b[None, :]
    if len(_CAUSAL_BLOCK_MASK_CACHE) >= 64:
        _CAUSAL_BLOCK_MASK_CACHE.clear()
    _CAUSAL_BLOCK_MASK_CACHE[key] = mask
    return mask


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
    # Validate m BEFORE ``T % m`` so that m=0 is rejected with a clear
    # ValueError rather than a bare ``ZeroDivisionError``.
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the check
    # survives ``python -O`` / ``PYTHONOPTIMIZE=1``; ``assert`` statements
    # are silently stripped under optimization.
    if m < 1:
        raise ValueError(f"compression factor m={m} must be >= 1")
    if T % m != 0:
        raise ValueError(f"T={T} must be divisible by m={m}")
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
    # Validate m BEFORE ``T % m`` (mirrors csa_compress_kv): m=0 is rejected
    # with a clear ValueError rather than a bare ``ZeroDivisionError``.
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the check
    # survives ``python -O`` / ``PYTHONOPTIMIZE=1``; ``assert`` statements
    # are silently stripped under optimization.
    if m < 1:
        raise ValueError(f"compression factor m={m} must be >= 1")
    if T % m != 0:
        raise ValueError(f"T={T} must be divisible by m={m}")
    n_blocks = T // m
    compute_dtype = torch.float64 if Ca.dtype == torch.float64 else torch.float
    # Degenerate case: empty sequence. The downstream
    # ``torch.cat([A_logits, Bb_logits], dim=2)`` crashes with
    # ``RuntimeError: Sizes of tensors must match except in dimension 2``
    # because ``A_logits`` is ``[B, 0, m, c]`` (from the Za + Ba broadcast)
    # while ``Bb_logits`` is ``[B, 1, m, c]`` (the -inf pad always inserts one
    # block). The public ``naive_csa`` guards T=0 with an early return, so this
    # path is unreachable through that API — but ``csa_compress_kv_overlapped``
    # is also a public function (no underscore prefix) imported directly by
    # ``run_correctness.py`` (e.g. ``test_overlap_causality`` and the
    # T=0 contract tests), so a defensive guard here makes the contract
    # match ``csa_compress_kv`` (which already handles T=0 correctly via the
    # view operation returning an empty [B, 0, m, c] tensor).
    if T == 0:
        return Ca.new_zeros(B_, 0, c)
    Ca = Ca.to(compute_dtype).view(B_, n_blocks, m, c)
    Cb = Cb.to(compute_dtype).view(B_, n_blocks, m, c)
    Za = Za.to(compute_dtype).view(B_, n_blocks, m, c)
    Zb = Zb.to(compute_dtype).view(B_, n_blocks, m, c)

    # Two-branch overlapped compression.
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


def _nan_safe_softmax(logits, dim=-1, all_masked_mask=None):
    """Softmax that returns zeros on rows where every entry is ``-inf``.

    Args:
        logits: the pre-softmax tensor.
        dim: the softmax reduction dim.
        all_masked_mask: optional precomputed boolean mask of fully-masked
            rows (shape broadcastable to ``logits`` with ``dim`` reduced).
            If ``None`` we compute it from ``logits.isinf().all(dim=...)``.

    Returns:
        Softmax along ``dim`` with fully-masked rows zeroed.
    """
    if all_masked_mask is None:
        all_masked = logits.isinf().all(dim=dim, keepdim=True)
    else:
        all_masked = all_masked_mask
    if not bool(all_masked.any()):
        # No fully-masked rows: the two ``masked_fill`` passes are no-ops,
        # so the plain softmax is bit-identical.
        return torch.softmax(logits, dim=dim)
    safe_logits = logits.masked_fill(all_masked, 0.0)
    soft = torch.softmax(safe_logits, dim=dim)
    return soft.masked_fill(all_masked, 0.0)


def _qk_normalize(x: torch.Tensor, mode: str = 'l2') -> torch.Tensor:
    """Normalize queries / compressed KV before the core attention.

    DeepSeek-V4 §2.3.3 ("Query and Key-Value Entry Normalization") specifies
    an RMSNorm on each head of the queries and on the single head of the
    compressed KV entries. The default is an L2-normalization (cosine
    attention).

    Modes
    -----
    * ``'l2'``  (default): ``x / ||x||_2`` — unit-norm vectors; callers use
      ``scale=1.0`` (cosine scores in [-1, 1]).
    * ``'rms'`` (paper §2.3.3 form): ``x / sqrt(mean(x^2) + eps)`` — the
      vectors keep norm ~sqrt(D). NOTE: (i) the attention VALUES are the
      normalized entries themselves, so the core-attention output is
      ~sqrt(D) larger than under 'l2'; (ii) the callers auto-select the
      STANDARD attention temperature ``scale = 1/sqrt(D)`` for this mode
      (mirroring how DeepSeek-V2/V3 combine per-head RMSNorm with standard
      scaling), which makes the logits ~sqrt(D)·cosine — sharper than the
      'l2' mode's plain cosine. The paper does not pin the core-attention temperature,
      so both choices are paper-compatible; pass an explicit ``scale=`` to
      pin any other temperature.
    """
    if mode == 'l2':
        return F.normalize(x, dim=-1)
    if mode == 'rms':
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    raise ValueError(
        f"qk_norm_mode={mode!r} must be 'l2' (cosine, historical) or "
        f"'rms' (paper §2.3.3 RMSNorm form)")


def _canonical_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k indices with a deterministic, shape-independent tie-break.

    ``torch.topk`` leaves the order of equal values unspecified, and its
    choice depends on the input's SIZE (verified on CPU: ``torch.zeros(3)``
    and ``torch.zeros(4)`` with one larger entry return different positions
    for the tied runner-up) and on the backend. That matters for CSA: the
    batched-prefill indexer scores ``n_blocks_full`` entries (causally
    invalid blocks masked to ``-inf``) while the per-token decode path
    scores only the already-completed ``n_blocks_inc`` blocks whose number
    grows with the position. With tied ReLU scores (common when the dot
    products are negative and the scores saturate at 0) the two paths
    could select DIFFERENT equal-valued blocks, so the incremental decode
    output stopped being a function of the model's scores alone.

    This selector makes the selection a pure function of the score vector
    by breaking ties on ASCENDING block index. Cost stays O(n), like
    ``torch.topk``: one ``topk`` call supplies the S-th largest score
    (the boundary value), then boolean/comparison masks pick the entries
    strictly above the boundary plus the smallest-index boundary ties
    needed to fill the S slots. No full sort is performed.

    The returned slots reproduce ``torch.topk``'s score-descending order
    (stable, so ties keep the ascending-index rule) and entries selected
    at ``-inf`` (fewer than ``k`` finite scores available) are reported as
    ``-1`` (invalid slot), matching the historical contract. ``k`` may
    exceed the score count; the caller pads the result.
    """
    n = scores.shape[-1]
    s = min(k, n)
    if s == 0:
        return scores.new_zeros(*scores.shape[:-1], 0, dtype=torch.long)
    # S-th largest score; ties exactly at this boundary are resolved below.
    boundary = torch.topk(scores, s, dim=-1).values[..., -1:]
    strictly_above = scores > boundary
    n_above = strictly_above.sum(dim=-1, keepdim=True)
    at_boundary = scores == boundary
    # Fill the remaining slots with the SMALLEST indices tied at the
    # boundary (the canonical rule). Exactly `s` entries are selected per
    # row (n_above + boundary ties >= s by the definition of the boundary).
    # Explicit ``dtype=torch.int32``: a bool cumsum would promote to
    # int64 on a [., T, n_blocks] tensor.
    tie_rank = at_boundary.cumsum(dim=-1, dtype=torch.int32)
    fill_ties = at_boundary & (tie_rank <= (s - n_above))
    selected = strictly_above | fill_ties
    # With exactly `s` selected entries per row, the row-major nonzero
    # coordinates group into [..., s] with ascending indices per row.
    indices = selected.nonzero()[:, -1].view(*scores.shape[:-1], s)
    values = scores.gather(-1, indices)
    # Restore the score-descending slot order (stable: boundary ties keep
    # the canonical ascending-index order).
    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    indices = torch.gather(indices, -1, order)
    values = torch.gather(values, -1, order)
    return indices.masked_fill(torch.isinf(values), -1)


def csa_lightning_indexer(
    q_idx: torch.Tensor,            # [B, T, HI, DI]
    k_idx: torch.Tensor,            # [B, n_blocks, DI]
    w_idx: torch.Tensor | None,     # [B, T, HI]
    topk: int,
    scale: float | None = None,
    causal_block_mask: torch.Tensor | None = None,   # [T, n_blocks]
    return_soft_weights: bool = False,
    *,
    ste_mode: str = 'topk_columns',
    normalize_qk: bool = False,
):
    """Top-k selection over compressed indexer keys (Eq. 13–17).

    Returns indices of shape ``[B, T, topk]`` (padded with -1).

    Selection contract — deterministic, size-independent tie-break
    ---------------------------------------------------------------
    The hard selection is computed by :func:`_canonical_topk_indices`,
    NOT by a bare ``torch.topk`` call: equal scores are ordered by
    ASCENDING block index, so the selected SET is a pure function of the
    score vector. This is what lets the batched-prefill and per-token
    incremental-decode paths agree index-for-index despite their
    differently-sized score tensors (see ``ops_decoding_cache``'s module
    docstring). Selected slots are ordered by descending score (ascending
    index among ties); entries selected at ``-inf`` are reported as ``-1``.

    Parameters
    ----------

    ste_mode : str, default ``'topk_columns'``
        Controls the backward-path behaviour of the straight-through
        estimator (STE) when ``return_soft_weights=True``:

        * ``'topk_columns'`` (default): the differentiable ``soft_weights``
          returned for the STE is gathered ONLY along the hard top-k
          columns. The gradient on non-selected blocks is exactly zero —
          this mirrors the hard selection's forward semantics but means
          the indexer never learns "you should have been selected". This
          is the simplified variant documented as the current behaviour
          in the README.
        * ``'full_softmax'``: the STE gradient additionally flows through
          the FULL softmax distribution over all blocks (including
          non-selected ones). The forward pass is unchanged (still the
          hard top-k gather, preserving sparsity), but the backward pass
          routes gradient through ``soft_weights`` for EVERY block — the
          indexer learns from non-selected blocks too, providing a denser
          training signal. This is implemented in ``naive_csa``'s STE
          block via a zero-in-forward addend whose backward is the full
          softmax's expected KV.
        * ``'aux_contrastive'``: reserved for a future implementation
          of DeepSeek-V4's contrastive auxiliary loss on the
          selection logits. Currently raises ``NotImplementedError``
          if explicitly requested.

        ``csa_lightning_indexer`` itself always returns the same
        ``soft_weights`` tensor (the full softmax over all blocks); the
        ``ste_mode`` parameter is consumed by ``naive_csa``'s STE gather
        block, which builds a different differentiable surrogate for each
        mode. The default ``'topk_columns'`` matches the published CSA
        reference implementation; the other modes are documented for
        extensibility and ablation.

    normalize_qk : bool, default ``False``
        If True, L2-normalize ``q_idx`` and ``k_idx`` along the
        per-head key dimension (``DI``) before computing the ReLU
        dot-product scores. This makes the indexer's top-k selection
        invariant to the magnitude of the indexer queries/keys,
        preventing a high-norm indexer query from dominating the
        top-k ranking purely by magnitude. NOTE: this normalization
        is a REPOSITORY-SIDE choice, NOT a paper requirement — the
        DeepSeek-V4 paper Eq. (15)–(16) define the index score as a
        plain weighted ReLU dot product with no normalization (see
        the matching NOTE at the normalization site in the function
        body, and the *Fairness notes* in the README). Setting to
        ``False`` (the default)
        selects the un-normalized behaviour; **new use cases should pass
        ``normalize_qk=True``** to match the
        magnitude-invariant contract of the CSA/HCA core attention
        (which L2-normalizes both ``q`` and ``C_comp``). The option
        is also exposed for ablation against the magnitude-sensitive
        baseline.

    .. note:: Straight-through estimator (STE) for the indexer.

        The returned ``idx`` tensor is an integer tensor produced by the
        deterministic top-k selector (``_canonical_topk_indices``).
        Integer indices do NOT propagate gradients, so an auxiliary path
        is needed for autograd to flow back from the loss through the
        selection to the indexer parameters
        (``W_IUQ``, ``W_w``, ``W_KV_idx``, ``W_Z_idx``, ``B_idx`` in
        ``CSAHybridLayer``).

        When
        ``return_soft_weights=True`` (the default in ``naive_csa``),
        this function ALSO returns ``soft_weights`` of shape
        ``[B, T, n_blocks]`` — a differentiable probability
        distribution over all compressed blocks (softmax of the indexer
        logits, masked by the causal block mask). The caller
        (``naive_csa``) uses these weights to construct a STE gather:
        the forward pass still uses the HARD top-k indices (so the
        algorithm remains genuinely sparse), but the backward pass
        routes gradients through ``soft_weights`` so the indexer
        parameters receive a training signal.

        This is the standard STE trick: ``forward = hard_topk``,
        ``backward = soft_softmax``. It does NOT change the algorithm's
        forward semantics (CSA is still sparse retrieval), but it makes
        the indexer *learnable* — the selection distribution is pushed
        toward blocks that reduce the task loss, and over training the
        top-k selection concentrates on the most relevant blocks.

        This matches the spirit of DeepSeek-V4's lightning indexer
        training signal (the paper uses an STE / contrastive auxiliary
        loss on the selection logits); the STE here is the simplest
        implementation that closes the gradient-flow gap without adding
        a separate auxiliary loss term.

    .. note:: topk=0 guard.

        ``topk=0`` is a valid degenerate configuration ("no sparse
        selection"). The downstream ``naive_csa`` guards against the
        ``scores.amax(-1)`` crash on an empty dim, and this function
        also guards here so a
        direct caller of ``csa_lightning_indexer`` (it is a PUBLIC
        function imported by ``run_correctness.py`` and
        ``method_analysis.py``) does not hit the underlying selector's
        empty-selection edge. With ``topk=0`` we return an indices tensor
        of shape ``[B, T, 0]`` (no valid slots) and, if requested, a
        ``soft_weights`` tensor of shape ``[B, T, n_blocks]`` (the
        full differentiable distribution — still useful for an
        auxiliary loss even when no hard selection is made).
    """
    # Validate topk BEFORE ``min(topk, n_blocks)``: a negative ``topk``
    # raises a clear ValueError here, whereas ``torch.topk`` with
    # ``S = min(-1, n_blocks) = -1`` raises
    # ``RuntimeError: selected index k out of range`` without naming the
    # offending parameter. The caller ``naive_csa`` already validates
    # topk >= 0, but ``csa_lightning_indexer`` is a PUBLIC function (no
    # underscore prefix) imported directly by ``run_correctness.py`` and
    # ``method_analysis.py``, so it must defend its own contract.
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the check
    # survives ``python -O`` / ``PYTHONOPTIMIZE=1``; ``assert`` statements
    # are silently stripped under optimization.
    if topk < 0:
        raise ValueError(
            f"topk={topk} must be >= 0 (0 selects zero blocks)")
    # Validate the indexer key dimension BEFORE computing
    # ``scale = DI ** -0.5``, which requires ``DI > 0``. ``naive_csa``
    # already validates ``c_I >= 1``, but this is a PUBLIC function
    # imported directly by ``run_correctness.py`` and
    # ``method_analysis.py``, so it must defend its own contract. Use
    # ``raise ValueError`` (NOT ``assert``) so the check survives
    # ``python -O``.
    if q_idx.shape[-1] < 1:
        raise ValueError(
            f"q_idx.shape[-1]={q_idx.shape[-1]} must be >= 1 "
            f"(indexer key dimension c_I must be positive)")
    # ``ste_mode`` is accepted here only for backwards-compatible call-site
    # ergonomics; the actual STE branching happens INSIDE ``naive_csa``. We
    # validate the value so a typo doesn't silently
    # fall through, but callers should be aware the parameter has NO effect
    # in this function.
    if ste_mode not in ('topk_columns', 'full_softmax', 'aux_contrastive'):
        raise ValueError(
            f"ste_mode={ste_mode!r} must be one of 'topk_columns', "
            f"'full_softmax', or 'aux_contrastive'.")
    if ste_mode == 'aux_contrastive':
        raise NotImplementedError(
            "ste_mode='aux_contrastive' is reserved for a future "
            "implementation of DeepSeek-V4's contrastive auxiliary loss "
            "on the indexer selection logits. Use 'topk_columns' (the "
            "default, simplified STE) or 'full_softmax' (dense STE) for now.")
    # Scale-selection policy:
    #   * If the caller explicitly passed a scale, honour it.
    #   * Otherwise, when ``normalize_qk=True``, scores are cosine
    #     similarities in [-1,1] and the canonical temperature is 1.0.
    #     Using 1/sqrt(c_I) (the un-normalized default) over-shrinks ReLU
    #     scores and makes the STE softmax nearly uniform, damping the
    #     straight-through gradient by ~sqrt(c_I) and systematically
    #     slowing indexer learning.
    #   * Otherwise (normalize_qk=False, the default) fall back to the
    #     classical 1/sqrt(c_I) dot-product scale.
    if scale is None:
        if normalize_qk:
            scale = 1.0
        else:
            scale = q_idx.shape[-1] ** -0.5
    B_, T, HI, DI = q_idx.shape
    n_blocks = k_idx.shape[1]
    compute_dtype = torch.float64 if q_idx.dtype == torch.float64 else torch.float
    # ``Tensor.to`` returns the tensor itself when the dtype already
    # matches, so guarding on a dtype comparison is value-identical.
    if q_idx.dtype != compute_dtype:
        q_idx = q_idx.to(compute_dtype)
    if k_idx.dtype != compute_dtype:
        k_idx = k_idx.to(compute_dtype)
    if w_idx is not None and w_idx.dtype != compute_dtype:
        w_idx = w_idx.to(compute_dtype)

    # L2-normalize indexer queries and keys so the top-k ranking
    # is magnitude-invariant (cosine-style scoring). NOTE: this is a
    # REPOSITORY-SIDE choice — the DeepSeek-V4 paper Eq. (15)–(16) define
    # the index score as a plain weighted ReLU dot product with no
    # normalization. Without normalization, a high-norm indexer query
    # dominates the top-k ranking purely by magnitude rather than by
    # semantic relevance, making the sparse selection sensitive to upstream
    # projection scaling. Set ``normalize_qk=False`` to restore the
    # un-normalized (paper-literal) behaviour (e.g. for ablation).
    if normalize_qk:
        q_idx = F.normalize(q_idx, dim=-1)
        k_idx = F.normalize(k_idx, dim=-1)

    # Head-wise similarities [B, HI, T, n_blocks]. Multiplying by a scale of
    # exactly 1.0 is an IEEE no-op (values and gradients are preserved
    # bit-for-bit).
    score = torch.einsum('b t h d, b n d -> b h t n', q_idx, k_idx)
    if not (isinstance(scale, (int, float)) and float(scale) == 1.0):
        score = score * scale
    score = F.relu(score)
    if w_idx is None:
        logits = score.sum(1)                                          # [B, T, n_blocks]
    else:
        logits = torch.einsum('b h t n, b t h -> b t n', score, w_idx)  # [B, T, n_blocks]
    if causal_block_mask is not None:
        # ``logits`` is fresh (einsum/sum output) and both consumers
        # (the selector and the softmax) read only the masked values, so
        # the fill can be written in place when no backward needs the
        # pre-fill tensor (inference / no-grad). Autograd-tracked logits
        # keep the out-of-place form.
        if logits.requires_grad:
            logits = logits.masked_fill(~causal_block_mask, float('-inf'))
        else:
            logits.masked_fill_(~causal_block_mask, float('-inf'))
    # topk=0 guard: return shape-[B, T, 0] indices directly so the
    # function is self-contained (does not rely on ``naive_csa`` to guard
    # the ``scores.amax(-1)`` call). ``torch.topk(logits, 0, ...)`` would
    # itself succeed (returning an empty tensor), but the explicit branch
    # documents the contract for the degenerate ``S = 0`` case.
    if topk == 0:
        idx = q_idx.new_zeros(B_, T, 0, dtype=torch.long)
        if return_soft_weights:
            # Use the shared _nan_safe_softmax helper.
            soft_weights = _nan_safe_softmax(logits, dim=-1)
            return idx, soft_weights
        return idx
    S = min(topk, n_blocks)
    # Deterministic, size-independent top-k (ties by ascending block
    # index) — see ``_canonical_topk_indices``. This is what keeps the
    # batched-prefill and incremental-decode selections identical.
    idx = _canonical_topk_indices(logits, S)
    if topk > S:
        idx = torch.cat([idx, idx.new_full((B_, T, topk - S), -1)], dim=-1)
    # Compute a differentiable soft distribution over ALL
    # blocks for the straight-through estimator. ``logits`` still has
    # grad-fn back to the indexer parameters (W_IUQ, W_w, W_KV_idx,
    # W_Z_idx, B_idx) because it is a function of q_idx / k_idx / w_idx
    # which are themselves functions of those parameters. The softmax
    # here is over the full n_blocks dimension (not just the top-k), so
    # gradients flow to every block's logit — the indexer learns which
    # blocks SHOULD have been selected, even for blocks not in the
    # current hard top-k.
    #
    # We use a numerically stable softmax that handles all-masked rows
    # (early query tokens with no preceding causal block) by replacing
    # their -inf entries with 0 before the exp, then zeroing the result.
    # This mirrors the NaN-safe softmax in ``naive_csa``'s else-branch.
    if return_soft_weights:
        # Use the shared _nan_safe_softmax helper.
        soft_weights = _nan_safe_softmax(logits, dim=-1)               # [B, T, n_blocks]
        return idx, soft_weights
    return idx


def naive_csa(
    H: torch.Tensor,               # [B, T, d]   input hidden states
    W_aKV: torch.Tensor,           # [c, d]   (nn.Linear.weight layout: [out, in])
    W_bKV: torch.Tensor,           # [c, d]
    W_aZ: torch.Tensor,            # [c, d]
    W_bZ: torch.Tensor,            # [c, d]
    Ba: torch.Tensor,              # [m, c]
    Bb: torch.Tensor,              # [m, c]
    W_DQ: torch.Tensor,            # [dc, d]
    W_UQ: torch.Tensor,            # [c*nh, dc]
    W_IUQ: torch.Tensor,           # [c_I*nIh, dc]
    W_w: torch.Tensor,             # [nIh, d]
    W_KV_idx: torch.Tensor,        # [c_I, d]   for indexer key compression
    W_Z_idx: torch.Tensor,         # [c_I, d]
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
    use_ste: bool = True,
    ste_mode: str = 'topk_columns',
    normalize_qk: bool = False,
    fuse_projections: bool = True,
    return_projections: bool = False,
    rope_dim: int | None = None,
    rope_base: float = 10000.0,
    qk_norm_mode: str = 'l2',
):
    """Full CSA forward (compression + indexer + sparse MQA core attention).

    Returns output ``[B, T, nh*c]``. NOTE: the caller
    performs a FULL output projection ``[B, T, nh*c] -> d`` — NOT the
    paper's grouped output projection (§2.3.1, which first splits the
    ``nh`` head outputs into ``g`` groups and projects each group to a
    ``d_g < c*nh/g`` intermediate). At this repository's experiment scales
    ``c*nh`` is small, so the full projection is used and the FLOPs
    accounting in ``run_kv_cache`` counts the full-projection cost
    (``2*T*c*nh*d``) — formula and implementation are consistent.

    When ``return_projections=True``, returns ``(output, projections)`` where
    ``projections`` is a tuple ``(Ca, Cb, Za, Zb, K_idx, Z_idx)`` of the 6
    per-token KV/indexer projections (each ``[B, original_T, c]`` or
    ``[B, original_T, c_I]``, trimmed to the input's original T before any
    right-padding). This lets incremental-decoding callers (e.g.
    ``run_decoding.CSAAttnDecoding``) populate a
    :class:`ops_decoding_cache.CSADecodingCache` directly from this call.

    **Weight layout**: all ``W_*`` tensors follow the
    ``nn.Linear.weight`` convention — shape ``[out_features, in_features]``.
    Internally we use ``F.linear(x, W)`` (which computes ``x @ W.T``).
    Callers pass ``self.W_aKV.weight`` directly.

    ``T`` does NOT need to be divisible by ``m``: the function right-pads the
    sequence with zeros up to the next multiple of ``m`` and trims the output
    back to the original length, mirroring the contract of
    ``naive_chunk_kda``. Real tokens keep their original positions; only the
    last partial block contains padding zeros, and the causal block mask
    ensures no real token attends to it.

    .. warning:: **``fuse_projections`` and the**
        ``use_ste=False`` **gradient contract.**

        By default (``fuse_projections=True``) this function concatenates
        the 6 KV/indexer-compression weights (``W_aKV, W_bKV, W_aZ, W_bZ,
        W_KV_idx, W_Z_idx``) into a single matmul (see the in-code comment
        above ``combined_weight``). The fused call preserves the forward
        values of the 6 separate ``F.linear`` calls, but it has an
        observable side effect on the **backward** pass when
        ``use_ste=False`` (STE disabled, e.g. for the untrained-indexer
        ablation): the indexer weights ``W_KV_idx`` / ``W_Z_idx`` receive a
        **zero-but-non-None** ``.grad`` after ``backward()``, instead of
        ``None``. Without the fusion, these two weights are ONLY consumed
        by the indexer's ``torch.topk`` selection, which does not
        propagate gradient at all when STE is off — so they would get
        ``None`` (never touched by autograd). With the fusion, they share
        one ``F.linear`` call with 4 OTHER weights that DO participate in
        the differentiable compression path, so autograd dutifully
        allocates and back-propagates a (all-zero) gradient into every
        slice of the merged weight, including the two indexer slices.

        The *substantive* contract — "the indexer does not learn without
        STE" — is unaffected either way (a zero gradient is still a
        no-op for the optimizer). But code that inspects
        ``param.grad is None`` to decide whether a parameter "participated
        in this forward pass" (e.g. custom gradient-clipping, gradient
        diagnostics, or ``torch.autograd.grad(..., allow_unused=True)``
        callers that branch on ``None`` vs a real tensor) will observe a
        difference depending on ``fuse_projections``. Pass
        ``fuse_projections=False`` to restore the 6-separate-matmul
        eager-autograd path (identical forward value, ``None`` grad on
        ``W_KV_idx``/``W_Z_idx`` under ``use_ste=False``) if your code
        depends on the ``None``-means-untouched convention. See
        ``test_csa_fuse_projections_grad_contract`` in
        ``run_correctness.py`` for a test pinning both branches.
    """
    B_, T, d = H.shape
    # Validate structural params up front so a caller passing m=0, topk=-1,
    # etc. gets a clear ValueError rather than a ZeroDivisionError or
    # IndexError deep inside the operator.
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the checks
    # survive ``python -O`` / ``PYTHONOPTIMIZE=1``; ``assert`` statements
    # are silently stripped under optimization.
    # ``raise ValueError`` is the standard exception for invalid user input
    # (``run_correctness.py::test_csa_hca_input_validation`` expects it).
    if m < 1:
        raise ValueError(f"compression factor m={m} must be >= 1")
    if topk < 0:
        raise ValueError(
            f"topk={topk} must be >= 0 (0 disables sparse selection)")
    if nh < 1:
        raise ValueError(f"nh={nh} must be >= 1")
    if c < 1:
        raise ValueError(f"c={c} must be >= 1")
    if dc < 1:
        raise ValueError(f"dc={dc} must be >= 1")
    # ``c_I`` (indexer key dim) and ``nIh`` (indexer head count) are
    # validated here, not inside ``csa_lightning_indexer``: the explicit
    # ``scale=c_I ** -0.5`` below requires c_I > 0, and ``nIh == 0`` makes
    # the indexer produce an all-zero ``logits`` tensor (the ``sum(1)``
    # over an empty head dim is 0), so top-k selects the first k blocks for
    # every query. Reject both up-front.
    if c_I < 1:
        raise ValueError(f"c_I={c_I} must be >= 1 (indexer key dim)")
    if nIh < 1:
        raise ValueError(f"nIh={nIh} must be >= 1 (indexer head count)")
    # ``sliding_window`` is gated by ``if sliding_window > 0`` below, so a
    # negative value silently skips the SW branch (looking like the caller
    # intentionally disabled it). A negative window is never a meaningful
    # configuration — reject it so the caller learns about the typo instead
    # of getting a model with no local-attention branch.
    if sliding_window < 0:
        raise ValueError(
            f"sliding_window={sliding_window} must be >= 0 "
            f"(0 disables the branch)")
    # Cosine-attention scale: when both ``q`` and ``C_comp`` are L2-normalized
    # (i.e. when ``normalize_qk=True``, see the normalization block below),
    # their dot product is a cosine similarity in ``[-1, 1]`` and the natural
    # temperature is 1.0 (cosFormer-style). Using 1/sqrt(c) further shrinks
    # the scores and makes softmax nearly uniform — defeating CSA's learned
    # sparse retrieval. When ``normalize_qk=False`` we fall back to the
    # classical 1/sqrt(c) dot-product scale (mirroring the indexer scale
    # policy further below). If a caller explicitly passes ``scale=`` we
    # honour it.
    #
    # NOTE: we do NOT assign ``scale`` here. The core-attention scale
    # auto-selection block below (after q/C_comp normalization) is the
    # canonical place where ``scale`` is resolved for the core attention.
    # Assigning it here would make that block dead code and prevent a
    # caller from passing ``scale=None`` to get the documented per-branch
    # auto-selection (indexer vs core attention). We only validate the
    # explicit-pass case here; the auto-selection is deferred.
    if scale is not None:
        # Caller explicitly pinned a temperature — honour it.
        pass
    if qk_norm_mode not in ('l2', 'rms'):
        raise ValueError(
            f"naive_csa: qk_norm_mode={qk_norm_mode!r} must be 'l2' (cosine, "
            f"historical) or 'rms' (paper §2.3.3 RMSNorm form)")
    # Partial RoPE (DeepSeek-V4 §2.3.3). ``rope_dim=None`` (default)
    # disables RoPE; the model-level wrappers (ops_fused / run_quality /
    # run_decoding) default to the paper's 64-dim partial RoPE (clamped to
    # the head dim for small geometries).
    rope_dim_eff = effective_rope_dim(rope_dim, c)
    if rope_base <= 0:
        raise ValueError(
            f"naive_csa: rope_base={rope_base!r} must be positive")
    device = H.device
    # Degenerate case: empty sequence. With n_blocks=0 the downstream
    # ``csa_compress_kv_overlapped`` cannot broadcast the [B, n_blocks, m, c]
    # reshape against the [m, c] positional bias (``Expected size 0 but got
    # size 1``). Return a zero-shaped output matching the contract.
    if T == 0:
        # Honor ``return_projections`` even in the degenerate T=0 case so
        # callers that unpack ``o, (Ca, Cb, Za, Zb, K_idx, Z_idx) =
        # naive_csa(..., return_projections=True)`` do not crash with
        # ``ValueError: not enough values to unpack``. The projections are
        # empty tensors of the correct feature dims so any downstream
        # cache-population code that iterates over the time axis is a no-op,
        # matching the contract of the T>0 path. Mirrors
        # ``ops_hca.py::naive_hca``'s handling of the same edge case.
        out_empty = torch.zeros(B_, 0, nh * c, dtype=H.dtype, device=device)
        if return_projections:
            Ca_empty = torch.zeros(B_, 0, c, dtype=H.dtype, device=device)
            Cb_empty = torch.zeros(B_, 0, c, dtype=H.dtype, device=device)
            Za_empty = torch.zeros(B_, 0, c, dtype=H.dtype, device=device)
            Zb_empty = torch.zeros(B_, 0, c, dtype=H.dtype, device=device)
            K_idx_empty = torch.zeros(B_, 0, c_I, dtype=H.dtype, device=device)
            Z_idx_empty = torch.zeros(B_, 0, c_I, dtype=H.dtype, device=device)
            return out_empty, (Ca_empty, Cb_empty, Za_empty, Zb_empty,
                               K_idx_empty, Z_idx_empty)
        return out_empty
    # Right-pad T up to a multiple of m so callers don't have to. Real tokens
    # keep their original positions; only the last partial block contains
    # padding zeros, and no real token attends to it (causal block mask).
    original_T = T
    pad = (-T) % m
    if pad:
        H = F.pad(H, (0, 0, 0, pad))
        T = T + pad
    n_blocks = T // m

    # --- 1. Compress KV (two-branch overlapped) ---
    # Use F.linear (computes H @ W.T) with W in nn.Linear.weight
    # layout [out, in].
    #
    # We concatenate all 6 weights along their out_features axis (dim=0) and
    # do ONE ``F.linear`` call, then ``tensor.split`` the result back into
    # the 6 named projections. ``split`` returns views into the concatenated
    # tensor. We do NOT cache the concatenated weight because the optimizer
    # may update the individual weights in-place between forward passes,
    # invalidating any cache.
    #
    # Numerical contract: equivalent to the 6 separate calls. ``F.linear`` is
    # ``x @ W.T``; concatenating the W's along dim=0 (out_features) and then
    # slicing the result along dim=-1 is mathematically identical to doing
    # each matmul independently, up to GEMM kernel reduction order.
    #
    # Autograd contract: under ``use_ste=False`` the indexer weights
    # ``W_KV_idx`` and ``W_Z_idx`` receive a ZERO (but non-None) grad,
    # because autograd flows backward through the single matmul into all
    # 6 weight slices. The substantive contract — "indexer params don't
    # learn without STE" — is preserved (the grad is zero, so the optimizer
    # step is a no-op).
    #
    # All 6 weights must share dtype & device with H (they come from the
    # same ``nn.Linear`` module hierarchy, so this is true by construction).
    # We add a defensive check so a misconfigured caller gets a clear error
    # instead of a cryptic ``Expected all tensors to be on the same device``
    # from inside ``torch.cat``.
    for _w_name, _w in [
        ('W_aKV', W_aKV), ('W_bKV', W_bKV), ('W_aZ', W_aZ), ('W_bZ', W_bZ),
        ('W_KV_idx', W_KV_idx), ('W_Z_idx', W_Z_idx),
    ]:
        if _w.device != H.device:
            raise ValueError(
                f"naive_csa: {_w_name}.device={_w.device} does not match "
                f"H.device={H.device}. All weights must be on the same "
                f"device as the input.")
        if _w.dtype != H.dtype:
            raise ValueError(
                f"naive_csa: {_w_name}.dtype={_w.dtype} does not match "
                f"H.dtype={H.dtype}. All weights must share the input's "
                f"dtype (cast before calling if needed).")
        if _w.shape[1] != d:
            raise ValueError(
                f"naive_csa: {_w_name}.shape={tuple(_w.shape)} — second dim "
                f"(in_features) must equal d={d} (H.shape[-1]).")
    if fuse_projections:
        # Concatenate all 6 weights along the out_features axis (dim=0). The
        # ``split_sizes`` list below MUST match this ordering — if you add or
        # reorder weights here, update both lists in lockstep.
        combined_weight = torch.cat(
            [W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx],
            dim=0,
        )                                                          # [4c + 2*c_I, d]
        combined_out = F.linear(H, combined_weight)                # [B, T, 4c + 2*c_I]
        # ``torch.split`` returns views into ``combined_out``.
        # split_sizes MUST match the order of weights in ``torch.cat`` above.
        Ca, Cb, Za, Zb, K_idx_raw, Z_idx = combined_out.split(
            [c, c, c, c, c_I, c_I], dim=-1,
        )
    else:
        # ``fuse_projections=False`` uses the 6 separate ``F.linear`` calls.
        # The forward VALUE is identical to the fused path (up to GEMM
        # reduction-order noise, well below the fp32 test tolerance), but
        # the BACKWARD path differs when ``use_ste=False``: here,
        # ``W_KV_idx``/``W_Z_idx`` are consumed ONLY by the (non-
        # differentiable-under-no-STE) indexer path, so they receive
        # ``None`` gradient instead of the fused path's zero-but-non-None
        # gradient. See the ``fuse_projections`` docstring warning above
        # for the full rationale and
        # ``test_csa_fuse_projections_grad_contract`` in run_correctness.py
        # for a test pinning this behaviour.
        Ca = F.linear(H, W_aKV)
        Cb = F.linear(H, W_bKV)
        Za = F.linear(H, W_aZ)
        Zb = F.linear(H, W_bZ)
        K_idx_raw = F.linear(H, W_KV_idx)
        Z_idx = F.linear(H, W_Z_idx)
    C_comp = csa_compress_kv_overlapped(Ca, Cb, Za, Zb, Ba, Bb, m)   # [B, n_blocks, c]
    # Partial RoPE on the compressed KV entries (DeepSeek-V4 §2.3.3): the
    # rotated entry serves as BOTH the attention key and the attention
    # value (shared-KV MQA). The anchor position of compressed entry ``s``
    # is ``s * m`` (block start; for the overlapped two-branch compression
    # this is also the a/b source-window boundary). Rotation happens
    # BEFORE normalization — ``ops_decoding_cache`` applies the same order
    # (rotate at append time, normalize at read time) so the incremental
    # decode path stays bit-identical to this full-sequence path.
    if rope_dim_eff > 0:
        C_comp = partial_rope(
            C_comp, block_positions(n_blocks, m, device),
            rope_dim_eff, rope_base, dim=1)

    # --- 2. Lightning indexer ---
    # compressed indexer keys via the same compression operation, applied
    # SINGLE-BRANCH (no a/b overlap). NOTE: the paper says CSA "performs
    # the same compression operation used for CComp" — i.e. the overlapped
    # two-branch compression — for the indexer keys as well; this reference
    # implementation deliberately keeps a single branch (one W_KV_idx /
    # W_Z_idx / B_idx instead of a/b pairs) as a documented simplification:
    # the indexer only produces a top-k RANKING, which is robust to the
    # compression variant. The RoPE is likewise NOT applied to the indexer
    # queries/keys — §2.3.3 scopes RoPE to the core-attention queries /
    # KV entries / outputs, not to the indexer scores (Eq. 15–16 have no
    # positional term).
    # K_idx_raw and Z_idx were computed in the merged matmul above.
    K_IComp = csa_compress_kv(K_idx_raw, Z_idx, B_idx, m)            # [B, n_blocks, c_I]
    # indexer queries (low-rank)
    cQ = F.linear(H, W_DQ)                                        # [B, T, dc]
    q_idx = F.linear(cQ, W_IUQ).view(B_, T, nIh, c_I)              # [B, T, nIh, c_I]
    w_idx = F.linear(H, W_w)                                      # [B, T, nIh]
    cbm = _causal_block_mask(T, n_blocks, m, device)
    # STE is a TRAINING-only surrogate for the non-differentiable top-k
    # indexer. It is forward-equivalent to the hard gather. Under
    # ``torch.no_grad()`` there is no backward path to preserve, so the STE
    # surrogate is gated on ``torch.is_grad_enabled()``: training keeps the
    # default behaviour, while no-grad inference uses the hard-index path.
    effective_use_ste = use_ste and torch.is_grad_enabled()
    # The lightning indexer scores are dot products over DI = c_I (not c).
    # Scale policy: csa_lightning_indexer AUTO-SELECTS the scale —
    #   * normalize_qk=True  -> scale=1.0 (cosine similarity)
    #   * normalize_qk=False -> scale=c_I**-0.5 (classical dot-product attention)
    # Callers pass scale=None to defer to that policy.
    indices = csa_lightning_indexer(q_idx, K_IComp, w_idx, topk,
                                    scale=None,
                                    causal_block_mask=cbm,
                                    return_soft_weights=effective_use_ste,
                                    ste_mode=ste_mode,
                                    normalize_qk=normalize_qk)     # [B, T, topk]
    # When STE is active, the indexer also returns a differentiable
    # ``soft_weights`` tensor of shape ``[B, T, n_blocks]`` for the
    # straight-through estimator. We keep it in a variable that is ``None``
    # when STE is disabled (or no-grad inference is active) so the gather code
    # below can branch cleanly.
    if effective_use_ste:
        indices, soft_weights = indices  # unpack the (idx, soft_weights) tuple
    else:
        soft_weights = None

    # --- 3. Shared-KV MQA core attention ---
    # attention queries (low-rank up-projection)
    #
    # Dtype consistency: ``C_comp`` is returned by ``csa_compress_kv_overlapped``
    # in ``compute_dtype`` (fp32 for fp16 inputs, fp64 for fp64 inputs). The
    # gathered ``kv`` and all downstream softmax/scores must therefore also be
    # in ``compute_dtype``. If we leave ``q`` in ``H.dtype`` (e.g. fp16),
    # ``torch.einsum`` does NOT auto-promote mixed dtypes and raises
    # ``RuntimeError: Expected object of scalar type Half but got scalar type
    # Float``. We cast ``q`` to ``compute_dtype`` before normalization so the
    # entire attention core runs in one consistent precision.
    compute_dtype = torch.float64 if H.dtype == torch.float64 else torch.float
    q = F.linear(cQ, W_UQ).view(B_, T, nh, c).to(compute_dtype)    # [B, T, nh, c]
    # Honor ``normalize_qk`` for the CORE attention as
    # well as the indexer. When ``normalize_qk=True`` (the default used by
    # every experiment runner — see run_quality, run_benchmark,
    # run_decoding, ops_fused), q and C_comp are L2-normalized and
    # ``scale`` defaults to 1.0 (cosine attention). When
    # ``normalize_qk=False``, we skip both normalizations and fall back to
    # the classical ``c ** -0.5`` dot-product scale (just like the indexer
    # branch does for its DI).
    # This makes the flag's semantics consistent across both branches and
    # gives ablations a genuine knob to turn off normalization.
    if normalize_qk:
        q = _qk_normalize(q, qk_norm_mode)
    C_comp_n = (_qk_normalize(C_comp, qk_norm_mode) if normalize_qk
                else C_comp.to(compute_dtype))
    # Partial RoPE on the core-attention QUERIES (§2.3.3): rotate each
    # query at its own token position AFTER normalization (the decoding
    # cache applies the same order: the caller normalizes ``q``, the cache
    # rotates it). RoPE is orthogonal, so normalize-then-rotate and
    # rotate-then-normalize agree up to fp rounding; the order is pinned
    # here only so the incremental decode path matches bit-exactly.
    # One shared position vector for every RoPE call in this forward pass
    # (queries, sliding-window keys, output inverse-rotation).
    tpos = token_positions(T, device) if rope_dim_eff > 0 else None
    if rope_dim_eff > 0:
        q = partial_rope(q, tpos, rope_dim_eff, rope_base, dim=1)
    # Core-attention scale auto-selection (mirrors the indexer scale
    # policy above). A caller can still override by passing an explicit
    # ``scale=``.
    # NOTE: this block runs BEFORE the topk=0 branch so ``scale`` is
    # resolved for the sliding-window branch (which runs AFTER the if/else
    # and uses ``scale`` unconditionally).
    if scale is None:
        if not normalize_qk:
            scale = c ** -0.5
        elif qk_norm_mode == 'rms':
            # Paper §2.3.3 RMSNorm form: the normalized vectors keep norm
            # ~sqrt(c), so the raw dot product is ~c * cosine. Use the
            # STANDARD attention temperature 1/sqrt(c) (mirroring how
            # DeepSeek-V2/V3 combine per-head RMSNorm with standard
            # scaling), giving logits ~sqrt(c) * cosine. Pass an explicit
            # ``scale=`` (e.g. 1.0) for any other temperature.
            scale = c ** -0.5
        else:
            scale = 1.0


    # Handle topk=0 (degenerate but valid: caller asks for no sparse
    # selection). ``scores`` has shape ``[B, T, nh, 0]``, so the sparse
    # branch contributes exactly zero; only the SW branch (if enabled)
    # produces non-zero output. The early return is placed AFTER ``q`` is
    # computed because the SW branch needs ``q``.
    if indices.shape[-1] == 0:
        scores = torch.empty(B_, T, nh, 0, dtype=compute_dtype, device=device)
        valid_mask = torch.empty(B_, T, 0, dtype=torch.bool, device=device)
        kv = torch.empty(B_, T, 0, c, dtype=compute_dtype, device=device)
    else:
        # --- Sparse MQA core attention ---
        # Gather the selected compressed KV entries for all (b, t).
        # indices: [B, T, topk], padded with -1 for invalid slots.
        valid_mask = indices >= 0                                        # [B, T, topk]
        idx_safe = indices.clamp(min=0)                                  # [B, T, topk]
        batch_idx = torch.arange(B_, device=device).view(B_, 1, 1)      # [B, 1, 1]
        kv = C_comp_n[batch_idx, idx_safe]                               # [B, T, topk, c]

        # Straight-through estimator (STE) for the indexer.
        # ``kv`` above is a hard gather: its value is correct (the
        # top-k compressed KV entries), but it has NO gradient path back
        # to the indexer parameters because ``indices`` is an integer
        # tensor from ``torch.topk``. We construct a differentiable
        # ``soft_kv`` of the SAME shape ``[B, T, topk, c]`` by gathering
        # the top-k columns of ``soft_weights`` (which IS differentiable)
        # and multiplying by ``C_comp_n``. Then we apply the STE identity:
        #
        #     kv_ste = soft_kv + (kv - soft_kv).detach()
        #
        # Forward value:  soft_kv + kv - soft_kv = kv           (hard gather)
        # Backward grad:  d(soft_kv) = the differentiable path  (soft gather)
        #
        # This makes the forward pass identical to the original hard
        # top-k gather (so the algorithm remains genuinely sparse), while
        # the backward pass routes gradients through ``soft_weights`` ->
        # ``logits`` -> ``q_idx/k_idx/w_idx`` -> indexer parameters
        # (``W_IUQ``, ``W_w``, ``W_KV_idx``, ``W_Z_idx``, ``B_idx``).
        # After ``backward()``, those parameters now have non-None
        # ``.grad`` and are updated by the optimizer.
        #
        # ``ste_mode`` is implemented as two genuinely distinct
        # branches instead of silently aliasing ``'full_softmax'`` to
        # ``'topk_columns'``. Both modes share the same STE identity
        #
        #     kv_ste = soft_kv + (kv - soft_kv).detach()
        #
        # which makes the forward pass identical to the hard top-k gather
        # (preserving sparsity) but routes the backward pass through
        # ``soft_kv`` (and therefore through ``soft_weights`` -> ``logits``
        # -> indexer parameters). The two modes differ in WHAT ``soft_kv``
        # depends on:
        #
        # * ``topk_columns``: ``soft_kv`` depends only on the top-k columns
        #   of ``soft_weights`` (via ``torch.gather``). The gradient on
        #   non-selected blocks is exactly zero — the indexer only learns
        #   "the selected blocks were right/wrong", never "you should have
        #   selected block j".
        #
        # * ``full_softmax``: ``soft_kv`` ADDITIONALLY depends on the full
        #   softmax distribution via a zero-in-forward addend:
        #
        #       aux = soft_full_expanded - soft_full_expanded.detach()
        #
        #   where ``soft_full = sum_b soft_weights[b,t,b] * C_comp_n[b,b,:]``
        #   is the full-softmax expected KV (shape [B,T,c], broadcast across
        #   the topk axis). Forward value of ``aux`` is exactly zero (so
        #   the forward pass is unchanged), but its backward is
        #   ``d(soft_full)`` which flows gradient to EVERY block's
        #   ``soft_weights`` entry. This gives the indexer a denser training
        #   signal: non-selected blocks now receive a (typically small)
        #   gradient proportional to their softmax probability.
        #
        # The two modes are forward-equivalent (same sparse top-k gather)
        # but backward-distinct (different gradient distributions), exactly
        # as documented in the ``ste_mode`` docstring above. A test in
        # ``run_correctness.py::test_csa_indexer_ste_full_softmax`` verifies
        # both the forward-equivalence and the backward-distinctness
        # (non-selected blocks receive non-zero gradient only under
        # ``full_softmax``).
        if effective_use_ste and soft_weights is not None:
            # Common path: gather top-k columns of soft_weights. This is
            # the differentiable surrogate for both modes (forward-equivalent
            # to the hard gather). ``full_softmax`` adds an additional
            # term below.
            soft_weights_selected = torch.gather(
                soft_weights, dim=-1, index=idx_safe)                # [B, T, topk]
            # Mask out invalid (-1) slots so they contribute zero
            # gradient (mirrors the hard path's valid_mask).
            soft_weights_selected = soft_weights_selected * \
                valid_mask.to(soft_weights_selected.dtype)
            # Build the soft gather: each selected block's contribution
            # weighted by its soft probability. This is differentiable
            # w.r.t. soft_weights (and therefore the indexer params).
            # soft_kv[b, t, k, :] = soft_weights_selected[b, t, k] * kv[b, t, k, :]
            soft_kv = soft_weights_selected.unsqueeze(-1) * kv.detach()  # [B, T, topk, c]

            if ste_mode == 'full_softmax':
                # Add a zero-in-forward term whose backward is the full
                # softmax's expected KV, so gradient flows to ALL blocks
                # (not just the top-k). This is the dense training signal
                # that distinguishes ``full_softmax`` from ``topk_columns``.
                #
                # ``C_comp_n`` is [B, n_blocks, c]; ``soft_weights`` is
                # [B, T, n_blocks]. Their contraction gives the expected
                # KV per (b, t): [B, T, c]. We then broadcast across the
                # topk axis so the shape matches ``soft_kv`` for the STE
                # identity below.
                soft_full = torch.einsum(
                    'btn,bnc->btc', soft_weights, C_comp_n)          # [B, T, c]
                # Expand to [B, T, topk, c] and apply the STE zero-in-forward
                # trick: forward = soft_full - soft_full.detach() = 0,
                # backward = d(soft_full) (flows to all blocks).
                topk_size = kv.shape[2]
                aux = (soft_full.unsqueeze(2).expand(-1, -1, topk_size, -1)
                       - soft_full.unsqueeze(2).expand(-1, -1, topk_size, -1).detach())
                # ``aux`` has zero forward value but its backward adds
                # ``d(soft_full)`` to ``d(soft_kv)``, so the total
                # differentiable path now depends on ALL blocks' soft_weights.
                soft_kv = soft_kv + aux

            # STE: forward = kv (hard), backward = soft_kv (differentiable).
            #
            # The form ``kv = soft_kv + (kv - soft_kv).detach()`` is
            # forward-equivalent to ``kv`` (hard gather) but in backward its
            # ``detach()`` ERASES the direct gradient from the hard gather
            # path, leaving only ``soft_weights * d_out`` as the gradient to
            # the selected ``C_comp_n`` entries — i.e. the selected KV would
            # receive a gradient shrunk by the soft probability (often < 1).
            # The identity used here is instead
            #
            #     kv_ste = hard_kv + (soft_surrogate - soft_surrogate.detach())
            #
            # Forward:  hard_kv + 0 = hard_kv                (hard gather value)
            # Backward: d(hard_kv) + d(soft_surrogate)       (full hard grad
            #           PLUS the differentiable soft path)
            #
            # so the hard gather's native gradient (which ``torch.gather``
            # already routes to the selected source positions of ``C_comp_n``)
            # is preserved, AND ``soft_weights`` receives a non-zero gradient
            # that flows back to the indexer parameters. The extra
            # ``soft_weights * d_out`` added to the hard path is the standard
            # STE bias and is benign for training stability. This matches the
            # contract documented in the ``ste_mode`` docstring: selected KV
            # entries receive the SAME gradient they receive under
            # ``use_ste=False``, plus an indexer-learning signal.
            kv = kv + (soft_kv - soft_kv.detach())

        # Per-head attention scores over the topk selected blocks.
        scores = torch.einsum('b t h d, b t k d -> b t h k', q, kv)  # [B, T, nh, topk]
        if scale != 1.0:
            # Multiplying by exactly 1.0 is an IEEE no-op; skipping the
            # multiply keeps values/gradients bit-for-bit identical.
            scores = scores * scale
        # Mask -1 padding so those slots get zero weight after softmax.
        # The einsum output is freshly allocated and only consumed by the
        # joint softmax below, so the mask can be written in place.
        scores.masked_fill_(~valid_mask[:, :, None, :], float('-inf'))

    # Sliding-window branch: gather per-query window SCORES (uncompressed
    # local KV) so they can share the compressed entries' softmax below.
    if sliding_window > 0:
        win = sliding_window
        H_proj = Ca  # [B, T, c], already == F.linear(H, W_aKV)
        if normalize_qk:
            C_local = _qk_normalize(H_proj.to(compute_dtype), qk_norm_mode)  # [B, T, c]
        else:
            C_local = H_proj.to(compute_dtype)
        # Partial RoPE on the sliding-window keys (§2.3.3: RoPE applies to
        # the KV entry vectors used in CSA/HCA — the SW entries are such
        # KV entries). Each key rotates at its own token position; the
        # decoding cache's ring buffer applies the same
        # normalize-then-rotate order at append time.
        if rope_dim_eff > 0:
            C_local = partial_rope(
                C_local, tpos, rope_dim_eff, rope_base, dim=1)
        scores_w, C_windows, win_valid = _sliding_window_scores(
            q, C_local, win, scale, device)
    else:
        scores_w = torch.empty(B_, T, nh, 0, dtype=compute_dtype, device=device)
        C_windows = torch.empty(B_, T, 0, c, dtype=compute_dtype, device=device)
        win_valid = torch.empty(T, 0, dtype=torch.bool, device=device)

    # JOINT softmax over the union of the top-k selected compressed
    # blocks and the sliding-window tokens (DeepSeek-V4 §2.3.1 Eq. 27):
    # one softmax over the concatenated scores, with the attention sink
    # in the single shared denominator.
    # ``torch.cat`` with a zero-width side is a pure copy of the other side;
    # the non-empty tensor is used directly instead.
    if scores.shape[-1] and scores_w.shape[-1]:
        cat = torch.cat([scores, scores_w], dim=-1)             # [B, T, nh, topk+win]
    elif scores_w.shape[-1]:
        cat = scores_w
    else:
        cat = scores
    # Per-row "no valid entry in either branch" flag: the OR of the
    # per-branch row ``any``s.
    all_invalid = ~(valid_mask.any(-1, keepdim=True).unsqueeze(2)
                    | win_valid.any(-1).view(1, T, 1, 1))
    if cat.shape[-1] == 0:
        out = torch.zeros(B_, T, nh, c, dtype=compute_dtype, device=device)
    else:
        if sink_logits is not None:
            # Numerically stable logsumexp approach — keep sink_logits in log
            # space (never exp it) and shift it by -row_max alongside every
            # score so the largest exp() argument is 0. Invalid slots hold
            # ``-inf`` from the masked fills above, so their ``exp`` is
            # exactly ``+0.0``.
            log_sink = sink_logits.view(1, 1, nh, 1).to(cat)      # [1, 1, nh, 1]
            row_max = cat.amax(-1, keepdim=True).clamp(min=0)     # [B, T, nh, 1]
            shifted = cat - row_max
            shifted_sink = log_sink - row_max
            log_sum_exp = torch.logsumexp(shifted, dim=-1, keepdim=True)
            log_denom = torch.logaddexp(log_sum_exp, shifted_sink)
            p = (shifted - log_denom).exp()
            p = p.masked_fill(all_invalid, 0.0)
        else:
            p = _nan_safe_softmax(cat, dim=-1, all_masked_mask=all_invalid)

        topk_size = scores.shape[-1]
        p_b = p[..., :topk_size]
        p_w = p[..., topk_size:]
        out = (torch.einsum('b t h k, b t k d -> b t h d', p_b, kv)
               + torch.einsum('b t h w, b t w d -> b t h d', p_w, C_windows))

    # Partial RoPE output countermeasure (§2.3.3): because the compressed
    # KV entries serve as both keys AND values, the naive output carries
    # absolute-position rotations; inverse-rotating at the NEGATED query
    # position makes each entry's contribution depend on the relative
    # (entry − query) distance only.
    if rope_dim_eff > 0:
        out = partial_rope(
            out, -tpos, rope_dim_eff, rope_base, dim=1)

    # Return the raw per-head core-attention output [B, T, nh, c] flattened to
    # [B, T, nh*c]; the caller performs the grouped output projection.
    # Cast to H.dtype at the very end so all intermediate computation
    # benefits from compute_dtype precision.
    # Trim the padded SUFFIX off the SEQUENCE axis (dim=1) so the output
    # matches the input's original T (right-padding added zeros at the end,
    # which never affect real-token outputs thanks to the causal block mask).
    out_final = out.reshape(B_, T, nh * c).to(H.dtype)[:, :original_T]
    if return_projections:
        # Return the 6 per-token KV/indexer projections (trimmed to
        # original_T) so incremental-decoding callers can populate a
        # CSADecodingCache without recomputing them. The projections are
        # in H.dtype (F.linear preserves dtype) and on the input's device.
        # ``Ca`` is also used internally for the SW branch (H_proj = Ca),
        # so we return it directly. ``K_idx_raw`` is the uncompressed
        # indexer key projection (before csa_compress_kv compresses it to
        # K_IComp) — that's what CSADecodingCache.append_step expects.
        projections = (
            Ca[:, :original_T],       # [B, original_T, c]
            Cb[:, :original_T],       # [B, original_T, c]
            Za[:, :original_T],       # [B, original_T, c]
            Zb[:, :original_T],       # [B, original_T, c]
            K_idx_raw[:, :original_T],# [B, original_T, c_I]
            Z_idx[:, :original_T],    # [B, original_T, c_I]
        )
        return out_final, projections
    return out_final
