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

import weakref

import torch
import torch.nn.functional as F

from ops_rope import (
    effective_rope_dim,
    partial_rope,
    token_positions,
    block_positions,
)


# ---------------------------------------------------------------------------
# bfloat16 numerical stability helpers (mirrors ops_kda_autograd.py)
# ---------------------------------------------------------------------------
def _bf16_safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Safe softmax for bfloat16: clamp logits before softmax to prevent Inf.

    bfloat16's 8-bit exponent can overflow on large positive logits. Clamping
    to a safe range (e.g., [-50, 50]) before softmax prevents this while
    preserving the relative probabilities for normal values.
    """
    if logits.dtype == torch.bfloat16:
        # Clamp to prevent exp overflow (exp(50) ~ 5e21, well below bf16 max ~3e38)
        # but large enough to not affect normal probability distributions.
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
    """Scale tensor down if its norm exceeds max_norm (bf16 only).

    Prevents gradual accumulation of large values over long sequences.
    """
    if x.dtype != torch.bfloat16:
        return x
    norms = x.norm(p=2, dim=-1, keepdim=True)
    scale = (max_norm / (norms + 1e-8)).clamp(max=1.0)
    return x * scale


def _bf16_safe_einsum_sum(
    x: torch.Tensor, dim: int, max_val: float = 1e4
) -> torch.Tensor:
    """Safe sum reduction for bfloat16: clamp before summing."""
    if x.dtype != torch.bfloat16:
        return x.sum(dim=dim)
    return x.clamp(min=-max_val, max=max_val).sum(dim=dim)


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


def _windows_contiguous(windows: torch.Tensor) -> torch.Tensor:
    """Densify a possibly sliding-strided window view without changing values.

    The ``unfold(...).permute(...)`` window view has overlapping strides
    (consecutive window rows start only ``c`` elements apart), so an
    explicit ``copy_`` into a fresh contiguous buffer materializes the
    dense layout that ``einsum`` consumes. Returning the input unchanged
    when it is already contiguous keeps the function allocation-free for
    the plain ``win == 1``/full-slice cases; consumers treat the result as
    read-only.
    """
    if windows.is_contiguous():
        return windows
    out = windows.new_empty(windows.shape)
    out.copy_(windows)
    return out


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
        C_windows = _windows_contiguous(C_windows[:, -ct:])   # [B, ct, win, c]
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
        p = _bf16_safe_softmax(scores, dim=-1)                   # [B, ct, nh, win]
        out[:, t_lo:t_hi] = torch.einsum('b t h w, b t w d -> b t h d', p, C_windows)
    return out


def _sw_query_chunk(T: int, win: int, c: int, chunk_t: int | None = None) -> int:
    """Query-axis tile size for the chunked sliding-window helpers.

    ``None`` targets ``CHUNKED_SW_THRESHOLD`` elements per ``[ct, win, c]``
    window slab (rounded up to a multiple of 64), mirroring
    :func:`_sliding_window_attention`. The tiling is row-wise: it changes
    only the transient buffer size, never any per-row value.
    """
    if chunk_t is None:
        target = max(64, CHUNKED_SW_THRESHOLD // max(1, win * c))
        chunk_t = min(T, ((target + 63) // 64) * 64)
    return max(1, min(chunk_t, T)) if T > 0 else 0


def _sliding_window_chunk_windows(
    C_padded: torch.Tensor,
    t_lo: int,
    t_hi: int,
    win: int,
) -> torch.Tensor:
    """Windows for queries ``[t_lo, t_hi)`` as a ``[B, ct, win, c]`` view.

    Row ``i`` is exactly ``C_padded[:, t_lo + i : t_lo + i + win]`` — the
    same slice the full-sequence ``unfold`` produces — so the per-row
    values are unchanged. The short-chunk branch mirrors
    :func:`_sliding_window_attention` (left-pad, unfold, drop the leading
    all-zero windows).
    """
    ct = t_hi - t_lo
    C_slice = C_padded[:, t_lo : t_hi + win - 1]           # [B, ct+win-1, c]
    if ct < win:
        C_slice = F.pad(C_slice, (0, 0, win - ct, 0))
    windows = C_slice.unfold(1, win, 1).permute(0, 1, 3, 2)
    return _windows_contiguous(windows[:, -ct:])


def _sliding_window_scores(
    q: torch.Tensor,           # [B, T, nh, c]   (L2-normalized)
    C_local: torch.Tensor,     # [B, T, c]       (L2-normalized)
    win: int,
    scale: float,
    device,
    *,
    return_windows: bool = True,
    chunk_t: int | None = None,
) -> tuple:
    """Compute per-query sliding-window attention SCORES for the joint softmax.

    Returns ``(scores_w, C_windows, win_valid)`` where
    ``scores_w`` is ``[B, T, nh, win]`` (already ``-inf``-masked for the
    out-of-sequence left-padding), ``C_windows`` is ``[B, T, win, c]`` (the
    per-query window keys), and ``win_valid`` is ``[T, win]`` (boolean, True =
    real position). The caller concatenates ``scores_w`` with the sparse-block
    scores and applies ONE softmax over the union (DeepSeek-V4 §2.3.1 Eq. 27),
    so the window entries share the compressed entries' denominator and sink.

    ``return_windows=False`` computes the SAME ``scores_w`` with a bounded
    query-axis tiling and returns ``None`` instead of materializing the
    ``[B, T, win, c]`` window tensor; callers that need the window output
    then use :func:`_sliding_window_output`, which re-derives each query
    tile's windows from ``C_local``. Every entry is evaluated by the same
    per-row dot product either way, so the values are unchanged; only the
    peak intermediate size is bounded.
    """
    B, T, nh, c = q.shape
    C_padded = F.pad(C_local, (0, 0, win - 1, 0))          # [B, T+win-1, c]
    # Cached [T, win] mask (read-only contract, see the helper).
    win_valid = _sliding_window_valid_mask(T, win, device)  # True=real
    if return_windows:
        C_windows = _windows_contiguous(
            C_padded.unfold(1, win, 1).permute(0, 1, 3, 2))  # [B, T, win, c]
        scores_w = torch.einsum('b t h d, b t w d -> b t h w', q, C_windows)
        if scale != 1.0:
            # Multiplying by exactly 1.0 is an IEEE no-op; skipping the
            # multiply keeps values/gradients bit-for-bit identical. The
            # einsum output is fresh, so inference (no autograd) may scale in
            # place; a recorded graph keeps the out-of-place multiply.
            if torch.is_grad_enabled():
                scores_w = scores_w * scale
            else:
                scores_w.mul_(scale)
        # ``scores_w`` is a fresh einsum (or freshly scaled) tensor that no
        # other consumer reads, so the mask can be written in place.
        scores_w.masked_fill_(~win_valid[None, :, None, :], float('-inf'))
        return scores_w, C_windows, win_valid
    scores_w = q.new_empty(B, T, nh, win)
    tile = _sw_query_chunk(T, win, c, chunk_t)
    for t_lo in range(0, T, tile):
        t_hi = min(T, t_lo + tile)
        windows = _sliding_window_chunk_windows(C_padded, t_lo, t_hi, win)
        chunk = torch.einsum(
            'b t h d, b t w d -> b t h w', q[:, t_lo:t_hi], windows)
        if scale != 1.0:
            chunk = chunk * scale
        scores_w[:, t_lo:t_hi] = chunk
    scores_w.masked_fill_(~win_valid[None, :, None, :], float('-inf'))
    return scores_w, None, win_valid


def _sliding_window_output(
    p_w: torch.Tensor,         # [B, T, nh, win] joint-softmax window weights
    C_local: torch.Tensor,     # [B, T, c]       (L2-normalized)
    win: int,
    device,
    *,
    chunk_t: int | None = None,
) -> torch.Tensor:
    """Window value output ``sum_w p_w[t, h, w] * C_local[t+w-win+1]``.

    Eager equivalent of the ``einsum`` against the full
    ``[B, T, win, c]`` window tensor, evaluated in bounded query tiles.
    Each output row is the same per-row contraction, so the values are
    unchanged; only the transient window slab is bounded.
    """
    B, T, nh, _ = p_w.shape
    c = C_local.shape[-1]
    if T == 0:
        return p_w.new_zeros(B, T, nh, c)
    C_padded = F.pad(C_local, (0, 0, win - 1, 0))
    tile = _sw_query_chunk(T, win, c, chunk_t)
    out = p_w.new_empty(B, T, nh, c)
    for t_lo in range(0, T, tile):
        t_hi = min(T, t_lo + tile)
        windows = _sliding_window_chunk_windows(C_padded, t_lo, t_hi, win)
        out[:, t_lo:t_hi] = torch.einsum(
            'b t h w, b t w d -> b t h d', p_w[:, t_lo:t_hi], windows)
    return out


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


# Cached negation of ``_causal_block_mask`` (True = INVALID/blocked). The
# negation is a pure function of the mask and is only ever read by
# ``masked_fill_`` / ``where`` consumers, so sharing one tensor avoids
# re-allocating a full-sized boolean mask per call. Same read-only contract
# as ``_CAUSAL_BLOCK_MASK_CACHE``.
_CAUSAL_BLOCK_MASK_INVALID_CACHE: dict = {}


def _causal_block_mask_invalid(T: int, n_blocks: int, m: int,
                               device) -> torch.Tensor:
    """Cached ``~_causal_block_mask(T, n_blocks, m, device)`` (read-only)."""
    key = (T, n_blocks, m, str(device))
    cached = _CAUSAL_BLOCK_MASK_INVALID_CACHE.get(key)
    if cached is not None:
        return cached
    invalid = ~_causal_block_mask(T, n_blocks, m, device)
    if len(_CAUSAL_BLOCK_MASK_INVALID_CACHE) >= 64:
        _CAUSAL_BLOCK_MASK_INVALID_CACHE.clear()
    _CAUSAL_BLOCK_MASK_INVALID_CACHE[key] = invalid
    return invalid


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
    S = _bf16_safe_softmax(logits, dim=2)                 # [B, n_blocks, m, c]
    if torch.is_grad_enabled():
        out = (S * C).sum(dim=2)                          # [B, n_blocks, c]
    else:
        # ``S`` is a fresh softmax output and ``C`` is only read: the in-place
        # multiply evaluates the same per-element product before the same
        # sum reduction.
        out = S.mul_(C).sum(dim=2)
    return _bf16_scale_if_needed(out)


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
    # b-branch entries shifted by one so block i uses block i-1; the first
    # block has no previous -> pad with -inf so softmax assigns 0 weight
    # (``F.pad`` writes the constant directly, no full-sized pad tensor).
    Zb_prev = F.pad(Zb[:, :-1], (0, 0, 0, 0, 1, 0),
                    value=float('-inf'))                          # [B, n_blocks, m, c]
    # a-branch logits for ALL blocks, then both branches concatenated along
    # the m axis. Without autograd the sums are written directly into the
    # two halves of one [B, n_blocks, 2m, c] buffer (the ``out=`` sums
    # compute the same elementwise adds). While a graph is recorded the
    # cat-of-sums form is kept (``out=`` does not build a graph).
    if torch.is_grad_enabled():
        A_logits = Za + Ba[None, None, :, :].to(Za)
        Bb_logits = Zb_prev + Bb[None, None, :, :].to(Zb_prev)
        all_logits = torch.cat([A_logits, Bb_logits], dim=2)
    else:
        all_logits = torch.empty(B_, n_blocks, 2 * m, c,
                                 device=Ca.device, dtype=compute_dtype)
        torch.add(Za, Ba[None, None, :, :].to(Za), out=all_logits[:, :, :m])
        torch.add(Zb_prev, Bb[None, None, :, :].to(Zb_prev),
                  out=all_logits[:, :, m:])
    S = _bf16_safe_softmax(all_logits, dim=2)                          # [B, n_blocks, 2m, c]
    Sa = S[:, :, :m]                                              # [B, n_blocks, m, c]
    Sb = S[:, :, m:]                                              # [B, n_blocks, m, c]

    # Cb shifted the same way: first block has no previous -> zero contribution.
    Cb_prev = F.pad(Cb[:, :-1], (0, 0, 0, 0, 1, 0),
                    value=0.0)                                    # [B, n_blocks, m, c]

    if torch.is_grad_enabled():
        out = (Sa * Ca).sum(2) + (Sb * Cb_prev).sum(2)            # [B, n_blocks, c]
    else:
        # ``Sa`` / ``Sb`` are views of the fresh softmax output ``S`` (no
        # other reader), and ``Ca`` / ``Cb_prev`` are only read; the in-place
        # products evaluate the identical elementwise chains before the same
        # sums.
        out = Sa.mul_(Ca).sum(2) + Sb.mul_(Cb_prev).sum(2)
    return _bf16_scale_if_needed(out)


def _csa_compress_kv_overlapped_prev(
    Ca: torch.Tensor,        # [B, n, m, c] current block's m tokens
    Za: torch.Tensor,        # [B, n, m, c]
    Cb_prev: torch.Tensor,   # [B, n, m, c] previous physical block per row
    Zb_prev: torch.Tensor,   # [B, n, m, c]
    Ba: torch.Tensor,
    Bb: torch.Tensor,
) -> torch.Tensor:
    """Overlapped compression with explicit per-row previous blocks.

    Evaluates exactly the value chain of :func:`csa_compress_kv_overlapped`
    for rows ``>= 1`` (the only rows ``_append_bulk`` keeps), taking the
    already-shifted previous-block ``Cb``/``Zb`` directly instead of padding
    and shifting the full sequence and recomputing/dropping row 0. Inputs
    are pre-blocked ``[B, n, m, c]`` so no concatenation is needed.
    """
    B_, n, m, c = Ca.shape
    compute_dtype = torch.float64 if Ca.dtype == torch.float64 else torch.float
    Ca = Ca.to(compute_dtype)
    Za = Za.to(compute_dtype)
    Cb_prev = Cb_prev.to(compute_dtype)
    Zb_prev = Zb_prev.to(compute_dtype)
    if torch.is_grad_enabled():
        A_logits = Za + Ba[None, None, :, :].to(Za)
        Bb_logits = Zb_prev + Bb[None, None, :, :].to(Zb_prev)
        all_logits = torch.cat([A_logits, Bb_logits], dim=2)
    else:
        # Same in-place layout as the public overlapped helper (identical
        # elementwise adds in the same cat layout).
        all_logits = torch.empty(B_, n, 2 * m, c,
                                 device=Ca.device, dtype=compute_dtype)
        torch.add(Za, Ba[None, None, :, :].to(Za), out=all_logits[:, :, :m])
        torch.add(Zb_prev, Bb[None, None, :, :].to(Zb_prev),
                  out=all_logits[:, :, m:])
    S = _bf16_safe_softmax(all_logits, dim=2)                      # [B, n, 2m, c]
    Sa = S[:, :, :m]
    Sb = S[:, :, m:]
    if torch.is_grad_enabled():
        out = (Sa * Ca).sum(2) + (Sb * Cb_prev).sum(2)
    else:
        out = Sa.mul_(Ca).sum(2) + Sb.mul_(Cb_prev).sum(2)
    return _bf16_scale_if_needed(out)


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
        # No fully-masked rows: use bf16-safe softmax.
        return _bf16_safe_softmax(logits, dim=dim)
    safe_logits = logits.masked_fill(all_masked, 0.0)
    soft = _bf16_safe_softmax(safe_logits, dim=dim)
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


def _qk_normalize_(x: torch.Tensor, mode: str = 'l2') -> torch.Tensor:
    """In-place :func:`_qk_normalize` on a freshly owned tensor.

    Evaluates the exact same per-element operations in the same order as
    :func:`_qk_normalize` (``x / clamp_min(||x||_2)`` for ``'l2'`` — the
    body of ``F.normalize`` — and ``x * rsqrt(mean(x^2) + 1e-6)`` for
    ``'rms'``), writing back into ``x``. The values are bit-identical to
    the out-of-place form.

    The caller must own ``x`` (a fresh cast / copy, never a view of cache
    storage or of a caller's tensor) and autograd must be disabled; the
    returned tensor is ``x`` itself.
    """
    if mode == 'l2':
        denom = x.norm(2, dim=-1, keepdim=True).clamp_min(1e-12)
        return x.div_(denom)
    if mode == 'rms':
        return x.mul_(torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6))
    raise ValueError(
        f"qk_norm_mode={mode!r} must be 'l2' (cosine, historical) or "
        f"'rms' (paper §2.3.3 RMSNorm form)")


# Cap on the number of (row, block) entries processed per iteration of the
# canonical top-k selector's boundary path. The rule is per-row, so the
# chunk size cannot change any selection; the cap only bounds the transient
# boolean / int32 masks (which would otherwise be allocated over the full
# [B, nIh, T, n_blocks] score tensor).
_TOPK_SELECT_CHUNK_ELEMS = 1 << 22


# Path-choice threshold for ``_canonical_topk_indices``: rows with at most
# this many entries take the stable sort, larger rows take the O(n)
# boundary path. Both paths implement the identical canonical rule, so the
# dispatch only selects which kernel runs.
_TOPK_SORT_MAX_ELEMS = 2048


def _boundary_selected_indices(blk: torch.Tensor, s: int) -> torch.Tensor:
    """Ascending indices of the canonical top-``s`` of each row of ``blk``.

    Boundary-path body of :func:`_canonical_topk_indices`. The returned
    ``[rows, s]`` index tensor holds, per row, the indices of every entry
    strictly above the S-th largest score followed by the smallest-index
    entries tied at that boundary value — always in ASCENDING index order.

    Exactly ``s`` valid indices are produced whenever ``blk`` contains no
    NaNs. NaNs are unselectable (they lose both the ``>`` and ``==``
    comparisons against the boundary), so a NaN row can under-fill: the
    missing slots then hold the sentinel ``n`` (the row length). The caller
    detects that with one range check and re-runs on a NaN-free copy.
    """
    n = blk.shape[-1]
    # One top-k supplies the boundary value AND the indices of the
    # strictly-above entries (every entry above the boundary is one of the
    # top-``s``). ``sorted=True`` orders both values and indices by
    # descending value; the values are needed only for the exact boundary
    # and above-count, so the tail is discarded.
    top_s, top_i = torch.topk(blk, s, dim=-1, sorted=True)
    boundary = top_s[..., -1:]
    above = top_s > boundary
    # Explicit ``dtype=torch.int32``: a bool sum would promote to int64,
    # and the (small) per-row count would then force a full-width int32 ->
    # int64 promotion below. Counts never exceed the row length, so int32
    # holds every value exactly.
    n_above = above.sum(dim=-1, keepdim=True, dtype=torch.int32)
    ar = torch.arange(n, device=blk.device)
    # Indices strictly above the boundary, ascending: sorting the small
    # ``[rows, s]`` index block (sentinel ``n`` pads the unused slots after
    # the valid ascending prefix) is cheaper than scanning the wide row.
    above_idx = torch.sort(
        torch.where(above, top_i, n), dim=-1).values
    # Fill the remaining slots with the SMALLEST indices tied at the
    # boundary (the canonical rule): one masked-index top-k gives them
    # already in ascending order.
    tie_idx = torch.topk(
        torch.where(blk == boundary, ar, n),
        s, dim=-1, largest=False, sorted=True).values
    slots = torch.arange(s, device=blk.device)
    tie_slot = (slots - n_above).clamp_(min=0)
    return torch.where(
        slots >= n_above, tie_idx.gather(-1, tie_slot), above_idx)


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
    by breaking ties on ASCENDING block index. The returned slots reproduce
    ``torch.topk``'s score-descending order (stable, so ties keep the
    ascending-index rule) and entries selected at ``-inf`` (fewer than
    ``k`` finite scores available) are reported as ``-1`` (invalid slot).
    ``k`` may exceed the score count; the caller pads the result.

    Path selection
    --------------
    Two interchangeable kernels produce the identical canonical rule (both
    paths are covered by the same parity tests, so the dispatch cannot
    change any result):

    * the stable descending sort, whose output IS the canonical rule
      (score-descending, ties ascending-index) — used for dense selections
      (``2*s >= n``, where nearly every entry is selected) and for rows up
      to ``_TOPK_SORT_MAX_ELEMS`` entries;
    * the ``O(n)`` boundary path (:func:`_boundary_selected_indices`) —
      a partial ``topk`` for the boundary value plus a few vectorized
      passes, no full sort — used for sparse
      selections on larger rows, INCLUDING the single-query-per-step
      decode shape once the block count exceeds the sort threshold, where
      the sort's ``O(n_blocks log n_blocks)`` term would dominate the
      per-token step.

    Non-finite scores
    -----------------
    NaNs are UNSELECTABLE: a NaN score never occupies a returned slot (the
    slot becomes ``-1`` when the row runs out of selectable entries), so a
    NaN anywhere in the score tensor can neither crash the selector nor
    silently occupy a real block index. ``-inf`` entries keep the documented
    "selected at ``-inf`` -> ``-1``" contract, and a NaN competes as
    ``-inf`` for the boundary/tie rules.
    """
    n = scores.shape[-1]
    s = min(k, n)
    if s == 0:
        return scores.new_zeros(*scores.shape[:-1], 0, dtype=torch.long)
    row_shape = scores.shape[:-1]
    rows_total = 1
    for d in row_shape:
        rows_total *= d
    # Dense selections and small arrays: one stable descending sort IS the
    # canonical rule (see docstring for why this split is cost-only).
    if 2 * s >= n or rows_total * n <= _TOPK_SORT_MAX_ELEMS:
        order = torch.sort(
            scores, dim=-1, descending=True, stable=True).indices[..., :s]
        values = scores.gather(-1, order)
        if bool(torch.isnan(values).any()):
            # NaN scores must be unselectable (see docstring). Checking the
            # gathered values — not the full score tensor — keeps the
            # finite-input hot path free of an ``O(n)`` NaN scan; this
            # branch only runs when a NaN actually entered the top-s slots.
            safe = scores.masked_fill(torch.isnan(scores), float('-inf'))
            order = torch.sort(
                safe, dim=-1, descending=True, stable=True).indices[..., :s]
            values = safe.gather(-1, order)
        return order.masked_fill(torch.isinf(values), -1)
    # Boundary path, processed in chunks of flattened leading rows so the
    # per-entry boolean / int32 masks below stay bounded. Every operation
    # here is row-wise (the S-th largest value is a per-row
    # order statistic, independent of how rows are grouped), so the chunk
    # size cannot change any selection.
    flat = scores.reshape(rows_total, n)
    chunk = max(1, _TOPK_SELECT_CHUNK_ELEMS // max(1, n))
    out = flat.new_empty(rows_total, s, dtype=torch.long)
    for lo in range(0, rows_total, chunk):
        hi = min(rows_total, lo + chunk)
        blk = flat[lo:hi]
        indices = _boundary_selected_indices(blk, s)
        # A NaN at the boundary under-fills the row (NaNs are unselectable,
        # so the missing slots hold the sentinel ``n``). Re-run on a
        # NaN-free copy (NaN -> -inf, so the canonical tie rule applies and
        # the slot count is exact); the check itself is one small per-row
        # reduction on the hot path.
        if bool((indices >= n).any()):
            blk = blk.masked_fill(torch.isnan(blk), float('-inf'))
            indices = _boundary_selected_indices(blk, s)
        values = blk.gather(-1, indices)
        # Restore the score-descending slot order (stable: boundary ties
        # keep the canonical ascending-index order).
        order = torch.argsort(values, dim=-1, descending=True, stable=True)
        indices = torch.gather(indices, -1, order)
        values = torch.gather(values, -1, order)
        out[lo:hi] = indices.masked_fill(torch.isinf(values), -1)
    return out.view(*row_shape, s)


def _lightning_indexer_chunk(
    q_idx_chunk: torch.Tensor,      # [B, T_c, HI, DI]  query slice
    k_idx: torch.Tensor,            # [B, n_blocks, DI]
    w_idx_chunk: torch.Tensor | None,   # [B, T_c, HI]
    topk: int,                      # >= 1 (topk=0 is handled by the caller)
    scale: float,
    causal_block_mask_chunk: torch.Tensor | None,   # [T_c, n_blocks]
    return_soft_weights: bool,
    causal_block_mask_invalid_chunk: torch.Tensor | None = None,
):
    """Score + select ONE query slice of the lightning indexer.

    This is the body of :func:`csa_lightning_indexer`, factored out so the
    public function can tile the query axis without duplicating the scoring,
    masking, canonical top-k and NaN-safe softmax logic. Every operation in
    here is row-wise along the query axis (the einsums reduce over ``DI`` /
    ``HI``; softmax and top-k are per query row), so evaluating a query
    slice yields results identical to evaluating the same rows inside the
    full tensor up to batched-GEMM reduction order. That identity is what
    makes ``query_chunk`` a pure memory optimization with no algorithmic
    effect — pinned by ``test_csa_indexer_query_chunk_equivalence`` in
    ``test_csa_chunk_fla_adapter.py``.

    ``topk`` must be >= 1; the ``topk=0`` contract (empty indices plus an
    optional full soft distribution) is handled by the caller, which also
    owns the query-chunk loop and the final concatenation.
    """
    # Same batched GEMM as the historical
    # ``einsum('b t h d, b n d -> b h t n', ...)``, but returned in the
    # ``[B, T_c, HI, n_blocks]`` layout: the GEMM needs no permute copy and
    # the following contraction over ``HI`` consumes this layout directly.
    # The contraction over ``DI`` (and therefore every value) is unchanged.
    score = torch.einsum(
        'b t h d, b n d -> b t h n',
        q_idx_chunk, k_idx)                                     # [B, T_c, HI, n]
    # Multiplying by exactly 1.0 is an IEEE no-op (values and gradients
    # preserved bit-for-bit); skip it to avoid an allocation.
    if not (isinstance(scale, (int, float)) and float(scale) == 1.0):
        # ``score`` is a fresh einsum output with no other readers; the
        # in-place multiply / ReLU evaluate the same elementwise ops
        # (bit-identical), but an autograd node would need the pre-op
        # tensor, so they are only used with autograd disabled.
        if torch.is_grad_enabled():
            score = score * scale
        else:
            score.mul_(scale)
    if torch.is_grad_enabled():
        score = F.relu(score)
    else:
        score.relu_()
    if w_idx_chunk is None:
        logits = score.sum(2)                                       # [B, T_c, n_blocks]
    else:
        # Same batched GEMM as the historical einsum (bit-identical forward
        # and backward), consuming ``score``'s [B, T_c, HI, n] layout
        # directly so no permute/copy is needed.
        logits = torch.matmul(
            w_idx_chunk.unsqueeze(-2), score).squeeze(-2)           # [B, T_c, n_blocks]
    if causal_block_mask_chunk is not None:
        # ``logits`` is fresh (einsum/sum output) and both consumers
        # (the selector and the softmax) read only the masked values, so
        # the fill can be written in place when no backward needs the
        # pre-fill tensor (inference / no-grad). Autograd-tracked logits
        # keep the out-of-place form.
        if causal_block_mask_invalid_chunk is not None:
            # Pre-negated mask supplied by the caller: the negation of the
            # full-sequence mask is cached, so only the slice is taken.
            invalid = causal_block_mask_invalid_chunk
        else:
            invalid = ~causal_block_mask_chunk
        if logits.requires_grad:
            logits = logits.masked_fill(invalid, float('-inf'))
        else:
            logits.masked_fill_(invalid, float('-inf'))
    B_, T_c, n_blocks = logits.shape
    S = min(topk, n_blocks)
    # Deterministic, size-independent top-k (ties by ascending block
    # index) — see ``_canonical_topk_indices``. This is what keeps the
    # batched-prefill and incremental-decode selections identical.
    idx = _canonical_topk_indices(logits, S)
    if topk > S:
        # One allocation with the left part copied in place (same values as
        # the former cat with a right-hand ``-1`` fill).
        padded = idx.new_full((B_, T_c, topk), -1)
        padded[..., :S].copy_(idx)
        idx = padded
    if return_soft_weights:
        # The differentiable soft distribution over ALL blocks for the
        # straight-through estimator. Softmax is row-wise, so computing it
        # per query chunk and concatenating is value-identical to one
        # softmax over the full batch.
        return idx, _nan_safe_softmax(logits, dim=-1)
    return idx


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
    query_chunk: int = 0,
    _causal_block_mask_invalid: torch.Tensor | None = None,
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

    query_chunk : int, default ``0``
        Optional query-axis tile size for the indexer scoring. ``0``
        (default) evaluates all ``T`` queries in one shot, exactly as the
        historical implementation did. A positive value scores at most
        ``query_chunk`` queries at a time and concatenates the per-tile
        selections, which bounds the peak size of the per-head score
        tensor from ``O(B · HI · T · n_blocks)`` to
        ``O(B · HI · query_chunk · n_blocks)``. This matters for
        long-context prefill: at ``T = 1M`` with ``m = 16``,
        ``HI = 2`` the untiled score tensor holds ~1.25e11 floats
        (~500 GB in fp32), while a 4096-token tile needs ~1.6M floats per
        head pair. The selection is unchanged: every operation in the
        indexer is row-wise along the query axis and the canonical
        tie-break (:func:`_canonical_topk_indices`) is
        shape-independent, so the tiled and untiled paths return the same
        indices (bit-compared in the regression suite) and the same soft
        weights (up to batched-GEMM reduction order). The ``soft_weights``
        OUTPUT is still ``[B, T, n_blocks]`` when requested — the tile
        bound applies to the intermediate per-head scores/logits, not to
        the STE distribution, which the caller may need in full.

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
    # ``query_chunk`` tiles the QUERY axis (see the docstring section of the
    # same name). Negative values are rejected rather than silently treated
    # as "disabled": a typo like ``query_chunk=-4096`` would otherwise run
    # the untiled path while looking like a memory bound was set.
    if not isinstance(query_chunk, int) or isinstance(query_chunk, bool):
        raise ValueError(
            f"query_chunk={query_chunk!r} must be an int (0 disables "
            f"query-axis tiling; a positive value is the tile size).")
    if query_chunk < 0:
        raise ValueError(
            f"query_chunk={query_chunk} must be >= 0 (0 disables "
            f"query-axis tiling).")
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

    # Query-axis tiling dispatch. ``query_chunk <= 0`` (the default) or a
    # tile at least as large as T evaluates everything in ONE call to
    # ``_lightning_indexer_chunk`` — the historical code path, just moved
    # into the shared helper (the operations inside are unchanged, so the
    # disabled path stays bit-identical). A positive tile smaller than T
    # loops over query slices and concatenates the per-tile results; every
    # indexer operation is row-wise along the query axis and the canonical
    # tie-break is shape-independent, so the result is the same while the
    # peak size of the per-head score tensor shrinks from
    # ``B·HI·T·n_blocks`` to ``B·HI·query_chunk·n_blocks``.
    tile = T if query_chunk <= 0 else min(query_chunk, T)

    def _run_chunk(start: int, end: int):
        mask_chunk = (None if causal_block_mask is None
                      else causal_block_mask[start:end])
        invalid_chunk = (None if _causal_block_mask_invalid is None
                         else _causal_block_mask_invalid[start:end])
        w_chunk = None if w_idx is None else w_idx[:, start:end]
        return _lightning_indexer_chunk(
            q_idx[:, start:end], k_idx, w_chunk, topk, scale,
            mask_chunk, return_soft_weights, invalid_chunk)

    if tile >= T:
        # ``topk=0`` is handled inside ``_lightning_indexer_chunk``: the
        # canonical selector returns shape-[B, T, 0] indices for ``S = 0``
        # and the softmax still produces the full differentiable
        # distribution when requested, preserving the historical contract.
        return _run_chunk(0, T)

    # Tiled path: accumulate per-tile selections along the query axis. The
    # concatenation output can be preallocated and written slice-by-slice
    # when no graph is recorded (the integer indices never need grad and
    # the soft weights are not retained by autograd); while a graph is
    # recorded the historical list-plus-cat form is kept.
    if not torch.is_grad_enabled():
        idx = torch.empty(B_, T, topk, dtype=torch.long,
                          device=q_idx.device)
        soft = (torch.empty(B_, T, n_blocks, dtype=compute_dtype,
                            device=q_idx.device)
                if return_soft_weights else None)
        for start in range(0, T, tile):
            end = min(start + tile, T)
            result = _run_chunk(start, end)
            if return_soft_weights:
                idx_t, soft_t = result
                soft[:, start:end] = soft_t
            else:
                idx_t = result
            idx[:, start:end] = idx_t
        return (idx, soft) if return_soft_weights else idx
    idx_tiles: list[torch.Tensor] = []
    soft_tiles: list[torch.Tensor] = []
    for start in range(0, T, tile):
        end = min(start + tile, T)
        result = _run_chunk(start, end)
        if return_soft_weights:
            idx_t, soft_t = result
            soft_tiles.append(soft_t)
        else:
            idx_t = result
        idx_tiles.append(idx_t)
    idx = torch.cat(idx_tiles, dim=1)
    if return_soft_weights:
        return idx, torch.cat(soft_tiles, dim=1)
    return idx


# Cached concatenation of the 6 KV/indexer projection weights used by
# ``naive_csa(fuse_projections=True)``. The concatenated tensor is only
# reused while every source weight is STILL THE SAME TENSOR at the SAME
# VERSION (an in-place optimizer update bumps ``_version``) and autograd is
# disabled — while a graph is recorded the per-forward ``cat`` is kept so
# backward sees the identical node. Weak references avoid pinning a
# discarded model's parameters in memory.
_FUSED_WEIGHT_CACHE: dict | None = None


def _fused_projection_weight(weights: tuple) -> torch.Tensor:
    """Return a cached ``cat(weights, dim=0)`` for inference forwards."""
    global _FUSED_WEIGHT_CACHE
    entry = _FUSED_WEIGHT_CACHE
    if entry is not None:
        refs, versions, combined = entry
        if (len(refs) == len(weights)
                and all(r() is w for r, w in zip(refs, weights))
                and versions == tuple(w._version for w in weights)
                and combined.shape[0] == sum(w.shape[0] for w in weights)
                and combined.dtype == weights[0].dtype
                and combined.device == weights[0].device):
            return combined
    combined = torch.cat(list(weights), dim=0)
    _FUSED_WEIGHT_CACHE = (
        tuple(weakref.ref(w) for w in weights),
        tuple(w._version for w in weights),
        combined,
    )
    return combined


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
    indexer_chunk: int = 0,
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

    ``indexer_chunk`` (default 0 = disabled) forwards a query-axis tile size
    to :func:`csa_lightning_indexer`'s ``query_chunk``: the lightning-indexer
    scores are then computed for at most ``indexer_chunk`` queries at a time,
    bounding the peak per-head score tensor to
    ``O(B · nIh · indexer_chunk · n_blocks)`` instead of
    ``O(B · nIh · T · n_blocks)``. The selection and STE soft weights are
    unchanged (pinned by ``test_csa_naive_indexer_chunk_end_to_end`` in
    ``test_csa_chunk_fla_adapter.py``); only the
    intermediate memory footprint changes. Useful for long-context prefill and
    for `torch.compile`'d call sites that need bounded intermediate buffers.

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
    # ``indexer_chunk`` tiles the indexer's QUERY axis (see
    # ``csa_lightning_indexer``'s ``query_chunk``). 0 = disabled (the
    # historical full-sequence scoring); a positive value bounds the peak
    # per-head score tensor to ``B · nIh · indexer_chunk · n_blocks``.
    # Negative values would silently run the untiled path (looking like a
    # memory bound was applied) — reject them.
    if not isinstance(indexer_chunk, int) or isinstance(indexer_chunk, bool):
        raise ValueError(
            f"indexer_chunk={indexer_chunk!r} must be an int (0 disables "
            f"query-axis tiling of the lightning indexer).")
    if indexer_chunk < 0:
        raise ValueError(
            f"indexer_chunk={indexer_chunk} must be >= 0 "
            f"(0 disables query-axis tiling).")
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
        #
        # The concatenation is a deterministic copy of read-only weights, so
        # with autograd disabled (inference / evaluation) it is cached across
        # calls, keyed on tensor identity + ``_version``; a recorded graph
        # keeps the per-forward ``torch.cat`` exactly as before (its
        # CatBackward must stay in the current graph).
        _weights = (W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx)
        if torch.is_grad_enabled():
            combined_weight = torch.cat(list(_weights), dim=0)
        else:
            combined_weight = _fused_projection_weight(_weights)
                                                                   # [4c + 2*c_I, d]
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
    compute_dtype = torch.float64 if H.dtype == torch.float64 else torch.float
    # Cast the a-branch projection to compute precision once and share it:
    # ``csa_compress_kv_overlapped`` consumes this tensor (its internal
    # ``.to(compute_dtype)`` becomes a no-op) and the sliding-window branch
    # below reuses the same storage instead of re-casting.
    Ca_c = Ca.to(compute_dtype)
    C_comp = csa_compress_kv_overlapped(
        Ca_c, Cb, Za, Zb, Ba, Bb, m)                                 # [B, n_blocks, c]
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
    # Cached negation of the full-sequence mask (read-only) so the indexer's
    # ``masked_fill_`` does not allocate its own [T, n_blocks] boolean.
    cbm_invalid = _causal_block_mask_invalid(T, n_blocks, m, device)
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
                                    normalize_qk=normalize_qk,
                                    query_chunk=indexer_chunk,
                                    _causal_block_mask_invalid=cbm_invalid
                                    )                              # [B, T, topk]
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
    # Both ``q`` (fresh linear + cast) and ``C_comp`` (fresh compression
    # output, optionally re-bound by ``partial_rope``) are exclusively owned
    # by this call and have no other readers once normalized, so with
    # autograd disabled the normalization runs in place (identical
    # elementwise chain, one fewer full-size temporary each). A recorded
    # graph keeps the out-of-place form.
    if normalize_qk:
        if torch.is_grad_enabled():
            q = _qk_normalize(q, qk_norm_mode)
        else:
            q = _qk_normalize_(q, qk_norm_mode)
    if normalize_qk:
        if torch.is_grad_enabled():
            C_comp_n = _qk_normalize(C_comp, qk_norm_mode)
        else:
            C_comp_n = _qk_normalize_(C_comp, qk_norm_mode)
    else:
        C_comp_n = C_comp.to(compute_dtype)
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
        # Row gather via a flattened ``index_select`` (equivalent to
        # ``C_comp_n[batch_idx, idx_safe]`` advanced indexing: each output
        # element is a copy of the same source element; the selector picks
        # DISTINCT blocks per row, so the backward scatter/accumulation is
        # unaffected).
        nblk = C_comp_n.shape[1]
        batch_off = torch.arange(B_, device=device).view(B_, 1, 1) * nblk
        # ``idx_safe`` is a fresh clamp output and has no later readers when
        # ``effective_use_ste`` is False (the STE path below re-gathers with
        # the un-offset indices), so that path folds the batch offset in
        # place instead of materializing a second index tensor.
        if effective_use_ste:
            flat_idx = (idx_safe + batch_off).reshape(-1)
        else:
            flat_idx = idx_safe.add_(batch_off).reshape(-1)
        kv = torch.index_select(
            C_comp_n.reshape(B_ * nblk, C_comp_n.shape[-1]), 0, flat_idx,
        ).view(B_, T, idx_safe.shape[-1], C_comp_n.shape[-1])           # [B, T, topk, c]

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
            # multiply keeps values/gradients bit-for-bit identical. The
            # einsum output is fresh, so inference (no autograd) may scale in
            # place; a recorded graph keeps the out-of-place multiply.
            if torch.is_grad_enabled():
                scores = scores * scale
            else:
                scores.mul_(scale)
        # Mask -1 padding so those slots get zero weight after softmax.
        # The einsum output is freshly allocated and only consumed by the
        # joint softmax below, so the mask can be written in place.
        scores.masked_fill_(~valid_mask[:, :, None, :], float('-inf'))

    # Sliding-window branch: gather per-query window SCORES (uncompressed
    # local KV) so they can share the compressed entries' softmax below.
    if sliding_window > 0:
        win = sliding_window
        H_proj = Ca_c  # [B, T, c], already == F.linear(H, W_aKV)
        if normalize_qk:
            # Exact same elementwise chain either way; the in-place form is
            # used only when this call exclusively owns the storage (the
            # cast produced a fresh copy) or the projection is not returned
            # (nothing else reads ``Ca`` afterwards).
            if (not torch.is_grad_enabled()) and (
                    H_proj is not Ca or not return_projections):
                C_local = _qk_normalize_(H_proj, qk_norm_mode)     # [B, T, c]
            else:
                C_local = _qk_normalize(H_proj, qk_norm_mode)      # [B, T, c]
        else:
            C_local = H_proj
        # Partial RoPE on the sliding-window keys (§2.3.3: RoPE applies to
        # the KV entry vectors used in CSA/HCA — the SW entries are such
        # KV entries). Each key rotates at its own token position; the
        # decoding cache's ring buffer applies the same
        # normalize-then-rotate order at append time.
        if rope_dim_eff > 0:
            C_local = partial_rope(
                C_local, tpos, rope_dim_eff, rope_base, dim=1)
        # Inference fast path: materialize the full [B, T, win, c] window
        # tensor ONCE when it fits the same per-tile element budget the
        # tiled path already uses (in its single-tile case the tiled
        # scoring path builds exactly this slab anyway); the window value
        # output then reuses it instead of re-deriving it tile-by-tile.
        # The per-row dot products are the same contraction, so the forward
        # values are bit-identical; only one densification copy is dropped.
        # GATED ON no_grad: under autograd the two-einsum-per-tile structure
        # is kept exactly as before so backward sees the identical graph
        # (the reuse path would sum the two window-gradient contributions
        # through one shared node, changing backward rounding by ~1 ulp).
        # Larger sequences keep the bounded-memory tiled path
        # (``return_windows=False``, window slab re-derived per tile under
        # ``CHUNKED_SW_THRESHOLD`` elements).
        use_full_windows = ((not torch.is_grad_enabled())
                            and (B_ * T * win * c) <= CHUNKED_SW_THRESHOLD)
        scores_w, C_windows_full, win_valid = _sliding_window_scores(
            q, C_local, win, scale, device,
            return_windows=use_full_windows)
    else:
        scores_w = torch.empty(B_, T, nh, 0, dtype=compute_dtype, device=device)
        win_valid = torch.empty(T, 0, dtype=torch.bool, device=device)
        C_windows_full = None

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
            if torch.is_grad_enabled():
                # Out-of-place: autograd's Sub/Exp backward save this chain's
                # operands/outputs, so in-place writes would trip the
                # version-counter check.
                shifted = cat - row_max
                shifted_sink = log_sink - row_max
                log_sum_exp = torch.logsumexp(shifted, dim=-1, keepdim=True)
                log_denom = torch.logaddexp(log_sum_exp, shifted_sink)
                p = (shifted - log_denom).exp()
                p = p.masked_fill(all_invalid, 0.0)
            else:
                # ``cat`` is freshly allocated by the einsum / window scoring
                # above (or is a fresh single-branch tensor) and has no other
                # readers left, so the shift/exp/fill chain can run in place:
                # same elementwise ops in the same order (bit-identical).
                shifted = cat.sub_(row_max)
                shifted_sink = log_sink - row_max
                log_sum_exp = torch.logsumexp(shifted, dim=-1, keepdim=True)
                log_denom = torch.logaddexp(log_sum_exp, shifted_sink)
                p = shifted.sub_(log_denom).exp_()
                p.masked_fill_(all_invalid, 0.0)
        else:
            p = _nan_safe_softmax(cat, dim=-1, all_masked_mask=all_invalid)

        topk_size = scores.shape[-1]
        p_b = p[..., :topk_size]
        p_w = p[..., topk_size:]
        # Accumulate the window contribution IN PLACE into the sparse
        # einsum output (same values as ``a + b``). The sparse output is
        # not saved by any autograd node (the add's backward needs neither
        # operand), so the in-place add is gradient-safe.
        out = torch.einsum('b t h k, b t k d -> b t h d', p_b, kv)
        if sliding_window > 0:
            if C_windows_full is not None:
                # Reuse the window slab materialized by the scoring pass
                # above (same per-row contraction as the tiled rebuild).
                out.add_(torch.einsum(
                    'b t h w, b t w d -> b t h d', p_w, C_windows_full))
            else:
                out.add_(_sliding_window_output(p_w, C_local, win, device))

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
    # The trim runs BEFORE the dtype cast so a padded call only casts the
    # real rows (elementwise cast, identical values on the kept rows).
    if original_T != T:
        out_final = out[:, :original_T].reshape(
            B_, original_T, nh * c).to(H.dtype)
    else:
        out_final = out.reshape(B_, T, nh * c).to(H.dtype)
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
