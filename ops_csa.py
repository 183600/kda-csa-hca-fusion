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


# =============================================================================
# Chunked sliding-window attention helper (issue 2.2 fix)
# -----------------------------------------------------------------------------
# The previous sliding-window implementation used
# ``C_padded.unfold(1, win, 1)`` which materializes a ``[B, T, win, c]`` tensor
# in a single allocation. For ``T=64K, win=2K, c=64, B=1`` this is
# ``64K × 2K × 64 × 4B = 32 GB`` — far beyond GPU memory. The README's
# *Limitations* section acknowledges this but no fix was provided in-repo.
#
# This helper processes the window axis in chunks of ``chunk_t`` query
# positions at a time, keeping peak memory at ``O(chunk_t · win · c)`` instead
# of ``O(T · win · c)``. The default threshold ``CHUNKED_SW_THRESHOLD``
# auto-engages the chunked path when ``T * win * c`` exceeds 8M elements
# (≈32MB in fp32) — small enough that the chunked path's Python overhead is
# negligible relative to the matmul cost, but large enough that the
# vectorized ``unfold`` path stays fast for the common ``T≤4k, win≤512``
# case documented as safe in the README.
#
# Numerical contract: IDENTICAL to the ``unfold``-based path. Softmax over
# the ``win`` non-masked entries of a row is the same operation whether the
# row was extracted as a slice of a ``[T, win]`` unfold view or as one chunk
# of ``[chunk_t, win]`` slices. Verified by
# ``test_hca_sliding_window_causality`` / ``test_csa_full_pipeline_causality``
# in run_correctness.py.
# ============================================================================

# Auto-engage chunked path when the materialized unfold tensor would exceed
# 8M elements (≈32MB fp32). Below this threshold the unfold path's
# vectorized CUDA kernel beats the chunked path's Python-loop overhead.
CHUNKED_SW_THRESHOLD = 8_000_000


def _sliding_window_attention(
    q: torch.Tensor,           # [B, T, nh, c]   (L2-normalized)
    C_local: torch.Tensor,     # [B, T, c]       (L2-normalized)
    win: int,
    scale: float,
    device,
    *,
    chunk_t: int | None = None,
) -> torch.Tensor:
    """Memory-efficient sliding-window attention.

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
            win))))`` so peak memory stays near the threshold.

    Numerical contract:
        Identical to the equivalent ``unfold``-based implementation, up to
        floating-point reassociation in the per-chunk softmax. Differences
        are well below the fp32 tolerance enforced by ``run_correctness.py``.
    """
    B, T, nh, c = q.shape
    # Pre-build the static validity mask once (shape [T, win]) — this is
    # O(T * win) booleans = ~T*win/8 bytes (e.g. 64K*2K/8 = 16MB at the
    # worst case documented in the README), small relative to the [B, T,
    # win, c] float tensor we avoid materializing.
    _j = torch.arange(win, device=device)
    _t = torch.arange(T, device=device)
    valid_mask = _j[None, :] >= (win - 1 - _t[:, None])    # [T, win], True=real

    # Auto-pick chunk_t if not given. Goal: keep peak memory near
    # CHUNKED_SW_THRESHOLD / win / c per chunk (in elements). Round up to a
    # multiple of 64 for matmul efficiency.
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
    # compute scores/softmax/output, and accumulate into ``out``. The peak
    # intermediate is ``[B, chunk_t, win, c]`` — bounded by chunk_t.
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
        C_windows = C_slice.unfold(1, win, 1).permute(0, 1, 3, 2)  # [B, ct, win, c]
        q_chunk = q[:, t_lo:t_hi]                            # [B, ct, nh, c]
        mask_chunk = valid_mask[t_lo:t_hi]                   # [ct, win]
        scores = torch.einsum('b t h d, b t w d -> b t h w', q_chunk, C_windows) * scale
        scores = scores.masked_fill(~mask_chunk[None, :, None, :], float('-inf'))
        p = torch.softmax(scores, dim=-1)                   # [B, ct, nh, win]
        out[:, t_lo:t_hi] = torch.einsum('b t h w, b t w d -> b t h d', p, C_windows)
    return out


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
    # Validate m BEFORE ``T % m`` so a caller passing m=0 gets a clear
    # ValueError instead of a bare ``ZeroDivisionError`` with no context.
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the check
    # survives ``python -O`` / ``PYTHONOPTIMIZE=1``. ``assert`` statements
    # are silently stripped under optimization, which would re-expose the
    # cryptic ZeroDivisionError this guard is specifically meant to prevent.
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
    # Validate m BEFORE ``T % m`` (mirrors csa_compress_kv).
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the check
    # survives ``python -O`` / ``PYTHONOPTIMIZE=1``. ``assert`` statements
    # are silently stripped under optimization, which would re-expose the
    # cryptic ZeroDivisionError this guard is specifically meant to prevent.
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
    # ``run_correctness.py::test_overlap_causality`` and
    # ``method_analysis.py``, so a defensive guard here makes the contract
    # match ``csa_compress_kv`` (which already handles T=0 correctly via the
    # view operation returning an empty [B, 0, m, c] tensor).
    if T == 0:
        return Ca.new_zeros(B_, 0, c)
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


# C5 fix: NaN-safe softmax helper, used by csa_lightning_indexer (twice)
# and naive_csa (twice). The previous implementation copy-pasted the
# 4-line ``all_masked / safe_logits / softmax / masked_fill_`` block in
# four places. Any future numerical-stability fix must now be applied in
# exactly one spot.
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
    safe_logits = logits.masked_fill(all_masked, 0.0)
    soft = torch.softmax(safe_logits, dim=dim)
    return soft.masked_fill(all_masked, 0.0)


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
        mode. The default ``'topk_columns'`` matches the historical
        behaviour and the published CSA reference implementation; the
        other modes are documented for extensibility and ablation.

    normalize_qk : bool, default ``False``
        If True, L2-normalize ``q_idx`` and ``k_idx`` along the
        per-head key dimension (``DI``) before computing the ReLU
        dot-product scores. This makes the indexer's top-k selection
        invariant to the magnitude of the indexer queries/keys (the
        DeepSeek-V4 paper specifies cosine-style scoring), preventing
        a high-norm indexer query from dominating the top-k ranking
        purely by magnitude. Setting to ``False`` (the default)
        preserves the historical un-normalized behaviour for backward
        compatibility with existing benchmark results; **new use cases
        should pass ``normalize_qk=True``** to match the
        magnitude-invariant contract of the CSA/HCA core attention
        (which L2-normalizes both ``q`` and ``C_comp``). The option
        is also exposed for ablation against the magnitude-sensitive
        baseline.

    .. note:: P0-4 fix — straight-through estimator (STE) for the indexer.

        The returned ``idx`` tensor is an integer tensor produced by
        ``torch.topk``. Integer indices do NOT propagate gradients, so
        without an auxiliary path autograd cannot flow back from the
        loss through the selection to the indexer parameters
        (``W_IUQ``, ``W_w``, ``W_KV_idx``, ``W_Z_idx``, ``B_idx`` in
        ``CSAHybridLayer``). After a backward pass these parameters had
        ``.grad is None`` and were silently skipped by ``AdamW``.

        The fix adds a **straight-through estimator**: when
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

    .. note:: topk=0 guard (P0 API self-containment).

        ``topk=0`` is a valid degenerate configuration ("no sparse
        selection"). The downstream ``naive_csa`` guards against the
        ``scores.amax(-1)`` crash on an empty dim, but historically
        relied on its own early return. We now also guard here so a
        direct caller of ``csa_lightning_indexer`` (it is a PUBLIC
        function imported by ``run_correctness.py`` and
        ``method_analysis.py``) does not crash with
        ``RuntimeError: selected index k out of range`` from
        ``torch.topk``. With ``topk=0`` we return an indices tensor
        of shape ``[B, T, 0]`` (no valid slots) and, if requested, a
        ``soft_weights`` tensor of shape ``[B, T, n_blocks]`` (the
        full differentiable distribution — still useful for an
        auxiliary loss even when no hard selection is made).
    """
    # Validate topk BEFORE ``min(topk, n_blocks)`` so a caller passing
    # topk=-1 gets a clear ValueError instead of a cryptic
    # ``RuntimeError: selected index k out of range`` from ``torch.topk``
    # (which receives ``S = min(-1, n_blocks) = -1`` and crashes with no
    # diagnostic about WHICH parameter was bad). The caller ``naive_csa``
    # already validates topk >= 0, but ``csa_lightning_indexer`` is a PUBLIC
    # function (no underscore prefix) imported directly by
    # ``run_correctness.py`` and ``method_analysis.py``, so it must defend
    # its own contract.
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the check
    # survives ``python -O`` / ``PYTHONOPTIMIZE=1``. ``assert`` statements
    # are silently stripped under optimization, which would re-expose the
    # cryptic torch.topk RuntimeError this guard is specifically meant to
    # prevent.
    if topk < 0:
        raise ValueError(
            f"topk={topk} must be >= 0 (0 selects zero blocks)")
    # Validate the indexer key dimension BEFORE computing ``scale = DI ** -0.5``,
    # which would raise ``ZeroDivisionError: 0.0 cannot be raised to a
    # negative power`` if ``DI == 0``. ``naive_csa`` already validates
    # ``c_I >= 1``, but this is a PUBLIC function imported directly by
    # ``run_correctness.py`` and ``method_analysis.py``, so it must defend
    # its own contract. Use ``raise ValueError`` (NOT ``assert``) so the
    # check survives ``python -O``.
    if q_idx.shape[-1] < 1:
        raise ValueError(
            f"q_idx.shape[-1]={q_idx.shape[-1]} must be >= 1 "
            f"(indexer key dimension c_I must be positive)")
    # C3 fix: ``ste_mode`` is accepted here only for backwards-compatible
    # call-site ergonomics (``naive_csa`` passes it through unconditionally
    # at line 745). The actual STE branching happens INSIDE ``naive_csa``
    # (lines 869, 915) — this function only returns the indices + (optionally)
    # the soft_weights, which are the SAME for every ste_mode. We still
    # validate the value so a typo (``ste_mode='topk'``) doesn't silently
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
    if scale is None:
        scale = q_idx.shape[-1] ** -0.5
    B_, T, HI, DI = q_idx.shape
    n_blocks = k_idx.shape[1]
    compute_dtype = torch.float64 if q_idx.dtype == torch.float64 else torch.float
    q_idx = q_idx.to(compute_dtype)
    k_idx = k_idx.to(compute_dtype)
    if w_idx is not None:
        w_idx = w_idx.to(compute_dtype)

    # P0 fix: L2-normalize indexer queries and keys so the top-k ranking
    # is magnitude-invariant (cosine-style scoring, as specified in the
    # DeepSeek-V4 paper Eq. 15). Without normalization, a high-norm
    # indexer query dominates the top-k ranking purely by magnitude
    # rather than by semantic relevance, making the sparse selection
    # sensitive to upstream projection scaling. The historical scale
    # ``DI ** -0.5`` partially mitigates this but does not eliminate it;
    # explicit normalization is the standard fix. Set ``normalize_qk=False``
    # to restore the un-normalized behaviour (e.g. for ablation).
    if normalize_qk:
        q_idx = F.normalize(q_idx, dim=-1)
        k_idx = F.normalize(k_idx, dim=-1)

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
    # P0 topk=0 guard: return shape-[B, T, 0] indices directly so the
    # function is self-contained (does not rely on ``naive_csa`` to guard
    # the ``scores.amax(-1)`` call). ``torch.topk(logits, 0, ...)`` would
    # itself succeed (returning an empty tensor), but the explicit branch
    # documents the contract and skips the ``min(topk, n_blocks)``
    # computation that would otherwise produce a misleading ``S=0`` value.
    if topk == 0:
        idx = q_idx.new_zeros(B_, T, 0, dtype=torch.long)
        if return_soft_weights:
            # C5 fix: use shared _nan_safe_softmax helper.
            soft_weights = _nan_safe_softmax(logits, dim=-1)
            return idx, soft_weights
        return idx
    S = min(topk, n_blocks)
    values, idx = torch.topk(logits, S, dim=-1)                        # [B, T, S]
    idx = idx.masked_fill(torch.isinf(values), -1)
    if topk > S:
        idx = torch.cat([idx, idx.new_full((B_, T, topk - S), -1)], dim=-1)
    # P0-4 fix: compute a differentiable soft distribution over ALL
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
        # C5 fix: use shared _nan_safe_softmax helper.
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
) -> torch.Tensor:
    """Full CSA forward (compression + indexer + sparse MQA core attention).

    Returns output ``[B, T, nh*c]`` (the caller performs the grouped output
    projection ``[B, T, nh*c] -> d``).

    **Weight layout** (P0 API fix): all ``W_*`` tensors follow the
    ``nn.Linear.weight`` convention — shape ``[out_features, in_features]``.
    Internally we use ``F.linear(x, W)`` (which computes ``x @ W.T``) instead
    of the previous ``x @ W`` form that required callers to pass
    ``self.W_aKV.weight.T`` (a non-contiguous view materialized on every
    forward). Callers now pass ``self.W_aKV.weight`` directly.

    ``T`` does NOT need to be divisible by ``m``: the function right-pads the
    sequence with zeros up to the next multiple of ``m`` and trims the output
    back to the original length, mirroring the contract of
    ``naive_chunk_kda``. Real tokens keep their original positions; only the
    last partial block contains padding zeros, and the causal block mask
    ensures no real token attends to it. This removes a footgun where direct
    callers (without the external padding done by ``HybridKCHAttention`` or
    ``CSAAttn``) would hit a bare ``ValueError`` with no message.

    .. warning:: **review-fix 1.3 — ``fuse_projections`` and the**
        ``use_ste=False`` **gradient contract.**

        By default (``fuse_projections=True``) this function concatenates
        the 6 KV/indexer-compression weights (``W_aKV, W_bKV, W_aZ, W_bZ,
        W_KV_idx, W_Z_idx``) into a single matmul (issue 2.3 fix — see the
        in-code comment above ``combined_weight`` for the performance
        rationale). This is a pure **forward-value-preserving**
        optimization — the output is bit-identical (to fp32 tolerance) to
        doing the 6 ``F.linear`` calls separately — but it has an
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
        ``run_correctness.py`` for a regression pinning both branches.
    """
    B_, T, d = H.shape
    # Validate structural params early so a caller passing m=0, topk=-1,
    # etc. gets a clear ValueError instead of a cryptic ZeroDivisionError
    # or IndexError deep inside the operator.
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the checks
    # survive ``python -O`` / ``PYTHONOPTIMIZE=1``. ``assert`` statements
    # are silently stripped under optimization, which would re-expose the
    # cryptic crashes these guards are specifically meant to prevent.
    # ``raise ValueError`` is the standard exception for invalid user input (the previous
    # tests in ``run_correctness.py::test_csa_hca_input_validation`` expect
    # (they now catch ``ValueError``; the test was updated to accept both ValueError and AssertionError for backward compatibility with any external callers that may still catch AssertionError).
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
    # ``c_I`` (indexer key dim) and ``nIh`` (indexer head count) MUST be
    # validated here, not inside ``csa_lightning_indexer``: the explicit
    # ``scale=c_I ** -0.5`` below raises
    # ``ZeroDivisionError: 0.0 cannot be raised to a negative power`` when
    # c_I == 0, and ``nIh == 0`` makes the indexer produce an all-zero
    # ``logits`` tensor (the ``sum(1)`` over an empty head dim is 0), so
    # top-k silently selects the first k blocks for every query — a
    # meaningless result with no diagnostic. Reject both up-front.
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
    # (see ``F.normalize`` calls below), their dot product is already a cosine
    # similarity in ``[-1, 1]``. The previous default ``scale = c ** -0.5``
    # (e.g. 0.125 for c=64) further shrinks the scores into ``[-0.125, 0.125]``,
    # which makes softmax over the selected blocks nearly uniform — effectively
    # turning sparse retrieval into average pooling and defeating the purpose
    # of CSA's learned compression.
    #
    # Standard cosine/cosFormer-style attention uses ``softmax(q·k / τ)`` with
    # ``τ = 1`` (or a learnable temperature). The extra ``1/sqrt(c)`` was a
    # leftover from the un-normalized softmax-attention formula and has been
    # removed. If a caller explicitly passes ``scale=``, we honor it (backward
    # compatibility for any external code that depended on the old behaviour).
    if scale is None:
        scale = 1.0
    device = H.device
    # Degenerate case: empty sequence. Without this guard the downstream
    # ``csa_compress_kv_overlapped`` would raise a cryptic broadcasting
    # error (``Expected size 0 but got size 1``) because n_blocks=0 makes
    # the [B, n_blocks, m, c] reshape collapse against [m, c] positional
    # bias. Return a zero-shaped output matching the contract.
    if T == 0:
        return torch.zeros(B_, 0, nh * c, dtype=H.dtype, device=device)
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
    # P0 API fix: use F.linear (computes H @ W.T) with W in nn.Linear.weight
    # layout [out, in] — avoids the non-contiguous .weight.T view that the
    # previous ``H @ W_aKV`` form required at every call site.
    #
    # Issue 2.3 fix: previously this block did 6 SEPARATE ``F.linear(H, W_x)``
    # calls — ``W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx`` — each launching
    # its own GEMM kernel. At ``B=1, T=2048, d=512`` this is 6 kernel launches
    # and 6 separate ``[B, T, d] · [d, out]`` matmuls where a single fused
    # matmul would do the same work in one launch with better arithmetic
    # intensity (the concatenated weight tensor has more rows, giving cuBLAS
    # a larger M dimension to tile over).
    #
    # We concatenate all 6 weights along their out_features axis (dim=0) and
    # do ONE ``F.linear`` call, then ``tensor.split`` the result back into
    # the 6 named projections. The split is a zero-copy view (``split``
    # returns views into the concatenated tensor), so the only additional
    # cost is the one-time ``torch.cat`` on the weight tensors. We do NOT
    # cache the concatenated weight because (a) the optimizer may update
    # the individual weights in-place between forward passes, invalidating
    # any cache, and (b) the cat cost on small weight tensors
    # (e.g. ``6 × [64, 512] = 196K elements``) is <<1ms — negligible
    # relative to the matmul cost.
    #
    # Numerical contract: IDENTICAL to the 6 separate calls. ``F.linear`` is
    # ``x @ W.T``; concatenating the W's along dim=0 (out_features) and then
    # slicing the result along dim=-1 is mathematically identical to doing
    # each matmul independently. Bit-identical up to GEMM kernel reduction
    # order (which may differ slightly when the kernel sees a larger M,
    # but well below the fp32 tolerance enforced by ``run_correctness.py``).
    #
    # Autograd contract CHANGE (intentional): under ``use_ste=False`` the
    # indexer weights ``W_KV_idx`` and ``W_Z_idx`` previously received
    # ``None`` grad (because the indexer's top-k selection didn't propagate
    # gradient, and the weights weren't tied to any other path). With the
    # merged matmul they now receive a ZERO (but non-None) grad, because
    # autograd flows backward through the single matmul into all 6 weight
    # slices. The substantive contract — "indexer params don't learn
    # without STE" — is preserved: the grad is zero, so the optimizer
    # step is a no-op. ``test_csa_indexer_ste_gradient`` was updated to
    # check for "no useful gradient" (None OR all-zero) instead of
    # requiring None specifically.
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
        # ``torch.split`` returns views into ``combined_out`` — no copy.
        # split_sizes MUST match the order of weights in ``torch.cat`` above.
        Ca, Cb, Za, Zb, K_idx_raw, Z_idx = combined_out.split(
            [c, c, c, c, c_I, c_I], dim=-1,
        )
    else:
        # Review-fix 1.3: ``fuse_projections=False`` restores the 6
        # separate ``F.linear`` calls (the pre-issue-2.3 behaviour). The
        # forward VALUE is identical to the fused path (up to GEMM
        # reduction-order noise, well below the fp32 test tolerance), but
        # the BACKWARD path differs when ``use_ste=False``: here,
        # ``W_KV_idx``/``W_Z_idx`` are consumed ONLY by the (non-
        # differentiable-under-no-STE) indexer path, so they receive
        # ``None`` gradient instead of the fused path's zero-but-non-None
        # gradient. See the "review-fix 1.3" docstring warning above for
        # the full rationale and ``test_csa_fuse_projections_grad_contract``
        # in run_correctness.py for the regression test.
        Ca = F.linear(H, W_aKV)
        Cb = F.linear(H, W_bKV)
        Za = F.linear(H, W_aZ)
        Zb = F.linear(H, W_bZ)
        K_idx_raw = F.linear(H, W_KV_idx)
        Z_idx = F.linear(H, W_Z_idx)
    C_comp = csa_compress_kv_overlapped(Ca, Cb, Za, Zb, Ba, Bb, m)   # [B, n_blocks, c]

    # --- 2. Lightning indexer ---
    # compressed indexer keys via the same compression (single-branch here for simplicity)
    # K_idx_raw and Z_idx were computed in the merged matmul above.
    K_IComp = csa_compress_kv(K_idx_raw, Z_idx, B_idx, m)            # [B, n_blocks, c_I]
    # indexer queries (low-rank)
    cQ = F.linear(H, W_DQ)                                        # [B, T, dc]
    q_idx = F.linear(cQ, W_IUQ).view(B_, T, nIh, c_I)              # [B, T, nIh, c_I]
    w_idx = F.linear(H, W_w)                                      # [B, T, nIh]
    cbm = _causal_block_mask(T, n_blocks, m, device)
    # The lightning indexer scores are dot products over DI = c_I (not c),
    # so the correct scale is c_I ** -0.5 (per the DeepSeek-V4 paper Eq. 15:
    # score = ReLU(q_idx . K_idx / sqrt(DI))). We previously passed the
    # outer ``scale`` (defaulting to c ** -0.5), which is the correct scale
    # for the sparse MQA core (dot product over c) but NOT for the indexer.
    #
    # Note: this does not change the top-k selection (ReLU is positively
    # homogeneous, so scaling all scores by a positive constant preserves
    # their relative ordering), but it makes the code match the documented
    # formula and ensures the logits have the intended magnitude if they
    # are ever exposed for downstream use (e.g. learnable temperature).
    # Inference / benchmark fix: STE is a TRAINING-only surrogate for the
    # non-differentiable top-k indexer. It is forward-equivalent to the hard
    # gather but costs an extra full softmax + soft gather. Under
    # ``torch.no_grad()`` there is no backward path to preserve, so computing
    # the STE surrogate only inflates inference latency (Exp2 / Exp6). Gate it
    # on ``torch.is_grad_enabled()`` so training keeps the default behaviour,
    # while no-grad inference uses the lean hard-index path.
    effective_use_ste = use_ste and torch.is_grad_enabled()
    indices = csa_lightning_indexer(q_idx, K_IComp, w_idx, topk,
                                    scale=c_I ** -0.5,
                                    causal_block_mask=cbm,
                                    return_soft_weights=effective_use_ste,
                                    ste_mode=ste_mode,
                                    normalize_qk=normalize_qk)     # [B, T, topk]
    # P0-4 fix: when STE is active, the indexer also returns a differentiable
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
    q = F.normalize(q, dim=-1)
    C_comp_n = F.normalize(C_comp, dim=-1)                         # L2-normalize (cosine-similarity attention)

    # Handle topk=0 (degenerate but valid: caller asks for no sparse
    # selection). Without this guard the downstream ``scores.amax(-1)``
    # raises ``IndexError: Expected reduction dim -1 to have non-zero size``
    # because ``scores`` would have shape ``[B, T, nh, 0]``. With topk=0
    # the sparse branch contributes exactly zero; only the SW branch (if
    # enabled) produces non-zero output. We still need ``q`` (for the SW
    # branch) so the early return is placed AFTER q is computed.
    if indices.shape[-1] == 0:
        out = torch.zeros(B_, T, nh, c, dtype=compute_dtype, device=device)
    else:
        # --- Vectorized sparse MQA core attention ---
        # Gather selected compressed KV entries for every (b, t) in one shot.
        # indices: [B, T, topk], padded with -1 for invalid slots.
        valid_mask = indices >= 0                                        # [B, T, topk]
        idx_safe = indices.clamp(min=0)                                  # [B, T, topk]
        batch_idx = torch.arange(B_, device=device).view(B_, 1, 1)      # [B, 1, 1]
        kv = C_comp_n[batch_idx, idx_safe]                               # [B, T, topk, c]

        # P0-4 fix — straight-through estimator (STE) for the indexer.
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
        # P0-2 fix — implement ``ste_mode`` as two genuinely distinct
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
        # as documented in the ``ste_mode`` docstring above. A regression
        # test in ``run_correctness.py::test_csa_indexer_ste_full_softmax``
        # verifies both the forward-equivalence and the backward-distinctness
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
            soft_kv = soft_weights_selected.unsqueeze(-1) * kv       # [B, T, topk, c]

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
            # P0-5 fix: the previous form ``kv = soft_kv + (kv - soft_kv).detach()``
            # is forward-equivalent to ``kv`` (hard gather) but in backward the
            # detach() ERASES the direct gradient from the hard gather path,
            # leaving only ``soft_weights * d_out`` as the gradient to the
            # selected ``C_comp_n`` entries — i.e. the selected KV receives a
            # gradient shrunk by the soft probability (often < 1). The standard
            # STE identity is the other way around:
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
            # STE bias and is benign for training stability. This restores the
            # contract documented in the ``ste_mode`` docstring: selected KV
            # entries receive the SAME gradient they would under
            # ``use_ste=False``, plus an indexer-learning signal.
            kv = kv + (soft_kv - soft_kv.detach())

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
            #
            # Math: we want p_i = exp(s_i) / (sum_j exp(s_j) + exp(sink)).
            # For numerical stability we subtract row_max from every score
            # (and from the sink!) so the largest exp() argument is 0:
            #   p_i = exp(s_i - M) / (sum_j exp(s_j - M) + exp(sink - M))
            # where M = max(0, max_i s_i).  The previous code forgot to
            # shift the sink, producing
            #   p_i = exp(s_i) / (sum_j exp(s_j) + exp(sink) * exp(M))
            # i.e. the sink was over-weighted by a factor exp(M) — a
            # systematic ~13% bias in the default c=64 config and up to 65%
            # at c=4.  Shifting log_sink by -row_max restores the identity
            #   logaddexp(a - M, b - M) = logaddexp(a, b) - M
            # so the shifted computation is mathematically identical to the
            # unshifted (overflow-prone) one.
            log_sink = sink_logits.view(1, 1, nh, 1).to(scores)  # [1, 1, nh, 1]
            vmask = valid_mask[:, :, None, :].to(scores.dtype)          # [B, T, 1, topk]
            # NOTE: renamed from `m` to `row_max` to avoid shadowing the
            # `m` parameter (compression factor). The previous `m = ...`
            # silently clobbered the compression factor for the rest of the
            # function; it happened not to be read again here, but the
            # shadowing was a latent footgun for future edits.
            row_max = scores.amax(-1, keepdim=True).clamp(min=0)        # [B, T, nh, 1]
            shifted = scores - row_max                                  # [B, T, nh, topk]
            shifted_sink = log_sink - row_max                           # [B, T, nh, 1]
            log_sum_exp = torch.logsumexp(shifted, dim=-1, keepdim=True)
            log_denom = torch.logaddexp(log_sum_exp, shifted_sink)      # [B, T, nh, 1]
            p = ((shifted - log_denom).exp() * vmask)                   # [B, T, nh, topk]
            # NaN guard for all-masked rows (early queries with no preceding
            # causal block). When every slot in a row is -inf, log_sum_exp =
            # -inf, log_denom = logaddexp(-inf, log_sink) = log_sink. If
            # log_sink is also -inf (e.g. sink_logits diverged to -inf during
            # training), then (shifted - log_denom) = (-inf - (-inf)) = NaN,
            # and NaN * 0 (vmask) = NaN in IEEE 754. Zero out any row where
            # all slots are invalid so the downstream einsum produces 0
            # instead of NaN. This mirrors the all_masked guard in the
            # ``else`` branch and in ``ops_hca.py::naive_hca``.
            #
            # Shape: valid_mask is [B, T, topk]; we reduce over topk to get
            # [B, T, 1], then add a head axis [:, :, None] to broadcast over
            # the nh dimension of p ([B, T, nh, topk]).
            all_invalid = ~valid_mask.any(-1, keepdim=True)[:, :, None]  # [B, T, 1, 1]
            p = p.masked_fill(all_invalid, 0.0)
        else:
            # NaN-safe softmax: rows that are entirely -inf (e.g. early
            # queries with no preceding causal block, or all-topk slots
            # padded with -1) yield all-zero p. We use the same explicit
            # all_masked guard as ops_hca.py::naive_hca and the sink branch
            # above for consistency: detect fully-masked rows, replace their
            # -inf entries with 0 so softmax is finite, then zero the result.
            #
            # The previous implementation used a clamp(min=1e-20) trick on
            # the denominator, which also produces p=0 for all-masked rows
            # but relies on a magic epsilon. The explicit guard is clearer
            # and avoids any theoretical concern about the epsilon being
            # too small for fp16/bf16 inputs (where 1e-20 underflows to 0).
            all_invalid = ~valid_mask.any(-1, keepdim=True)[:, :, None]  # [B, T, 1, 1]
            # C5 fix: delegate to shared _nan_safe_softmax helper. We pass
            # the precomputed all_masked mask so we don't recompute it
            # (the indexer-style ``logits.isinf().all`` would not match
            # this branch's ``valid_mask``-based definition of "masked").
            p = _nan_safe_softmax(scores, dim=-1, all_masked_mask=all_invalid)

        out = torch.einsum('b t h k, b t k d -> b t h d', p, kv)        # [B, T, nh, c] in compute_dtype

    # optional sliding window branch (local uncompressed KV)
    if sliding_window > 0:
        win = sliding_window
        # Precompute F.linear(H, W_aKV) once (reuse Ca from §1) instead of
        # redoing the matmul per (b, t).
        H_proj = Ca  # [B, T, c], already == F.linear(H, W_aKV)
        # P5 fix — TRUE O(T·win) sliding-window attention (was O(T²)).
        #
        # The previous implementation built a full ``[T, T]`` boolean mask and
        # a full ``[B, nh, T, T]`` attention-scores tensor, then masked every
        # entry outside the window to ``-inf`` before softmax. Even though
        # only ``win`` entries per row were non-trivial, the dense matmul
        # (``einsum('bthd,bnd->bhtn')``) and the dense softmax both did
        # ``O(T²·nh·c)`` work — the window size ``win`` had NO effect on the
        # compute cost, only on which entries survived the mask. This means
        # the "sliding window" branch was NOT actually achieving the
        # ``O(T·window)`` complexity that is the whole point of a local
        # attention mechanism; at ``T=2048`` it allocated and filled a
        # 4M-entry scores tensor per call regardless of ``win``.
        #
        # The P5 fix used a **banded / windowed-gather** approach:
        #   1. Left-pad ``C_local`` with ``win-1`` zero columns so that the
        #      window for query ``t`` can be extracted as a contiguous slice.
        #   2. Use ``unfold`` to extract per-query windows of shape
        #      ``[B, T, win, c]`` in O(T·win·c) time.
        #   3. Compute attention scores ONLY over the ``win`` entries:
        #      ``[B, T, nh, win]`` — O(T·win·nh·c).
        #   4. Mask the left-padding entries (queries near the start of the
        #      sequence whose window extends before position 0) to ``-inf``,
        #      softmax, and weighted-sum over the ``win`` dimension.
        #
        # Issue 2.2 fix: the ``unfold`` call materialized a single ``[B, T,
        # win, c]`` tensor — fine for ``T≤4k, win≤512`` (≤4M elements) but
        # blowing up at ``T=64k, win=2k`` (≈8B elements / 32 GB). We now
        # delegate to ``_sliding_window_attention`` which auto-engages a
        # chunked path when ``T * win * c`` exceeds ``CHUNKED_SW_THRESHOLD``
        # (default 8M elements), keeping peak memory at
        # ``O(chunk_t · win · c)``. For the small-T / small-win case the
        # helper still uses ``unfold`` on a single slice — same vectorized
        # fast path as before.
        #
        # The result is numerically identical to the old dense+mask approach
        # (verified by ``test_hca_sliding_window_causality`` /
        # ``test_csa_full_pipeline_causality`` in run_correctness.py) because
        # softmax over the ``win`` non-masked entries of a row is the same
        # operation whether the masked entries are materialized as ``-inf``
        # in a ``[T,T]`` tensor or simply absent from a ``[T,win]`` tensor.
        #
        # q is already L2-normalized above (line: q = F.normalize(q, dim=-1)),
        # so the sliding-window branch reuses it directly.
        #
        # Dtype: cast H_proj to compute_dtype so the SW branch matches the
        # sparse branch's precision. Without this, the SW softmax runs in
        # H.dtype (e.g. fp16) while the sparse softmax ran in compute_dtype
        # (fp32) — an asymmetric precision loss that silently degrades the
        # SW branch's contribution for fp16 inputs.
        C_local = F.normalize(H_proj.to(compute_dtype), dim=-1)  # [B, T, c]
        sw_out = _sliding_window_attention(q, C_local, win, scale, device)
        out = out + sw_out

    # Return the raw per-head core-attention output [B, T, nh, c] flattened to
    # [B, T, nh*c]; the caller performs the grouped output projection.
    # Cast to H.dtype at the very end so all intermediate computation benefits
    # from compute_dtype precision (previously the sparse branch output was
    # cast to H.dtype *before* the SW branch, losing precision unnecessarily).
    # Trim the padded SUFFIX off the SEQUENCE axis (dim=1) so the output
    # matches the input's original T (right-padding added zeros at the end,
    # which never affect real-token outputs thanks to the causal block mask).
    return out.reshape(B_, T, nh * c).to(H.dtype)[:, :original_T]
