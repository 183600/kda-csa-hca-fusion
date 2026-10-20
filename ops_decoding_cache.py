"""Incremental decoding caches for CSA and HCA — naive PyTorch reference.

This module implements the *incremental decoding cache* that lets
``naive_csa`` / ``naive_hca`` (and the corresponding ``CSAHybridLayer`` /
``HCAHybridLayer``) participate in token-by-token autoregressive decoding,
closing the scope gap documented in the README's "Fairness notes" #4 and
the inline warning at the top of ``run_decoding.py``.

The cache has three pieces of state, mirroring the spec in the issue:

  1. **Partial block accumulator** (Python ``list``). The compression
     factors ``m`` (CSA) / ``m2`` (HCA) require ``m`` consecutive KV
     entries before a compressed block can be produced. During decoding
     we receive one token at a time, so we accumulate the per-token
     projections here until we have ``m`` of them, then call
     ``csa_compress_kv(_overlapped)`` to produce ONE new compressed block
     and push it into the compressed block cache (item 2). The
     accumulator is then cleared (and, for CSA, the just-consumed ``Cb``
     and ``Zb`` are stashed as "previous block" state for the overlapped
     compression of the NEXT block).
  2. **Compressed block cache** (``torch.Tensor`` ``[B, n_blocks, c]``).
     Grows by one row every time the accumulator fills. For CSA this also
     includes a parallel **indexer key cache** ``[B, n_blocks, c_I]`` so
     the lightning indexer can score the new block against the current
     query.
  3. **Sliding-window ring buffer** (``torch.Tensor`` ``[B, win, c]``).
     A fixed-size FIFO of the most recent ``win`` *uncompressed* local
     keys (the ``Ca`` projection for CSA, ``C`` for HCA, normalized with
     the cache's ``qk_norm_mode`` — 'l2' by default).
     Writes wrap around; reads return the contents in causal order
     (oldest first). This backs the sliding-window branch of
     ``naive_csa`` / ``naive_hca`` during decoding.

Numerical contract
------------------
For a sequence of ``T`` tokens fed ONE AT A TIME through ``append_step``,
the per-token outputs produced by :meth:`CSADecodingCache.forward_step`
/ :meth:`HCADecodingCache.forward_step` are **bit-identical (to fp32
tolerance) to the corresponding rows of ``naive_csa`` / ``naive_hca``
called once on the full ``[B, T, d]`` sequence** — including when the
CSA indexer's scores tie (see "Deterministic tie-breaking" below).

The bit-equivalence holds because:

  * The compressed block cache stores EXACTLY the ``C_comp`` tensor that
    ``naive_csa`` would have computed from the full sequence (the
    overlapped compression is associative across block boundaries: the
    softmax over ``[Za_block_i ; Zb_block_{i-1}]`` is the same whether
    we compute it incrementally or all-at-once, since each block's
    softmax is independent — see :func:`_csa_compress_kv_overlapped_single`).
  * The sliding-window ring buffer stores EXACTLY the same ``Ca`` /
    ``C`` entries that ``naive_csa`` / ``naive_hca`` would have used for
    the SW branch. The per-query window in the naive path is
    ``[max(0, t-win+1), t]``; the ring buffer holds exactly these
    ``min(win, t+1)`` entries in causal order, so the softmax is over
    the same set of keys.
  * The indexer's top-k selection is over the same set of compressed
    blocks (the paper-exact causal block mask ``b < t // m`` admits only
    STRICTLY PRECEDING blocks; the block containing ``t`` — partial or
    just-completed — remains unavailable through the sparse branch and is
    served by the sliding-window branch instead).

Deterministic tie-breaking
--------------------------
The hard selection is computed by ``ops_csa._canonical_topk_indices``,
which breaks equal indexer scores on ASCENDING block index instead of
delegating to a bare ``torch.topk`` (whose tie choice depends on the
tensor's size). The full path scores ``n_blocks_full = T_padded // m``
entries (including ``-inf``-masked future blocks) while the incremental
path scores only the ``n_blocks_inc`` completed blocks, so with tied
ReLU scores (e.g. dot products saturating at 0) only a canonical
selector keeps the two paths on the same equal-valued blocks: the
selected SET is a pure function of the score vector, so prefill and
incremental decode agree index-for-index even under ties.
``run_correctness.py`` pins this with
``test_csa_indexer_canonical_tiebreak`` and
``test_csa_decoding_cache_sparse_topk_ties``.

What this is NOT
----------------
This is a *correctness-first* reference, not a production kernel. The
``forward_step`` path uses Python-level control flow (append-to-list,
conditional compress, top-k gather). It is suitable for:

  * Closing the Exp 6 scope gap (CSA / HCA decode vs softmax /
    KDA, on the same Kaggle T4).
  * Correctness verification (incremental == full-prefill).
  * Small-scale autoregressive generation (a few hundred tokens).

For production decoding, the same data structures would be implemented
in CUDA / Triton with fused matmuls; the Python reference here is the
spec they would be validated against.
"""

from __future__ import annotations

import torch

from ops_cache import normal_tensor_scope
from ops_rope import effective_rope_dim, partial_rope
from ops_csa import (csa_compress_kv, _csa_compress_kv_overlapped_prev,
                     csa_lightning_indexer, _qk_normalize, _qk_normalize_)


def _validate_cache_dtype(dtype) -> None:
    """Reject integer/bool cache storage before normalize or matmul paths."""
    try:
        is_float = torch.empty((), dtype=dtype).is_floating_point()
    except (TypeError, RuntimeError) as exc:
        raise TypeError(f'cache dtype must be a floating torch.dtype, got {dtype!r}') from exc
    if not is_float:
        raise TypeError(f'cache dtype must be floating-point, got {dtype!r}')


def _geometric_capacity(n_rows: int) -> int:
    """Round ``n_rows`` up to the next power of two (0 stays 0).

    This is the single growth step used by every compressed-cache
    allocation so the RETAINED capacity depends only on the number of
    completed blocks, not on the fill history (one bulk prefill append vs
    token-by-token appends). ``run_kv_cache.geometric_capacity`` is the
    analytical twin used by the Exp 3 accounting; keeping the two
    formulas identical is what makes the live-vs-analytic audit
    (``cache_accounting``) exact instead of history-dependent.
    """
    if not isinstance(n_rows, int) or isinstance(n_rows, bool):
        raise TypeError(
            f'_geometric_capacity: n_rows must be an int, got {n_rows!r}')
    if n_rows < 0:
        raise ValueError(
            f'_geometric_capacity: n_rows must be >= 0, got {n_rows}')
    if n_rows == 0:
        return 0
    return 1 << (n_rows - 1).bit_length()


# =============================================================================
# Internal helpers
# =============================================================================

def _tensor_nbytes(t: torch.Tensor | None) -> int:
    """Bytes retained by a tensor (0 for ``None``)."""
    return 0 if t is None else int(t.numel() * t.element_size())


def _valid_nbytes(t: torch.Tensor | None, valid_elems: int) -> int:
    """Bytes of the logically live prefix of a capacity-grown tensor."""
    return 0 if t is None else int(valid_elems) * t.element_size()


def _to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``x.to(dtype)``, returning ``x`` unchanged when the dtype matches.

    ``Tensor.to`` returns the SAME tensor when the dtype already matches,
    so the guard is value-identical.
    """
    if x.dtype == dtype:
        return x
    return x.to(dtype)


def _to_cache(x: torch.Tensor, dtype: torch.dtype, device) -> torch.Tensor:
    """``x.to(dtype=dtype, device=device)``, returning ``x`` when it matches.

    Value-identical to the unconditional call (a matching dtype/device
    ``.to`` returns ``x`` itself).
    """
    if x.dtype == dtype and x.device == device:
        return x
    if torch.is_inference_mode_enabled():
        # A cache tensor created under ``torch.inference_mode`` cannot be
        # mutated later outside inference mode. Convert in a normal scope so
        # the retained cache stays ordinary (see ``normal_tensor_scope``).
        with normal_tensor_scope(), torch.no_grad():
            return x.to(dtype=dtype, device=device)
    return x.to(dtype=dtype, device=device)


# Shape-keyed memo for the ``[n_blocks]`` causal block mask
# ``b < b_threshold`` (paper Eq. 16). Depends only on
# ``(n_blocks, b_threshold, device)`` and is read-only at every use site
# (``~cbm`` / ``cbm[None, :]`` produce views/copies, and
# ``csa_lightning_indexer`` negates without mutating), so the same storage
# can be shared across decode steps.
_BLOCK_VALID_MEMO: dict = {}

# Shape-keyed memo for all-True boolean masks (the sliding-window validity
# block of the joint-softmax mask). Read-only at every use site.
_BOOL_ONES_MEMO: dict = {}

# Per-device single-element int64 scratch used for the position arguments
# of ``partial_rope``. Only reused outside autograd: when grad is enabled
# the graph may retain the position tensor, so a fresh tensor is returned.
_POS_SCRATCH: dict = {}


def _block_valid_mask(n_blocks: int, b_threshold: int,
                      device) -> torch.Tensor:
    """Memoized ``arange(n_blocks, device=device) < b_threshold``."""
    key = (n_blocks, b_threshold, str(device))
    mask = _BLOCK_VALID_MEMO.get(key)
    if mask is None:
        with normal_tensor_scope():
            mask = torch.arange(n_blocks, device=device) < b_threshold
        if len(_BLOCK_VALID_MEMO) >= 128:
            _BLOCK_VALID_MEMO.clear()
        _BLOCK_VALID_MEMO[key] = mask
    return mask


def _bool_ones(B: int, n: int, device) -> torch.Tensor:
    """Memoized ``torch.ones(B, 1, 1, n, dtype=torch.bool, device=device)``."""
    key = (B, n, str(device))
    mask = _BOOL_ONES_MEMO.get(key)
    if mask is None:
        with normal_tensor_scope():
            mask = torch.ones(B, 1, 1, n, dtype=torch.bool, device=device)
        if len(_BOOL_ONES_MEMO) >= 64:
            _BOOL_ONES_MEMO.clear()
        _BOOL_ONES_MEMO[key] = mask
    return mask


def _pos_tensor(value: int, device) -> torch.Tensor:
    """``torch.tensor([value], device=device)``, with a no-grad scratch.

    The scratch is overwritten (``fill_``) immediately before every use and
    is only handed to read-only consumers, so the values seen are identical
    to a fresh tensor. Outside autograd nothing can retain it; under grad a
    fresh tensor is returned so a later ``fill_`` can never rewrite a value
    kept alive by the graph.
    """
    if torch.is_grad_enabled():
        return torch.tensor([value], device=device)
    key = str(device)
    buf = _POS_SCRATCH.get(key)
    if buf is None:
        if len(_POS_SCRATCH) >= 8:
            _POS_SCRATCH.clear()
        # Ordinary tensor: the scratch outlives the inference-mode call that
        # created it, so a later ``no_grad`` call must be allowed to write it.
        with normal_tensor_scope():
            buf = torch.zeros(1, dtype=torch.long, device=device)
        _POS_SCRATCH[key] = buf
    buf.fill_(value)
    return buf


def _chunk_window_keys(keys: torch.Tensor, n_pre: int,
                       win: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query sliding windows for a whole continuation chunk.

    ``keys`` is the causal-order concatenation
    ``[pre-chunk SW tail ; chunk keys]`` of shape ``[B, n_pre + T, c]``.
    Query ``i`` of the chunk sits at concatenated index ``n_pre + i`` and
    sees the last ``win`` entries ending there (the same window
    ``_SlidingWindowRingBuffer.get()`` exposes after appending that
    query). Returns ``(windows, valid)`` with shapes
    ``[B, T, win, c]`` and ``[T, win]``. Left-padding (fewer than ``win``
    entries available) is zero-filled and marked invalid, so the joint
    softmax masks it exactly like the per-token path's absent entries.
    """
    if keys.dim() != 3:
        raise ValueError(
            f'_chunk_window_keys: keys must be [B, N, c], got '
            f'{tuple(keys.shape)}')
    if win < 0:
        raise ValueError(f'_chunk_window_keys: win={win} must be >= 0')
    total = keys.shape[1]
    T = total - n_pre
    if T < 0:
        raise ValueError(
            f'_chunk_window_keys: n_pre={n_pre} exceeds keys length {total}')
    B, _, c = keys.shape
    if win == 0 or T == 0:
        return (keys.new_zeros(B, T, 0, c),
                torch.zeros(T, 0, dtype=torch.bool, device=keys.device))
    ends = n_pre + torch.arange(T, device=keys.device)       # [T]
    starts = (ends - win + 1).clamp(min=0)                   # [T]
    offsets = torch.arange(win, device=keys.device)          # [win]
    idx = starts[:, None] + offsets[None, :]                 # [T, win]
    valid = idx <= ends[:, None]
    idx = idx.clamp(max=max(total - 1, 0))
    flat = idx.reshape(-1)
    windows = keys[:, flat, :].reshape(B, T, win, c)
    return windows, valid


def _joint_softmax_chunk(logits: torch.Tensor, valid: torch.Tensor,
                         sink_logits: torch.Tensor | None) -> torch.Tensor:
    """Stable joint softmax over ``[compressed ; window]`` for a chunk.

    Mirrors the per-token ``forward_step`` chain: shift by
    ``max(0, row_max)`` so the largest exponent argument is <= 0, keep
    the sink in log space (denominator only), and zero all-invalid rows.
    ``valid`` broadcasts to ``logits``'s shape.
    """
    masked = logits.masked_fill(~valid, float('-inf'))
    all_invalid = ~valid.any(-1, keepdim=True)
    if sink_logits is not None:
        nh = masked.shape[2]
        log_sink = sink_logits.view(1, 1, nh, 1).to(masked)
        row_max = masked.amax(-1, keepdim=True).clamp(min=0)
        shifted = masked - row_max
        shifted_sink = log_sink - row_max
        lse = torch.logsumexp(shifted, dim=-1, keepdim=True)
        log_denom = torch.logaddexp(lse, shifted_sink)
        p = (shifted - log_denom).exp()
        return p.masked_fill(all_invalid, 0.0)
    safe = masked.masked_fill(all_invalid, 0.0)
    p = torch.softmax(safe, dim=-1)
    return p.masked_fill(all_invalid, 0.0)


def _csa_compress_kv_overlapped_single(
    Ca_new: torch.Tensor,   # [B, m, c]
    Cb_new: torch.Tensor,   # [B, m, c]
    Za_new: torch.Tensor,   # [B, m, c]
    Zb_new: torch.Tensor,   # [B, m, c]
    Cb_prev: torch.Tensor | None,  # [B, m, c] or None for the first block
    Zb_prev: torch.Tensor | None,  # [B, m, c] or None
    Ba: torch.Tensor,       # [m, c]
    Bb: torch.Tensor,       # [m, c]
) -> torch.Tensor:
    """Two-branch overlapped compression for ONE new block of ``m`` tokens.

    Mirrors :func:`ops_csa.csa_compress_kv_overlapped` but operates on a
    single block with the *previous* block's ``Cb`` / ``Zb`` supplied
    explicitly (rather than being computed internally via a shift-and-pad
    on a multi-block tensor). This is the incremental-decoding analogue
    of the batch function: when the partial-token accumulator reaches
    ``m`` tokens, we have one new block to compress and need to combine
    it with the previous block's ``Cb`` / ``Zb`` (stashed in the cache
    state) to honor the overlap.

    Args:
        Ca_new, Cb_new, Za_new, Zb_new: ``[B, m, c]`` — the ``m`` tokens
            of the new block, projected through ``W_aKV`` / ``W_bKV`` /
            ``W_aZ`` / ``W_bZ``.
        Cb_prev, Zb_prev: ``[B, m, c]`` or ``None``. The PREVIOUS
            block's ``Cb`` / ``Zb``. ``None`` for the very first block
            (no overlap partner); the function then uses ``-inf`` pad
            for the b-branch logits so ``Sb = 0`` and the result
            reduces to single-branch ``csa_compress_kv`` with ``Ba``
            (mathematically identical to passing the first block
            through ``csa_compress_kv_overlapped`` as block 0 of a
            multi-block input).
        Ba, Bb: ``[m, c]`` learnable positional biases.

    Returns:
        ``[B, c]`` — one compressed KV entry for the new block.
    """
    B, m, c = Ca_new.shape
    compute_dtype = torch.float64 if Ca_new.dtype == torch.float64 else torch.float
    Ca_new = _to_dtype(Ca_new, compute_dtype)
    Cb_new = _to_dtype(Cb_new, compute_dtype)
    Za_new = _to_dtype(Za_new, compute_dtype)
    Zb_new = _to_dtype(Zb_new, compute_dtype)
    A_logits = Za_new + _to_cache(
        Ba[None, :, :], Za_new.dtype, Za_new.device)            # [B, m, c]
    if Cb_prev is None:
        # First block: no overlap partner. Pad b-branch logits with -inf
        # so softmax assigns 0 weight to the (non-existent) previous
        # block. Cb_prev contribution is zero (sum of 0 * anything).
        Bb_logits = torch.full(
            (B, m, c), float('-inf'),
            device=Ca_new.device, dtype=compute_dtype,
        )
        Cb_prev_eff = torch.zeros_like(Cb_new)
    else:
        Cb_prev = _to_dtype(Cb_prev, compute_dtype)
        Zb_prev = _to_dtype(Zb_prev, compute_dtype)
        Bb_logits = Zb_prev + _to_cache(
            Bb[None, :, :], Zb_prev.dtype, Zb_prev.device)      # [B, m, c]
        Cb_prev_eff = Cb_prev
    # Joint softmax over the concatenated 2m entries (matches the batch
    # function's ``torch.cat([A_logits, Bb_logits], dim=2)`` then
    # ``softmax(dim=2)``).
    all_logits = torch.cat([A_logits, Bb_logits], dim=1)        # [B, 2m, c]
    S = torch.softmax(all_logits, dim=1)                        # [B, 2m, c]
    Sa = S[:, :m]                                               # [B, m, c]
    Sb = S[:, m:]                                               # [B, m, c]
    if torch.is_grad_enabled():
        out = (Sa * Ca_new).sum(1) + (Sb * Cb_prev_eff).sum(1)  # [B, c]
    else:
        # ``S`` is fresh and its two halves have no other readers; the
        # second sum is added in place into the fresh first sum (same
        # addition, one fewer [B, c] temporary).
        out = Sa.mul_(Ca_new).sum(1)
        out.add_(Sb.mul_(Cb_prev_eff).sum(1))
    return out


def _hca_compress_kv_single(
    C_new: torch.Tensor,    # [B, m2, c]
    Z_new: torch.Tensor,    # [B, m2, c]
    B_pos: torch.Tensor,    # [m2, c]
) -> torch.Tensor:
    """Single-branch KV compression for ONE new block of ``m2`` tokens.

    Mirrors :func:`ops_csa.csa_compress_kv` (which is what HCA uses
    internally) but operates on a single block — the incremental-decoding
    analogue. HCA has no overlap, so this is a straightforward softmax-
    weighted sum of the ``m2`` tokens.

    Returns:
        ``[B, c]`` — one compressed KV entry.
    """
    B, m2, c = C_new.shape
    compute_dtype = torch.float64 if C_new.dtype == torch.float64 else torch.float
    C_new = _to_dtype(C_new, compute_dtype)
    Z_new = _to_dtype(Z_new, compute_dtype)
    logits = Z_new + _to_cache(
        B_pos[None, :, :], Z_new.dtype, Z_new.device)           # [B, m2, c]
    S = torch.softmax(logits, dim=1)                            # [B, m2, c]
    return (S * C_new).sum(1)                                   # [B, c]


def _indexer_compress_single(
    K_idx_new: torch.Tensor,  # [B, m, c_I]
    Z_idx_new: torch.Tensor,  # [B, m, c_I]
    B_idx: torch.Tensor,      # [m, c_I]
) -> torch.Tensor:
    """Single-branch compression of the indexer keys for ONE new block.

    Mirrors the ``csa_compress_kv`` call on indexer keys inside
    ``naive_csa`` (``K_IComp = csa_compress_kv(K_idx_raw, Z_idx, B_idx, m)``)
    but operates on a single block.

    Returns:
        ``[B, c_I]`` — one compressed indexer key.
    """
    B, m, c_I = K_idx_new.shape
    compute_dtype = (
        torch.float64 if K_idx_new.dtype == torch.float64 else torch.float
    )
    K_idx_new = _to_dtype(K_idx_new, compute_dtype)
    Z_idx_new = _to_dtype(Z_idx_new, compute_dtype)
    logits = Z_idx_new + _to_cache(
        B_idx[None, :, :], Z_idx_new.dtype, Z_idx_new.device)   # [B, m, c_I]
    S = torch.softmax(logits, dim=1)                            # [B, m, c_I]
    return (S * K_idx_new).sum(1)                               # [B, c_I]


# =============================================================================
# Sliding-window ring buffer
# =============================================================================

class _SlidingWindowRingBuffer:
    """Fixed-size FIFO ring buffer for the sliding-window KV cache.

    The buffer holds at most ``win`` entries of shape ``[B, c]`` (one
    per cached token). Entries are written in arrival order; once the
    buffer is full, a new entry evicts the OLDEST entry (FIFO).

    Reads return the current contents in **causal order** (oldest first,
    newest last), which is the order ``_sliding_window_attention`` in
    ``ops_csa.py`` expects when computing ``scores = q · C_local^T`` and
    ``out = p · C_local``.

    The buffer is pre-allocated as ``[B, win, c]`` so that ``.to(device)``
    moves it along with the layer's parameters (the caller is responsible
    for moving — see :class:`CSADecodingCache` / :class:`HCADecodingCache`
    which expose ``to`` / ``device`` / ``dtype`` helpers).

    Numerical contract
    ------------------
    The contents returned by :meth:`get` are EXACTLY the last
    ``min(n_written, win)`` entries written, in arrival order. This is
    the same set of entries that ``_sliding_window_attention`` would
    extract via ``C_padded[:, t_lo : t_hi + win - 1].unfold(...)`` for
    query ``t = n_written - 1`` in the naive path.

    The ``win - 1 - t`` left-edge masking in the naive path (which
    zeroes out positions before 0 for the first ``win - 1`` queries) is
    handled implicitly here: when ``n_written < win``, the buffer simply
    has fewer entries, and the caller's softmax is over the actual
    entries present (no masking needed).
    """

    def __init__(self, B: int, win: int, c: int, device, dtype):
        # Validate win BEFORE allocating — a negative win would produce
        # a zero-size buffer that silently never caches anything, and
        # ``win = 0`` (a valid "disable SW" sentinel in HybridConfig)
        # must be handled by the CALLER (the cache classes skip the SW
        # branch entirely when ``win == 0``, so this class is never
        # instantiated with ``win = 0``).
        if win < 1:
            raise ValueError(
                f"_SlidingWindowRingBuffer: win={win} must be >= 1 "
                f"(win=0 disables the SW branch and should bypass this "
                f"class entirely).")
        self.B, self.win, self.c = B, win, c
        self.device, self.dtype = torch.device(device), dtype
        # Pre-allocated buffer. Entries at indices >= _sw_len are stale
        # (left over from a previous write cycle before a reset) and
        # MUST NOT be read; ``get`` only ever returns the first
        # ``_sw_len`` entries of the causal-order view.
        with normal_tensor_scope():
            self._buf = torch.zeros(
                B, win, c, device=self.device, dtype=dtype)
        # Number of valid entries currently in the buffer (0..win).
        self._sw_len = 0
        # Index in ``self._buf`` of the OLDEST valid entry. Stays at 0
        # while ``_sw_len < win`` (entries are written sequentially at
        # positions 0, 1, 2, ...). Once the buffer is full, each new
        # write goes to ``self._sw_head`` (overwriting the oldest) and
        # ``_sw_head`` advances by the number of written entries (mod
        # ``win``).
        self._sw_head = 0
        # Lazily-allocated causal-order mirror used by ``get()`` when the
        # ring has wrapped (the physical layout is then split in two).
        # ``_version`` bumps on every mutation of the ring contents;
        # ``_window_version`` records which version the mirror holds.
        self._window = None
        self._window_version = -1
        self._version = 0

    def reset(self) -> None:
        """Clear all state (does not deallocate the buffer)."""
        # The buffer is zeroed defensively so a stale read after reset
        # returns zeros rather than the previous session's keys (which
        # could leak across sequences).
        self._buf.zero_()
        self._sw_len = 0
        self._sw_head = 0
        self._version += 1

    def append(self, x: torch.Tensor) -> None:
        """Append one or more new entries.

        Args:
            x: ``[B, T_new, c]`` — the new entries, in arrival order
                (``x[:, 0]`` is the oldest of the new entries,
                ``x[:, -1]`` is the newest).
        """
        if x.dim() != 3 or x.shape[0] != self.B or x.shape[-1] != self.c:
            raise ValueError(
                f"_SlidingWindowRingBuffer.append: expected "
                f"[{self.B}, T_new, {self.c}], got {tuple(x.shape)}.")
        if x.device != self._buf.device or x.dtype != self._buf.dtype:
            # Cast defensively — the caller is expected to pass
            # matching dtype/device, but a stale buffer after a
            # ``model.to(cuda)`` would otherwise fail inside
            # ``self._buf[:, ...] = x`` with a device-mismatch error.
            x = x.to(device=self._buf.device, dtype=self._buf.dtype)
        self.append_bulk(x)

    def append_bulk(self, x: torch.Tensor) -> None:
        """Append ``[B, T_new, c]`` entries in arrival order.

        Equivalent to calling :meth:`append` once per token — the final
        causal-order contents are exactly the last ``min(total, win)``
        entries written. Only the physical placement inside ``_buf``
        may differ (any placement with the same causal-order view is
        valid for :meth:`get`).

        The writes go straight into the ring (at most two vectorized
        slice copies); no concatenation of the previous contents is
        needed to preserve the causal-order contract.
        """
        if x.dim() != 3 or x.shape[0] != self.B or x.shape[-1] != self.c:
            raise ValueError(
                f"_SlidingWindowRingBuffer.append_bulk: expected "
                f"[{self.B}, T_new, {self.c}], got {tuple(x.shape)}.")
        if x.device != self._buf.device or x.dtype != self._buf.dtype:
            x = x.to(device=self._buf.device, dtype=self._buf.dtype)
        t_new = x.shape[1]
        if t_new == 0:
            return
        if t_new >= self.win:
            # Every entry cached before this call (and every entry of ``x``
            # except the newest ``win``) is evicted before it can be read,
            # so the final content is simply the newest ``win`` entries.
            self._buf[:, :self.win] = x[:, -self.win:]
            self._sw_head = 0
            self._sw_len = self.win
            self._version += 1
            return
        if self._sw_len < self.win:
            # Not yet full -> ``_sw_head == 0`` and entries are at
            # ``[0, _sw_len)``. Write the new entries after them; if they
            # do not fit, the overflow wraps to the front, overwriting the
            # oldest entries (which are evicted by the append).
            space = self.win - self._sw_len
            if t_new <= space:
                self._buf[:, self._sw_len:self._sw_len + t_new] = x
                self._sw_len += t_new
            else:
                self._buf[:, self._sw_len:self.win] = x[:, :space]
                k = t_new - space
                self._buf[:, :k] = x[:, space:]
                self._sw_head = k
                self._sw_len = self.win
        else:
            # Full: overwrite the ``T_new`` oldest entries starting at the
            # head (wrapping once if needed) and advance the head past
            # them.
            head = self._sw_head
            write = min(t_new, self.win - head)
            self._buf[:, head:head + write] = x[:, :write]
            if write < t_new:
                self._buf[:, :t_new - write] = x[:, write:]
            self._sw_head = (head + t_new) % self.win
        self._version += 1

    def get(self) -> torch.Tensor:
        """Return the current contents in causal order (oldest first).

        Returns:
            ``[B, sw_len, c]`` where ``sw_len = min(n_written, win)``.
            For ``sw_len == 0`` returns an empty ``[B, 0, c]`` tensor
            (NOT ``None`` — the caller's einsum handles the empty case
            via ``softmax(-inf) = NaN`` guards, but a zero-shape tensor
            is more convenient for the downstream einsum path).
        """
        if self._sw_len == 0:
            return self._buf[:, :0]
        # Two cases:
        #  1. No wrap (head + len <= win): entries are at
        #     ``[head : head + len]`` in causal order.
        #  2. Wrap (head + len > win): entries are at
        #     ``[head : win]`` (oldest) followed by ``[0 : head + len - win]``
        #     (newer). The two pieces are copied into the reusable
        #     causal-order mirror instead of allocating a fresh tensor via
        #     ``torch.cat`` on every call; the mirror is refilled only when
        #     the ring contents changed since the last fill. While a graph
        #     is being recorded the ``cat`` form is kept so the result
        #     stays connected to the buffer entries' autograd history.
        if self._sw_head + self._sw_len <= self.win:
            return self._buf[:, self._sw_head : self._sw_head + self._sw_len]
        tail_len = self.win - self._sw_head                       # oldest
        head_len = self._sw_head + self._sw_len - self.win        # newer
        if torch.is_grad_enabled():
            return torch.cat([self._buf[:, self._sw_head:],
                              self._buf[:, :head_len]], dim=1)
        if (self._window is None
                or self._window.shape[1:] != (self.win, self.c)
                or self._window.device != self._buf.device
                or self._window.dtype != self._buf.dtype):
            with normal_tensor_scope():
                self._window = torch.empty(
                    self.B, self.win, self.c,
                    device=self._buf.device, dtype=self._buf.dtype)
            self._window_version = -1
        if self._window_version != self._version:
            self._window[:, :tail_len].copy_(self._buf[:, self._sw_head:])
            self._window[:, tail_len:tail_len + head_len].copy_(
                self._buf[:, :head_len])
            self._window_version = self._version
        return self._window[:, :self._sw_len]

    @property
    def n_valid(self) -> int:
        """Number of valid entries currently in the buffer (0..win)."""
        return self._sw_len

    def to(self, device=None, dtype=None) -> '_SlidingWindowRingBuffer':
        """Move the buffer to a new device / dtype (in-place)."""
        if device is not None:
            device = torch.device(device)
        if device is not None and device != self._buf.device:
            self._buf = _to_cache(self._buf, self._buf.dtype, device)
            self.device = self._buf.device
        if dtype is not None and dtype != self._buf.dtype:
            self._buf = _to_cache(self._buf, dtype, self._buf.device)
            self.dtype = self._buf.dtype
        # The mirror is lazily rebuilt on the next wrapped ``get()`` (its
        # device/dtype checks would catch a move anyway; dropping it also
        # keeps stale contents from being returned).
        self._window = None
        self._window_version = -1
        self._version += 1
        return self


# =============================================================================
# CSADecodingCache
# =============================================================================

class CSADecodingCache:
    """Incremental decoding cache for CSA layers.

    State
    -----
    Three pieces of state, matching the spec in the issue:

    1. **Partial block accumulator** (Python lists of ``[B, 1, c]`` and
       ``[B, 1, c_I]`` tensors). Holds 0..m-1 tokens whose projections
       have been computed but not yet compressed into a block. Cleared
       after every ``m`` tokens; the just-consumed ``Cb`` and ``Zb`` are
       stashed as the "previous block" overlap partner for the next
       compression.
    2. **Compressed block cache** (valid prefix ``[B, n_blocks, c]``) and
       **indexer key cache** (valid prefix ``[B, n_blocks, c_I]``). Their
       backing storage grows geometrically as the accumulator fills.
       Read by the sparse-attention branch (compressed block cache) and
       the lightning indexer (indexer key cache) during ``forward_step``.
    3. **Sliding-window ring buffer** (:class:`_SlidingWindowRingBuffer`).
       Holds the most recent ``win`` ``qk_norm_mode``-normalized ``Ca``
       entries, in FIFO order. Read by the SW branch during
       ``forward_step``.

    The cache also tracks two counters: ``n_tokens_seen`` (total tokens
    appended) and ``n_blocks`` (compressed blocks produced). The former
    is used to compute the causal block mask ``b < t // m`` for
    the indexer; the latter is the length of the compressed block cache.

    Lifecycle
    ---------
    1. Construct with ``B, c, c_I, m, win, device, dtype`` matching the
       layer's config.
    2. Call :meth:`append_step` for each new token (or chunk of tokens)
       to update the cache state (accumulator + compressed blocks + SW
       buffer). This does NOT compute the attention output — it only
       updates the cache.
    3. Call :meth:`forward_step` to compute the attention output for the
       CURRENT query (the last token appended), using the current cache
       state. The query's projections (``q``, ``q_idx``, ``w_idx``) are
       passed as arguments since they depend on the layer's parameters
       (``W_DQ``, ``W_UQ``, ``W_IUQ``, ``W_w``) which the cache does
       not own.
    4. Call :meth:`reset` between independent sequences.

    Alternatively, the typical usage pattern (see :class:`CSAAttnDecoding`
    in ``run_decoding.py``) interleaves 2 and 3: for each new token,
    call ``append_step`` then ``forward_step``.

    Numerical contract
    ------------------
    For a sequence of ``T`` tokens fed one at a time through
    ``append_step`` + ``forward_step``, the per-token outputs are bit-
    identical (to fp32 tolerance) to the corresponding rows of
    ``naive_csa`` called once on the full ``[B, T, d]`` sequence,
    including under tied indexer scores. See the module-level docstring
    ("Deterministic tie-breaking") and the regression tests
    ``test_csa_decoding_cache_correctness`` /
    ``test_csa_decoding_cache_sparse_topk_ties`` in ``run_correctness.py``
    for the verification (the latter exercises sparse ``topk < n_blocks``
    with maximally tied scores).
    """

    def __init__(
        self,
        B: int,
        c: int,
        c_I: int,
        m: int,
        win: int,
        device,
        dtype,
        rope_dim: int | None = None,
        rope_base: float = 10000.0,
        qk_norm_mode: str = 'l2',
    ):
        # Validate structural params up-front so a misconfigured cache
        # raises at construction (not deep inside ``append_step``).
        # NOTE: use ``raise ValueError`` (NOT ``assert``) so the checks
        # survive ``python -O``.
        if B < 1:
            raise ValueError(f"B={B} must be >= 1")
        if c < 1:
            raise ValueError(f"c={c} must be >= 1")
        if c_I < 1:
            raise ValueError(f"c_I={c_I} must be >= 1")
        if m < 1:
            raise ValueError(f"m={m} must be >= 1")
        if win < 0:
            raise ValueError(f"win={win} must be >= 0 (0 disables SW)")
        self.B, self.c, self.c_I, self.m, self.win = B, c, c_I, m, win
        _validate_cache_dtype(dtype)
        self.device, self.dtype = torch.device(device), dtype
        # normalize_qk flag: set by the caller after construction (or
        # default True to match the repo-wide convention). The cache
        # needs this to decide whether to apply the ``qk_norm_mode``
        # normalization to keys before appending to the sliding-window
        # ring buffer.
        self._normalize_qk: bool = True
        # Partial RoPE (DeepSeek-V4 §2.3.3): the cache rotates the
        # compressed KV entries at their block-anchor positions (``s*m``)
        # when a block completes, rotates the sliding-window keys at
        # their token positions when they are appended, rotates the query
        # at the current token position inside ``forward_step``, and
        # inverse-rotates the output at the negated query position — the
        # same contract as the full-sequence ``naive_csa`` path, so the
        # incremental decode stays equivalent to prefill. ``rope_dim=None``
        # (default) disables RoPE.
        if rope_base <= 0:
            raise ValueError(
                f"CSADecodingCache: rope_base={rope_base!r} must be positive")
        # qk_norm_mode must match the mode the full-sequence ``naive_csa``
        # call uses; otherwise the incremental decode would normalize keys
        # differently from prefill and silently disagree with it.
        if qk_norm_mode not in ('l2', 'rms'):
            raise ValueError(
                f"CSADecodingCache: qk_norm_mode={qk_norm_mode!r} must be "
                f"'l2' (cosine, historical) or 'rms' (paper §2.3.3)")
        self.qk_norm_mode = qk_norm_mode
        self.rope_dim_eff = effective_rope_dim(rope_dim, c)
        self.rope_base = float(rope_base)
        self.reset()

    def reset(self) -> None:
        """Clear all cache state (does not deallocate tensors)."""
        # Partial accumulator (Python list of [B, 1, c] / [B, 1, c_I]).
        self._acc_Ca: list[torch.Tensor] = []
        self._acc_Cb: list[torch.Tensor] = []
        self._acc_Za: list[torch.Tensor] = []
        self._acc_Zb: list[torch.Tensor] = []
        self._acc_K_idx: list[torch.Tensor] = []
        self._acc_Z_idx: list[torch.Tensor] = []
        # Previous block's Cb / Zb for the overlapped compression of the
        # NEXT block. ``None`` until the first block is completed.
        self._prev_Cb: torch.Tensor | None = None  # [B, m, c]
        self._prev_Zb: torch.Tensor | None = None  # [B, m, c]
        # Compressed block caches use geometrically-grown storage. Only the
        # valid prefix is exposed through the public properties.
        self._C_comp: torch.Tensor | None = None   # [B, capacity, c]
        self._K_IComp: torch.Tensor | None = None  # [B, capacity, c_I]
        # SW ring buffer (None if win == 0).
        if self.win > 0:
            self._sw = _SlidingWindowRingBuffer(
                self.B, self.win, self.c, self.device, self.dtype,
            )
        else:
            self._sw = None
        # Counters.
        self._n_tokens_seen = 0
        self._n_blocks = 0

    def _ensure_compressed_capacity(self, required: int) -> None:
        """Ensure CSA compressed/indexer storage can hold ``required`` rows.

        Capacity grows geometrically with the completed-block count. Only
        the valid prefix ``:self._n_blocks`` is copied; unused capacity is
        not exposed through the public properties or attention path.

        ``required`` is ALWAYS rounded up to the geometric capacity even
        when a bulk prefill (``_append_bulk``) jumps across many block
        boundaries in one call, so the retained capacity is a function of
        the completed-block count only (bulk prefill and token-by-token
        fills allocate identically). Keep the two formulas identical — see
        ``cache_accounting``.
        """
        if self._C_comp is not None and required <= self._C_comp.shape[1]:
            return
        old_capacity = 0 if self._C_comp is None else self._C_comp.shape[1]
        new_capacity = max(_geometric_capacity(required), old_capacity * 2)
        with normal_tensor_scope():
            new_C = torch.empty(
                self.B, new_capacity, self.c,
                device=self.device, dtype=self.dtype,
            )
            new_K = torch.empty(
                self.B, new_capacity, self.c_I,
                device=self.device, dtype=self.dtype,
            )
        if self._n_blocks:
            new_C[:, :self._n_blocks] = self._C_comp[:, :self._n_blocks]
            new_K[:, :self._n_blocks] = self._K_IComp[:, :self._n_blocks]
        self._C_comp = new_C
        self._K_IComp = new_K

    # ----- properties ----------------------------------------------------

    @property
    def n_tokens_seen(self) -> int:
        return self._n_tokens_seen

    @property
    def n_blocks(self) -> int:
        return self._n_blocks

    @property
    def accumulator_len(self) -> int:
        """Number of tokens currently in the partial accumulator (0..m-1)."""
        return len(self._acc_Ca)

    @property
    def C_comp(self) -> torch.Tensor | None:
        """Current valid compressed KV rows ``[B, n_blocks, c]`` (or None)."""
        if self._C_comp is None:
            return None
        return self._C_comp[:, :self._n_blocks]

    @property
    def K_IComp(self) -> torch.Tensor | None:
        """Current valid indexer key rows ``[B, n_blocks, c_I]`` (or None)."""
        if self._K_IComp is None:
            return None
        return self._K_IComp[:, :self._n_blocks]

    @property
    def sw_buffer(self) -> _SlidingWindowRingBuffer | None:
        """The sliding-window ring buffer (or None if ``win == 0``)."""
        return self._sw

    # ----- state accounting ----------------------------------------------

    @property
    def nbytes(self) -> int:
        """Physical bytes retained by the cache's backing tensors.

        Counts ALLOCATED storage (including geometric spare capacity and
        the full sliding-window ring buffer), i.e. what the process holds.
        ``nbytes_valid`` counts only the logically live entries. Mirrors
        :class:`Mamba2DecodingState`'s accounting surface.
        """
        return self.nbytes_breakdown()['total_allocated_bytes']

    @property
    def nbytes_valid(self) -> int:
        """Bytes of logically live state (valid prefix + accumulator)."""
        return self.nbytes_breakdown()['total_valid_bytes']

    def nbytes_breakdown(self) -> dict:
        """Component-wise decode-state byte accounting (JSON-safe ints).

        Components (allocated / valid):

        * ``compressed_kv`` — the compressed block cache ``[B, cap, c]``
          (valid prefix ``n_blocks``);
        * ``indexer_kv`` — the CSA indexer key cache ``[B, cap, c_I]``;
        * ``sliding_window`` — the fixed ``[B, win, c]`` ring buffer;
        * ``partial_tail`` — the ``n_tokens_seen % m`` partial-token
          accumulator rows (allocated == valid);
        * ``overlap`` — the previous block's ``Cb``/``Zb`` stashed for the
          next overlapped compression.

        The allocated totals are exactly ``run_kv_cache``'s
        ``full_accounting`` components for one CSA layer at the same
        ``(T, m, c, c_I, win)``, so the live and analytical models can be
        cross-checked (see ``cache_accounting``).
        """
        B, c, c_I = self.B, self.c, self.c_I
        cap = 0 if self._C_comp is None else self._C_comp.shape[1]
        compressed_valid_elems = self._n_blocks * c * B
        indexer_valid_elems = self._n_blocks * c_I * B
        compressed_alloc = _tensor_nbytes(self._C_comp)
        indexer_alloc = _tensor_nbytes(self._K_IComp)
        compressed_valid = _valid_nbytes(self._C_comp, compressed_valid_elems)
        indexer_valid = _valid_nbytes(self._K_IComp, indexer_valid_elems)
        sw_alloc = _tensor_nbytes(self._sw._buf) if self._sw is not None else 0
        sw_valid = (_valid_nbytes(self._sw._buf, self._sw._sw_len * c * B)
                    if self._sw is not None else 0)
        partial_alloc = sum(
            _tensor_nbytes(t)
            for group in (self._acc_Ca, self._acc_Cb, self._acc_Za,
                          self._acc_Zb, self._acc_K_idx, self._acc_Z_idx)
            for t in group)
        partial_valid = partial_alloc
        overlap_alloc = _tensor_nbytes(self._prev_Cb) + \
            _tensor_nbytes(self._prev_Zb)
        overlap_valid = overlap_alloc
        total_alloc = (compressed_alloc + indexer_alloc + sw_alloc
                       + partial_alloc + overlap_alloc)
        total_valid = (compressed_valid + indexer_valid + sw_valid
                       + partial_valid + overlap_valid)
        return {
            'compressed_kv_allocated_bytes': compressed_alloc,
            'compressed_kv_valid_bytes': compressed_valid,
            'indexer_kv_allocated_bytes': indexer_alloc,
            'indexer_kv_valid_bytes': indexer_valid,
            'sliding_window_allocated_bytes': sw_alloc,
            'sliding_window_valid_bytes': sw_valid,
            'partial_tail_allocated_bytes': partial_alloc,
            'partial_tail_valid_bytes': partial_valid,
            'overlap_allocated_bytes': overlap_alloc,
            'overlap_valid_bytes': overlap_valid,
            'total_allocated_bytes': total_alloc,
            'total_valid_bytes': total_valid,
            'capacity_rows': int(cap),
            'n_blocks': int(self._n_blocks),
            'n_tokens_seen': int(self._n_tokens_seen),
        }

    # ----- state migration -----------------------------------------------

    def to(self, device=None, dtype=None) -> 'CSADecodingCache':
        """Move all cache tensors to a new device / dtype (in-place)."""
        if device is not None:
            device = torch.device(device)
        if device is not None and device != self.device:
            self.device = device
        if dtype is not None and dtype != self.dtype:
            self.dtype = dtype
        # Move partial accumulator.
        self._acc_Ca = [_to_cache(t, self.dtype, self.device)
                        for t in self._acc_Ca]
        self._acc_Cb = [_to_cache(t, self.dtype, self.device)
                        for t in self._acc_Cb]
        self._acc_Za = [_to_cache(t, self.dtype, self.device)
                        for t in self._acc_Za]
        self._acc_Zb = [_to_cache(t, self.dtype, self.device)
                        for t in self._acc_Zb]
        self._acc_K_idx = [_to_cache(t, self.dtype, self.device)
                           for t in self._acc_K_idx]
        self._acc_Z_idx = [_to_cache(t, self.dtype, self.device)
                           for t in self._acc_Z_idx]
        if self._prev_Cb is not None:
            self._prev_Cb = _to_cache(self._prev_Cb, self.dtype, self.device)
        if self._prev_Zb is not None:
            self._prev_Zb = _to_cache(self._prev_Zb, self.dtype, self.device)
        if self._C_comp is not None:
            self._C_comp = _to_cache(self._C_comp, self.dtype, self.device)
        if self._K_IComp is not None:
            self._K_IComp = _to_cache(self._K_IComp, self.dtype, self.device)
        if self._sw is not None:
            self._sw.to(device=self.device, dtype=self.dtype)
        return self

    # ----- core: append new tokens ---------------------------------------

    def append_step(
        self,
        Ca_new: torch.Tensor,    # [B, T_new, c]
        Cb_new: torch.Tensor,    # [B, T_new, c]
        Za_new: torch.Tensor,    # [B, T_new, c]
        Zb_new: torch.Tensor,    # [B, T_new, c]
        K_idx_new: torch.Tensor, # [B, T_new, c_I]
        Z_idx_new: torch.Tensor, # [B, T_new, c_I]
        Ba: torch.Tensor,        # [m, c]
        Bb: torch.Tensor,        # [m, c]
        B_idx: torch.Tensor,     # [m, c_I]
    ) -> list[int]:
        """Append ``T_new`` new tokens' projections to the cache.

        For each new token:
          1. Append its 6 projections to the partial accumulator.
          2. Append its (``qk_norm_mode``-normalized) ``Ca`` to the SW
             ring buffer.
          3. If the accumulator now has ``m`` tokens, compress them
             into one new block (using the previous block's ``Cb`` /
             ``Zb`` for the overlapped b-branch), push the result to
             the compressed block cache + indexer key cache, then clear
             the accumulator and stash the just-consumed ``Cb`` / ``Zb``
             as the new "previous block".

        Args:
            Ca_new, Cb_new, Za_new, Zb_new: ``[B, T_new, c]``
            K_idx_new, Z_idx_new: ``[B, T_new, c_I]``
            Ba, Bb: ``[m, c]`` learnable positional biases.
            B_idx: ``[m, c_I]`` learnable indexer positional bias.

        Returns:
            List of indices of newly-compressed blocks (in the order
            they were produced). Empty if no block was completed during
            this call. The indices are absolute block indices (i.e.
            ``n_blocks - 1`` after the push, NOT relative to the call).
        """
        # Validate shapes.
        B, T_new, c = Ca_new.shape
        if B != self.B:
            raise ValueError(
                f"CSADecodingCache.append_step: batch size {B} does not "
                f"match cache's B={self.B}. Call reset() between sequences "
                f"with different batch sizes.")
        if c != self.c:
            raise ValueError(
                f"CSADecodingCache.append_step: c={c} does not match "
                f"cache's c={self.c}.")
        if K_idx_new.shape[-1] != self.c_I:
            raise ValueError(
                f"CSADecodingCache.append_step: c_I={K_idx_new.shape[-1]} "
                f"does not match cache's c_I={self.c_I}.")
        # Validate T_new consistency across ALL inputs, not just
        # ``Ca_new``: a mismatch (e.g. ``Ca_new`` has T_new=4 but
        # ``Cb_new`` has T_new=3) would otherwise raise deep inside the
        # per-token Python loop with a cryptic IndexError.
        for _name, _t in [
            ('Cb_new', Cb_new), ('Za_new', Za_new), ('Zb_new', Zb_new),
            ('K_idx_new', K_idx_new), ('Z_idx_new', Z_idx_new),
        ]:
            if _t.shape[1] != T_new:
                raise ValueError(
                    f"CSADecodingCache.append_step: {_name}.shape[1]="
                    f"{_t.shape[1]} does not match Ca_new.shape[1]="
                    f"{T_new}. All inputs must share the same T_new.")
        # Cast to cache dtype/device defensively.
        Ca_new = _to_cache(Ca_new, self.dtype, self.device)
        Cb_new = _to_cache(Cb_new, self.dtype, self.device)
        Za_new = _to_cache(Za_new, self.dtype, self.device)
        Zb_new = _to_cache(Zb_new, self.dtype, self.device)
        K_idx_new = _to_cache(K_idx_new, self.dtype, self.device)
        Z_idx_new = _to_cache(Z_idx_new, self.dtype, self.device)

        if T_new > 1:
            return self._append_bulk(
                Ca_new, Cb_new, Za_new, Zb_new, K_idx_new, Z_idx_new,
                Ba, Bb, B_idx,
            )
        new_block_indices: list[int] = []
        for i in range(T_new):
            # Slice out token i.
            ca_i = Ca_new[:, i:i+1]    # [B, 1, c]
            cb_i = Cb_new[:, i:i+1]
            za_i = Za_new[:, i:i+1]
            zb_i = Zb_new[:, i:i+1]
            k_i = K_idx_new[:, i:i+1]  # [B, 1, c_I]
            z_i = Z_idx_new[:, i:i+1]
            # Append to accumulator.
            self._acc_Ca.append(ca_i)
            self._acc_Cb.append(cb_i)
            self._acc_Za.append(za_i)
            self._acc_Zb.append(zb_i)
            self._acc_K_idx.append(k_i)
            self._acc_Z_idx.append(z_i)
            self._n_tokens_seen += 1
            # Append Ca to the SW ring buffer. When normalize_qk=True
            # (the default), normalize with ``qk_norm_mode`` (L2 by
            # default, RMS under the paper §2.3.3 mode) so the SW branch
            # matches the full-sequence naive_csa path. When
            # normalize_qk=False, store raw (unnormalized) Ca so the SW
            # branch runs in dot-product mode, matching the sparse branch
            # and the full-sequence naive_csa path.
            if self._sw is not None:
                # The SW key normalize/rotate path runs in the cache's
                # compute dtype (fp64 for float64 caches) so a float64 cache
                # stays faithful to the float64 full-sequence path. (An
                # unconditional fp32 round-trip here would silently
                # quantize float64 SW entries to fp32 precision.)
                compute_dtype = (
                    torch.float64 if self.dtype == torch.float64 else torch.float)
                if self._normalize_qk:
                    ca_src_i = _to_dtype(ca_i.squeeze(1), compute_dtype)
                    if (not torch.is_grad_enabled()
                            and self.dtype != compute_dtype):
                        # ``ca_src_i`` is a fresh compute-dtype cast copy
                        # (storage dtype differs from compute dtype), so
                        # the per-row normalize runs in place with the
                        # identical per-element chain.
                        ca_norm_i = _qk_normalize_(
                            ca_src_i, self.qk_norm_mode)
                    else:
                        ca_norm_i = _qk_normalize(
                            ca_src_i, self.qk_norm_mode)
                    ca_store_i = ca_norm_i.unsqueeze(1)
                else:
                    ca_store_i = ca_i
                # Partial RoPE (§2.3.3): the SW key rotates at its own
                # token position AFTER normalization — the same
                # normalize-then-rotate order as ``naive_csa``'s
                # sliding-window branch.
                if self.rope_dim_eff > 0:
                    pos_t = _pos_tensor(self._n_tokens_seen - 1, self.device)
                    ca_store_i = partial_rope(
                        _to_dtype(ca_store_i, compute_dtype), pos_t,
                        self.rope_dim_eff, self.rope_base, dim=1)
                self._sw.append(_to_dtype(ca_store_i, self.dtype))
            # If the accumulator is full, compress and push.
            if len(self._acc_Ca) == self.m:
                # Stack the accumulated projections into [B, m, c] / [B, m, c_I].
                Ca_block = torch.cat(self._acc_Ca, dim=1)   # [B, m, c]
                Cb_block = torch.cat(self._acc_Cb, dim=1)
                Za_block = torch.cat(self._acc_Za, dim=1)
                Zb_block = torch.cat(self._acc_Zb, dim=1)
                K_idx_block = torch.cat(self._acc_K_idx, dim=1)  # [B, m, c_I]
                Z_idx_block = torch.cat(self._acc_Z_idx, dim=1)
                # Compress KV (overlapped two-branch) with the previous
                # block's Cb / Zb (None for the first block).
                new_C = _csa_compress_kv_overlapped_single(
                    Ca_block, Cb_block, Za_block, Zb_block,
                    self._prev_Cb, self._prev_Zb, Ba, Bb,
                )                                                  # [B, c]
                # Compress indexer keys (single-branch).
                new_K_I = _indexer_compress_single(
                    K_idx_block, Z_idx_block, B_idx,
                )                                                  # [B, c_I]
                # Partial RoPE (§2.3.3): rotate the just-completed
                # compressed KV entry at its block-anchor position
                # ``block_idx * m`` (block_idx == self._n_blocks, the
                # row this entry is about to occupy). The rotated entry
                # serves as both key and value; rotation happens BEFORE
                # storage — ``forward_step`` normalizes at read time,
                # mirroring the rotate-then-normalize order of
                # ``naive_csa``. The indexer key cache is NOT rotated
                # (the paper scopes RoPE to the core attention, not
                # the indexer scores).
                if self.rope_dim_eff > 0:
                    pos_b = _pos_tensor(self._n_blocks * self.m, self.device)
                    new_C = partial_rope(
                        new_C.unsqueeze(1), pos_b,
                        self.rope_dim_eff, self.rope_base,
                        dim=1).squeeze(1)                           # [B, c]
                # The compress helpers upcast to fp32/fp64 internally
                # (compute_dtype) and return that dtype. Cast back to the
                # cache's storage dtype before writing into the
                # geometrically-grown storage so all rows stay uniform.
                # Storage grows geometrically with the number of completed
                # blocks. The public cache exposes only the valid prefix,
                # while the storage tensors retain spare capacity for
                # future blocks. ``copy_`` casts with exactly the same
                # rounding as ``.to`` and writes straight into the slot, so
                # the intermediate ``[B, 1, c]`` row temporaries disappear.
                self._ensure_compressed_capacity(self._n_blocks + 1)
                self._C_comp[:, self._n_blocks:self._n_blocks + 1].copy_(
                    _to_dtype(new_C.unsqueeze(1), self.dtype))
                self._K_IComp[:, self._n_blocks:self._n_blocks + 1].copy_(
                    _to_dtype(new_K_I.unsqueeze(1), self.dtype))
                self._n_blocks += 1
                new_block_indices.append(self._n_blocks - 1)
                # Stash the just-consumed Cb / Zb as the previous
                # block's overlap partner for the NEXT compression.
                # Detach so we don't retain the autograd graph across
                # decode steps (mirrors the KDA state detach pattern in
                # ops_fused.py::KDAHybridLayer.forward_functional). The
                # ``cat`` above already produced freshly-owned storage
                # that nothing else aliases or mutates, so ``detach()``
                # alone is enough here (the bulk path keeps its clone:
                # there the source is a view into the caller's tensor).
                self._prev_Cb = Cb_block.detach()
                self._prev_Zb = Zb_block.detach()
                # Clear the accumulator for the next block.
                self._acc_Ca.clear()
                self._acc_Cb.clear()
                self._acc_Za.clear()
                self._acc_Zb.clear()
                self._acc_K_idx.clear()
                self._acc_Z_idx.clear()
        return new_block_indices

    def _append_bulk(
        self,
        Ca_new: torch.Tensor,
        Cb_new: torch.Tensor,
        Za_new: torch.Tensor,
        Zb_new: torch.Tensor,
        K_idx_new: torch.Tensor,
        Z_idx_new: torch.Tensor,
        Ba: torch.Tensor,
        Bb: torch.Tensor,
        B_idx: torch.Tensor,
    ) -> list[int]:
        """Batched ``append_step`` body for ``T_new > 1`` (prefill).

        Splits the incoming tokens at exactly the same block boundaries as
        the per-token loop (a new block completes every ``m`` tokens counted
        from the very first token appended to this cache), reuses the same
        compression helpers, and leaves the same accumulator / ring-buffer /
        compressed-row state.
        """
        B, T_new, c = Ca_new.shape
        m = self.m
        dtype = self.dtype
        # ----- sliding-window branch (batched) -----
        if self._sw is not None:
            # Compute dtype, not storage dtype: a float64 cache must not be
            # silently rounded through fp32 (see the matching note in
            # ``append_step``).
            compute_dtype = torch.float64 if dtype == torch.float64 else torch.float
            if self._normalize_qk:
                ca_src = _to_dtype(Ca_new, compute_dtype)
                if (not torch.is_grad_enabled()
                        and self.dtype != compute_dtype):
                    # Fresh cast copy (see the matching note in
                    # ``append_step``): normalize in place.
                    ca_store = _qk_normalize_(ca_src, self.qk_norm_mode)
                else:
                    ca_store = _qk_normalize(ca_src, self.qk_norm_mode)
            else:
                ca_store = Ca_new
            if self.rope_dim_eff > 0:
                pos = torch.arange(
                    self._n_tokens_seen, self._n_tokens_seen + T_new,
                    device=self.device)
                ca_store = partial_rope(
                    _to_dtype(ca_store, compute_dtype), pos,
                    self.rope_dim_eff, self.rope_base, dim=1)
            self._sw.append_bulk(_to_dtype(ca_store, dtype))
        self._n_tokens_seen += T_new
        # ----- accumulate / compress into blocks -----
        acc = len(self._acc_Ca)
        first_needed = m - acc
        if T_new < first_needed:
            # Still short of the next block boundary: stash every token.
            self._acc_Ca += list(Ca_new.split(1, dim=1))
            self._acc_Cb += list(Cb_new.split(1, dim=1))
            self._acc_Za += list(Za_new.split(1, dim=1))
            self._acc_Zb += list(Zb_new.split(1, dim=1))
            self._acc_K_idx += list(K_idx_new.split(1, dim=1))
            self._acc_Z_idx += list(Z_idx_new.split(1, dim=1))
            return []
        if acc:
            Ca_first = torch.cat(
                list(self._acc_Ca) + list(Ca_new[:, :first_needed].split(1, dim=1)), dim=1)
            Cb_first = torch.cat(
                list(self._acc_Cb) + list(Cb_new[:, :first_needed].split(1, dim=1)), dim=1)
            Za_first = torch.cat(
                list(self._acc_Za) + list(Za_new[:, :first_needed].split(1, dim=1)), dim=1)
            Zb_first = torch.cat(
                list(self._acc_Zb) + list(Zb_new[:, :first_needed].split(1, dim=1)), dim=1)
            K_first = torch.cat(
                list(self._acc_K_idx) + list(K_idx_new[:, :first_needed].split(1, dim=1)), dim=1)
            Zk_first = torch.cat(
                list(self._acc_Z_idx) + list(Z_idx_new[:, :first_needed].split(1, dim=1)), dim=1)
        else:
            Ca_first = Ca_new[:, :first_needed]
            Cb_first = Cb_new[:, :first_needed]
            Za_first = Za_new[:, :first_needed]
            Zb_first = Zb_new[:, :first_needed]
            K_first = K_idx_new[:, :first_needed]
            Zk_first = Z_idx_new[:, :first_needed]
        new_C_first = _csa_compress_kv_overlapped_single(
            Ca_first, Cb_first, Za_first, Zb_first,
            self._prev_Cb, self._prev_Zb, Ba, Bb,
        )                                                          # [B, c]
        new_K_first = _indexer_compress_single(
            K_first, Zk_first, B_idx)                              # [B, c_I]
        rest = T_new - first_needed
        n_rest = rest // m
        tail = rest % m
        if n_rest:
            rest_end = first_needed + n_rest * m
            Ca_rest = Ca_new[:, first_needed:rest_end].reshape(B, n_rest, m, c)
            Cb_rest = Cb_new[:, first_needed:rest_end].reshape(B, n_rest, m, c)
            Za_rest = Za_new[:, first_needed:rest_end].reshape(B, n_rest, m, c)
            Zb_rest = Zb_new[:, first_needed:rest_end].reshape(B, n_rest, m, c)
            # The first kept block's overlap partner is the first physical
            # block (a real block, unlike the -inf row the full-sequence
            # helper uses for its dropped row 0). Build only those two
            # shifted sequences: no full-sequence concatenation, no
            # recomputation of the already-handled first block.
            Cb_prev_rest = torch.cat(
                [Cb_first.unsqueeze(1), Cb_rest[:, :-1]], dim=1)
            Zb_prev_rest = torch.cat(
                [Zb_first.unsqueeze(1), Zb_rest[:, :-1]], dim=1)
            new_C_rest = _csa_compress_kv_overlapped_prev(
                Ca_rest, Za_rest, Cb_prev_rest, Zb_prev_rest, Ba, Bb,
            )                                                      # [B, n_rest, c]
            new_K_rest = csa_compress_kv(
                K_idx_new[:, first_needed:rest_end].reshape(
                    B, n_rest * m, self.c_I),
                Z_idx_new[:, first_needed:rest_end].reshape(
                    B, n_rest * m, self.c_I),
                B_idx, m,
            )                                                      # [B, n_rest, c_I]
            last_Cb = Cb_rest[:, -1]
            last_Zb = Zb_rest[:, -1]
        else:
            new_C_rest = None
            new_K_rest = None
            last_Cb = Cb_first
            last_Zb = Zb_first
        n_new = 1 + n_rest
        block_idx0 = self._n_blocks
        # Partial RoPE: rotate the just-completed compressed entries at
        # their block-anchor positions ``block_idx * m`` (same positions
        # and normalize-then-rotate order as the per-token path). The first
        # block and the rest are rotated separately and written straight
        # into storage slices; the per-row elements are identical.
        pos_b = (torch.arange(block_idx0, block_idx0 + n_new,
                              device=self.device) * m
                 if self.rope_dim_eff > 0 else None)
        first_C = new_C_first.unsqueeze(1)
        if pos_b is not None:
            first_C = partial_rope(
                first_C, pos_b[:1], self.rope_dim_eff, self.rope_base, dim=1)
            if new_C_rest is not None:
                new_C_rest = partial_rope(
                    new_C_rest, pos_b[1:], self.rope_dim_eff, self.rope_base,
                    dim=1)
        self._ensure_compressed_capacity(block_idx0 + n_new)
        self._C_comp[:, block_idx0:block_idx0 + 1].copy_(
            _to_dtype(first_C, dtype))
        self._K_IComp[:, block_idx0:block_idx0 + 1].copy_(
            _to_dtype(new_K_first.unsqueeze(1), dtype))
        if new_C_rest is not None:
            self._C_comp[:, block_idx0 + 1:block_idx0 + n_new].copy_(
                _to_dtype(new_C_rest, dtype))
            self._K_IComp[:, block_idx0 + 1:block_idx0 + n_new].copy_(
                _to_dtype(new_K_rest, dtype))
        self._n_blocks += n_new
        new_block_indices = list(range(block_idx0, block_idx0 + n_new))
        # Stash the LAST block's Cb / Zb as the overlap partner for the next
        # compression (detached: decode must not retain the prefill graph).
        self._prev_Cb = last_Cb.detach().clone()
        self._prev_Zb = last_Zb.detach().clone()
        # Leave the tokens past the last full block in the accumulator.
        if tail:
            self._acc_Ca = list(Ca_new[:, -tail:].split(1, dim=1))
            self._acc_Cb = list(Cb_new[:, -tail:].split(1, dim=1))
            self._acc_Za = list(Za_new[:, -tail:].split(1, dim=1))
            self._acc_Zb = list(Zb_new[:, -tail:].split(1, dim=1))
            self._acc_K_idx = list(K_idx_new[:, -tail:].split(1, dim=1))
            self._acc_Z_idx = list(Z_idx_new[:, -tail:].split(1, dim=1))
        else:
            self._acc_Ca = []
            self._acc_Cb = []
            self._acc_Za = []
            self._acc_Zb = []
            self._acc_K_idx = []
            self._acc_Z_idx = []
        return new_block_indices

    # ----- core: compute attention output for the current query ---------

    def forward_step(
        self,
        q: torch.Tensor,          # [B, 1, nh, c]  (qk_norm_mode-normalized iff normalize_qk)
        q_idx: torch.Tensor,      # [B, 1, nIh, c_I]
        w_idx: torch.Tensor,      # [B, 1, nIh]
        *,
        topk: int,
        nh: int,
        nIh: int,
        scale: float | None = None,
        sink_logits: torch.Tensor | None = None,  # [nh]
        use_ste: bool = True,
        ste_mode: str = 'topk_columns',
        normalize_qk: bool = True,
    ) -> torch.Tensor:
        """Compute the CSA attention output for the CURRENT query.

        Uses the current cache state (compressed blocks + indexer keys +
        SW buffer). The query's projections (``q``, ``q_idx``, ``w_idx``)
        are passed as arguments because they depend on the layer's
        parameters (``W_DQ``, ``W_UQ``, ``W_IUQ``, ``W_w``) which the
        cache does not own.

        Args:
            q: ``[B, 1, nh, c]`` per-head attention queries. When
                ``normalize_qk=True`` the caller MUST pass queries
                normalized with the cache's ``qk_norm_mode`` ('l2' for
                every runner in this repo); when ``normalize_qk=False``
                raw queries are expected and we fall back to dot-product
                scale ``c ** -0.5``.
            q_idx: ``[B, 1, nIh, c_I]`` indexer queries (low-rank).
            w_idx: ``[B, 1, nIh]`` indexer head weights.
            topk: number of compressed blocks to select per query.
            nh, nIh: head counts (validated against ``q`` / ``q_idx``).
            scale: cosine-attention scale (default 1.0, matches
                ``naive_csa``).
            sink_logits: ``[nh]`` learnable attention-sink logits, or
                None to disable.
            use_ste, ste_mode, normalize_qk: forwarded to
                ``csa_lightning_indexer`` (same semantics as
                ``naive_csa``).

        Returns:
            ``[B, 1, nh, c]`` per-head attention output (in
            ``compute_dtype`` — the caller casts to ``H.dtype`` and
            applies the grouped output projection).
        """
        B = q.shape[0]
        T_new = q.shape[1]   # must be 1
        if T_new != 1:
            raise ValueError(
                f"CSADecodingCache.forward_step: T_new={T_new} must be 1. "
                f"For multi-token chunks, call append_step + forward_step "
                f"in a loop (or use naive_csa on the full chunk and "
                f"populate the cache via the prefill path).")
        if B != self.B:
            raise ValueError(
                f"CSADecodingCache.forward_step: batch size {B} does not "
                f"match cache's B={self.B}.")
        if q.shape[2] != nh:
            raise ValueError(
                f"q.shape[2]={q.shape[2]} does not match nh={nh}.")
        if q_idx.shape[2] != nIh:
            raise ValueError(
                f"q_idx.shape[2]={q_idx.shape[2]} does not match nIh={nIh}.")
        # Validate ``q.device == self.device`` so a caller passing a GPU
        # query to a CPU cache (or vice versa) gets a clear error instead
        # of a cryptic cross-device einsum error deep inside.
        if q.device != self.device:
            raise ValueError(
                f"CSADecodingCache.forward_step: q.device={q.device} does "
                f"not match cache's device={self.device}. Call "
                f"cache.to(device=q.device) or move q to the cache's device.")
        device, dtype = q.device, q.dtype
        compute_dtype = (
            torch.float64 if dtype == torch.float64 else torch.float
        )
        # The current query is at absolute position t = n_tokens_seen - 1
        # (we just appended it in ``append_step``). Its causal block
        # mask is ``b < t // m`` (paper Eq. 16): a query attends only to
        # STRICTLY PRECEDING compressed blocks — the query's own block
        # (complete or not) is excluded; its local context is served by
        # the sliding-window branch instead. This matches
        # ``ops_csa._causal_block_mask`` exactly, so incremental decoding
        # is equivalent to the full-sequence ``naive_csa`` path.
        t = self._n_tokens_seen - 1
        # Partial RoPE (§2.3.3): rotate the query at its own token
        # position AFTER the caller's normalization (the same
        # normalize-then-rotate order as ``naive_csa``'s query path).
        # The rotation applies to the last ``rope_dim_eff`` dims of each
        # per-head vector; the head axis is untouched.
        if self.rope_dim_eff > 0:
            pos_t = _pos_tensor(t, device)
            q = partial_rope(
                _to_dtype(q, compute_dtype), pos_t,
                self.rope_dim_eff, self.rope_base, dim=1)          # [B, 1, nh, c]
        # The block containing t is ``t // m``. It is visible only from
        # the NEXT block's queries onward (strictly-preceding rule), so a
        # partial (or just-completed) block stays out of the sparse path
        # for the queries inside it.
        #
        # The causal_block_mask passed to ``csa_lightning_indexer`` is
        # ``[T, n_blocks]`` (one row per query). With T=1, it is
        # ``[1, n_blocks]`` with True for the strictly-preceding blocks.
        # Core-attention scale: compute ONCE here so it is available to
        # BOTH the sparse branch (below) and the sliding-window branch
        # (further down). The assignment must not live inside either
        # branch: ``core_scale`` has to be bound even when
        # ``self._C_comp is None or self._n_blocks == 0`` (the first ``m``
        # decode steps before the first compressed block is formed).
        # Auto-selection mirrors ``naive_csa``: cosine scale 1.0 for 'l2',
        # the standard 1/sqrt(c) temperature for the RMS form (whose
        # normalized vectors keep norm ~sqrt(c)), and the classical
        # dot-product scale when normalization is off.
        if scale is None:
            if not normalize_qk or self.qk_norm_mode == 'rms':
                core_scale = self.c ** -0.5
            else:
                core_scale = 1.0
        else:
            core_scale = scale
        # Shared by the sparse-block and sliding-window branches.
        q_compute = _to_dtype(q, compute_dtype)                        # [B, 1, nh, c]

        if self._C_comp is None or self._n_blocks == 0:
            # No compressed blocks yet — no block scores.
            scores = torch.empty(
                B, 1, nh, 0, dtype=compute_dtype, device=device,
            )
            valid_mask = torch.empty(
                B, 1, 0, dtype=torch.bool, device=device,
            )
            kv = torch.empty(
                B, 1, 0, self.c, dtype=compute_dtype, device=device,
            )
        else:
            n_blocks = self._n_blocks
            # Causal block mask (paper Eq. 16): block b is valid iff
            # b < t // m — only STRICTLY PRECEDING blocks are visible to
            # the query at position t. The block containing t (partial or
            # just-completed) is excluded; its content is reachable via
            # the sliding-window branch.
            b_threshold = t // self.m
            cbm = _block_valid_mask(
                n_blocks, b_threshold, device)[None, :]                 # [1, n_blocks]
            # Honor ``normalize_qk`` for the CORE attention (as in
            # ``naive_csa``). When normalize_qk=True compressed KV are
            # normalized with ``qk_norm_mode`` (cosine-similarity
            # attention for 'l2', the paper §2.3.3 RMSNorm form for
            # 'rms'). When False, skip normalization and fall back to the
            # classical 1/sqrt(c) dot-product scale.
            #
            # ``_qk_normalize`` is a per-row map, so normalizing only the
            # ``topk`` gathered rows (below) is bit-identical to
            # normalizing all ``n_blocks`` rows first and gathering after;
            # the compressed rows are therefore gathered straight out of
            # the storage-dtype cache and only the selected rows are
            # upcast (a cast commutes with row selection).
            C_comp_storage = self._C_comp[:, :self._n_blocks]
            if torch.is_grad_enabled():
                # Every read-path op that builds a graph (gather,
                # normalize, einsum) saves its input for backward. When
                # the cast/gather aliases the compressed storage, a later
                # ``append_step`` overwrites that storage in place and
                # backward raises a version-counter error. Keep a stable
                # copy while autograd is recording.
                C_comp_storage = C_comp_storage.clone()
            # The indexer scores ALL blocks (it must, to select), so its
            # keys are upcast whole. When autograd is recording, the same
            # version-counter contract applies to the indexer's
            # normalization/einsums, hence the clone.
            K_IComp_valid = _to_dtype(
                self._K_IComp[:, :self._n_blocks], compute_dtype)
            if (torch.is_grad_enabled()
                    and self.dtype == compute_dtype):
                # The clone is only required when the cast above returned
                # a view aliasing the cache storage (dtype match); with a
                # dtype mismatch the cast already allocated an unaliased
                # copy.
                K_IComp_valid = K_IComp_valid.clone()
            # Lightning indexer: score the current query against ALL
            # cached indexer keys, select top-k (respecting the causal
            # block mask). STE is only useful when autograd is enabled; in
            # no-grad inference the two forms are forward-equivalent, so
            # gate it just like ``naive_csa``.
            effective_use_ste = use_ste and torch.is_grad_enabled()
            if (normalize_qk and not torch.is_grad_enabled()
                    and self.dtype != compute_dtype):
                # ``K_IComp_valid`` is a fresh fp32/fp64 copy here, and
                # the indexer would normalize it with ``F.normalize``
                # (mode 'l2') as its first step. Normalizing the copy in
                # place performs the identical per-element chain and
                # avoids the second full-sized temporary. The query is
                # normalized the same way (it is a single [B, 1, nIh, c_I]
                # row per step).
                _qk_normalize_(K_IComp_valid, 'l2')
                q_idx_n = torch.nn.functional.normalize(
                    _to_dtype(q_idx, compute_dtype), dim=-1)
                # ``scale=1.0`` + ``normalize_qk=False`` reproduces the
                # indexer's ``normalize_qk=True`` auto-scale exactly.
                indexer_result = csa_lightning_indexer(
                    q_idx_n,
                    K_IComp_valid,
                    _to_dtype(w_idx, compute_dtype),
                    topk,
                    scale=1.0,
                    causal_block_mask=cbm,
                    return_soft_weights=effective_use_ste,
                    ste_mode=ste_mode,
                    normalize_qk=False,
                )
            else:
                indexer_result = csa_lightning_indexer(
                    _to_dtype(q_idx, compute_dtype),                    # [B, 1, nIh, c_I]
                    K_IComp_valid,                                      # [B, n_blocks, c_I]
                    _to_dtype(w_idx, compute_dtype),                    # [B, 1, nIh]
                    topk,
                    # Defer scale selection to csa_lightning_indexer — it
                    # picks 1.0 when normalize_qk=True (cosine),
                    # 1/sqrt(c_I) otherwise.
                    scale=None,
                    causal_block_mask=cbm,
                    return_soft_weights=effective_use_ste,
                    ste_mode=ste_mode,
                    normalize_qk=normalize_qk,
                )
            if effective_use_ste:
                indices, soft_weights = indexer_result
            else:
                indices = indexer_result
                soft_weights = None
            # ``indices``: [B, 1, topk], padded with -1 for invalid slots.
            # ``soft_weights``: [B, 1, n_blocks] (differentiable).
            if indices.shape[-1] == 0:
                # topk=0: no block scores.
                scores = torch.empty(
                    B, 1, nh, 0, dtype=compute_dtype, device=device,
                )
                valid_mask = torch.empty(
                    B, 1, 0, dtype=torch.bool, device=device,
                )
                kv = torch.empty(
                    B, 1, 0, self.c, dtype=compute_dtype, device=device,
                )
            else:
                valid_mask = indices >= 0                              # [B, 1, topk]
                idx_safe = indices.clamp(min=0)                        # [B, 1, topk]
                # Per-batch ``dim=1`` gather straight out of the
                # storage-dtype cache (value-identical to the flat
                # ``index_select`` used by ``naive_csa``; distinct blocks
                # per row). Gathering before the upcast keeps the compute
                # dtype allocation at [B, 1, topk, c] instead of
                # [B, n_blocks, c], and avoids materializing the
                # ``[B*n_blocks, c]`` reshape copy when B > 1.
                nblk = C_comp_storage.shape[1]
                dim = C_comp_storage.shape[-1]
                kv = torch.gather(
                    C_comp_storage.unsqueeze(1).expand(B, 1, nblk, dim),
                    2,
                    idx_safe.unsqueeze(-1).expand(
                        B, 1, idx_safe.shape[-1], dim),
                )                                                      # [B, 1, topk, c]
                kv = _to_dtype(kv, compute_dtype)
                if normalize_qk:
                    # Per-row normalization after the gather (see the note
                    # at ``C_comp_storage`` above). ``kv`` is the fresh
                    # gather output, so outside autograd the normalization
                    # runs in place (bit-identical per-element chain).
                    if torch.is_grad_enabled():
                        kv = _qk_normalize(kv, self.qk_norm_mode)
                    else:
                        kv = _qk_normalize_(kv, self.qk_norm_mode)
                # STE for the indexer (mirrors ``naive_csa``).
                if effective_use_ste and soft_weights is not None:
                    soft_weights_selected = torch.gather(
                        soft_weights, dim=-1, index=idx_safe,
                    )                                                  # [B, 1, topk]
                    soft_weights_selected = soft_weights_selected * \
                        valid_mask.to(soft_weights_selected.dtype)
                    soft_kv = soft_weights_selected.unsqueeze(-1) * kv.detach()
                    if ste_mode == 'full_softmax':
                        # The full-softmax expected KV reads every block,
                        # so only this branch needs the fully normalized
                        # compressed rows (identical per-row values).
                        C_comp_n = _to_dtype(C_comp_storage, compute_dtype)
                        if normalize_qk:
                            C_comp_n = _qk_normalize(
                                C_comp_n, self.qk_norm_mode)
                        soft_full = torch.einsum(
                            'btn,bnc->btc', soft_weights, C_comp_n,
                        )                                              # [B, 1, c]
                        topk_size = kv.shape[2]
                        # The same expansion serves the value and its
                        # detached copy.
                        soft_full_exp = soft_full.unsqueeze(2).expand(
                            -1, -1, topk_size, -1)
                        aux = soft_full_exp - soft_full_exp.detach()
                        soft_kv = soft_kv + aux
                    kv = kv + (soft_kv - soft_kv.detach())
                # Per-head attention scores over the topk selected blocks.
                scores = torch.einsum(
                    'b t h d, b t k d -> b t h k', q_compute, kv,
                )                                                      # [B, 1, nh, topk]
                if core_scale != 1.0:
                    # Multiplying by exactly 1.0 is an IEEE no-op, so the
                    # guarded multiply is value-identical. Outside autograd
                    # the fresh einsum output is scaled in place.
                    if torch.is_grad_enabled():
                        scores = scores * core_scale
                    else:
                        scores.mul_(core_scale)
                # The einsum output (or its freshly scaled copy) is not
                # saved by any autograd node, so the mask can be written
                # in place (mirrors ``naive_csa``).
                scores.masked_fill_(
                    ~valid_mask[:, :, None, :], float('-inf'))

        # --- Sliding-window branch: window scores for the joint softmax ---
        if self._sw is None or self._sw.n_valid == 0:
            scores_w = torch.empty(
                B, 1, nh, 0, dtype=compute_dtype, device=device,
            )
            C_local = torch.empty(B, 0, self.c, dtype=compute_dtype, device=device)
        else:
            # SW buffer contents in causal order: [B, sw_len, c]. When
            # normalize_qk=True (default), the buffer stores keys
            # normalized with ``qk_norm_mode`` (applied at append time).
            # When normalize_qk=False the caller should have appended
            # raw (unnormalized) keys; we read whatever is stored.
            # ``clone()``: ``get()`` returns a view into the ring buffer,
            # and the einsum's below save it to compute gradients. A later
            # ``append_step`` overwrites the buffer in place, which would
            # invalidate that saved view (backward would raise "modified by
            # an inplace operation"). Materialize a stable copy so one
            # backward pass can span several decode steps.
            # The copy is needed whenever grad is enabled: the
            # ``sw_out`` einsum saves the view for the gradient w.r.t. the
            # softmax weights ``p_w``, which can require grad even when
            # neither ``q_compute`` nor the buffer entries do (e.g. a
            # learnable attention sink). Outside grad mode nothing can
            # save the view, so the copy is skipped.
            C_local = _to_dtype(self._sw.get(), compute_dtype)         # [B, sw_len, c]
            if torch.is_grad_enabled():
                C_local = C_local.clone()
            sw = torch.einsum(
                'b h d, b s d -> b h s', q_compute[:, 0], C_local,
            )                                                          # [B, nh, sw_len]
            if core_scale != 1.0:
                # Multiplying by exactly 1.0 is an IEEE no-op, so the
                # guarded multiply is value-identical. Outside autograd the
                # fresh einsum output is scaled in place.
                if torch.is_grad_enabled():
                    sw = sw * core_scale
                else:
                    sw.mul_(core_scale)
            scores_w = sw.unsqueeze(1)                                 # [B, 1, nh, sw_len]

        # JOINT softmax over [blocks; window] with the sink in the single
        # shared denominator (matches full-sequence naive_csa; DeepSeek-V4
        # §2.3.1 Eq. 27). The SW buffer holds EXACTLY the valid window for
        # the newest query, so no -inf padding is needed for the window part.
        n_win = scores_w.shape[-1]
        cat = torch.cat([scores, scores_w], dim=-1)                    # [B, 1, nh, topk+sw_len]
        if n_win:
            cat_valid = torch.cat(
                [valid_mask[:, :, None, :],
                 _bool_ones(B, n_win, device)],
                dim=-1)
        else:
            cat_valid = valid_mask[:, :, None, :]
        if cat.shape[-1] == 0:
            return torch.zeros(B, 1, nh, self.c, dtype=compute_dtype, device=device)
        if sink_logits is not None:
            log_sink = sink_logits.view(1, 1, nh, 1).to(cat)
            vmask = cat_valid.to(cat.dtype)
            row_max = cat.amax(-1, keepdim=True).clamp(min=0)
            shifted = cat - row_max
            shifted_sink = log_sink - row_max
            log_sum_exp = torch.logsumexp(shifted, dim=-1, keepdim=True)
            log_denom = torch.logaddexp(log_sum_exp, shifted_sink)
            p = (shifted - log_denom).exp() * vmask
            all_invalid = ~cat_valid.any(-1, keepdim=True)
            if torch.is_grad_enabled():
                p = p.masked_fill(all_invalid, 0.0)
            else:
                p.masked_fill_(all_invalid, 0.0)
        else:
            all_invalid = ~cat_valid.any(-1, keepdim=True)
            if not torch.is_grad_enabled():
                # ``cat`` / the softmax output are fresh tensors of this
                # step; outside autograd the zeroing runs in place.
                safe = cat.masked_fill_(all_invalid, 0.0)
                p = torch.softmax(safe, dim=-1)
                p.masked_fill_(all_invalid, 0.0)
            else:
                safe = cat.masked_fill(all_invalid, 0.0)
                p = torch.softmax(safe, dim=-1)
                p = p.masked_fill(all_invalid, 0.0)

        topk_size = scores.shape[-1]
        p_b = p[..., :topk_size]
        p_w = p[..., topk_size:]
        sparse_out = torch.einsum(
            'b t h k, b t k d -> b t h d', p_b, kv,
        )                                                              # [B, 1, nh, c]
        # Accumulate the window contribution in place (same values as
        # ``a + b``; the add's backward saves neither operand).
        sparse_out.add_(torch.einsum(
            'b t h w, b w d -> b t h d', p_w, C_local,
        ))                                                             # [B, 1, nh, c]
        out = sparse_out
        # Partial RoPE output countermeasure (§2.3.3): inverse-rotate the
        # core-attention output at the NEGATED query position so each KV
        # entry's contribution carries the relative (entry − query)
        # distance — mirroring ``naive_csa``'s output path.
        if self.rope_dim_eff > 0:
            pos_t = _pos_tensor(t, device)
            out = partial_rope(
                out, -pos_t, self.rope_dim_eff, self.rope_base, dim=1)
        return out                                                     # [B, 1, nh, c]


    def forward_chunk(
        self,
        Ca_new: torch.Tensor,
        Cb_new: torch.Tensor,
        Za_new: torch.Tensor,
        Zb_new: torch.Tensor,
        K_idx_new: torch.Tensor,
        Z_idx_new: torch.Tensor,
        Ba: torch.Tensor,
        Bb: torch.Tensor,
        B_idx: torch.Tensor,
        q: torch.Tensor,
        q_idx: torch.Tensor,
        w_idx: torch.Tensor,
        *,
        topk: int,
        nh: int,
        nIh: int,
        scale: float | None = None,
        sink_logits: torch.Tensor | None = None,
        use_ste: bool = True,
        ste_mode: str = 'topk_columns',
        normalize_qk: bool = True,
    ) -> torch.Tensor:
        """Batched equivalent of ``append_step`` + ``forward_step`` per token.

        Feeds a whole continuation chunk through the cache and returns all
        ``T_new`` outputs at once.  The cache state after the call is
        IDENTICAL to the per-token ``append_step`` loop (same block
        boundaries, same compression helpers); only the attention is
        evaluated with a batched block mask and batched per-query sliding
        windows.  This is the chunked-prefill fast path: one projection
        GEMM (done by the caller) plus one attention call per chunk
        instead of one per token.

        The per-query window is reconstructed from a snapshot of the
        pre-chunk SW tail plus the chunk keys.  Queries inside the chunk
        therefore see exactly the history they would see in the
        token-by-token path; future tokens of the same chunk are excluded
        by construction (window end index == query index).

        Args:
            Ca_new, Cb_new, Za_new, Zb_new: ``[B, T_new, c]`` projections.
            K_idx_new, Z_idx_new: ``[B, T_new, c_I]`` indexer projections.
            Ba, Bb, B_idx: positional biases (same shapes as
                :meth:`append_step`).
            q: ``[B, T_new, nh, c]`` queries (already normalized iff
                ``normalize_qk``); rotated inside at absolute positions.
            q_idx: ``[B, T_new, nIh, c_I]`` indexer queries.
            w_idx: ``[B, T_new, nIh]`` indexer head weights.
            topk, nh, nIh, scale, sink_logits, use_ste, ste_mode,
                normalize_qk: same semantics as :meth:`forward_step`.

        Returns:
            ``[B, T_new, nh, c]`` attention output in ``compute_dtype``.
        """
        B, T_new, c = Ca_new.shape
        if T_new < 1:
            raise ValueError(
                f'CSADecodingCache.forward_chunk: T_new={T_new} must be >= 1')
        if B != self.B:
            raise ValueError(
                f'CSADecodingCache.forward_chunk: batch size {B} does not '
                f"match cache's B={self.B}.")
        if c != self.c:
            raise ValueError(
                f'CSADecodingCache.forward_chunk: c={c} does not match '
                f"cache's c={self.c}.")
        if q.shape != (B, T_new, nh, c):
            raise ValueError(
                f'CSADecodingCache.forward_chunk: q shape {tuple(q.shape)} '
                f'does not match {(B, T_new, nh, c)}.')
        if q_idx.shape[0] != B or q_idx.shape[1] != T_new \
                or q_idx.shape[2] != nIh:
            raise ValueError(
                f'CSADecodingCache.forward_chunk: q_idx shape '
                f'{tuple(q_idx.shape)} does not match '
                f'{(B, T_new, nIh, q_idx.shape[-1])}.')
        if w_idx.shape != (B, T_new, nIh):
            raise ValueError(
                f'CSADecodingCache.forward_chunk: w_idx shape '
                f'{tuple(w_idx.shape)} does not match '
                f'{(B, T_new, nIh)}.')
        if q.device != self.device:
            raise ValueError(
                f'CSADecodingCache.forward_chunk: q.device={q.device} does '
                f"not match cache's device={self.device}.")
        if T_new == 1:
            # Keep the canonical decode step for the singleton case.
            self.append_step(
                Ca_new, Cb_new, Za_new, Zb_new, K_idx_new, Z_idx_new,
                Ba, Bb, B_idx)
            return self.forward_step(
                q, q_idx, w_idx, topk=topk, nh=nh, nIh=nIh, scale=scale,
                sink_logits=sink_logits, use_ste=use_ste,
                ste_mode=ste_mode, normalize_qk=normalize_qk)

        # Cast once so the SW-key preparation below sees exactly the
        # tensors ``append_step`` will store.
        Ca_new = _to_cache(Ca_new, self.dtype, self.device)
        Cb_new = _to_cache(Cb_new, self.dtype, self.device)
        Za_new = _to_cache(Za_new, self.dtype, self.device)
        Zb_new = _to_cache(Zb_new, self.dtype, self.device)
        K_idx_new = _to_cache(K_idx_new, self.dtype, self.device)
        Z_idx_new = _to_cache(Z_idx_new, self.dtype, self.device)
        n0 = self._n_tokens_seen
        compute_dtype = (
            torch.float64 if self.dtype == torch.float64 else torch.float)
        # Snapshot the pre-chunk SW tail BEFORE ``append_step`` overwrites
        # the ring (clone: the ring is mutated in place).
        pre_sw = None
        if self._sw is not None and self._sw.n_valid:
            pre_sw = self._sw.get().clone()
        # Reproduce exactly the SW key preparation in ``_append_bulk``:
        # normalize (when enabled) then rotate at absolute positions.
        ca_store_store = None
        if self._sw is not None:
            if self._normalize_qk:
                ca_src = _to_dtype(Ca_new, compute_dtype)
                if (not torch.is_grad_enabled()
                        and self.dtype != compute_dtype):
                    ca_store = _qk_normalize_(ca_src, self.qk_norm_mode)
                else:
                    ca_store = _qk_normalize(ca_src, self.qk_norm_mode)
            else:
                ca_store = Ca_new
            if self.rope_dim_eff > 0:
                pos = torch.arange(
                    n0, n0 + T_new, device=self.device)
                ca_store = partial_rope(
                    _to_dtype(ca_store, compute_dtype), pos,
                    self.rope_dim_eff, self.rope_base, dim=1)
            ca_store_store = _to_dtype(ca_store, self.dtype)
        # Advance the cache exactly as the per-token loop would.
        self.append_step(
            Ca_new, Cb_new, Za_new, Zb_new, K_idx_new, Z_idx_new,
            Ba, Bb, B_idx)
        t_pos = torch.arange(n0, n0 + T_new, device=self.device)
        q_compute = _to_dtype(q, compute_dtype)
        if self.rope_dim_eff > 0:
            q_compute = partial_rope(
                q_compute, t_pos, self.rope_dim_eff, self.rope_base, dim=1)
        if scale is None:
            core_scale = (self.c ** -0.5
                          if (not normalize_qk or self.qk_norm_mode == 'rms')
                          else 1.0)
        else:
            core_scale = scale

        # --- compressed-block branch (indexer top-k) -------------------
        n_blocks = self._n_blocks
        if n_blocks:
            cbm = (torch.arange(n_blocks, device=self.device)[None, :]
                   < (t_pos // self.m)[:, None])                    # [T, nb]
            K_I = _to_dtype(
                self._K_IComp[:, :n_blocks], compute_dtype)
            if (torch.is_grad_enabled()
                    and self.dtype == compute_dtype):
                # Same grad-mode clone contract as ``forward_step``: the
                # cast above aliases the cache storage when the dtypes
                # match, and the indexer's normalization/einsums save
                # these keys for backward. A later ``append_step`` would
                # overwrite that storage in place.
                K_I = K_I.clone()
            effective_use_ste = use_ste and torch.is_grad_enabled()
            if (normalize_qk and not torch.is_grad_enabled()
                    and self.dtype != compute_dtype):
                _qk_normalize_(K_I, 'l2')
                q_idx_n = torch.nn.functional.normalize(
                    _to_dtype(q_idx, compute_dtype), dim=-1)
                indexer_result = csa_lightning_indexer(
                    q_idx_n, K_I, _to_dtype(w_idx, compute_dtype), topk,
                    scale=1.0, causal_block_mask=cbm,
                    return_soft_weights=effective_use_ste,
                    ste_mode=ste_mode, normalize_qk=False)
            else:
                indexer_result = csa_lightning_indexer(
                    _to_dtype(q_idx, compute_dtype), K_I,
                    _to_dtype(w_idx, compute_dtype), topk, scale=None,
                    causal_block_mask=cbm,
                    return_soft_weights=effective_use_ste,
                    ste_mode=ste_mode, normalize_qk=normalize_qk)
            if effective_use_ste:
                indices, soft_weights = indexer_result
            else:
                indices = indexer_result
                soft_weights = None
            if indices.shape[-1] == 0:
                scores = torch.empty(
                    B, T_new, nh, 0, dtype=compute_dtype,
                    device=self.device)
                valid_mask = torch.empty(
                    B, T_new, 0, dtype=torch.bool, device=self.device)
                kv = torch.empty(
                    B, T_new, 0, self.c, dtype=compute_dtype,
                    device=self.device)
            else:
                valid_mask = indices >= 0                       # [B,T,topk]
                idx_safe = indices.clamp(min=0)
                C_storage = self._C_comp[:, :n_blocks]
                if torch.is_grad_enabled():
                    # Same grad-mode clone contract as ``forward_step``:
                    # gather/normalize/einsum save their input for
                    # backward, so the read must not alias storage that a
                    # later ``append_step`` overwrites in place.
                    C_storage = C_storage.clone()
                dim = C_storage.shape[-1]
                kv = torch.gather(
                    C_storage.unsqueeze(1).expand(
                        B, T_new, n_blocks, dim),
                    2,
                    idx_safe.unsqueeze(-1).expand(
                        B, T_new, idx_safe.shape[-1], dim),
                )                                               # [B,T,topk,c]
                kv = _to_dtype(kv, compute_dtype)
                if normalize_qk:
                    kv = _qk_normalize(kv, self.qk_norm_mode)
                if effective_use_ste and soft_weights is not None:
                    soft_sel = torch.gather(
                        soft_weights, dim=-1, index=idx_safe)
                    soft_sel = soft_sel * valid_mask.to(soft_sel.dtype)
                    soft_kv = soft_sel.unsqueeze(-1) * kv.detach()
                    if ste_mode == 'full_softmax':
                        C_all = _to_dtype(C_storage, compute_dtype)
                        if normalize_qk:
                            C_all = _qk_normalize(
                                C_all, self.qk_norm_mode)
                        soft_full = torch.einsum(
                            'btn,bnc->btc', soft_weights, C_all)
                        topk_size = kv.shape[2]
                        soft_full_exp = soft_full.unsqueeze(2).expand(
                            -1, -1, topk_size, -1)
                        aux = soft_full_exp - soft_full_exp.detach()
                        soft_kv = soft_kv + aux
                    kv = kv + (soft_kv - soft_kv.detach())
                scores = torch.einsum(
                    'bthd,btkd->bthk', q_compute, kv)
                if core_scale != 1.0:
                    scores = scores * core_scale
                scores = scores.masked_fill(
                    ~valid_mask[:, :, None, :], float('-inf'))
        else:
            cbm = torch.empty(
                T_new, 0, dtype=torch.bool, device=self.device)
            scores = torch.empty(
                B, T_new, nh, 0, dtype=compute_dtype, device=self.device)
            valid_mask = torch.empty(
                B, T_new, 0, dtype=torch.bool, device=self.device)
            kv = torch.empty(
                B, T_new, 0, self.c, dtype=compute_dtype,
                device=self.device)

        # --- sliding-window branch ------------------------------------
        if self._sw is not None and ca_store_store is not None:
            if pre_sw is None:
                keys = ca_store_store
                n_pre = 0
            else:
                keys = torch.cat([pre_sw, ca_store_store], dim=1)
                n_pre = pre_sw.shape[1]
            windows, valid_w = _chunk_window_keys(
                keys, n_pre, self.win)
            windows = _to_dtype(windows, compute_dtype)
            scores_w = torch.einsum(
                'bthd,btwd->bthw', q_compute, windows)
            if core_scale != 1.0:
                scores_w = scores_w * core_scale
            scores_w = scores_w.masked_fill(
                ~valid_w[None, :, None, :], float('-inf'))
        else:
            windows = torch.empty(
                B, T_new, 0, self.c, dtype=compute_dtype,
                device=self.device)
            scores_w = torch.empty(
                B, T_new, nh, 0, dtype=compute_dtype, device=self.device)
            valid_w = torch.empty(
                T_new, 0, dtype=torch.bool, device=self.device)

        # --- joint softmax + output -----------------------------------
        n_sparse = scores.shape[-1]
        n_win = scores_w.shape[-1]
        if n_sparse + n_win == 0:
            out = torch.zeros(
                B, T_new, nh, self.c, dtype=compute_dtype,
                device=self.device)
        else:
            logits = torch.cat([scores, scores_w], dim=-1)
            valid = torch.cat([
                valid_mask[:, :, None, :],
                valid_w[None, :, None, :].expand(
                    B, T_new, 1, valid_w.shape[-1]),
            ], dim=-1)
            p = _joint_softmax_chunk(logits, valid, sink_logits)
            p_b = p[..., :n_sparse]
            p_w = p[..., n_sparse:]
            out = torch.zeros(
                B, T_new, nh, self.c, dtype=compute_dtype,
                device=self.device)
            if n_sparse:
                out = out + torch.einsum(
                    'bthk,btkd->bthd', p_b, kv)
            if n_win:
                out = out + torch.einsum(
                    'bthw,btwd->bthd', p_w, windows)
        if self.rope_dim_eff > 0:
            out = partial_rope(
                out, -t_pos, self.rope_dim_eff, self.rope_base, dim=1)
        return out


# =============================================================================
# HCADecodingCache
# =============================================================================

class HCADecodingCache:
    """Incremental decoding cache for HCA layers.

    State
    -----
    Mirrors :class:`CSADecodingCache` but simplified (no indexer, no
    overlap):

    1. **Partial block accumulator** (Python lists of ``[B, 1, c]``
       tensors). Holds 0..m2-1 tokens whose ``C`` and ``Z`` projections
       have been computed but not yet compressed. Cleared after every
       ``m2`` tokens.
    2. **Compressed block cache** (``[B, n_blocks, c]``). Grows by one
       row each time the accumulator fills. Read by the dense-attention
       branch during ``forward_step``.
    3. **Sliding-window ring buffer** (:class:`_SlidingWindowRingBuffer`).
       Holds the most recent ``win`` ``qk_norm_mode``-normalized ``C``
       entries.

    See :class:`CSADecodingCache` for the lifecycle and numerical
    contract. The same bit-equivalence guarantee applies (verified by
    ``test_hca_decoding_cache_correctness`` in ``run_correctness.py``).
    """

    def __init__(
        self,
        B: int,
        c: int,
        m2: int,
        win: int,
        device,
        dtype,
        rope_dim: int | None = None,
        rope_base: float = 10000.0,
        qk_norm_mode: str = 'l2',
    ):
        if B < 1:
            raise ValueError(f"B={B} must be >= 1")
        if c < 1:
            raise ValueError(f"c={c} must be >= 1")
        if m2 < 1:
            raise ValueError(f"m2={m2} must be >= 1")
        if win < 0:
            raise ValueError(f"win={win} must be >= 0 (0 disables SW)")
        self.B, self.c, self.m2, self.win = B, c, m2, win
        _validate_cache_dtype(dtype)
        self.device, self.dtype = torch.device(device), dtype
        # normalize_qk flag: set by the caller after construction (or
        # default True to match the repo-wide convention). The cache
        # needs this to decide whether to apply the ``qk_norm_mode``
        # normalization to keys before appending to the sliding-window
        # ring buffer.
        self._normalize_qk: bool = True
        # Partial RoPE (DeepSeek-V4 §2.3.3) — same contract as
        # CSADecodingCache / naive_hca: compressed entries rotate at their
        # block-anchor positions (``s*m2``), SW keys at their token
        # positions, the query at the current position, and the output is
        # inverse-rotated at the negated query position. ``rope_dim=None``
        # (default) disables RoPE.
        if rope_base <= 0:
            raise ValueError(
                f"HCADecodingCache: rope_base={rope_base!r} must be positive")
        # qk_norm_mode must match the mode the full-sequence ``naive_hca``
        # call uses; otherwise the incremental decode would normalize keys
        # differently from prefill and silently disagree with it.
        if qk_norm_mode not in ('l2', 'rms'):
            raise ValueError(
                f"HCADecodingCache: qk_norm_mode={qk_norm_mode!r} must be "
                f"'l2' (cosine, historical) or 'rms' (paper §2.3.3)")
        self.qk_norm_mode = qk_norm_mode
        self.rope_dim_eff = effective_rope_dim(rope_dim, c)
        self.rope_base = float(rope_base)
        self.reset()

    def reset(self) -> None:
        self._acc_C: list[torch.Tensor] = []
        self._acc_Z: list[torch.Tensor] = []
        # Geometrically-grown storage; only the valid prefix is exposed.
        self._C_comp: torch.Tensor | None = None   # [B, capacity, c]
        if self.win > 0:
            self._sw = _SlidingWindowRingBuffer(
                self.B, self.win, self.c, self.device, self.dtype,
            )
        else:
            self._sw = None
        self._n_tokens_seen = 0
        self._n_blocks = 0

    def _ensure_compressed_capacity(self, required: int) -> None:
        """Grow HCA compressed storage geometrically when needed.

        Same contract as :meth:`CSADecodingCache._ensure_compressed_capacity`:
        ``required`` is rounded up to the geometric capacity so the
        retained capacity is a function of the completed-block count only
        (bulk prefill and token-by-token fills allocate identically), which
        keeps the Exp 3 ``full_accounting`` model exact.
        """
        if self._C_comp is not None and required <= self._C_comp.shape[1]:
            return
        old_capacity = 0 if self._C_comp is None else self._C_comp.shape[1]
        new_capacity = max(_geometric_capacity(required), old_capacity * 2)
        with normal_tensor_scope():
            new_C = torch.empty(
                self.B, new_capacity, self.c,
                device=self.device, dtype=self.dtype,
            )
        if self._n_blocks:
            new_C[:, :self._n_blocks] = self._C_comp[:, :self._n_blocks]
        self._C_comp = new_C

    # ----- properties ----------------------------------------------------

    @property
    def n_tokens_seen(self) -> int:
        return self._n_tokens_seen

    @property
    def n_blocks(self) -> int:
        return self._n_blocks

    @property
    def accumulator_len(self) -> int:
        return len(self._acc_C)

    @property
    def C_comp(self) -> torch.Tensor | None:
        if self._C_comp is None:
            return None
        return self._C_comp[:, :self._n_blocks]

    @property
    def sw_buffer(self) -> _SlidingWindowRingBuffer | None:
        return self._sw

    # ----- state accounting ----------------------------------------------

    @property
    def nbytes(self) -> int:
        """Physical bytes retained by the cache's backing tensors.

        Counts ALLOCATED storage (geometric spare capacity + the full
        sliding-window ring buffer); ``nbytes_valid`` counts the live
        entries only. Mirrors :class:`Mamba2DecodingState`'s surface.
        """
        return self.nbytes_breakdown()['total_allocated_bytes']

    @property
    def nbytes_valid(self) -> int:
        """Bytes of logically live state (valid prefix + accumulator)."""
        return self.nbytes_breakdown()['total_valid_bytes']

    def nbytes_breakdown(self) -> dict:
        """Component-wise decode-state byte accounting (JSON-safe ints).

        Components (allocated / valid): ``compressed_kv`` (the heavy
        ``[B, cap, c]`` block cache), ``sliding_window`` (fixed ring
        buffer), and ``partial_tail`` (the ``n_tokens_seen % m2``
        accumulator rows). The allocated totals match ``run_kv_cache``'s
        ``full_accounting`` components for one HCA layer at the same
        ``(T, m2, c, win)``.
        """
        B, c = self.B, self.c
        cap = 0 if self._C_comp is None else self._C_comp.shape[1]
        compressed_valid_elems = self._n_blocks * c * B
        compressed_alloc = _tensor_nbytes(self._C_comp)
        compressed_valid = _valid_nbytes(self._C_comp, compressed_valid_elems)
        sw_alloc = _tensor_nbytes(self._sw._buf) if self._sw is not None else 0
        sw_valid = (_valid_nbytes(self._sw._buf, self._sw._sw_len * c * B)
                    if self._sw is not None else 0)
        partial_alloc = sum(
            _tensor_nbytes(t)
            for group in (self._acc_C, self._acc_Z)
            for t in group)
        partial_valid = partial_alloc
        total_alloc = compressed_alloc + sw_alloc + partial_alloc
        total_valid = compressed_valid + sw_valid + partial_valid
        return {
            'compressed_kv_allocated_bytes': compressed_alloc,
            'compressed_kv_valid_bytes': compressed_valid,
            'sliding_window_allocated_bytes': sw_alloc,
            'sliding_window_valid_bytes': sw_valid,
            'partial_tail_allocated_bytes': partial_alloc,
            'partial_tail_valid_bytes': partial_valid,
            'total_allocated_bytes': total_alloc,
            'total_valid_bytes': total_valid,
            'capacity_rows': int(cap),
            'n_blocks': int(self._n_blocks),
            'n_tokens_seen': int(self._n_tokens_seen),
        }

    # ----- state migration -----------------------------------------------

    def to(self, device=None, dtype=None) -> 'HCADecodingCache':
        if device is not None:
            device = torch.device(device)
        if device is not None and device != self.device:
            self.device = device
        if dtype is not None and dtype != self.dtype:
            self.dtype = dtype
        self._acc_C = [_to_cache(t, self.dtype, self.device)
                       for t in self._acc_C]
        self._acc_Z = [_to_cache(t, self.dtype, self.device)
                       for t in self._acc_Z]
        if self._C_comp is not None:
            self._C_comp = _to_cache(self._C_comp, self.dtype, self.device)
        if self._sw is not None:
            self._sw.to(device=self.device, dtype=self.dtype)
        return self

    # ----- core: append new tokens ---------------------------------------

    def append_step(
        self,
        C_new: torch.Tensor,     # [B, T_new, c]
        Z_new: torch.Tensor,     # [B, T_new, c]
        B_pos: torch.Tensor,     # [m2, c]
    ) -> list[int]:
        """Append ``T_new`` new tokens' projections to the cache.

        For each new token:
          1. Append its ``C`` and ``Z`` projections to the accumulator.
          2. Append its (``qk_norm_mode``-normalized) ``C`` to the SW
             ring buffer.
          3. If the accumulator now has ``m2`` tokens, compress them
             into one new block, push to the compressed block cache,
             clear the accumulator.

        Returns:
            List of indices of newly-compressed blocks (absolute block
            indices). Empty if no block was completed.
        """
        B, T_new, c = C_new.shape
        if B != self.B:
            raise ValueError(
                f"HCADecodingCache.append_step: batch size {B} does not "
                f"match cache's B={self.B}. Call reset() between sequences "
                f"with different batch sizes.")
        if c != self.c:
            raise ValueError(
                f"HCADecodingCache.append_step: c={c} does not match "
                f"cache's c={self.c}.")
        # Validate T_new consistency between C_new and Z_new.
        if Z_new.shape[1] != T_new:
            raise ValueError(
                f"HCADecodingCache.append_step: Z_new.shape[1]="
                f"{Z_new.shape[1]} does not match C_new.shape[1]={T_new}.")
        C_new = _to_cache(C_new, self.dtype, self.device)
        Z_new = _to_cache(Z_new, self.dtype, self.device)
        if T_new > 1:
            return self._append_bulk(C_new, Z_new, B_pos)

        new_block_indices: list[int] = []
        for i in range(T_new):
            c_i = C_new[:, i:i+1]    # [B, 1, c]
            z_i = Z_new[:, i:i+1]
            self._acc_C.append(c_i)
            self._acc_Z.append(z_i)
            self._n_tokens_seen += 1
            if self._sw is not None:
                # Run the SW key normalize/rotate path in the cache's
                # compute dtype (fp64 for float64 caches), matching the
                # compressed-entry path and the full-sequence naive_hca;
                # an unconditional fp32 round-trip would silently quantize
                # float64 SW entries to fp32 precision.
                compute_dtype = (
                    torch.float64 if self.dtype == torch.float64 else torch.float)
                if self._normalize_qk:
                    c_src_i = _to_dtype(c_i.squeeze(1), compute_dtype)
                    if (not torch.is_grad_enabled()
                            and self.dtype != compute_dtype):
                        # Fresh cast copy (see the matching note in the
                        # CSA cache): normalize in place.
                        c_norm_i = _qk_normalize_(
                            c_src_i, self.qk_norm_mode)
                    else:
                        c_norm_i = _qk_normalize(
                            c_src_i, self.qk_norm_mode)
                    c_store_i = c_norm_i.unsqueeze(1)
                else:
                    c_store_i = c_i
                # Partial RoPE (§2.3.3): the SW key rotates at its own
                # token position AFTER normalization — the same
                # normalize-then-rotate order as ``naive_hca``'s SW
                # branch.
                if self.rope_dim_eff > 0:
                    pos_t = _pos_tensor(self._n_tokens_seen - 1, self.device)
                    c_store_i = partial_rope(
                        _to_dtype(c_store_i, compute_dtype), pos_t,
                        self.rope_dim_eff, self.rope_base, dim=1)
                self._sw.append(_to_dtype(c_store_i, self.dtype))
            if len(self._acc_C) == self.m2:
                C_block = torch.cat(self._acc_C, dim=1)              # [B, m2, c]
                Z_block = torch.cat(self._acc_Z, dim=1)
                new_C = _hca_compress_kv_single(C_block, Z_block, B_pos)  # [B, c]
                # Partial RoPE (§2.3.3): rotate the just-completed heavy
                # compressed entry at its block-anchor position
                # ``block_idx * m2`` (block_idx == self._n_blocks).
                # Rotation happens BEFORE storage; ``forward_step``
                # normalizes at read time, mirroring the
                # rotate-then-normalize order of ``naive_hca``.
                if self.rope_dim_eff > 0:
                    pos_b = _pos_tensor(self._n_blocks * self.m2, self.device)
                    new_C = partial_rope(
                        new_C.unsqueeze(1), pos_b,
                        self.rope_dim_eff, self.rope_base,
                        dim=1).squeeze(1)                           # [B, c]
                # ``_hca_compress_kv_single`` returns fp32/fp64
                # (compute_dtype) even when the cache stores fp16. Cast
                # back to the storage dtype before writing the valid row.
                self._ensure_compressed_capacity(self._n_blocks + 1)
                self._C_comp[:, self._n_blocks:self._n_blocks + 1].copy_(
                    _to_dtype(new_C.unsqueeze(1), self.dtype))
                self._n_blocks += 1
                new_block_indices.append(self._n_blocks - 1)
                self._acc_C.clear()
                self._acc_Z.clear()
        return new_block_indices

    def _append_bulk(
        self,
        C_new: torch.Tensor,
        Z_new: torch.Tensor,
        B_pos: torch.Tensor,
    ) -> list[int]:
        """Batched ``append_step`` body for ``T_new > 1`` (prefill).

        Mirrors :meth:`CSADecodingCache._append_bulk`: identical block
        boundaries (every ``m2`` tokens from the first token ever appended)
        and identical compression helpers, with whole sweeps of blocks
        compressed in one batched call.
        """
        B, T_new, c = C_new.shape
        m2 = self.m2
        dtype = self.dtype
        # ----- sliding-window branch (batched) -----
        if self._sw is not None:
            # Compute dtype, not storage dtype (see the matching note in
            # ``append_step``).
            compute_dtype = torch.float64 if dtype == torch.float64 else torch.float
            if self._normalize_qk:
                c_src = _to_dtype(C_new, compute_dtype)
                if (not torch.is_grad_enabled()
                        and self.dtype != compute_dtype):
                    # Fresh cast copy (see the matching note in the CSA
                    # cache's ``_append_bulk``): normalize in place.
                    c_store = _qk_normalize_(c_src, self.qk_norm_mode)
                else:
                    c_store = _qk_normalize(c_src, self.qk_norm_mode)
            else:
                c_store = C_new
            if self.rope_dim_eff > 0:
                pos = torch.arange(
                    self._n_tokens_seen, self._n_tokens_seen + T_new,
                    device=self.device)
                c_store = partial_rope(
                    _to_dtype(c_store, compute_dtype), pos,
                    self.rope_dim_eff, self.rope_base, dim=1)
            self._sw.append_bulk(_to_dtype(c_store, dtype))
        self._n_tokens_seen += T_new
        # ----- accumulate / compress into blocks -----
        acc = len(self._acc_C)
        first_needed = m2 - acc
        if T_new < first_needed:
            self._acc_C += list(C_new.split(1, dim=1))
            self._acc_Z += list(Z_new.split(1, dim=1))
            return []
        if acc:
            C_first = torch.cat(
                list(self._acc_C) + list(C_new[:, :first_needed].split(1, dim=1)), dim=1)
            Z_first = torch.cat(
                list(self._acc_Z) + list(Z_new[:, :first_needed].split(1, dim=1)), dim=1)
        else:
            C_first = C_new[:, :first_needed]
            Z_first = Z_new[:, :first_needed]
        new_C_first = _hca_compress_kv_single(C_first, Z_first, B_pos)  # [B, c]
        rest = T_new - first_needed
        n_rest = rest // m2
        tail = rest % m2
        if n_rest:
            rest_end = first_needed + n_rest * m2
            new_C_rest = csa_compress_kv(
                C_new[:, first_needed:rest_end].reshape(B, n_rest * m2, c),
                Z_new[:, first_needed:rest_end].reshape(B, n_rest * m2, c),
                B_pos, m2,
            )                                                      # [B, n_rest, c]
        n_new = 1 + n_rest
        block_idx0 = self._n_blocks
        # Partial RoPE: rotate the just-completed heavy compressed entries
        # at their block-anchor positions ``block_idx * m2`` (same positions
        # and rotate-before-store order as the per-token path). The
        # rotation is a per-row map along ``dim=1``, so rotating the first
        # block and the rest separately and storing each slice straight
        # into its slot is elementwise-identical to rotating the
        # concatenation (no full ``[B, n_new, c]`` concatenation needed;
        # same split as the CSA cache).
        pos_b = (torch.arange(block_idx0, block_idx0 + n_new,
                              device=self.device) * m2
                 if self.rope_dim_eff > 0 else None)
        first_C = new_C_first.unsqueeze(1)
        if pos_b is not None:
            first_C = partial_rope(
                first_C, pos_b[:1], self.rope_dim_eff, self.rope_base, dim=1)
            if n_rest:
                new_C_rest = partial_rope(
                    new_C_rest, pos_b[1:],
                    self.rope_dim_eff, self.rope_base, dim=1)
        self._ensure_compressed_capacity(block_idx0 + n_new)
        self._C_comp[:, block_idx0:block_idx0 + 1].copy_(
            _to_dtype(first_C, dtype))
        if n_rest:
            self._C_comp[:, block_idx0 + 1:block_idx0 + n_new].copy_(
                _to_dtype(new_C_rest, dtype))
        self._n_blocks += n_new
        new_block_indices = list(range(block_idx0, block_idx0 + n_new))
        # Leave the tokens past the last full block in the accumulator.
        if tail:
            self._acc_C = list(C_new[:, -tail:].split(1, dim=1))
            self._acc_Z = list(Z_new[:, -tail:].split(1, dim=1))
        else:
            self._acc_C = []
            self._acc_Z = []
        return new_block_indices

    # ----- core: compute attention output for the current query ---------

    def forward_step(
        self,
        q: torch.Tensor,          # [B, 1, nh, c]  (qk_norm_mode-normalized iff normalize_qk)
        *,
        nh: int,
        scale: float | None = None,
        sink_logits: torch.Tensor | None = None,  # [nh]
        normalize_qk: bool = True,
    ) -> torch.Tensor:
        """Compute the HCA attention output for the CURRENT query.

        Uses the current cache state (compressed blocks + SW buffer).
        Dense attention over ALL causally-valid compressed blocks (no
        indexer, no top-k — HCA is dense MQA).

        Returns:
            ``[B, 1, nh, c]`` per-head attention output.
        """
        B = q.shape[0]
        T_new = q.shape[1]
        if T_new != 1:
            raise ValueError(
                f"HCADecodingCache.forward_step: T_new={T_new} must be 1.")
        if B != self.B:
            raise ValueError(
                f"HCADecodingCache.forward_step: batch size {B} does not "
                f"match cache's B={self.B}.")
        if q.shape[2] != nh:
            raise ValueError(
                f"q.shape[2]={q.shape[2]} does not match nh={nh}.")
        # Validate device consistency (mirrors CSA cache).
        if q.device != self.device:
            raise ValueError(
                f"HCADecodingCache.forward_step: q.device={q.device} does "
                f"not match cache's device={self.device}. Call "
                f"cache.to(device=q.device) or move q to the cache's device.")
        device, dtype = q.device, q.dtype
        compute_dtype = (
            torch.float64 if dtype == torch.float64 else torch.float
        )
        # Core-attention scale: compute ONCE here so it is available to
        # BOTH the dense branch (below) and the sliding-window branch
        # (further down). Auto-selection mirrors ``naive_hca``: cosine
        # scale 1.0 for 'l2', the standard 1/sqrt(c) temperature for the
        # RMS form (whose normalized vectors keep norm ~sqrt(c)), and the
        # classical dot-product scale when normalization is off.
        if scale is None:
            if not normalize_qk or self.qk_norm_mode == 'rms':
                core_scale = self.c ** -0.5
            else:
                core_scale = 1.0
        else:
            core_scale = scale
        t = self._n_tokens_seen - 1
        # Partial RoPE (§2.3.3): rotate the query at its own token
        # position AFTER the caller's normalization (the same
        # normalize-then-rotate order as ``naive_hca``'s query path).
        if self.rope_dim_eff > 0:
            pos_t = _pos_tensor(t, device)
            q = partial_rope(
                _to_dtype(q, compute_dtype), pos_t,
                self.rope_dim_eff, self.rope_base, dim=1)          # [B, 1, nh, c]
        # Shared by the dense-block and sliding-window branches.
        q_compute = _to_dtype(q, compute_dtype)                    # [B, 1, nh, c]
        if self._C_comp is None or self._n_blocks == 0:
            scores = torch.empty(
                B, 1, nh, 0, dtype=compute_dtype, device=device,
            )
            C_comp_n = torch.empty(B, 0, self.c, dtype=compute_dtype, device=device)
        else:
            n_blocks = self._n_blocks
            # Causal block mask (paper Eq. 16, HCA variant): a
            # heavy-compression block is visible iff ``b < t // m2`` —
            # only strictly preceding blocks (the query's own block is
            # excluded; its content is served by the SW branch). Matches
            # ``ops_csa._causal_block_mask`` and ``naive_hca``.
            b_threshold = t // self.m2
            cbm = _block_valid_mask(
                n_blocks, b_threshold, device)[None, None, :]         # [1, 1, n_blocks]
            # Honor ``normalize_qk`` for the CORE attention (as in
            # ``naive_hca``). When normalize_qk=True compressed KV are
            # normalized with ``qk_norm_mode`` (cosine-similarity
            # attention for 'l2', the paper §2.3.3 RMSNorm form for
            # 'rms'). When False, skip normalization and fall back
            # to classical 1/sqrt(c) dot-product scale — this gives
            # ablations a real off-switch rather than a silently-ignored
            # flag.
            if normalize_qk:
                C_comp_n = _to_dtype(
                    self._C_comp[:, :self._n_blocks], compute_dtype)
                if torch.is_grad_enabled() and self.dtype == compute_dtype:
                    # The cast aliases the cache storage; normalization
                    # saves its input for backward, and a later
                    # ``append_step`` overwrites that storage in place.
                    C_comp_n = C_comp_n.clone()
                if not torch.is_grad_enabled() and self.dtype != compute_dtype:
                    # ``C_comp_n`` is a fresh compute-dtype copy here
                    # (the cast above allocated it); normalize it in
                    # place to avoid a second full-sized temporary. Same
                    # per-element chain as ``_qk_normalize``.
                    C_comp_n = _qk_normalize_(C_comp_n, self.qk_norm_mode)
                else:
                    C_comp_n = _qk_normalize(C_comp_n, self.qk_norm_mode)  # [B, n_blocks, c]
            else:
                C_comp_n = _to_dtype(
                    self._C_comp[:, :self._n_blocks], compute_dtype)
                if torch.is_grad_enabled() and self.dtype == compute_dtype:
                    # Clone only when the cast aliases the cache storage
                    # (see the matching note in CSA ``forward_step``).
                    C_comp_n = C_comp_n.clone()
            # scores: [B, nh, 1, n_blocks] -> [B, 1, nh, n_blocks]
            scores = torch.einsum(
                'b t h d, b n d -> b t h n', q_compute, C_comp_n,
            )                                                          # [B, 1, nh, n_blocks]
            if core_scale != 1.0:
                # Multiplying by exactly 1.0 is an IEEE no-op, so the
                # guarded multiply is value-identical. Outside autograd the
                # fresh einsum output is scaled in place.
                if torch.is_grad_enabled():
                    scores = scores * core_scale
                else:
                    scores.mul_(core_scale)
            # In place: the einsum output is freshly allocated and not
            # saved by any autograd node (mirrors ``naive_hca``).
            scores.masked_fill_(~cbm[:, :, None, :], float('-inf'))

        # --- Sliding-window branch: window scores for the joint softmax ---
        if self._sw is None or self._sw.n_valid == 0:
            scores_w = torch.empty(
                B, 1, nh, 0, dtype=compute_dtype, device=device,
            )
            C_local = torch.empty(B, 0, self.c, dtype=compute_dtype, device=device)
        else:
            # ``clone()``: same autograd version guard as the CSA cache —
            # ``get()`` returns a live view into the ring buffer, and the
            # einsum's below save it to compute gradients; a later
            # ``append_step`` overwrites the buffer in place and would
            # invalidate that saved view. The copy is needed whenever grad
            # is enabled: the ``sw_out`` einsum
            # saves the view for the gradient w.r.t. the softmax weights
            # ``p_w``, which can require grad even when neither
            # ``q_compute`` nor the buffer entries do (e.g. a learnable
            # attention sink). Outside grad mode nothing can save the
            # view, so the copy is skipped.
            C_local = _to_dtype(self._sw.get(), compute_dtype)         # [B, sw_len, c]
            if torch.is_grad_enabled():
                C_local = C_local.clone()
            sw = torch.einsum(
                'b h d, b s d -> b h s', q_compute[:, 0], C_local,
            )                                                          # [B, nh, sw_len]
            if core_scale != 1.0:
                # Multiplying by exactly 1.0 is an IEEE no-op, so the
                # guarded multiply is value-identical. Outside autograd the
                # fresh einsum output is scaled in place.
                if torch.is_grad_enabled():
                    sw = sw * core_scale
                else:
                    sw.mul_(core_scale)
            scores_w = sw.unsqueeze(1)                                 # [B, 1, nh, sw_len]

        # JOINT softmax over [compressed; window] with the sink in the single
        # shared denominator (matches full-sequence naive_hca; DeepSeek-V4
        # §2.3.1 Eq. 27).
        cat = torch.cat([scores, scores_w], dim=-1)                    # [B, 1, nh, n_blocks+sw_len]
        if cat.shape[-1] == 0:
            return torch.zeros(B, 1, nh, self.c, dtype=compute_dtype, device=device)
        if sink_logits is not None:
            log_sink = sink_logits.view(1, 1, nh, 1).to(cat)
            row_max = cat.amax(-1, keepdim=True).clamp(min=0)
            shifted = cat - row_max
            shifted_sink = log_sink - row_max
            lse = torch.logsumexp(shifted, dim=-1, keepdim=True)
            log_denom = torch.logaddexp(lse, shifted_sink)
            p = (shifted - log_denom).exp()
            all_masked = torch.isinf(cat).all(-1, keepdim=True)
            if torch.is_grad_enabled():
                p = p.masked_fill(all_masked, 0.0)
            else:
                p.masked_fill_(all_masked, 0.0)
        else:
            all_masked = torch.isinf(cat).all(-1, keepdim=True)
            if not torch.is_grad_enabled():
                # ``cat`` / the softmax output are fresh tensors of this
                # step; outside autograd the zeroing runs in place.
                safe = cat.masked_fill_(all_masked, 0.0)
                p = torch.softmax(safe, dim=-1)
                p.masked_fill_(all_masked, 0.0)
            else:
                safe = cat.masked_fill(all_masked, 0.0)
                p = torch.softmax(safe, dim=-1)
                p = p.masked_fill(all_masked, 0.0)
        n_blk = scores.shape[-1]
        p_dense = p[..., :n_blk]
        p_w = p[..., n_blk:]
        dense_out = torch.einsum(
            'b t h n, b n d -> b t h d', p_dense, C_comp_n,
        )                                                              # [B, 1, nh, c]
        # Accumulate the window contribution in place (same values as
        # ``a + b``; the add's backward saves neither operand).
        dense_out.add_(torch.einsum(
            'b t h w, b w d -> b t h d', p_w, C_local,
        ))                                                             # [B, 1, nh, c]
        out = dense_out
        # Partial RoPE output countermeasure (§2.3.3): inverse-rotate the
        # core-attention output at the NEGATED query position so each KV
        # entry's contribution carries the relative (entry − query)
        # distance — mirroring ``naive_hca``'s output path.
        if self.rope_dim_eff > 0:
            pos_t = _pos_tensor(t, device)
            out = partial_rope(
                out, -pos_t, self.rope_dim_eff, self.rope_base, dim=1)
        return out


    def forward_chunk(
        self,
        C_new: torch.Tensor,
        Z_new: torch.Tensor,
        B_pos: torch.Tensor,
        q: torch.Tensor,
        *,
        nh: int,
        scale: float | None = None,
        sink_logits: torch.Tensor | None = None,
        normalize_qk: bool = True,
    ) -> torch.Tensor:
        """Batched equivalent of ``append_step`` + ``forward_step`` per token.

        See :meth:`CSADecodingCache.forward_chunk` for the contract. HCA
        has no indexer/top-k, so the dense compressed-block scores are
        computed for all queries at once with the per-query strictly-
        preceding block mask; the sliding-window branch is reconstructed
        from a pre-append snapshot plus the chunk keys.
        """
        B, T_new, c = C_new.shape
        if T_new < 1:
            raise ValueError(
                f'HCADecodingCache.forward_chunk: T_new={T_new} must be >= 1')
        if B != self.B:
            raise ValueError(
                f'HCADecodingCache.forward_chunk: batch size {B} does not '
                f"match cache's B={self.B}.")
        if c != self.c:
            raise ValueError(
                f'HCADecodingCache.forward_chunk: c={c} does not match '
                f"cache's c={self.c}.")
        if q.shape != (B, T_new, nh, c):
            raise ValueError(
                f'HCADecodingCache.forward_chunk: q shape {tuple(q.shape)} '
                f'does not match {(B, T_new, nh, c)}.')
        if Z_new.shape[1] != T_new:
            raise ValueError(
                f'HCADecodingCache.forward_chunk: Z_new.shape[1]='
                f'{Z_new.shape[1]} does not match C_new.shape[1]={T_new}.')
        if q.device != self.device:
            raise ValueError(
                f'HCADecodingCache.forward_chunk: q.device={q.device} does '
                f"not match cache's device={self.device}.")
        if T_new == 1:
            self.append_step(C_new, Z_new, B_pos)
            return self.forward_step(
                q, nh=nh, scale=scale, sink_logits=sink_logits,
                normalize_qk=normalize_qk)

        C_new = _to_cache(C_new, self.dtype, self.device)
        Z_new = _to_cache(Z_new, self.dtype, self.device)
        n0 = self._n_tokens_seen
        compute_dtype = (
            torch.float64 if self.dtype == torch.float64 else torch.float)
        pre_sw = None
        if self._sw is not None and self._sw.n_valid:
            pre_sw = self._sw.get().clone()
        c_store_store = None
        if self._sw is not None:
            if self._normalize_qk:
                c_src = _to_dtype(C_new, compute_dtype)
                if (not torch.is_grad_enabled()
                        and self.dtype != compute_dtype):
                    c_store = _qk_normalize_(c_src, self.qk_norm_mode)
                else:
                    c_store = _qk_normalize(c_src, self.qk_norm_mode)
            else:
                c_store = C_new
            if self.rope_dim_eff > 0:
                pos = torch.arange(
                    n0, n0 + T_new, device=self.device)
                c_store = partial_rope(
                    _to_dtype(c_store, compute_dtype), pos,
                    self.rope_dim_eff, self.rope_base, dim=1)
            c_store_store = _to_dtype(c_store, self.dtype)
        self.append_step(C_new, Z_new, B_pos)
        t_pos = torch.arange(n0, n0 + T_new, device=self.device)
        q_compute = _to_dtype(q, compute_dtype)
        if self.rope_dim_eff > 0:
            q_compute = partial_rope(
                q_compute, t_pos, self.rope_dim_eff, self.rope_base, dim=1)
        if scale is None:
            core_scale = (self.c ** -0.5
                          if (not normalize_qk or self.qk_norm_mode == 'rms')
                          else 1.0)
        else:
            core_scale = scale

        n_blocks = self._n_blocks
        if n_blocks:
            cbm = (torch.arange(n_blocks, device=self.device)[None, :]
                   < (t_pos // self.m2)[:, None])                    # [T, nb]
            C_comp_n = _to_dtype(
                self._C_comp[:, :n_blocks], compute_dtype)
            if torch.is_grad_enabled() and self.dtype == compute_dtype:
                # Same grad-mode clone contract as ``forward_step``: the
                # cast aliases the cache storage when the dtypes match,
                # and normalization/einsums save their input for
                # backward, while a later ``append_step`` overwrites that
                # storage in place.
                C_comp_n = C_comp_n.clone()
            if normalize_qk:
                C_comp_n = _qk_normalize(C_comp_n, self.qk_norm_mode)
            scores = torch.einsum(
                'bthd,bnd->bthn', q_compute, C_comp_n)
            if core_scale != 1.0:
                scores = scores * core_scale
            scores = scores.masked_fill(
                ~cbm[None, :, None, :], float('-inf'))
        else:
            cbm = torch.empty(
                T_new, 0, dtype=torch.bool, device=self.device)
            scores = torch.empty(
                B, T_new, nh, 0, dtype=compute_dtype, device=self.device)
            C_comp_n = torch.empty(
                B, 0, self.c, dtype=compute_dtype, device=self.device)

        if self._sw is not None and c_store_store is not None:
            if pre_sw is None:
                keys = c_store_store
                n_pre = 0
            else:
                keys = torch.cat([pre_sw, c_store_store], dim=1)
                n_pre = pre_sw.shape[1]
            windows, valid_w = _chunk_window_keys(
                keys, n_pre, self.win)
            windows = _to_dtype(windows, compute_dtype)
            scores_w = torch.einsum(
                'bthd,btwd->bthw', q_compute, windows)
            if core_scale != 1.0:
                scores_w = scores_w * core_scale
            scores_w = scores_w.masked_fill(
                ~valid_w[None, :, None, :], float('-inf'))
        else:
            windows = torch.empty(
                B, T_new, 0, self.c, dtype=compute_dtype,
                device=self.device)
            scores_w = torch.empty(
                B, T_new, nh, 0, dtype=compute_dtype, device=self.device)
            valid_w = torch.empty(
                T_new, 0, dtype=torch.bool, device=self.device)

        n_dense = scores.shape[-1]
        n_win = scores_w.shape[-1]
        if n_dense + n_win == 0:
            out = torch.zeros(
                B, T_new, nh, self.c, dtype=compute_dtype,
                device=self.device)
        else:
            logits = torch.cat([scores, scores_w], dim=-1)
            valid = torch.cat([
                cbm[None, :, None, :],
                valid_w[None, :, None, :],
            ], dim=-1)
            p = _joint_softmax_chunk(logits, valid, sink_logits)
            p_dense = p[..., :n_dense]
            p_w = p[..., n_dense:]
            out = torch.zeros(
                B, T_new, nh, self.c, dtype=compute_dtype,
                device=self.device)
            if n_dense:
                out = out + torch.einsum(
                    'bthn,bnd->bthd', p_dense, C_comp_n)
            if n_win:
                out = out + torch.einsum(
                    'bthw,btwd->bthd', p_w, windows)
        if self.rope_dim_eff > 0:
            out = partial_rope(
                out, -t_pos, self.rope_dim_eff, self.rope_base, dim=1)
        return out


class Mamba2DecodingState:
    """Runtime state container for a Mamba-2 memory layer during decoding.

    Unlike CSA / HCA, whose incremental state is a structured cache
    (partial-token accumulator + compressed blocks + sliding-window ring
    buffer), a Mamba-2 layer carries exactly TWO flat tensors:

      * ``ssm_state``  — ``[B, nheads, headdim, d_state]``, the per-head
        SSD recurrent state (compute precision, fp32 by default);
      * ``conv_state`` — ``[B, d_inner, d_conv - 1]``, the causal short
        conv's left context (input dtype).

    ``Mamba2Mixer.forward_functional`` / ``prefill`` / ``step`` already
    thread these tensors directly, so this container is deliberately thin:
    it exists to give decoding / accounting code one object with ``reset()``
    + ``nbytes`` (mirroring the CSA/HCA caches' accounting surface) and to
    make the per-request state ownership explicit — each request owns its
    own ``Mamba2DecodingState``; nothing is attached to the shared module.

    The KDA layers' state (recurrent matrix ``[B, HV, K, V]`` + short-conv
    lookback) uses its own representation: the stacked
    ``HybridAttentionStack._kda_state`` buffer + per-layer
    ``_conv_lookback``. Wrapping KDA in this class would be a pure
    refactor with no behavioural gain, so the mixed stack presents both
    representations side by side (documented in docs/mamba2-framework.md).
    """

    def __init__(self, ssm_state, conv_state):
        self.ssm_state = ssm_state
        self.conv_state = conv_state

    @classmethod
    def from_state_dict(cls, state: dict | None):
        """Build from the dict returned by ``Mamba2Mixer.prefill``/``step``."""
        if state is None:
            return cls(None, None)
        if not isinstance(state, dict):
            raise TypeError(
                f"Mamba2DecodingState.from_state_dict: expected the dict "
                f"returned by Mamba2Mixer.prefill/step (or None), got "
                f"{type(state).__name__}.")
        return cls(state.get('ssm_state'), state.get('conv_state'))

    def to_state_dict(self) -> dict:
        """Return the dict consumed by ``Mamba2Mixer.step``."""
        return {'ssm_state': self.ssm_state, 'conv_state': self.conv_state}

    def reset(self) -> None:
        """Drop both tensors (call between independent sequences)."""
        self.ssm_state = None
        self.conv_state = None

    @property
    def nbytes(self) -> int:
        """Total bytes retained by this state (0 for the unset entries)."""
        total = 0
        for t in (self.ssm_state, self.conv_state):
            if t is not None:
                total += t.numel() * t.element_size()
        return total

    def nbytes_breakdown(self) -> dict:
        """Component-wise byte accounting for ``run_kv_cache``'s
        recursive-vs-conv-state split (fp32 state / input-dtype conv)."""
        def _bytes(t):
            return None if t is None else t.numel() * t.element_size()
        return {
            'ssm_state_bytes': _bytes(self.ssm_state),
            'conv_state_bytes': _bytes(self.conv_state),
            'total_bytes': self.nbytes,
        }
