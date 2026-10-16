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
     query without recomputing the full history.
  3. **Sliding-window ring buffer** (``torch.Tensor`` ``[B, win, c]``).
     A fixed-size FIFO of the most recent ``win`` *uncompressed* local
     keys (the ``Ca`` projection for CSA, ``C`` for HCA, normalized with
     the cache's ``qk_norm_mode`` — 'l2' by default).
     Writes wrap around; reads return the contents in causal order
     (oldest first). This backs the sliding-window branch of
     ``naive_csa`` / ``naive_hca`` during decoding without materializing
     a ``[T, win, c]`` unfold per step.

Numerical contract
------------------
For a sequence of ``T`` tokens fed ONE AT A TIME through ``append_step``,
the per-token outputs produced by :meth:`CSADecodingCache.forward_step`
/ :meth:`HCADecodingCache.forward_step` are **bit-identical (to fp32
tolerance) to the corresponding rows of ``naive_csa`` / ``naive_hca``
called once on the full ``[B, T, d]`` sequence**, PROVIDED that the
CSA indexer's top-k selection is unambiguous (no tied scores).

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

Known limitation: torch.topk tie-breaking
-----------------------------------------
When the CSA indexer's ReLU scores have many exact ties at 0 (which
happens when the dot products are negative — common with random
untrained weights but rare in trained models where the indexer learns
discriminative scores), ``torch.topk``'s tie-breaking depends on the
tensor SIZE. The full path's tensor has ``n_blocks_full = T_padded // m``
entries (including -inf-masked future blocks); the incremental path's
tensor has only ``n_blocks_inc`` entries (no padding). With the same
underlying scores but different tensor sizes, ``torch.topk`` may pick
DIFFERENT tied blocks, leading to different gathered ``kv`` and
different sparse-attention outputs.

This is a ``torch.topk`` implementation artifact, NOT a correctness bug
in the cache — both paths select valid blocks with the highest scores,
just different tie-breaking. In a trained model the indexer's scores
are discriminative (the whole point of the indexer is to learn which
blocks to select), so ties are rare and the issue does not arise in
practice.

For correctness testing, the regression tests use ``topk >= n_blocks``
(so ALL valid blocks are selected, making the output
permutation-invariant over the selected set and thus independent of
tie-breaking). For production decoding (Exp 6 latency benchmark), the
tie-breaking difference is irrelevant — we measure latency, not
bit-exact outputs.

What this is NOT
----------------
This is a *correctness-first* reference, not a production kernel. The
``forward_step`` path uses Python-level control flow (append-to-list,
conditional compress, top-k gather) and is dominated by interpreter
overhead at small ``T_new``. It is suitable for:

  * Closing the Exp 6 scope gap (CSA / HCA decode latency vs softmax /
    KDA, on the same Kaggle T4).
  * Correctness verification (incremental == full-prefill).
  * Small-scale autoregressive generation (a few hundred tokens).

For production decoding, the same data structures would be implemented
in CUDA / Triton with fused matmuls; the Python reference here is the
spec they would be validated against.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ops_rope import effective_rope_dim, partial_rope
from ops_csa import (csa_compress_kv, csa_compress_kv_overlapped,
                     csa_lightning_indexer, _qk_normalize)


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

def _to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``x.to(dtype)`` with the dispatcher skipped when it is a no-op.

    ``Tensor.to`` returns the SAME tensor when the dtype already matches,
    so guarding on a dtype comparison is value-identical — it only avoids
    the per-call dispatch overhead, which accumulates on the decode hot
    path (thousands of no-op conversions per benchmark).
    """
    if x.dtype == dtype:
        return x
    return x.to(dtype)


def _to_cache(x: torch.Tensor, dtype: torch.dtype, device) -> torch.Tensor:
    """``x.to(dtype=dtype, device=device)`` skipping the no-op case.

    Value-identical to the unconditional call (a matching dtype/device
    ``.to`` returns ``x`` itself); the guard only removes the dispatch.
    """
    if x.dtype == dtype and x.device == device:
        return x
    return x.to(dtype=dtype, device=device)


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
    out = (Sa * Ca_new).sum(1) + (Sb * Cb_prev_eff).sum(1)      # [B, c]
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
        self._buf = torch.zeros(B, win, c, device=self.device, dtype=dtype)
        # Number of valid entries currently in the buffer (0..win).
        self._sw_len = 0
        # Index in ``self._buf`` of the OLDEST valid entry. Stays at 0
        # while ``_sw_len < win`` (entries are written sequentially at
        # positions 0, 1, 2, ...). Once the buffer is full, each new
        # write goes to ``_sw_head`` (overwriting the oldest) and
        # ``_sw_head`` advances by 1 (mod ``win``).
        self._sw_head = 0

    def reset(self) -> None:
        """Clear all state (does not deallocate the buffer)."""
        # We zero the buffer defensively so a stale read after reset
        # returns zeros rather than the previous session's keys (which
        # could leak across sequences). The cost is one ``win * c``
        # element zeroing, negligible for the typical ``win=16``.
        self._buf.zero_()
        self._sw_len = 0
        self._sw_head = 0

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
            # ``model.to(cuda)`` could otherwise crash with a cryptic
            # device-mismatch error inside ``self._buf[:, ...] = x``.
            x = x.to(device=self._buf.device, dtype=self._buf.dtype)
        _, T_new, _ = x.shape
        for i in range(T_new):
            self._append_one(x[:, i])

    def append_bulk(self, x: torch.Tensor) -> None:
        """Append ``[B, T_new, c]`` entries in arrival order (vectorized).

        Equivalent to calling :meth:`append` once per token — the final
        causal-order contents are exactly the last ``min(total, win)``
        entries written — but without the per-token Python loop, so a
        prefill of thousands of tokens does not pay thousands of tiny
        ``copy_`` dispatches. Only the physical placement inside ``_buf``
        may differ (any placement with the same causal-order view is
        valid for :meth:`get`).
        """
        if x.dim() != 3 or x.shape[0] != self.B or x.shape[-1] != self.c:
            raise ValueError(
                f"_SlidingWindowRingBuffer.append_bulk: expected "
                f"[{self.B}, T_new, {self.c}], got {tuple(x.shape)}.")
        if x.device != self._buf.device or x.dtype != self._buf.dtype:
            x = x.to(device=self._buf.device, dtype=self._buf.dtype)
        if x.shape[1] == 0:
            return
        if x.shape[1] >= self.win:
            # Every entry cached before this call (and every entry of ``x``
            # except the newest ``win``) is evicted before it can be read,
            # so the final content is simply the newest ``win`` entries.
            self._buf[:, :self.win] = x[:, -self.win:]
            self._sw_head = 0
            self._sw_len = self.win
            return
        # Small append (T_new < win): keep the last
        # ``min(_sw_len + T_new, win)`` entries of (previous + new).
        combined = torch.cat([self.get(), x], dim=1)
        keep = combined[:, -self.win:]
        self._buf[:, :keep.shape[1]] = keep
        self._sw_head = 0
        self._sw_len = keep.shape[1]

    def _append_one(self, x_one: torch.Tensor) -> None:
        """Append a single ``[B, c]`` entry."""
        if self._sw_len < self.win:
            # Buffer not yet full: append at the next free slot.
            # ``_sw_head == 0`` and ``_sw_len < win``, so the write
            # position is simply ``_sw_head + _sw_len``.
            write_pos = self._sw_head + self._sw_len
            self._buf[:, write_pos] = x_one
            self._sw_len += 1
        else:
            # Buffer full: overwrite the oldest entry, advance head.
            # ``_sw_head`` is the oldest; write there, then advance so
            # the NEXT-oldest becomes the new oldest.
            self._buf[:, self._sw_head] = x_one
            self._sw_head = (self._sw_head + 1) % self.win

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
        #     (newer), concatenated.
        if self._sw_head + self._sw_len <= self.win:
            return self._buf[:, self._sw_head : self._sw_head + self._sw_len]
        else:
            tail = self._buf[:, self._sw_head:]                       # oldest
            head_len = self._sw_head + self._sw_len - self.win
            head = self._buf[:, :head_len]                            # newer
            return torch.cat([tail, head], dim=1)

    @property
    def n_valid(self) -> int:
        """Number of valid entries currently in the buffer (0..win)."""
        return self._sw_len

    def to(self, device=None, dtype=None) -> '_SlidingWindowRingBuffer':
        """Move the buffer to a new device / dtype (in-place)."""
        if device is not None:
            device = torch.device(device)
        if device is not None and device != self._buf.device:
            self._buf = self._buf.to(device=device)
            self.device = self._buf.device
        if dtype is not None and dtype != self._buf.dtype:
            self._buf = self._buf.to(dtype=dtype)
            self.dtype = self._buf.dtype
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
       backing storage grows geometrically as the accumulator fills, avoiding
       a full copy on every new block. Read by the sparse-attention branch
       (compressed block cache) and the lightning indexer (indexer key
       cache) during ``forward_step``.
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
    PROVIDED the indexer's top-k selection is unambiguous. See the
    module-level docstring for the ``torch.topk`` tie-breaking caveat
    and the regression test ``test_csa_decoding_cache_correctness`` in
    ``run_correctness.py`` for the verification (which uses
    ``topk >= n_blocks`` to sidestep the tie-breaking issue).
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
        # crashes at construction (not deep inside ``append_step``).
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
        # (default) disables RoPE and preserves the historical behaviour.
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

        Capacity grows geometrically, so appending one completed block is
        amortized O(1) in the number of previously cached blocks. Only the
        valid prefix ``:self._n_blocks`` is copied; unused capacity is not
        exposed through the public properties or attention path.

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
        self._acc_Ca = [t.to(device=self.device, dtype=self.dtype)
                        for t in self._acc_Ca]
        self._acc_Cb = [t.to(device=self.device, dtype=self.dtype)
                        for t in self._acc_Cb]
        self._acc_Za = [t.to(device=self.device, dtype=self.dtype)
                        for t in self._acc_Za]
        self._acc_Zb = [t.to(device=self.device, dtype=self.dtype)
                        for t in self._acc_Zb]
        self._acc_K_idx = [t.to(device=self.device, dtype=self.dtype)
                           for t in self._acc_K_idx]
        self._acc_Z_idx = [t.to(device=self.device, dtype=self.dtype)
                           for t in self._acc_Z_idx]
        if self._prev_Cb is not None:
            self._prev_Cb = self._prev_Cb.to(
                device=self.device, dtype=self.dtype)
        if self._prev_Zb is not None:
            self._prev_Zb = self._prev_Zb.to(
                device=self.device, dtype=self.dtype)
        if self._C_comp is not None:
            self._C_comp = self._C_comp.to(device=self.device, dtype=self.dtype)
        if self._K_IComp is not None:
            self._K_IComp = self._K_IComp.to(
                device=self.device, dtype=self.dtype)
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
        # ``Cb_new`` has T_new=3) would otherwise crash deep inside the
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
                if self._normalize_qk:
                    ca_norm_i = _qk_normalize(
                        _to_dtype(ca_i.squeeze(1), torch.float),
                        self.qk_norm_mode)
                    ca_store_i = ca_norm_i.unsqueeze(1)
                else:
                    ca_store_i = ca_i
                # Partial RoPE (§2.3.3): the SW key rotates at its own
                # token position AFTER normalization — the same
                # normalize-then-rotate order as ``naive_csa``'s
                # sliding-window branch.
                if self.rope_dim_eff > 0:
                    pos_t = torch.tensor(
                        [self._n_tokens_seen - 1], device=self.device)
                    ca_store_i = partial_rope(
                        _to_dtype(ca_store_i, torch.float), pos_t,
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
                    pos_b = torch.tensor(
                        [self._n_blocks * self.m], device=self.device)
                    new_C = partial_rope(
                        new_C.unsqueeze(1), pos_b,
                        self.rope_dim_eff, self.rope_base,
                        dim=1).squeeze(1)                           # [B, c]
                # The compress helpers upcast to fp32/fp64 internally
                # (compute_dtype) and return that dtype. Cast back to the
                # cache's storage dtype before writing into the
                # geometrically-grown storage so all rows stay uniform.
                new_C_row = _to_dtype(new_C.unsqueeze(1), self.dtype)  # [B, 1, c]
                new_K_I_row = _to_dtype(new_K_I.unsqueeze(1), self.dtype)  # [B, 1, c_I]
                # Geometric growth avoids copying the complete compressed
                # history on every block. The public cache exposes only the
                # valid prefix, while the storage tensors retain spare
                # capacity for future blocks.
                self._ensure_compressed_capacity(self._n_blocks + 1)
                self._C_comp[:, self._n_blocks:self._n_blocks + 1] = new_C_row
                self._K_IComp[:, self._n_blocks:self._n_blocks + 1] = new_K_I_row
                self._n_blocks += 1
                new_block_indices.append(self._n_blocks - 1)
                # Stash the just-consumed Cb / Zb as the previous
                # block's overlap partner for the NEXT compression.
                # Detach so we don't retain the autograd graph across
                # decode steps (mirrors the KDA state detach pattern in
                # ops_fused.py::KDAHybridLayer.forward_functional).
                self._prev_Cb = Cb_block.detach().clone()
                self._prev_Zb = Zb_block.detach().clone()
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
        """Vectorized ``append_step`` body for ``T_new > 1`` (prefill).

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
            if self._normalize_qk:
                ca_store = _qk_normalize(
                    _to_dtype(Ca_new, torch.float), self.qk_norm_mode)
            else:
                ca_store = Ca_new
            if self.rope_dim_eff > 0:
                pos = torch.arange(
                    self._n_tokens_seen, self._n_tokens_seen + T_new,
                    device=self.device)
                ca_store = partial_rope(
                    _to_dtype(ca_store, torch.float), pos,
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
            # Prepend the already-compressed first block so that the batched
            # overlap shift gives the remaining blocks the correct partner
            # (row 0 of the output is the first block itself and is dropped).
            Ca_seq = torch.cat([Ca_first.unsqueeze(1), Ca_rest], dim=1)
            Cb_seq = torch.cat([Cb_first.unsqueeze(1), Cb_rest], dim=1)
            Za_seq = torch.cat([Za_first.unsqueeze(1), Za_rest], dim=1)
            Zb_seq = torch.cat([Zb_first.unsqueeze(1), Zb_rest], dim=1)
            new_C_rest = csa_compress_kv_overlapped(
                Ca_seq.reshape(B, (n_rest + 1) * m, c),
                Cb_seq.reshape(B, (n_rest + 1) * m, c),
                Za_seq.reshape(B, (n_rest + 1) * m, c),
                Zb_seq.reshape(B, (n_rest + 1) * m, c),
                Ba, Bb, m,
            )[:, 1:]                                               # [B, n_rest, c]
            new_K_rest = csa_compress_kv(
                K_idx_new[:, first_needed:rest_end].reshape(
                    B, n_rest * m, self.c_I),
                Z_idx_new[:, first_needed:rest_end].reshape(
                    B, n_rest * m, self.c_I),
                B_idx, m,
            )                                                      # [B, n_rest, c_I]
            new_C_all = torch.cat(
                [new_C_first.unsqueeze(1), new_C_rest], dim=1)
            new_K_all = torch.cat(
                [new_K_first.unsqueeze(1), new_K_rest], dim=1)
            last_Cb = Cb_rest[:, -1]
            last_Zb = Zb_rest[:, -1]
        else:
            new_C_all = new_C_first.unsqueeze(1)
            new_K_all = new_K_first.unsqueeze(1)
            last_Cb = Cb_first
            last_Zb = Zb_first
        n_new = 1 + n_rest
        # Partial RoPE: rotate the just-completed compressed entries at
        # their block-anchor positions ``block_idx * m`` (same positions
        # and normalize-then-rotate order as the per-token path).
        if self.rope_dim_eff > 0:
            pos_b = torch.arange(
                self._n_blocks, self._n_blocks + n_new, device=self.device) * m
            new_C_all = partial_rope(
                new_C_all, pos_b,
                self.rope_dim_eff, self.rope_base, dim=1)
        block_idx0 = self._n_blocks
        self._ensure_compressed_capacity(block_idx0 + n_new)
        self._C_comp[:, block_idx0:block_idx0 + n_new] = _to_dtype(new_C_all, dtype)
        self._K_IComp[:, block_idx0:block_idx0 + n_new] = _to_dtype(new_K_all, dtype)
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
            pos_t = torch.tensor([t], device=device)
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
            cbm = torch.arange(n_blocks, device=device) < b_threshold   # [n_blocks]
            cbm = cbm[None, :]                                          # [1, n_blocks]
            # Honor ``normalize_qk`` for the CORE attention (as in
            # ``naive_csa``). When normalize_qk=True compressed KV are
            # normalized with ``qk_norm_mode`` (cosine-similarity
            # attention for 'l2', the paper §2.3.3 RMSNorm form for
            # 'rms'). When False, skip normalization and fall back to the
            # classical 1/sqrt(c) dot-product scale.
            if normalize_qk:
                C_comp_valid = _to_dtype(
                    self._C_comp[:, :self._n_blocks], compute_dtype)
                C_comp_n = _qk_normalize(
                    C_comp_valid,
                    self.qk_norm_mode)                                # [B, n_blocks, c]
            else:
                C_comp_n = _to_dtype(
                    self._C_comp[:, :self._n_blocks], compute_dtype)
            # Lightning indexer: score the current query against ALL
            # cached indexer keys, select top-k (respecting the causal
            # block mask). STE is only useful when autograd is enabled; in
            # no-grad inference it is forward-equivalent but adds avoidable
            # softmax/gather work, so gate it just like ``naive_csa``.
            effective_use_ste = use_ste and torch.is_grad_enabled()
            indexer_result = csa_lightning_indexer(
                _to_dtype(q_idx, compute_dtype),                       # [B, 1, nIh, c_I]
                _to_dtype(self._K_IComp[:, :self._n_blocks],
                          compute_dtype),                              # [B, n_blocks, c_I]
                _to_dtype(w_idx, compute_dtype),                       # [B, 1, nIh]
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
                batch_idx = torch.arange(B, device=device).view(B, 1, 1)
                kv = C_comp_n[batch_idx, idx_safe]                     # [B, 1, topk, c]
                # STE for the indexer (mirrors ``naive_csa``).
                if effective_use_ste and soft_weights is not None:
                    soft_weights_selected = torch.gather(
                        soft_weights, dim=-1, index=idx_safe,
                    )                                                  # [B, 1, topk]
                    soft_weights_selected = soft_weights_selected * \
                        valid_mask.to(soft_weights_selected.dtype)
                    soft_kv = soft_weights_selected.unsqueeze(-1) * kv.detach()
                    if ste_mode == 'full_softmax':
                        soft_full = torch.einsum(
                            'btn,bnc->btc', soft_weights, C_comp_n,
                        )                                              # [B, 1, c]
                        topk_size = kv.shape[2]
                        # Cache the expansion once: ``expand`` returns a
                        # cheap view, but computing it twice per decode step
                        # (once for the value, once for the detached value)
                        # still pays the duplicate view-creation cost.
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
                    # Multiplying by exactly 1.0 is an IEEE no-op; skipping
                    # the multiply keeps values bit-for-bit identical.
                    scores = scores * core_scale
                scores = scores.masked_fill(
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
            C_local = _to_dtype(self._sw.get(), compute_dtype)         # [B, sw_len, c]
            sw = torch.einsum(
                'b h d, b s d -> b h s', q_compute[:, 0], C_local,
            )                                                          # [B, nh, sw_len]
            if core_scale != 1.0:
                # Multiplying by exactly 1.0 is an IEEE no-op; skipping the
                # multiply keeps values bit-for-bit identical.
                sw = sw * core_scale
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
                 torch.ones(B, 1, 1, n_win, dtype=torch.bool, device=device)],
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
            p = p.masked_fill(all_invalid, 0.0)
        else:
            all_invalid = ~cat_valid.any(-1, keepdim=True)
            safe = cat.masked_fill(all_invalid, 0.0)
            p = torch.softmax(safe, dim=-1)
            p = p.masked_fill(all_invalid, 0.0)

        topk_size = scores.shape[-1]
        p_b = p[..., :topk_size]
        p_w = p[..., topk_size:]
        sparse_out = torch.einsum(
            'b t h k, b t k d -> b t h d', p_b, kv,
        )                                                              # [B, 1, nh, c]
        sw_out = torch.einsum(
            'b t h w, b w d -> b t h d', p_w, C_local,
        )                                                              # [B, 1, nh, c]
        out = sparse_out + sw_out
        # Partial RoPE output countermeasure (§2.3.3): inverse-rotate the
        # core-attention output at the NEGATED query position so each KV
        # entry's contribution carries the relative (entry − query)
        # distance — mirroring ``naive_csa``'s output path.
        if self.rope_dim_eff > 0:
            pos_t = torch.tensor([t], device=device)
            out = partial_rope(
                out, -pos_t, self.rope_dim_eff, self.rope_base, dim=1)
        return out                                                     # [B, 1, nh, c]


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
        # (default) disables RoPE and preserves the historical behaviour.
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

    # ----- state migration -----------------------------------------------

    def to(self, device=None, dtype=None) -> 'HCADecodingCache':
        if device is not None:
            device = torch.device(device)
        if device is not None and device != self.device:
            self.device = device
        if dtype is not None and dtype != self.dtype:
            self.dtype = dtype
        self._acc_C = [t.to(device=self.device, dtype=self.dtype)
                       for t in self._acc_C]
        self._acc_Z = [t.to(device=self.device, dtype=self.dtype)
                       for t in self._acc_Z]
        if self._C_comp is not None:
            self._C_comp = self._C_comp.to(device=self.device, dtype=self.dtype)
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
                if self._normalize_qk:
                    c_norm_i = _qk_normalize(
                        _to_dtype(c_i.squeeze(1), torch.float),
                        self.qk_norm_mode)
                    c_store_i = c_norm_i.unsqueeze(1)
                else:
                    c_store_i = c_i
                # Partial RoPE (§2.3.3): the SW key rotates at its own
                # token position AFTER normalization — the same
                # normalize-then-rotate order as ``naive_hca``'s SW
                # branch.
                if self.rope_dim_eff > 0:
                    pos_t = torch.tensor(
                        [self._n_tokens_seen - 1], device=self.device)
                    c_store_i = partial_rope(
                        _to_dtype(c_store_i, torch.float), pos_t,
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
                    pos_b = torch.tensor(
                        [self._n_blocks * self.m2], device=self.device)
                    new_C = partial_rope(
                        new_C.unsqueeze(1), pos_b,
                        self.rope_dim_eff, self.rope_base,
                        dim=1).squeeze(1)                           # [B, c]
                # ``_hca_compress_kv_single`` returns fp32/fp64
                # (compute_dtype) even when the cache stores fp16. Cast
                # back to the storage dtype before writing the valid row.
                new_C_row = _to_dtype(new_C.unsqueeze(1), self.dtype)  # [B, 1, c]
                self._ensure_compressed_capacity(self._n_blocks + 1)
                self._C_comp[:, self._n_blocks:self._n_blocks + 1] = new_C_row
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
        """Vectorized ``append_step`` body for ``T_new > 1`` (prefill).

        Mirrors :meth:`CSADecodingCache._append_bulk`: identical block
        boundaries (every ``m2`` tokens from the first token ever appended)
        and identical compression helpers, with whole sweeps of blocks
        compressed in a single batched call instead of one call per token.
        """
        B, T_new, c = C_new.shape
        m2 = self.m2
        dtype = self.dtype
        # ----- sliding-window branch (batched) -----
        if self._sw is not None:
            if self._normalize_qk:
                c_store = _qk_normalize(
                    _to_dtype(C_new, torch.float), self.qk_norm_mode)
            else:
                c_store = C_new
            if self.rope_dim_eff > 0:
                pos = torch.arange(
                    self._n_tokens_seen, self._n_tokens_seen + T_new,
                    device=self.device)
                c_store = partial_rope(
                    _to_dtype(c_store, torch.float), pos,
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
            new_C_all = torch.cat(
                [new_C_first.unsqueeze(1), new_C_rest], dim=1)
        else:
            new_C_all = new_C_first.unsqueeze(1)
        n_new = 1 + n_rest
        # Partial RoPE: rotate the just-completed heavy compressed entries
        # at their block-anchor positions ``block_idx * m2`` (same positions
        # and rotate-before-store order as the per-token path).
        if self.rope_dim_eff > 0:
            pos_b = torch.arange(
                self._n_blocks, self._n_blocks + n_new, device=self.device) * m2
            new_C_all = partial_rope(
                new_C_all, pos_b,
                self.rope_dim_eff, self.rope_base, dim=1)
        block_idx0 = self._n_blocks
        self._ensure_compressed_capacity(block_idx0 + n_new)
        self._C_comp[:, block_idx0:block_idx0 + n_new] = _to_dtype(new_C_all, dtype)
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
            pos_t = torch.tensor([t], device=device)
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
            cbm = torch.arange(n_blocks, device=device) < b_threshold
            cbm = cbm[None, None, :]                                  # [1, 1, n_blocks]
            # Honor ``normalize_qk`` for the CORE attention (as in
            # ``naive_hca``). When normalize_qk=True compressed KV are
            # normalized with ``qk_norm_mode`` (cosine-similarity
            # attention for 'l2', the paper §2.3.3 RMSNorm form for
            # 'rms'). When False, skip normalization and fall back
            # to classical 1/sqrt(c) dot-product scale — this gives
            # ablations a real off-switch rather than a silently-ignored
            # flag.
            if normalize_qk:
                C_comp_n = _qk_normalize(
                    _to_dtype(self._C_comp[:, :self._n_blocks],
                              compute_dtype),
                    self.qk_norm_mode)                                # [B, n_blocks, c]
            else:
                C_comp_n = _to_dtype(
                    self._C_comp[:, :self._n_blocks], compute_dtype)
            # scores: [B, nh, 1, n_blocks] -> [B, 1, nh, n_blocks]
            scores = torch.einsum(
                'b t h d, b n d -> b t h n', q_compute, C_comp_n,
            )                                                          # [B, 1, nh, n_blocks]
            if core_scale != 1.0:
                # Multiplying by exactly 1.0 is an IEEE no-op; skipping the
                # multiply keeps values bit-for-bit identical.
                scores = scores * core_scale
            scores = scores.masked_fill(~cbm[:, :, None, :], float('-inf'))

        # --- Sliding-window branch: window scores for the joint softmax ---
        if self._sw is None or self._sw.n_valid == 0:
            scores_w = torch.empty(
                B, 1, nh, 0, dtype=compute_dtype, device=device,
            )
            C_local = torch.empty(B, 0, self.c, dtype=compute_dtype, device=device)
        else:
            C_local = _to_dtype(self._sw.get(), compute_dtype)         # [B, sw_len, c]
            sw = torch.einsum(
                'b h d, b s d -> b h s', q_compute[:, 0], C_local,
            )                                                          # [B, nh, sw_len]
            if core_scale != 1.0:
                # Multiplying by exactly 1.0 is an IEEE no-op; skipping the
                # multiply keeps values bit-for-bit identical.
                sw = sw * core_scale
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
            p = p.masked_fill(all_masked, 0.0)
        else:
            all_masked = torch.isinf(cat).all(-1, keepdim=True)
            safe = cat.masked_fill(all_masked, 0.0)
            p = torch.softmax(safe, dim=-1)
            p = p.masked_fill(all_masked, 0.0)
        n_blk = scores.shape[-1]
        p_dense = p[..., :n_blk]
        p_w = p[..., n_blk:]
        dense_out = torch.einsum(
            'b t h n, b n d -> b t h d', p_dense, C_comp_n,
        )                                                              # [B, 1, nh, c]
        sw_out = torch.einsum(
            'b t h w, b w d -> b t h d', p_w, C_local,
        )                                                              # [B, 1, nh, c]
        out = dense_out + sw_out
        # Partial RoPE output countermeasure (§2.3.3): inverse-rotate the
        # core-attention output at the NEGATED query position so each KV
        # entry's contribution carries the relative (entry − query)
        # distance — mirroring ``naive_hca``'s output path.
        if self.rope_dim_eff > 0:
            pos_t = torch.tensor([t], device=device)
            out = partial_rope(
                out, -pos_t, self.rope_dim_eff, self.rope_base, dim=1)
        return out
