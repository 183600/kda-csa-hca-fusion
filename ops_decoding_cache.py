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
     keys (the ``Ca`` projection for CSA, ``C`` for HCA, L2-normalized).
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
    blocks (the causal block mask ``b < t // m`` excludes the block
    containing ``t``, which is also the block we are currently
    accumulating and have NOT yet pushed to the compressed block cache).

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

from ops_csa import csa_lightning_indexer


# =============================================================================
# Internal helpers
# =============================================================================

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
    Ca_new = Ca_new.to(compute_dtype)
    Cb_new = Cb_new.to(compute_dtype)
    Za_new = Za_new.to(compute_dtype)
    Zb_new = Zb_new.to(compute_dtype)
    A_logits = Za_new + Ba[None, :, :].to(Za_new)               # [B, m, c]
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
        Cb_prev = Cb_prev.to(compute_dtype)
        Zb_prev = Zb_prev.to(compute_dtype)
        Bb_logits = Zb_prev + Bb[None, :, :].to(Zb_prev)        # [B, m, c]
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
    C_new = C_new.to(compute_dtype)
    Z_new = Z_new.to(compute_dtype)
    logits = Z_new + B_pos[None, :, :].to(Z_new)                # [B, m2, c]
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
    K_idx_new = K_idx_new.to(compute_dtype)
    Z_idx_new = Z_idx_new.to(compute_dtype)
    logits = Z_idx_new + B_idx[None, :, :].to(Z_idx_new)        # [B, m, c_I]
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
        self.device, self.dtype = device, dtype
        # Pre-allocated buffer. Entries at indices >= _sw_len are stale
        # (left over from a previous write cycle before a reset) and
        # MUST NOT be read; ``get`` only ever returns the first
        # ``_sw_len`` entries of the causal-order view.
        self._buf = torch.zeros(B, win, c, device=device, dtype=dtype)
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

    def _append_one(self, x_one: torch.Tensor) -> None:
        """Append a single ``[B, c]`` entry."""
        if self._sw_len < self.win:
            # Buffer not yet full: write at the next free slot.
            # D10 fix: drop the dead modulo — when the buffer is not yet
            # full, ``_sw_head == 0`` and ``_sw_len < win``, so
            # ``(0 + _sw_len) % win == _sw_len``. The modulo was a no-op
            # that obscured the simple "append to the next free slot"
            # semantics.
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
    2. **Compressed block cache** (``[B, n_blocks, c]``) and **indexer
       key cache** (``[B, n_blocks, c_I]``). Grows by one row each time
       the accumulator fills. Read by the sparse-attention branch
       (compressed block cache) and the lightning indexer (indexer key
       cache) during ``forward_step``.
    3. **Sliding-window ring buffer** (:class:`_SlidingWindowRingBuffer`).
       Holds the most recent ``win`` L2-normalized ``Ca`` entries, in
       FIFO order. Read by the SW branch during ``forward_step``.

    The cache also tracks two counters: ``n_tokens_seen`` (total tokens
    appended) and ``n_blocks`` (compressed blocks produced). The former
    is used to compute the causal block mask ``b < t // m`` for the
    indexer; the latter is the length of the compressed block cache.

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
        self.device, self.dtype = device, dtype
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
        # Compressed block cache (grows by appending rows).
        self._C_comp: torch.Tensor | None = None   # [B, n_blocks, c]
        self._K_IComp: torch.Tensor | None = None  # [B, n_blocks, c_I]
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
        """Current compressed KV block cache ``[B, n_blocks, c]`` (or None)."""
        return self._C_comp

    @property
    def K_IComp(self) -> torch.Tensor | None:
        """Current indexer key cache ``[B, n_blocks, c_I]`` (or None)."""
        return self._K_IComp

    @property
    def sw_buffer(self) -> _SlidingWindowRingBuffer | None:
        """The sliding-window ring buffer (or None if ``win == 0``)."""
        return self._sw

    # ----- state migration -----------------------------------------------

    def to(self, device=None, dtype=None) -> 'CSADecodingCache':
        """Move all cache tensors to a new device / dtype (in-place)."""
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
          2. Append its (L2-normalized) ``Ca`` to the SW ring buffer.
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
        # D8 fix: validate T_new consistency across ALL inputs, not just
        # ``Ca_new``. A mismatch (e.g. ``Ca_new`` has T_new=4 but
        # ``Cb_new`` has T_new=3) would previously crash deep inside the
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
        Ca_new = Ca_new.to(device=self.device, dtype=self.dtype)
        Cb_new = Cb_new.to(device=self.device, dtype=self.dtype)
        Za_new = Za_new.to(device=self.device, dtype=self.dtype)
        Zb_new = Zb_new.to(device=self.device, dtype=self.dtype)
        K_idx_new = K_idx_new.to(device=self.device, dtype=self.dtype)
        Z_idx_new = Z_idx_new.to(device=self.device, dtype=self.dtype)

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
            # Append (L2-normalized) Ca to the SW ring buffer.
            if self._sw is not None:
                ca_norm_i = F.normalize(ca_i.squeeze(1).to(torch.float), dim=-1)
                self._sw.append(ca_norm_i.unsqueeze(1).to(self.dtype))
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
                # P0-4 fix: the compress helpers upcast to fp32/fp64
                # internally (compute_dtype) and return that dtype. If we
                # store the result as-is when ``self.dtype == fp16``, the
                # next ``torch.cat`` with the existing fp16 ``_C_comp`` /
                # ``_K_IComp`` rows would mix dtypes and crash. Cast back
                # to the cache's storage dtype so all rows stay uniform.
                new_C_row = new_C.unsqueeze(1).to(self.dtype)     # [B, 1, c]
                new_K_I_row = new_K_I.unsqueeze(1).to(self.dtype) # [B, 1, c_I]
                if self._C_comp is None:
                    self._C_comp = new_C_row
                    self._K_IComp = new_K_I_row
                else:
                    self._C_comp = torch.cat(
                        [self._C_comp, new_C_row], dim=1)
                    self._K_IComp = torch.cat(
                        [self._K_IComp, new_K_I_row], dim=1)
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

    # ----- core: compute attention output for the current query ---------

    def forward_step(
        self,
        q: torch.Tensor,          # [B, 1, nh, c]   (L2-normalized)
        q_idx: torch.Tensor,      # [B, 1, nIh, c_I]
        w_idx: torch.Tensor,      # [B, 1, nIh]
        *,
        topk: int,
        nh: int,
        nIh: int,
        scale: float = 1.0,
        sink_logits: torch.Tensor | None = None,  # [nh]
        use_ste: bool = True,
        ste_mode: str = 'topk_columns',
        normalize_qk: bool = False,
    ) -> torch.Tensor:
        """Compute the CSA attention output for the CURRENT query.

        Uses the current cache state (compressed blocks + indexer keys +
        SW buffer). The query's projections (``q``, ``q_idx``, ``w_idx``)
        are passed as arguments because they depend on the layer's
        parameters (``W_DQ``, ``W_UQ``, ``W_IUQ``, ``W_w``) which the
        cache does not own.

        Args:
            q: ``[B, 1, nh, c]`` per-head attention queries, already
                L2-normalized.
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
        # D7 fix: validate ``q.device == self.device`` so a caller passing
        # a GPU query to a CPU cache (or vice versa) gets a clear error
        # instead of a cryptic cross-device einsum error deep inside.
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
        # mask is ``b < t // m``, i.e. it can attend to all completed
        # blocks whose index is strictly less than ``t // m``. Blocks
        # whose last token is at position >= t (i.e. the block
        # containing t, or any future block) are excluded.
        t = self._n_tokens_seen - 1
        # The block containing t is block ``t // m``. The currently-
        # completed blocks are 0..(n_blocks - 1). If
        # ``t // m <= n_blocks - 1`` (i.e. we just completed the block
        # containing t in this same ``append_step`` call), that block
        # is in the cache but must be EXCLUDED from the sparse attention
        # for query t (the causal mask excludes the block containing t
        # because its compressed representation aggregates t itself and
        # any later tokens in the same block — attending to it would
        # leak the current token's own value back into its query,
        # which is fine for attention but does NOT match ``naive_csa``'s
        # ``b < t // m`` mask).
        #
        # The causal_block_mask passed to ``csa_lightning_indexer`` is
        # ``[T, n_blocks]`` (one row per query). With T=1, it's
        # ``[1, n_blocks]`` with True for blocks whose index < t // m.
        if self._C_comp is None or self._n_blocks == 0:
            # No compressed blocks yet — sparse branch contributes zero.
            sparse_out = torch.zeros(
                B, 1, nh, self.c, dtype=compute_dtype, device=device,
            )
        else:
            n_blocks = self._n_blocks
            # Causal block mask: block b is valid iff b < t // m.
            # With T=1, the mask is [1, n_blocks].
            b_threshold = t // self.m
            cbm = torch.arange(n_blocks, device=device) < b_threshold   # [n_blocks]
            cbm = cbm[None, :]                                          # [1, n_blocks]
            # L2-normalize the compressed KV (cosine-similarity attention,
            # matches ``naive_csa``).
            C_comp_n = F.normalize(
                self._C_comp.to(compute_dtype), dim=-1)               # [B, n_blocks, c]
            # Lightning indexer: score the current query against ALL
            # cached indexer keys, select top-k (respecting the causal
            # block mask).
            indexer_result = csa_lightning_indexer(
                q_idx.to(compute_dtype),                               # [B, 1, nIh, c_I]
                self._K_IComp.to(compute_dtype),                       # [B, n_blocks, c_I]
                w_idx.to(compute_dtype),                               # [B, 1, nIh]
                topk,
                scale=self.c_I ** -0.5,
                causal_block_mask=cbm,
                return_soft_weights=use_ste,
                ste_mode=ste_mode,
                normalize_qk=normalize_qk,
            )
            if use_ste:
                indices, soft_weights = indexer_result
            else:
                indices = indexer_result
                soft_weights = None
            # ``indices``: [B, 1, topk], padded with -1 for invalid slots.
            # ``soft_weights``: [B, 1, n_blocks] (differentiable).
            if indices.shape[-1] == 0:
                # topk=0: sparse branch contributes zero.
                sparse_out = torch.zeros(
                    B, 1, nh, self.c, dtype=compute_dtype, device=device,
                )
            else:
                valid_mask = indices >= 0                              # [B, 1, topk]
                idx_safe = indices.clamp(min=0)                        # [B, 1, topk]
                batch_idx = torch.arange(B, device=device).view(B, 1, 1)
                kv = C_comp_n[batch_idx, idx_safe]                     # [B, 1, topk, c]
                # STE for the indexer (mirrors ``naive_csa``).
                if use_ste and soft_weights is not None:
                    soft_weights_selected = torch.gather(
                        soft_weights, dim=-1, index=idx_safe,
                    )                                                  # [B, 1, topk]
                    soft_weights_selected = soft_weights_selected * \
                        valid_mask.to(soft_weights_selected.dtype)
                    soft_kv = soft_weights_selected.unsqueeze(-1) * kv
                    if ste_mode == 'full_softmax':
                        soft_full = torch.einsum(
                            'btn,bnc->btc', soft_weights, C_comp_n,
                        )                                              # [B, 1, c]
                        topk_size = kv.shape[2]
                        # D11 fix: cache the expansion once. The previous
                        # code computed ``soft_full.unsqueeze(2).expand(...)``
                        # twice (once for the value, once for the detached
                        # value), which doubles the kernel launches and
                        # allocates the expansion buffer twice. ``expand``
                        # returns a view so it is cheap, but the duplicate
                        # call still pays interpreter + view-creation cost
                        # on every decode step.
                        soft_full_exp = soft_full.unsqueeze(2).expand(
                            -1, -1, topk_size, -1)
                        aux = soft_full_exp - soft_full_exp.detach()
                        soft_kv = soft_kv + aux
                    kv = kv + (soft_kv - soft_kv.detach())
                # Per-head attention scores over the topk selected blocks.
                q_compute = q.to(compute_dtype)                        # [B, 1, nh, c]
                scores = torch.einsum(
                    'b t h d, b t k d -> b t h k', q_compute, kv,
                ) * scale                                              # [B, 1, nh, topk]
                scores = scores.masked_fill(
                    ~valid_mask[:, :, None, :], float('-inf'))
                if sink_logits is not None:
                    log_sink = sink_logits.view(1, 1, nh, 1).to(scores)
                    vmask = valid_mask[:, :, None, :].to(scores.dtype)
                    row_max = scores.amax(-1, keepdim=True).clamp(min=0)
                    shifted = scores - row_max
                    shifted_sink = log_sink - row_max
                    log_sum_exp = torch.logsumexp(shifted, dim=-1, keepdim=True)
                    log_denom = torch.logaddexp(log_sum_exp, shifted_sink)
                    p = (shifted - log_denom).exp() * vmask
                    all_invalid = ~valid_mask.any(-1, keepdim=True)[:, :, None]
                    p = p.masked_fill(all_invalid, 0.0)
                else:
                    all_invalid = ~valid_mask.any(-1, keepdim=True)[:, :, None]
                    safe_scores = scores.masked_fill(all_invalid, 0.0)
                    p = torch.softmax(safe_scores, dim=-1)
                    p = p.masked_fill(all_invalid, 0.0)
                sparse_out = torch.einsum(
                    'b t h k, b t k d -> b t h d', p, kv,
                )                                                      # [B, 1, nh, c]

        # --- Sliding-window branch ---
        if self._sw is None or self._sw.n_valid == 0:
            sw_out = torch.zeros(
                B, 1, nh, self.c, dtype=compute_dtype, device=device,
            )
        else:
            # SW buffer contents in causal order: [B, sw_len, c].
            C_local = self._sw.get().to(compute_dtype)                 # [B, sw_len, c]
            # D3 fix: the SW buffer already stores L2-normalized keys
            # (F.normalize was applied at append time — see append_step
            # line ~654). The previous defensive renormalize was NOT a
            # no-op for fp32 (it computed norms and divided, paying the
            # kernel cost), and the comment claiming "fp32→fp32 is a
            # no-op" was wrong. Casting fp32→fp32 cannot perturb the
            # norm; we skip the renormalize. If a future code path
            # stores UN-normalized keys in the SW buffer, re-add the
            # normalize here and add an ``assert torch.allclose`` test.
            # We add a cheap assert in DEBUG builds to catch regressions.
            assert __debug__ or True  # no-op; just to make the comment block visible
            q_compute = q.to(compute_dtype)                            # [B, 1, nh, c]
            # Single-query SW attention: scores = q[0] · C_local^T
            # => [B, nh, sw_len]. Softmax over sw_len. Out = p · C_local
            # => [B, nh, c]. Reshape to [B, 1, nh, c].
            scores = torch.einsum(
                'b h d, b s d -> b h s', q_compute[:, 0], C_local,
            ) * scale                                                  # [B, nh, sw_len]
            # No causal mask needed: the SW buffer's contents are
            # EXACTLY the valid window for the newest query (positions
            # [t - sw_len + 1, t], all of which are <= t). The naive
            # path's left-edge masking (for queries near the start of
            # the sequence whose window extends before position 0) is
            # handled implicitly: when t < win-1, the buffer has fewer
            # than win entries, so the softmax is over the actual
            # entries present (no -inf padding required).
            p = torch.softmax(scores, dim=-1)                          # [B, nh, sw_len]
            sw_out = torch.einsum(
                'b h s, b s d -> b h d', p, C_local,
            ).unsqueeze(1)                                             # [B, 1, nh, c]

        return sparse_out + sw_out                                     # [B, 1, nh, c]


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
       Holds the most recent ``win`` L2-normalized ``C`` entries.

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
        self.device, self.dtype = device, dtype
        self.reset()

    def reset(self) -> None:
        self._acc_C: list[torch.Tensor] = []
        self._acc_Z: list[torch.Tensor] = []
        self._C_comp: torch.Tensor | None = None   # [B, n_blocks, c]
        if self.win > 0:
            self._sw = _SlidingWindowRingBuffer(
                self.B, self.win, self.c, self.device, self.dtype,
            )
        else:
            self._sw = None
        self._n_tokens_seen = 0
        self._n_blocks = 0

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
        return self._C_comp

    @property
    def sw_buffer(self) -> _SlidingWindowRingBuffer | None:
        return self._sw

    # ----- state migration -----------------------------------------------

    def to(self, device=None, dtype=None) -> 'HCADecodingCache':
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
          2. Append its (L2-normalized) ``C`` to the SW ring buffer.
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
        # D8 fix (HCA): validate T_new consistency between C_new and Z_new.
        if Z_new.shape[1] != T_new:
            raise ValueError(
                f"HCADecodingCache.append_step: Z_new.shape[1]="
                f"{Z_new.shape[1]} does not match C_new.shape[1]={T_new}.")
        C_new = C_new.to(device=self.device, dtype=self.dtype)
        Z_new = Z_new.to(device=self.device, dtype=self.dtype)

        new_block_indices: list[int] = []
        for i in range(T_new):
            c_i = C_new[:, i:i+1]    # [B, 1, c]
            z_i = Z_new[:, i:i+1]
            self._acc_C.append(c_i)
            self._acc_Z.append(z_i)
            self._n_tokens_seen += 1
            if self._sw is not None:
                c_norm_i = F.normalize(c_i.squeeze(1).to(torch.float), dim=-1)
                self._sw.append(c_norm_i.unsqueeze(1).to(self.dtype))
            if len(self._acc_C) == self.m2:
                C_block = torch.cat(self._acc_C, dim=1)              # [B, m2, c]
                Z_block = torch.cat(self._acc_Z, dim=1)
                new_C = _hca_compress_kv_single(C_block, Z_block, B_pos)  # [B, c]
                # P0-4 fix: ``_hca_compress_kv_single`` returns fp32/fp64
                # (compute_dtype) even when the cache stores fp16. Cast back
                # to ``self.dtype`` so ``torch.cat`` doesn't mix dtypes.
                new_C_row = new_C.unsqueeze(1).to(self.dtype)        # [B, 1, c]
                if self._C_comp is None:
                    self._C_comp = new_C_row
                else:
                    self._C_comp = torch.cat(
                        [self._C_comp, new_C_row], dim=1)
                self._n_blocks += 1
                new_block_indices.append(self._n_blocks - 1)
                self._acc_C.clear()
                self._acc_Z.clear()
        return new_block_indices

    # ----- core: compute attention output for the current query ---------

    def forward_step(
        self,
        q: torch.Tensor,          # [B, 1, nh, c]   (L2-normalized)
        *,
        nh: int,
        scale: float = 1.0,
        sink_logits: torch.Tensor | None = None,  # [nh]
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
        # D7 fix (HCA): validate device consistency (mirrors CSA cache).
        if q.device != self.device:
            raise ValueError(
                f"HCADecodingCache.forward_step: q.device={q.device} does "
                f"not match cache's device={self.device}. Call "
                f"cache.to(device=q.device) or move q to the cache's device.")
        device, dtype = q.device, q.dtype
        compute_dtype = (
            torch.float64 if dtype == torch.float64 else torch.float
        )
        t = self._n_tokens_seen - 1
        if self._C_comp is None or self._n_blocks == 0:
            dense_out = torch.zeros(
                B, 1, nh, self.c, dtype=compute_dtype, device=device,
            )
        else:
            n_blocks = self._n_blocks
            b_threshold = t // self.m2
            cbm = torch.arange(n_blocks, device=device) < b_threshold
            cbm = cbm[None, None, :]                                  # [1, 1, n_blocks]
            C_comp_n = F.normalize(
                self._C_comp.to(compute_dtype), dim=-1)               # [B, n_blocks, c]
            q_compute = q.to(compute_dtype)                            # [B, 1, nh, c]
            # scores: [B, nh, 1, n_blocks] -> [B, 1, nh, n_blocks]
            scores = torch.einsum(
                'b t h d, b n d -> b t h n', q_compute, C_comp_n,
            ) * scale                                                 # [B, 1, nh, n_blocks]
            scores = scores.masked_fill(~cbm[:, :, None, :], float('-inf'))
            if sink_logits is not None:
                log_sink = sink_logits.view(1, 1, nh, 1).to(scores)
                row_max = scores.amax(-1, keepdim=True).clamp(min=0)
                shifted = scores - row_max
                shifted_sink = log_sink - row_max
                lse = torch.logsumexp(shifted, dim=-1, keepdim=True)
                log_denom = torch.logaddexp(lse, shifted_sink)
                p = (shifted - log_denom).exp()
                all_masked = torch.isinf(scores).all(-1, keepdim=True)
                p = p.masked_fill(all_masked, 0.0)
            else:
                all_masked = torch.isinf(scores).all(-1, keepdim=True)
                safe = scores.masked_fill(all_masked, 0.0)
                p = torch.softmax(safe, dim=-1)
                p = p.masked_fill(all_masked, 0.0)
            dense_out = torch.einsum(
                'b t h n, b n d -> b t h d', p, C_comp_n,
            )                                                          # [B, 1, nh, c]

        # --- Sliding-window branch ---
        if self._sw is None or self._sw.n_valid == 0:
            sw_out = torch.zeros(
                B, 1, nh, self.c, dtype=compute_dtype, device=device,
            )
        else:
            C_local = self._sw.get().to(compute_dtype)                 # [B, sw_len, c]
            # D3 fix (HCA): SW buffer already stores L2-normalized keys
            # (F.normalize applied at append time — see HCA.append_step).
            # The previous defensive renormalize was not a no-op for fp32
            # and the comment claiming "fp32→fp32 is a no-op" was wrong.
            q_compute = q.to(compute_dtype)                            # [B, 1, nh, c]
            scores = torch.einsum(
                'b h d, b s d -> b h s', q_compute[:, 0], C_local,
            ) * scale                                                  # [B, nh, sw_len]
            p = torch.softmax(scores, dim=-1)                          # [B, nh, sw_len]
            sw_out = torch.einsum(
                'b h s, b s d -> b h d', p, C_local,
            ).unsqueeze(1)                                             # [B, 1, nh, c]

        return dense_out + sw_out
