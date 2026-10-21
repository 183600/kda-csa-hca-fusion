"""Online-softmax attention — streaming reference primitive.

Why this module?
----------------
CSA and HCA both compute a **joint softmax** over the union of the
selected/compressed KV entries, the sliding-window entries, and the
attention sink (DeepSeek-V4 §2.3.1 Eq 27; see ``ops_csa._joint_p`` and
``ops_hca._joint_p``).  The reference implementations materialize the
whole ``[B, nh, T, n_blocks + win]`` score row before the softmax, which
is the right thing for a *correctness* reference but is exactly what a
flash-style kernel must avoid.

This module is the streaming (online-softmax) formulation of the same
semantics, kept as an oracle for such kernels:

* per-query state ``(m, den, acc)`` with ``m`` the running max, ``den``
  the running ``sum exp(s - m)`` and ``acc`` the running
  ``sum exp(s - m) * v``;
* ``online_softmax_merge`` folds two states, so key chunks can be
  processed (and re-associated) in any order / granularity;
* ``online_softmax_attention`` streams over key chunks and never holds
  more than ``B * nh * T * key_chunk`` score elements;
* the sink enters **only the denominator** — a virtual key whose value is
  zero — matching ``ops_hca``: ``out = sum exp(s) v / (sum exp(s) +
  exp(sink))``.

Arithmetic identity: online softmax performs the same QK^T and P@V MACs
and the same number of per-score exponentials as the dense two-pass
softmax; only the memory footprint changes (pinned by the audit tests via
``flops_audit``).  Chunk merging adds one ``exp(m_old - m_new)`` per row
per chunk, which is why the primitive is not *bit*-identical to the dense
path — it is value-equal up to fp round-off, and it is exactly
re-association-invariant.
"""
from __future__ import annotations

import torch

__all__ = [
    'online_softmax_attention',
    'online_softmax_finalize',
    'online_softmax_merge',
    'online_softmax_state',
]

_NEG_INF = float('-inf')


def _shrink(shift: torch.Tensor) -> torch.Tensor:
    """``exp(shift)`` with the empty-row cases mapped to ``+0``.

    A row with no valid entry has ``m == -inf``; when the merged max is
    also ``-inf`` the naive difference ``-inf - -inf`` is ``nan``.  Such
    rows have ``l == 0`` and contribute nothing, so the correct carry
    factor is 0.  Non-finite shifts can only arise from all-masked rows.
    """
    safe = torch.where(torch.isfinite(shift), shift,
                       torch.full_like(shift, _NEG_INF))
    return safe.exp()


def online_softmax_merge(state_a: tuple, state_b: tuple) -> tuple:
    """Associative merge of two ``(m, l, acc)`` online-softmax states.

    Shapes: ``m`` / ``den``: ``[B, H, T, 1]``; ``acc``: ``[B, H, T, Dv]``.
    """
    m_a, den_a, acc_a = state_a
    m_b, den_b, acc_b = state_b
    if m_a.shape != m_b.shape or acc_a.shape != acc_b.shape:
        raise ValueError(
            f'online_softmax_merge: state shapes differ '
            f'({tuple(m_a.shape)} vs {tuple(m_b.shape)}).')
    m = torch.maximum(m_a, m_b)
    a_a = _shrink(m_a - m)
    a_b = _shrink(m_b - m)
    den = den_a * a_a + den_b * a_b
    acc = acc_a * a_a + acc_b * a_b
    # A row containing ``+inf`` scores puts all its mass on those entries,
    # so a state with ``m == +inf`` dominates any finite state and two
    # ``+inf`` states add. The carry factors above are 0 (or NaN turned
    # into 0 by ``_shrink``) for these rows, so replace their state here
    # instead of losing the mass.
    pos_inf_a = torch.isposinf(m_a)
    pos_inf_b = torch.isposinf(m_b)
    pos_inf = pos_inf_a | pos_inf_b
    if bool(pos_inf.any()):
        both = pos_inf_a & pos_inf_b
        den_inf = torch.where(
            both, den_a + den_b,
            torch.where(pos_inf_a, den_a, den_b))
        acc_inf = torch.where(
            both, acc_a + acc_b,
            torch.where(pos_inf_a, acc_a, acc_b))
        den = torch.where(pos_inf, den_inf, den)
        acc = torch.where(pos_inf, acc_inf, acc)
    return m, den, acc


def online_softmax_state(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         *, scale: float | None = None,
                         mask: torch.Tensor | None = None) -> tuple:
    """Online-softmax state for one key block.

    ``q``: ``[B, H, T, D]``; ``k`` / ``v``: ``[B, H, Nc, D]``.
    ``mask``: optional boolean ``[T, Nc]`` broadcastable to
    ``[B, H, T, Nc]`` where ``True`` = keep (used for the causal /
    block-causal cases).  Returns ``(m, den, acc)``.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            'online_softmax_state: q/k/v must be 4-D [B, H, T, D]; got '
            f'{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}.')
    if k.shape != v.shape:
        raise ValueError(
            'online_softmax_state: k and v must share shape (shared-KV / '
            f'MQA contract); got {tuple(k.shape)} vs {tuple(v.shape)}.')
    if k.shape[0] != q.shape[0] or k.shape[1] != q.shape[1]:
        raise ValueError(
            'online_softmax_state: batch/head dims of q and k must match.')
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            'online_softmax_state: q/k feature dims must match; got '
            f'{q.shape[-1]} vs {k.shape[-1]}.')
    if scale is None:
        scale = q.shape[-1] ** -0.5
    scores = torch.einsum('b h t d, b h n d -> b h t n', q, k) * scale
    if mask is not None:
        # The mask contract is boolean (``True`` = keep). An integer mask
        # would make ``~mask`` a BITWISE inversion (1 -> -2, 0 -> -1), so
        # the mask is validated instead of silently inverted.
        if mask.dtype != torch.bool:
            raise ValueError(
                'online_softmax_state: mask must be a boolean tensor '
                '(True = keep); got dtype '
                f'{mask.dtype}.')
        scores = scores.masked_fill(~mask, _NEG_INF)
    m = scores.amax(dim=-1, keepdim=True)
    pos_inf = torch.isposinf(scores)
    has_pos_inf = pos_inf.any(dim=-1, keepdim=True)
    if bool(has_pos_inf.any()):
        # ``+inf`` scores make ``scores - m`` an ``inf - inf`` NaN. Keep
        # the ordinary softmax for finite rows and use the limit (mass
        # split evenly over the +inf entries) on the others. Empty rows
        # also get a finite shift so the discarded ``exp`` branch cannot
        # poison backward.
        empty = torch.isneginf(m)
        special = empty | has_pos_inf
        m_safe = torch.where(special, torch.zeros_like(m), m)
        shifted = scores - m_safe
        shifted = torch.where(special, torch.zeros_like(shifted), shifted)
        p = shifted.exp()
        p = torch.where(empty, torch.zeros_like(p), p)
        p = torch.where(has_pos_inf, pos_inf.to(p.dtype), p)
    else:
        shifted = scores - m
        p = torch.where(torch.isneginf(m), torch.zeros_like(shifted),
                        shifted.exp())
    den = p.sum(dim=-1, keepdim=True)
    acc = torch.einsum('b h t n, b h n d -> b h t d', p, v)
    return m, den, acc


def _align_sink_logits(sink_logits: torch.Tensor,
                       m: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-head sink logit against ``m`` of shape [B, H, T, 1].

    ``[H]`` is the canonical form; the documented contract also accepts any
    tensor broadcastable to ``[B, H, T, 1]`` after the head axis has been
    made explicit (e.g. ``[B, H, 1, 1]`` or ``[1, H, 1, 1]``).
    """
    sink = sink_logits.to(device=m.device, dtype=m.dtype)
    H = m.shape[1]
    if sink.dim() == 1:
        if sink.shape[0] != H:
            raise ValueError(
                f'online_softmax: sink_logits has {sink.shape[0]} entries '
                f'but the attention has {H} heads.')
        return sink.view(1, H, 1, 1)
    if sink.dim() == 2:
        if sink.shape[-1] == H:
            return sink.view(*sink.shape, 1, 1)
        if sink.shape[-1] == 1 and sink.shape[0] == H:
            return sink.view(1, H, 1, 1)
        raise ValueError(
            'online_softmax: 2-D sink_logits must be [B, H] or [H, 1]; '
            f'got {tuple(sink.shape)} for H={H}.')
    if sink.dim() == 3:
        if sink.shape[-2:] == (1, 1):
            if sink.shape[0] == H:
                return sink.view(1, H, 1, 1)
            return sink.view(*sink.shape, 1)
        if sink.shape[-1] == 1:
            return sink.unsqueeze(-1)
        raise ValueError(
            'online_softmax: 3-D sink_logits must be [B, H, 1] or '
            f'[H, 1, 1]; got {tuple(sink.shape)} for H={H}.')
    if sink.dim() > 4:
        raise ValueError(
            'online_softmax: sink_logits must be [H] or broadcastable to '
            f'[B, H, T, 1]; got rank {sink.dim()} '
            f'(shape={tuple(sink.shape)}).')
    return sink


def online_softmax_finalize(m: torch.Tensor, den: torch.Tensor,
                            acc: torch.Tensor,
                            sink_logits: torch.Tensor | None = None):
    """Turn an online state into attention output.

    ``sink_logits``: optional ``[H]`` (or broadcastable) per-head virtual
    logit added to the **denominator only**, matching ``ops_hca`` /
    ``ops_csa``.  Rows with no valid entry (``den == 0``) return zeros.
    """
    empty = (den == 0)
    pos_inf = torch.isposinf(m)
    # Substitute a finite state for all-empty and ``+inf``-max rows BEFORE
    # forming the exponent. Empty rows are zeroed at the end anyway, and a
    # ``+inf`` row is the softmax limit ``acc / den`` (the sink is
    # irrelevant because its finite logit loses to the +inf scores).
    # Without the substitution ``-inf - (-inf)`` / ``inf - inf`` produces a
    # NaN/Inf that poisons the backward pass even though ``torch.where``
    # discards that branch forward.
    special = empty | pos_inf
    m_safe = torch.where(special, torch.zeros_like(m), m)
    den_safe = torch.where(empty, torch.ones_like(den), den)
    log_denom = m_safe + torch.log(
        den_safe.clamp_min(torch.finfo(den.dtype).tiny))
    if sink_logits is not None:
        sink = _align_sink_logits(sink_logits, m)
        sink = torch.where(pos_inf, torch.full_like(sink, _NEG_INF), sink)
        log_denom = torch.logaddexp(log_denom, sink)
    out = acc * (m_safe - log_denom).exp()
    return torch.where(empty.expand_as(out), torch.zeros_like(out), out)


def online_softmax_attention(q: torch.Tensor, k: torch.Tensor,
                             v: torch.Tensor, *,
                             key_chunk: int = 0,
                             scale: float | None = None,
                             mask: torch.Tensor | None = None,
                             sink_logits: torch.Tensor | None = None):
    """Streaming softmax attention over key chunks.

    Same value as ``softmax(q k^T / sqrt(D)) v`` with the optional
    ``mask`` (``True`` = keep) and an attention sink that appears in the
    denominator only.  ``key_chunk=0`` (default) uses one chunk (still
    online, just not memory-sharded).
    """
    if key_chunk < 0:
        raise ValueError(
            f'online_softmax_attention: key_chunk={key_chunk} must be >= 0 '
            f'(0 = single chunk).')
    n = k.shape[2]
    chunk = n if key_chunk == 0 else min(key_chunk, n)
    if n == 0:
        raise ValueError(
            'online_softmax_attention: the key/value sequence is empty.')
    merged = None
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        # ``mask`` may be [T, Nc] or any leading-batch/head broadcastable
        # form ([B, H, T, Nc]); only the key axis is chunked here.  A mask
        # whose last axis is a broadcast singleton (size 1, e.g. a
        # per-query keep/drop-all flag) must NOT be sliced: slicing a
        # size-1 axis past its start yields an empty block, which the
        # masked_fill/amax chain cannot reduce.  Such a mask is already
        # broadcastable to every chunk, so it is passed through whole.
        if mask is None:
            block_mask = None
        elif mask.shape[-1] == n:
            block_mask = mask[..., lo:hi]
        else:
            block_mask = mask
        state = online_softmax_state(
            q, k[:, :, lo:hi], v[:, :, lo:hi], scale=scale, mask=block_mask)
        merged = state if merged is None else online_softmax_merge(merged, state)
    return online_softmax_finalize(merged[0], merged[1], merged[2],
                                   sink_logits=sink_logits)
