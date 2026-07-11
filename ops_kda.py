"""Standalone KDA (Kimi Delta Attention) naive reference implementation.

Adapted from flash-linear-attention/fla/ops/kda/naive.py
(MIT licensed, Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li,
modified with the support of the Moonshot AI Team).

KDA recurrence (per head, per step t):

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

where alpha_t = exp(g_t) is the per-channel fine-grained forget gate (in log
space as ``g``) and beta_t is the delta-rule learning rate. ``Diag(alpha_t)``
is the key novelty of KDA over Gated DeltaNet (which uses a single scalar
forget gate per head).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """Naive step-by-step recurrent KDA (reference, O(T) sequential).

    Args:
        q: ``[B, T, H, K]``
        k: ``[B, T, H, K]``
        v: ``[B, T, HV, V]`` (HV must be divisible by H, enabling GVA)
        g: per-channel log-decay gate ``[B, T, HV, K]``
        beta: delta-rule learning rate ``[B, T, HV]``
        scale: defaults to ``1/sqrt(K)``
        initial_state: ``[B, HV, K, V]``
        output_final_state: return the final recurrent state.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    # Validate H BEFORE ``G = HV // H`` so H=0 produces a clear ValueError
    # instead of a bare ZeroDivisionError. Also validates non-negativity of
    # head dims so a malformed caller gets an informative message.
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the checks
    # survive ``python -O`` / ``PYTHONOPTIMIZE=1``. ``assert`` statements
    # are silently stripped under optimization, which would re-expose the
    # cryptic ZeroDivisionError this guard is specifically meant to prevent.
    if H < 1:
        raise ValueError(
            f"H={H} must be >= 1 (would cause ZeroDivisionError in HV // H)")
    if K < 1:
        raise ValueError(f"K={K} must be >= 1")
    if V < 1:
        raise ValueError(f"V={V} must be >= 1")
    G = HV // H
    if HV % H != 0:
        raise ValueError(
            f"HV={HV} must be divisible by H={H} (GVA factor)")
    # Validate g and beta head dimensions match HV. Without this, a mismatched
    # g (e.g. [B, T, H, K] instead of [B, T, HV, K]) would silently broadcast
    # or crash deep inside the recurrence loop with a cryptic einsum error.
    # Use ``raise ValueError`` (NOT ``assert``) so the checks survive ``-O``.
    if g.shape[2] != HV:
        raise ValueError(
            f"g.shape[2]={g.shape[2]} must equal HV={HV} "
            f"(g must be [B, T, HV, K], got {tuple(g.shape)})")
    if g.shape[-1] != K:
        raise ValueError(
            f"g.shape[-1]={g.shape[-1]} must equal K={K} "
            f"(g must be [B, T, HV, K], got {tuple(g.shape)})")
    if beta.shape[2] != HV:
        raise ValueError(
            f"beta.shape[2]={beta.shape[2]} must equal HV={HV} "
            f"(beta must be [B, T, HV], got {tuple(beta.shape)})")
    if scale is None:
        scale = K ** -0.5

    # Compute in at least float32 for numerical stability, but preserve
    # float64 when the caller asks for it (gradient checks, high-precision
    # correctness tests).
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float
    q, k, v, g, beta = map(lambda x: x.to(compute_dtype), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale   # [B, T, HV, K]
    k = k.repeat_interleave(G, dim=2)           # [B, T, HV, K]

    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        # Cast initial_state to compute_dtype AND move it to S's device before
        # adding. Previously we only cast dtype (``.to(compute_dtype)``); a
        # device mismatch (e.g. caller manually moves the model to CUDA but
        # passes a stale CPU state) would raise ``RuntimeError: Expected all
        # tensors to be on the same device``. Mirrors the more defensive
        # ``.to(Z)`` pattern used in ops_csa.py / ops_hca.py.
        #
        # Use out-of-place ``S = S + ...`` (NOT in-place ``S += ...``) so a
        # caller passing a leaf ``initial_state`` with ``requires_grad=True``
        # (e.g. BPTT across call boundaries) does not hit
        # ``RuntimeError: one of the variables needed for gradient
        # computation has been modified by an inplace operation`` on the
        # next backward(). ``Tensor.to(...)`` returns the *same* tensor when
        # dtype and device already match, so the in-place variant would
        # mutate a tensor that may participate in the user's autograd graph.
        # Mirrors the out-of-place pattern documented in the recurrence
        # loop below (see the comment block starting at the ``for i in
        # range(0, T)`` loop).
        S = S + initial_state.to(device=S.device, dtype=compute_dtype)
    # Degenerate case: empty sequence. The for-loop body would not execute,
    # and the function would *happen* to return correct shapes by accident
    # (o=torch.zeros_like(v) is [B, 0, HV, V]; S is uninitialized [B, HV, K, V]).
    # But the S.to(dtype) cast at the end of the function would run
    # unnecessarily and a future edit touching the loop body could break the
    # accident. Guard explicitly for clarity and consistency with
    # naive_csa / naive_hca / naive_chunk_kda.
    if T == 0:
        o = torch.zeros_like(v)
        if not output_final_state:
            S = None
        else:
            S = S.to(dtype)
        return o.to(dtype), S
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i,
                             v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    else:
        # Cast the returned state back to the caller's dtype for consistency
        # (compute_dtype may be fp32/fp64 while the caller passed fp16/bf16).
        # This avoids dtype-mismatch surprises if the state is reused as
        # initial_state in a subsequent call with a different input dtype.
        S = S.to(dtype)
    return o.to(dtype), S


def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    """Chunkwise-parallel KDA (reference). Matches ``naive_recurrent_kda`` up to fp error."""
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    # Validate H and chunk_size BEFORE the divisions ``HV // H`` and
    # ``T // BT`` so zero or negative values produce a clear AssertionError
    # instead of a bare ZeroDivisionError. Mirrors naive_recurrent_kda.
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the checks
    # survive ``python -O`` / ``PYTHONOPTIMIZE=1``. ``assert`` statements
    # are silently stripped under optimization, which would re-expose the
    # cryptic ZeroDivisionError this guard is specifically meant to prevent.
    if H < 1:
        raise ValueError(
            f"H={H} must be >= 1 (would cause ZeroDivisionError in HV // H)")
    if K < 1:
        raise ValueError(f"K={K} must be >= 1")
    if V < 1:
        raise ValueError(f"V={V} must be >= 1")
    if chunk_size < 1:
        raise ValueError(
            f"chunk_size={chunk_size} must be >= 1 "
            f"(would cause ZeroDivisionError in T // chunk_size and "
            f"(-T) % chunk_size)")
    G = HV // H
    if HV % H != 0:
        raise ValueError(
            f"HV={HV} must be divisible by H={H} (GVA factor)")
    # Validate g and beta head dimensions match HV (mirrors
    # naive_recurrent_kda). Without this, a mismatched g or beta would
    # crash deep inside the chunk computation with a cryptic einsum error.
    # Use ``raise ValueError`` (NOT ``assert``) so the checks survive ``-O``.
    if g.shape[2] != HV:
        raise ValueError(
            f"g.shape[2]={g.shape[2]} must equal HV={HV} "
            f"(g must be [B, T, HV, K], got {tuple(g.shape)})")
    if g.shape[-1] != K:
        raise ValueError(
            f"g.shape[-1]={g.shape[-1]} must equal K={K} "
            f"(g must be [B, T, HV, K], got {tuple(g.shape)})")
    if beta.shape[2] != HV:
        raise ValueError(
            f"beta.shape[2]={beta.shape[2]} must equal HV={HV} "
            f"(beta must be [B, T, HV], got {tuple(beta.shape)})")
    BT = chunk_size
    original_T = T
    # Degenerate case: empty sequence. The downstream
    # ``torch.linalg.solve_triangular`` on an empty NT=0 batch raises
    # ``RuntimeError: solve_triangular: A and b must have the same number
    # of rows``. Guard explicitly (mirrors naive_recurrent_kda /
    # naive_csa / naive_hca).
    if T == 0:
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float
        # Allocate S in ``compute_dtype`` (NOT q.dtype) so the in-place add
        # below does not crash with ``RuntimeError: result type Float cannot
        # be cast to the desired output type Half`` for fp16/bf16 callers
        # that pass a carried-over ``initial_state`` (e.g. a streaming
        # decoder whose first chunk happens to be empty). Mirrors the
        # ordering in ``naive_recurrent_kda`` (lines 74-86), where the
        # inputs are cast to ``compute_dtype`` *before* ``S = q.new_zeros``
        # so ``S`` is already in ``compute_dtype`` when the add runs.
        S = q.new_zeros(B, HV, K, V, dtype=compute_dtype, device=q.device)
        if initial_state is not None:
            # Out-of-place add (not ``S += ...``) for autograd safety: a
            # caller passing a leaf ``initial_state`` with
            # ``requires_grad=True`` would otherwise hit "one of the
            # variables needed for gradient computation has been modified
            # by an inplace operation" on the next backward(). Mirrors the
            # out-of-place pattern documented in the recurrence loop
            # (see the ``S = S + ...`` state-update step inside the
            # ``for i in range(0, NT)`` loop below).
            S = S + initial_state.to(device=S.device, dtype=compute_dtype)
        o = q.new_zeros(B, 0, HV, V, dtype=compute_dtype, device=q.device)
        if not output_final_state:
            S = None
        else:
            S = S.to(dtype)
        return o.to(dtype), S
    pad = (-T) % BT
    if pad:
        # Right-pad T up to a multiple of BT so callers don't have to.
        # q/k/g are [B, T, H, K] (4D); v is [B, T, HV, V] (4D);
        # beta is [B, T, HV] (3D).
        q    = F.pad(q,    (0, 0, 0, 0, 0, pad))
        k    = F.pad(k,    (0, 0, 0, 0, 0, pad))
        v    = F.pad(v,    (0, 0, 0, 0, 0, pad))
        g    = F.pad(g,    (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
        T = T + pad
    NT = T // BT
    if scale is None:
        scale = K ** -0.5

    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float
    q, k = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(compute_dtype) for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(compute_dtype) for x in [v, g, beta]]
    q = q.repeat_interleave(G, dim=1) * scale
    k = k.repeat_interleave(G, dim=1)
    g = g.cumsum(-2)

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    A = torch.zeros(*g.shape[:-1], BT, dtype=compute_dtype, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]
    A = -A.masked_fill(mask, 0)
    # Vectorized Neumann series. At this point ``A`` is strictly lower
    # triangular (call it ``N``). The in-place forward-substitution loop
    #   for i in range(1, BT):
    #       A[..., i, :i] += (A[..., i, :, None] * A[..., :, :i]).sum(-2)
    # computes ``N + N^2 + N^3 + ... = (I - N)^{-1} N`` (each row is updated
    # using the already-finalized rows above it). A single batched triangular
    # solve evaluates the same quantity without BT iterations x 3 clones.
    # Standardize both ``torch.eye`` calls on (compute_dtype, q.device) so a
    # future cast of ``A`` to a different dtype (e.g. for memory) does not
    # silently diverge the two eyes and break the ``I - A`` / ``A + I`` math.
    A = torch.linalg.solve_triangular(
        torch.eye(BT, dtype=compute_dtype, device=q.device) - A, A, upper=False
    )
    A = (A + torch.eye(BT, dtype=compute_dtype, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        # Cast initial_state to compute_dtype AND move to S's device before
        # adding (mirrors the fix in naive_recurrent_kda: prevents both
        # dtype- and device-mismatch RuntimeErrors). Use out-of-place
        # ``S = S + ...`` (NOT in-place ``S += ...``) for autograd safety
        # — same rationale as the ``S = S + ...`` state-update step inside
        # the ``for i in range(0, NT)`` loop below. ``Tensor.to(...)`` may
        # return the *same* tensor when dtype/device already match, so the
        # in-place variant could mutate a tensor that participates in the
        # caller's autograd graph (e.g. BPTT across call boundaries).
        S = S + initial_state.to(device=S.device, dtype=compute_dtype)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i = u[:, :, i]
        g_i = g[:, :, i]
        w_i = w[:, :, i]
        diff = g_i.unsqueeze(-2) - g_i.unsqueeze(-3)
        Aqk = (q_i.unsqueeze(-2) * diff.exp() * k_i.unsqueeze(-3)).sum(-1)
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        # Use out-of-place ``S = S + ...`` (not in-place ``S += ...``) so the
        # state-update step is safe under future gradient-checkpointing. The
        # in-place variant would raise "one of the variables needed for
        # gradient computation has been modified by an inplace operation" if
        # anyone ever wraps this loop with checkpoint(). Mirrors the
        # out-of-place pattern in naive_recurrent_kda (line: S = S + einsum(...)).
        S = S + rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    if not output_final_state:
        S = None
    else:
        # Cast the returned state back to the caller's dtype for consistency
        # (mirrors naive_recurrent_kda).
        S = S.to(dtype)
    o = rearrange(o, 'b h n c d -> b (n c) h d').to(dtype)
    return o[:, :original_T], S
