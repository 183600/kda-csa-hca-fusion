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
    G = HV // H
    assert HV % H == 0, f"HV={HV} must be divisible by H={H} (GVA factor)"
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
        # Cast initial_state to compute_dtype before adding. If the caller
        # passes a higher-precision state (e.g. fp64 state with fp32 inputs
        # where compute_dtype=fp32), the in-place add would raise
        # ``RuntimeError: result type Double can't be cast to the desired
        # output type Float``. This can happen when the caller changes dtype
        # between calls (e.g. fp64 first call -> fp32 second call) and reuses
        # the returned state as initial_state.
        S += initial_state.to(compute_dtype)
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
    G = HV // H
    assert HV % H == 0, f"HV={HV} must be divisible by H={H} (GVA factor)"
    BT = chunk_size
    original_T = T
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
    A = torch.linalg.solve_triangular(
        torch.eye(BT, dtype=A.dtype, device=A.device) - A, A, upper=False
    )
    A = (A + torch.eye(BT, dtype=compute_dtype, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        # Cast initial_state to compute_dtype before adding (same reason as
        # naive_recurrent_kda: prevents dtype-mismatch RuntimeError when the
        # caller reuses a state from a different-precision call).
        S += initial_state.to(compute_dtype)
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
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    if not output_final_state:
        S = None
    else:
        # Cast the returned state back to the caller's dtype for consistency
        # (mirrors naive_recurrent_kda).
        S = S.to(dtype)
    o = rearrange(o, 'b h n c d -> b (n c) h d').to(dtype)
    return o[:, :original_T], S
