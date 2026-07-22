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

import warnings as _warnings

import torch
import torch.nn.functional as F
from einops import rearrange


def _validate_kda_inputs(q, k, v, g, beta, fn_name='naive_recurrent_kda'):
    """Centralized shape / device / dtype contract validation for KDA inputs.

    Validated:
      * ``q``, ``k``: rank 4, shape ``[B, T, H, K]``.
      * ``v``: rank 4, shape ``[B, T, HV, V]``.
      * ``g``: rank 4, shape ``[B, T, HV, K]``.
      * ``beta``: rank 3, shape ``[B, T, HV]``.
      * B and T consistent across all tensors.
      * All tensors on the same device.
      * All tensors of the same dtype.

    Args:
        q, k, v, g, beta: the KDA inputs (see ``naive_recurrent_kda``'s
            docstring for the expected shapes).
        fn_name: the calling function name, used in error messages so the
            user knows which operator rejected the input.

    Raises:
        ValueError: with a message naming the misconfigured input.
    """
    for name, t, expected_rank in [
        ('q', q, 4), ('k', k, 4), ('v', v, 4), ('g', g, 4),
        ('beta', beta, 3),
    ]:
        if t.dim() != expected_rank:
            raise ValueError(
                f"{fn_name}: {name} must have rank {expected_rank} but got "
                f"rank {t.dim()} (shape={tuple(t.shape)}). "
                f"Expected shape: " +
                ("[B, T, H, K]" if name in ('q', 'k') else
                 "[B, T, HV, V]" if name == 'v' else
                 "[B, T, HV, K]" if name == 'g' else
                 "[B, T, HV]"))
    B_q, T_q = q.shape[0], q.shape[1]
    for name, t in [('k', k), ('v', v), ('g', g), ('beta', beta)]:
        if t.shape[0] != B_q or t.shape[1] != T_q:
            raise ValueError(
                f"{fn_name}: {name}.shape[0:2]={tuple(t.shape[0:2])} does not "
                f"match q.shape[0:2]=({B_q}, {T_q}). All inputs must share "
                f"the same batch (B) and sequence (T) dimensions.")
    ref_device = q.device
    for name, t in [('k', k), ('v', v), ('g', g), ('beta', beta)]:
        if t.device != ref_device:
            raise ValueError(
                f"{fn_name}: {name}.device={t.device} does not match "
                f"q.device={ref_device}. All inputs must be on the same "
                f"device.")
    ref_dtype = q.dtype
    for name, t in [('k', k), ('v', v), ('g', g), ('beta', beta)]:
        if t.dtype != ref_dtype:
            raise ValueError(
                f"{fn_name}: {name}.dtype={t.dtype} does not match "
                f"q.dtype={ref_dtype}. All inputs must share the same "
                f"dtype (cast before calling if needed).")


def _validate_kda_shapes(q, k, v, g, beta, fn_name, chunk_size=None):
    """Validate KDA shape dimensions.

    Args:
        chunk_size: if not None, also validates chunk_size >= 1 (chunk path
            only).
    """
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    if H < 1:
        raise ValueError(
            f"{fn_name}: H={H} must be >= 1 "
            f"(would cause ZeroDivisionError in HV // H)")
    if K < 1:
        raise ValueError(f"{fn_name}: K={K} must be >= 1")
    if V < 1:
        raise ValueError(f"{fn_name}: V={V} must be >= 1")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(
            f"{fn_name}: chunk_size={chunk_size} must be >= 1 "
            f"(would cause ZeroDivisionError in T // chunk_size and "
            f"(-T) % chunk_size)")
    if HV % H != 0:
        raise ValueError(
            f"{fn_name}: HV={HV} must be divisible by H={H} (GVA factor)")
    if g.shape[2] != HV:
        raise ValueError(
            f"{fn_name}: g.shape[2]={g.shape[2]} must equal HV={HV} "
            f"(g must be [B, T, HV, K], got {tuple(g.shape)})")
    if g.shape[-1] != K:
        raise ValueError(
            f"{fn_name}: g.shape[-1]={g.shape[-1]} must equal K={K} "
            f"(g must be [B, T, HV, K], got {tuple(g.shape)})")
    if beta.shape[2] != HV:
        raise ValueError(
            f"{fn_name}: beta.shape[2]={beta.shape[2]} must equal HV={HV} "
            f"(beta must be [B, T, HV], got {tuple(beta.shape)})")


def _compute_dtype(dtype):
    """Pick the compute dtype for KDA math."""
    return torch.float64 if dtype == torch.float64 else torch.float


def _is_compiling_safely() -> bool:
    """Return True if we are inside a torch.compile / torch.export trace."""
    try:
        return torch.compiler.is_compiling()  # type: ignore[attr-defined]
    except AttributeError:
        pass
    except Exception:
        return False
    try:
        import torch._dynamo as _dm
        fn = getattr(_dm, "is_compiling", None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    return False


def _warn_if_nonfinite(o, fn_name, stacklevel=3):
    """Surface non-finite KDA outputs with an actionable hint."""
    if _is_compiling_safely():
        return
    if not torch.isfinite(o).all():
        _warnings.warn(
            f"{fn_name}: output contains NaN/Inf. The KDA delta-rule "
            f"recurrence is only numerically bounded for unit-norm "
            f"q/k (L2-normalize along the last dim before calling, as "
            f"every regression test in this repo does) and "
            f"moderate-magnitude v; un-normalized or large-magnitude "
            f"inputs can make the recurrent state S diverge over a long "
            f"sequence, producing NaN (recurrent path) or NaN/Inf (chunk "
            f"path — the two paths amplify divergence differently, so "
            f"they are NOT expected to agree once either produces "
            f"non-finite values). If this is unexpected, check that q/k "
            f"passed to {fn_name} are L2-normalized and that g_clamp_min "
            f"is not disabled (g_clamp_min=-inf).",
            stacklevel=stacklevel,
        )


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    *,
    g_clamp_min: float = -10.0,
    state_dtype: torch.dtype | None = None,
):
    """Naive step-by-step recurrent KDA (reference, O(T) sequential).

    .. warning:: **Input contract — q/k should be unit-norm.**

        This function does NOT itself normalize ``q``/``k``, but the
        delta-rule recurrence is only numerically bounded when ``q``/``k``
        are unit-norm along the last (``K``) axis. Callers should
        ``F.normalize(q, dim=-1)`` / ``F.normalize(k, dim=-1)`` before
        calling this function unless they have a specific reason not to.

    Args:
        q: ``[B, T, H, K]``
        k: ``[B, T, H, K]``
        v: ``[B, T, HV, V]`` (HV must be divisible by H, enabling GVA)
        g: per-channel log-decay gate ``[B, T, HV, K]``
        beta: delta-rule learning rate ``[B, T, HV]``
        scale: defaults to ``1/sqrt(K)``
        initial_state: ``[B, HV, K, V]``
        output_final_state: return the final recurrent state.
        g_clamp_min: lower bound for the log-decay gate ``g``.
        state_dtype: dtype of the RETURNED recurrent state.

    .. note::

        This is a **Python-loop reference implementation**.
    """
    dtype = v.dtype
    _validate_kda_inputs(q, k, v, g, beta, fn_name='naive_recurrent_kda')
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    _validate_kda_shapes(q, k, v, g, beta, fn_name='naive_recurrent_kda')
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    compute_dtype = _compute_dtype(dtype)
    q, k, v, g, beta = map(lambda x: x.to(compute_dtype), [q, k, v, g, beta])
    if g_clamp_min > -float('inf'):
        g = g.clamp(min=float(g_clamp_min))
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    if T > 8192:
        _warnings.warn(
            f"naive_recurrent_kda: T={T} > 8192; the Python for-loop is "
            f"interpreter-overhead-bound.",
            stacklevel=2,
        )
    elif T > 1024 and q.is_cuda:
        _warnings.warn(
            f"naive_recurrent_kda: T={T} > 1024 on CUDA.",
            stacklevel=2,
        )

    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state.to(device=S.device, dtype=compute_dtype)
    if T == 0:
        o = torch.zeros_like(v)
        if not output_final_state:
            S = None
        else:
            target_state_dtype = (state_dtype if state_dtype is not None
                                  else compute_dtype)
            S = S.to(target_state_dtype)
        return o.to(dtype), S
    outs = []
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i,
                             v_i - (k_i[..., None] * S).sum(-2))
        outs.append(torch.einsum('b h k, b h k v -> b h v', q_i, S))
    o = torch.stack(outs, dim=1)
    if not output_final_state:
        S = None
    else:
        target_state_dtype = state_dtype if state_dtype is not None else compute_dtype
        S = S.to(target_state_dtype)
    o = o.to(dtype)
    _warn_if_nonfinite(o, 'naive_recurrent_kda')
    return o, S


import collections as _collections
_COMPILED_KDA_CACHE: '_collections.OrderedDict' = _collections.OrderedDict()
_COMPILED_CACHE_MAX = 32


def compiled_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    *,
    g_clamp_min: float = -10.0,
    state_dtype: torch.dtype | None = None,
    mode: str | None = None,
    dynamic: bool = True,
    fullgraph: bool = False,
):
    """``torch.compile``-wrapped :func:`naive_recurrent_kda`."""
    _validate_kda_inputs(q, k, v, g, beta, fn_name='compiled_recurrent_kda')
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    cache_key = (
        B, T, H, K, HV, V,
        q.dtype, q.device.type,
        q.requires_grad or k.requires_grad or v.requires_grad
        or g.requires_grad or beta.requires_grad,
        mode, dynamic, fullgraph,
        bool(output_final_state),
        bool(initial_state is not None),
        str(state_dtype),
        repr(scale),
        repr(g_clamp_min),
    )
    compiled_fn = _COMPILED_KDA_CACHE.get(cache_key)
    if compiled_fn is None:
        if len(_COMPILED_KDA_CACHE) >= _COMPILED_CACHE_MAX:
            _COMPILED_KDA_CACHE.popitem(last=False)
        def _kda_kernel(
            q, k, v, g, beta,
            scale_val, g_clamp_min_val, output_final_state_val,
            initial_state_arg,
        ):
            return naive_recurrent_kda(
                q, k, v, g, beta,
                scale=scale_val,
                initial_state=initial_state_arg,
                output_final_state=output_final_state_val,
                g_clamp_min=g_clamp_min_val,
                state_dtype=state_dtype,
            )

        compile_kwargs = {"dynamic": dynamic, "fullgraph": fullgraph}
        if mode is not None:
            compile_kwargs["mode"] = mode
        compiled_fn = torch.compile(_kda_kernel, **compile_kwargs)
        _COMPILED_KDA_CACHE[cache_key] = compiled_fn
    else:
        _COMPILED_KDA_CACHE.move_to_end(cache_key)

    return compiled_fn(
        q, k, v, g, beta,
        scale, g_clamp_min, output_final_state,
        initial_state,
    )


def _chunk_kda_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int,
    g_clamp_min: float,
    state_dtype: torch.dtype | None,
    fn_name: str,
):
    """Shared setup for the chunk-parallel KDA path."""
    dtype = v.dtype
    _validate_kda_inputs(q, k, v, g, beta, fn_name=fn_name)
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    _validate_kda_shapes(q, k, v, g, beta, fn_name=fn_name, chunk_size=chunk_size)
    G = HV // H
    BT = chunk_size
    original_T = T
    if T == 0:
        compute_dtype = _compute_dtype(dtype)
        S = q.new_zeros(B, HV, K, V, dtype=compute_dtype, device=q.device)
        if initial_state is not None:
            S = S + initial_state.to(device=S.device, dtype=compute_dtype)
        o = q.new_zeros(B, 0, HV, V, dtype=compute_dtype, device=q.device)
        if not output_final_state:
            S = None
        else:
            target_state_dtype = (state_dtype if state_dtype is not None
                                  else compute_dtype)
            S = S.to(target_state_dtype)
        return {'early': True, 'o': o.to(dtype), 'S': S}
    pad = (-T) % BT
    if pad:
        q    = F.pad(q,    (0, 0, 0, 0, 0, pad))
        k    = F.pad(k,    (0, 0, 0, 0, 0, pad))
        v    = F.pad(v,    (0, 0, 0, 0, 0, pad))
        g    = F.pad(g,    (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
        T = T + pad
    NT = T // BT
    if scale is None:
        scale = K ** -0.5

    compute_dtype = _compute_dtype(dtype)
    q, k = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(compute_dtype) for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(compute_dtype) for x in [v, g, beta]]
    q = q.repeat_interleave(G, dim=1) * scale
    k = k.repeat_interleave(G, dim=1)
    if g_clamp_min > -float('inf'):
        g = g.clamp(min=float(g_clamp_min))
    g = g.cumsum(-2)

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    A = torch.zeros(*g.shape[:-1], BT, dtype=compute_dtype, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i, :]
        g_diff = (g - g_i.unsqueeze(-2)).clamp(max=50.0)
        A[..., i] = (k * g_diff.exp() * k_i.unsqueeze(-2)).sum(-1)
    A = A * beta[..., None]
    A = -A.masked_fill(mask, 0)
    A = torch.linalg.solve_triangular(
        torch.eye(BT, dtype=compute_dtype, device=q.device) - A, A, upper=False
    )
    A = (A + torch.eye(BT, dtype=compute_dtype, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state.to(device=S.device, dtype=compute_dtype)
    upper_mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)

    return {
        'early': False,
        'q': q, 'k': k, 'u': u, 'g': g, 'w': w, 'S': S,
        'upper_mask': upper_mask, 'NT': NT,
        'dtype': dtype, 'compute_dtype': compute_dtype,
        'original_T': original_T,
    }


def _chunk_kda_finalize(
    chunk_outs,
    S: torch.Tensor,
    *,
    dtype: torch.dtype,
    compute_dtype: torch.dtype,
    state_dtype: torch.dtype | None,
    output_final_state: bool,
    original_T: int,
    fn_name: str,
):
    """Shared tail for the chunk-parallel KDA path."""
    o = torch.stack(chunk_outs, dim=2)
    if not output_final_state:
        S = None
    else:
        target_state_dtype = state_dtype if state_dtype is not None else compute_dtype
        S = S.to(target_state_dtype)
    o = rearrange(o, 'b h n c d -> b (n c) h d').to(dtype)
    o = o[:, :original_T]
    _warn_if_nonfinite(o, fn_name, stacklevel=4)
    return o, S


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
    *,
    g_clamp_min: float = -10.0,
    state_dtype: torch.dtype | None = None,
):
    """Chunkwise-parallel KDA (reference). Matches ``naive_recurrent_kda`` up to fp error."""
    prep = _chunk_kda_prepare(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        chunk_size, g_clamp_min, state_dtype, fn_name='naive_chunk_kda',
    )
    if prep['early']:
        return prep['o'], prep['S']
    chunk_outs, S = _chunk_kda_inner_loop(
        prep['q'], prep['k'], prep['u'], prep['g'], prep['w'], prep['S'],
        prep['upper_mask'], prep['NT'],
    )
    return _chunk_kda_finalize(
        chunk_outs, S,
        dtype=prep['dtype'], compute_dtype=prep['compute_dtype'],
        state_dtype=state_dtype, output_final_state=output_final_state,
        original_T=prep['original_T'], fn_name='naive_chunk_kda',
    )


def _chunk_kda_inner_loop(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    w: torch.Tensor,
    S: torch.Tensor,
    mask: torch.Tensor,
    NT: int,
) -> tuple:
    """TorchScript-compatible inner loop of ``naive_chunk_kda``."""
    chunk_outs = []
    for i in range(NT):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i = u[:, :, i]
        g_i = g[:, :, i]
        w_i = w[:, :, i]
        diff = g_i.unsqueeze(-2) - g_i.unsqueeze(-3)
        Aqk = (q_i.unsqueeze(-2) * diff.clamp(max=50.0).exp() * k_i.unsqueeze(-3)).sum(-1)
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        chunk_outs.append((q_i * g_i.exp()) @ S + Aqk @ v_i)
        S = S * g_i[:, :, -1].exp().unsqueeze(-1)
        update = torch.einsum('bhnci,bhncj->bhncij', (g_i - g_i[:, :, -1:]).exp() * k_i, v_i)
        S = S + update.sum(-3)
    return chunk_outs, S


def scripted_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    *,
    g_clamp_min: float = -10.0,
    state_dtype: torch.dtype | None = None,
    use_script: bool = True,
):
    """``naive_chunk_kda`` with a ``torch.jit.script``-compiled inner loop."""
    if not use_script:
        return naive_chunk_kda(
            q, k, v, g, beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            g_clamp_min=g_clamp_min,
            state_dtype=state_dtype,
        )

    prep = _chunk_kda_prepare(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        chunk_size, g_clamp_min, state_dtype, fn_name='scripted_chunk_kda',
    )
    if prep['early']:
        return prep['o'], prep['S']

    try:
        scripted_inner = torch.jit.script(_chunk_kda_inner_loop)
        chunk_outs, S = scripted_inner(
            prep['q'], prep['k'], prep['u'], prep['g'], prep['w'], prep['S'],
            prep['upper_mask'], prep['NT'],
        )
    except Exception as exc:
        _warnings.warn(
            f"scripted_chunk_kda: torch.jit.script failed ({type(exc).__name__}: "
            f"{exc}); falling back to the eager inner loop.",
            stacklevel=2,
        )
        chunk_outs, S = _chunk_kda_inner_loop(
            prep['q'], prep['k'], prep['u'], prep['g'], prep['w'], prep['S'],
            prep['upper_mask'], prep['NT'],
        )

    return _chunk_kda_finalize(
        chunk_outs, S,
        dtype=prep['dtype'], compute_dtype=prep['compute_dtype'],
        state_dtype=state_dtype, output_final_state=output_final_state,
        original_T=prep['original_T'], fn_name='scripted_chunk_kda',
    )
