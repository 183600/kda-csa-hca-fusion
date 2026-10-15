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

_KDA_NONFINITE_WARN = True


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
        # Probe torch._dynamo FIRST. ``torch._dynamo.is_compiling()`` is
        # tracable inside a ``fullgraph=True`` compile on every supported
        # torch release (2.2+), whereas attribute access on
        # ``torch.compiler.is_compiling`` raises InternalTorchDynamoError on
        # torch 2.2 (the attribute exists only from 2.3, and touching it in a
        # traced function aborts the compile). Reordering the probes keeps the
        # fullgraph compile on 2.2 from failing on the guard itself.
        import torch._dynamo as _dm
        if callable(_dm.is_compiling):
            return bool(_dm.is_compiling())
    except Exception:
        pass
    try:
        return torch.compiler.is_compiling()  # type: ignore[attr-defined]
    except Exception:
        return False


def _warn_if_nonfinite(o, fn_name, stacklevel=3):
    """Surface non-finite KDA outputs with an actionable hint."""
    if not _KDA_NONFINITE_WARN:
        return
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
    # GVA expansion. With G == 1 (H == HV, the default geometry) the
    # repeat_interleave is an identity copy, so skip it entirely — values
    # are unchanged and one full q / k copy per call is avoided.
    if G == 1:
        q = q * scale
    else:
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
    # Preallocate the output and write each step's result into its slice —
    # avoids the extra full-sequence copy that ``torch.stack`` performs.
    o = q.new_empty(B, T, HV, V)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i,
                             v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
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
    recompute_attn: bool = False,
):
    """Shared setup for the chunk-parallel KDA path.

    ``recompute_attn=True`` selects the low-memory *recompute* policy: the
    q-weighted intra-chunk attention matrices are NOT cached for the whole
    sequence here; :func:`_chunk_kda_inner_loop_recompute` rebuilds each
    chunk's matrix from ``q`` / ``k`` / the cumsum gates inside the
    sequential loop (see that function's docstring). The math is unchanged.
    """
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
    # Note: ``repeat_interleave`` (even with G == 1) is kept because it also
    # materializes contiguous q / k tensors; the rearranged views feed several
    # downstream matmuls that regress when handed non-contiguous operands.
    q = q.repeat_interleave(G, dim=1) * scale
    k = k.repeat_interleave(G, dim=1)
    if g_clamp_min > -float('inf'):
        g = g.clamp(min=float(g_clamp_min))
    g = g.cumsum(-2)

    # NOTE: ``diagonal=0`` (NOT ``diagonal=1``) is required here to match the
    # FLA-referenced chunkwise-parallel KDA math (fla/ops/kda/naive.py). The
    # mask covers the main diagonal AND the strict upper triangle; the
    # subsequent ``A.masked_fill(mask, 0)`` then leaves ``A`` STRICTLY lower
    # triangular (diagonal zeroed). With ``diagonal=1`` the diagonal entries
    # ``A[i, i] = beta[i] * (k_i·k_i) * exp(0) = beta[i]`` (k is L2-normalised)
    # survive into ``eye - A``, which decouples the triangular solve from the
    # correct recurrent delta-rule and makes the chunk output/state diverge
    # from ``naive_recurrent_kda`` by O(1e-2) — empirically confirmed:
    # diagonal=1 gives max|o_diff|≈0.019, diagonal=0 matches to fp32 round-off
    # (~1e-9). Examining the git history shows earlier review rounds had
    # documented this same point and switched to diagonal=0; a later LLM
    # bug-fix round reverted it to diagonal=1 and reintroduced the regression. Do not revert.
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    upper_mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    # Every column is written exactly once below, so ``empty`` avoids the
    # redundant zero-fill of the (potentially large) A tensors.
    A = torch.empty(*g.shape[:-1], BT, dtype=compute_dtype, device=q.device)
    # ``Aqk`` (the q-weighted strictly-attention matrix consumed by the inner
    # loop) shares the ``exp(clamp(g_j - g_i))`` pair grid with ``A``: the
    # grid is materialized once here and folded into both column updates,
    # instead of being recomputed inside the inner loop for every chunk. The
    # per-element product order ``(x * exp) * k_i`` and the contiguous
    # last-dim reduction are unchanged, so both outputs are bit-identical to
    # the previous two-pass computation.
    #
    # In ``recompute_attn`` mode this cache is intentionally DROPPED: the
    # persistent ``Aqk`` tensor is O(NT * BT * BT) = O(T * BT) elements per
    # head (e.g. ~33 MB for B=1, HV=8, T=8192, BT=64 in fp32, growing
    # linearly in T). The inner loop then rebuilds each chunk's matrix from
    # the already-returned ``q`` / ``k`` / cumsum ``g`` — same flops, but
    # peak extra memory O(BT * BT) per head. This is the "recompute versus
    # cache" dial for long-context prefill / inference.
    Aqk = None if recompute_attn else torch.empty(
        *g.shape[:-1], BT, dtype=compute_dtype, device=q.device)
    # ``clamp(max=50.0)`` below is provably an identity when the largest
    # positive pair difference ``max(g) - min(g)`` never exceeds 50: the
    # clamp then returns its input bit-for-bit, and its backward (an
    # elementwise ``lt`` + ``where`` per column) can be skipped as well.
    # NaN / +inf ranges fail this ``<=`` test (comparisons are False), so
    # degenerate inputs keep the full clamp path. The recompute inner loop
    # receives ``clamp_g_diff`` so it makes the exact same clamp decision.
    clamp_g_diff = not bool((g.amax(-2) - g.amin(-2)).max() <= 50.0)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i, :]
        g_diff = g - g_i.unsqueeze(-2)
        if clamp_g_diff:
            g_diff = g_diff.clamp(max=50.0)
        e = g_diff.exp()
        A[..., i] = (k * e * k_i.unsqueeze(-2)).sum(-1)
        if not recompute_attn:
            Aqk[..., i] = (q * e * k_i.unsqueeze(-2)).sum(-1)
    if not recompute_attn:
        Aqk = Aqk.masked_fill(upper_mask, 0)
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

    return {
        'early': False,
        'q': q, 'k': k, 'u': u, 'g': g, 'w': w, 'S': S,
        'Aqk': Aqk, 'NT': NT,
        # Needed by ``_chunk_kda_inner_loop_recompute`` when ``Aqk`` is None.
        'upper_mask': upper_mask,
        'clamp_g_diff': clamp_g_diff,
        'recompute_attn': recompute_attn,
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
    recompute_attn: bool = False,
):
    """Chunkwise-parallel KDA (reference). Matches ``naive_recurrent_kda`` up to fp error.

    Args:
        recompute_attn: when False (default) the q-weighted intra-chunk
            attention matrices are cached for the whole sequence in
            ``_chunk_kda_prepare`` (faster: the ``exp(g_j - g_i)`` pair grid
            is shared with the ``A`` matrix). When True the cache is
            dropped and each chunk's matrix is recomputed inside the
            sequential loop, trading extra pointwise work for a peak
            memory reduction from ``O(T * chunk_size)`` to
            ``O(chunk_size ** 2)`` extra elements per head. Both settings
            perform the same per-element operations in the same order and
            are bit-identical on the tested CPU geometries (locked by
            ``test_recompute_attn.py``).
    """
    prep = _chunk_kda_prepare(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        chunk_size, g_clamp_min, state_dtype, fn_name='naive_chunk_kda',
        recompute_attn=recompute_attn,
    )
    if prep['early']:
        return prep['o'], prep['S']
    if recompute_attn:
        chunk_outs, S = _chunk_kda_inner_loop_recompute(
            prep['q'], prep['k'], prep['u'], prep['g'], prep['w'], prep['S'],
            prep['upper_mask'], prep['clamp_g_diff'], prep['NT'],
        )
    else:
        chunk_outs, S = _chunk_kda_inner_loop(
            prep['q'], prep['k'], prep['u'], prep['g'], prep['w'], prep['S'],
            prep['Aqk'], prep['NT'],
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
    Aqk: torch.Tensor,
    NT: int,
) -> tuple:
    """TorchScript-compatible inner loop of ``naive_chunk_kda``.

    ``Aqk`` (the masked q-weighted within-chunk attention matrix, precomputed
    by :func:`_chunk_kda_prepare` from the shared ``exp(clamp(g_j - g_i))``
    pair grid) is sliced per chunk below.
    """
    chunk_outs = []
    for i in range(NT):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i = u[:, :, i]
        g_i = g[:, :, i]
        w_i = w[:, :, i]
        v_i = u_i - w_i @ S
        chunk_outs.append((q_i * g_i.exp()) @ S + Aqk[:, :, i] @ v_i)
        S = S * g_i[:, :, -1].exp().unsqueeze(-1)
        # NOTE: decay direction is ``g_last - g_pos`` (NOT ``g_pos - g_last``).
        # Inside a chunk, position ``c`` contributes to the carried state via
        # `exp(g_cumsum[last] - g_cumsum[c]) * k_c ⊗ v_c`, where ``g_cumsum``
        # is the running sum of the per-channel log-decay. The recurrence multiplies the
        # state by ``exp(g_t)`` AFTER step ``t``, so a token written at
        # position ``c`` is decayed by the SUM of gates from ``c+1`` through
        # the chunk's last position, i.e. by ``exp(g_cumsum[last] -
        # g_cumsum[c])``. The old code used ``exp(g_cumsum[c] -
        # g_cumsum[last])`` (the reciprocal), making the contribution from
        # later positions OVER-emphasised and earlier positions sharply
        # under-emphasised within each chunk — diverging from
        # ``naive_recurrent_kda`` by O(1e-2) in state and output. The sign
        # matches the upstream FLA reference (fla/ops/kda/naive.py).
        update = torch.einsum('bhck,bhcv->bhckv', (g_i[:, :, -1:] - g_i).exp() * k_i, v_i)
        S = S + update.sum(-3)
    return chunk_outs, S


def _chunk_kda_inner_loop_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    w: torch.Tensor,
    S: torch.Tensor,
    upper_mask: torch.Tensor,
    clamp_g_diff: bool,
    NT: int,
) -> tuple:
    """TorchScript-compatible recompute variant of ``_chunk_kda_inner_loop``.

    The default chunk path caches every chunk's q-weighted attention matrix
    ``Aqk`` in ``_chunk_kda_prepare`` (shape ``[B, HV, NT, BT, BT]`` —
    O(T * BT) extra elements per head). This variant rebuilds ``Aqk`` for
    the current chunk only, from the already-materialized ``q`` / ``k`` and
    the cumsum gates ``g``.

    The column update is deliberately kept textually aligned with
    ``_chunk_kda_prepare``'s loop body (``diff = g - g_col``,
    ``(q * exp(diff)) * k_col`` reduced over the contiguous last dim), so
    the two policies perform the *same* per-element operations in the same
    order — the recompute output is bit-identical to the cache output for
    the tested geometries (verified by ``test_recompute_attn.py``). The
    per-column ``[B, HV, BT, K]`` slice also keeps the transient footprint
    at O(BT * K) instead of materializing the full O(BT * BT * K) pair
    grid, which is why this form is used rather than a single broadcast
    expression.

    ``clamp_g_diff`` must be the flag computed by ``_chunk_kda_prepare``
    (``max(g) - min(g) > 50`` anywhere), so both policies clamp identically.
    """
    chunk_outs = []
    B, HV, BT = q.shape[0], q.shape[1], q.shape[3]
    for i in range(NT):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i = u[:, :, i]
        g_i = g[:, :, i]
        w_i = w[:, :, i]
        Aqk_i = q_i.new_empty(B, HV, BT, BT)
        for j in range(BT):
            diff = g_i - g_i[..., j:j + 1, :]
            if clamp_g_diff:
                diff = diff.clamp(max=50.0)
            Aqk_i[..., j] = (q_i * diff.exp() * k_i[..., j:j + 1, :]).sum(-1)
        Aqk_i = Aqk_i.masked_fill(upper_mask, 0)
        v_i = u_i - w_i @ S
        chunk_outs.append((q_i * g_i.exp()) @ S + Aqk_i @ v_i)
        S = S * g_i[:, :, -1].exp().unsqueeze(-1)
        update = torch.einsum('bhck,bhcv->bhckv', (g_i[:, :, -1:] - g_i).exp() * k_i, v_i)
        S = S + update.sum(-3)
    return chunk_outs, S


def _assoc_scan_chunk_states(
    M: torch.Tensor,
    b: torch.Tensor,
    S0: torch.Tensor,
) -> tuple:
    """TorchScript-compatible associative (prefix) scan of affine chunk maps.

    Each chunk ``n`` carries an affine transition on the recurrent state::

        S_{n+1} = M_n S_n + b_n

    with ``M_n`` of shape ``[B, HV, K, K]`` and ``b_n`` of shape
    ``[B, HV, K, V]``. Affine maps compose associatively::

        (M_2, b_2) ∘ (M_1, b_1) = (M_2 M_1, M_2 b_1 + b_2)

    so the per-chunk *incoming* states can be produced by a prefix scan
    whose combine is a ``K×K`` matmul plus a ``K×V`` matmul, instead of the
    sequential ``for n in range(NT)`` loop used by
    :func:`_chunk_kda_inner_loop`. This implementation uses a Hillis–Steele
    (doubling) scan: ``ceil(log2(NT))`` levels, each level one vectorized
    ``matmul`` over ``[B, HV, NT, ...]`` — so the Python-level sequential
    depth grows logarithmically in the sequence length and every level is a
    single batched kernel, which is what GPU utilization wants. (The
    reference CPU path deliberately keeps the sequential loop as the
    default; see :func:`assoc_scan_chunk_kda`'s docstring for the trade.)

    Returns:
        ``(S_in, S_final, levels)`` where ``S_in`` has shape
        ``[B, HV, NT, K, V]`` (the state entering each chunk, i.e.
        ``S_in[:, :, 0] == S0``), ``S_final`` is the state after the last
        chunk, and ``levels`` is the number of doubling steps performed
        (``ceil(log2(NT))``, 0 for ``NT == 1``) — returned so regression
        tests can lock the logarithmic depth contract.
    """
    B, HV, NT, K = M.shape[0], M.shape[1], M.shape[2], M.shape[3]
    V = b.shape[-1]
    # Inclusive prefix scan: E[:, :, n] / F[:, :, n] describe the composed
    # map of chunks 0..n (applied to S0).
    E = M
    Fv = b
    off = 1
    levels = 0
    while off < NT:
        levels = levels + 1
        E_shift = E[:, :, :-off]
        F_shift = Fv[:, :, :-off]
        E_tail = E[:, :, off:]
        F_tail = Fv[:, :, off:]
        # NOTE: compute the composed source from the OLD E_tail (chunk n's
        # own map), then overwrite. Reusing one variable for both would
        # apply the newly-composed transition to the source too.
        new_E_tail = E_tail @ E_shift
        new_F_tail = E_tail @ F_shift + F_tail
        E = torch.cat([E[:, :, :off], new_E_tail], dim=2)
        Fv = torch.cat([Fv[:, :, :off], new_F_tail], dim=2)
        off = off * 2
    # Exclusive form: S_in[:, :, 0] = S0 (identity map, zero source), while
    # chunk n >= 1 consumes the inclusive prefix of chunks 0..n-1.
    eye = torch.eye(K, dtype=M.dtype, device=M.device)
    eye = eye.expand(B, HV, 1, K, K)
    zero = torch.zeros(B, HV, 1, K, V, dtype=M.dtype, device=M.device)
    E_pre = torch.cat([eye, E[:, :, :-1]], dim=2)
    F_pre = torch.cat([zero, Fv[:, :, :-1]], dim=2)
    S_in = E_pre @ S0.unsqueeze(2) + F_pre
    # NOTE: keep both operands 4-D here. ``E[:, :, -1] @ S0.unsqueeze(2)``
    # would broadcast the batch dims asymmetrically ((B,HV) vs (B,HV,1))
    # and silently pair each head's matrix with the wrong initial state;
    # with S0 == 0 (the common case) the error is invisible.
    S_final = E[:, :, -1] @ S0 + Fv[:, :, -1]
    return S_in, S_final, levels


def assoc_scan_chunk_kda(
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
    """Chunkwise KDA via an **associative scan over chunks** (reference).

    Same math as :func:`naive_chunk_kda`, reorganized so that the state
    recurrence is expressed as per-chunk *affine maps* and resolved by a
    parallel prefix (associative) scan instead of the sequential inner
    loop. For chunk ``n`` let ``gexp = exp(g_cumsum)`` and
    ``kexp = k * exp(g_cumsum[last] - g_cumsum)`` (the per-channel log-decay
    factors to the chunk end). The delta-rule/wY transform gives::

        S_{n+1} = M_n S_n + b_n,     M_n = Diag(gexp[last]) - kexp^T w_n
                                     b_n = kexp^T u_n
        o_n     = C_n S_n + a_n,     C_n = q * gexp - Aqk_n w_n
                                     a_n = Aqk_n u_n

    where ``w_n`` / ``u_n`` / ``Aqk_n`` come from the standard triangular
    solve in :func:`_chunk_kda_prepare`, exactly as in the sequential path.
    The maps ``(M_n, b_n)`` compose associatively, so
    :func:`_assoc_scan_chunk_states` recovers all incoming chunk states in
    ``ceil(log2 NT)`` batched levels (Hillis–Steele) — no Python loop over
    the sequence length. This is the *parallel-depth* complement of the
    sequential reference loop: numerically it agrees with
    ``naive_chunk_kda`` to fp round-off (~1e-8 fp32, ~1e-17 fp64 on the
    locked test geometries) but is NOT bit-exact, because the composition
    order of the chunk PREfix differs from the running multiplication in
    the loop.

    Memory: the scan materializes the per-chunk operators ``[B, HV, NT, K,
    K]`` (+ two such levels during the doubling), on top of the cached
    ``Aqk``/``A`` from ``_chunk_kda_prepare``. It is therefore NOT the
    low-memory option — that is ``naive_chunk_kda(recompute_attn=True)``.
    ``recompute_attn`` is incompatible here because the readout terms
    ``C_n`` / ``a_n`` are computed from the full ``Aqk`` tensor in one
    vectorized shot.

    The scan core is TorchScript-compatible; the surrounding wrapper stays
    eager. The output is assembled by :func:`_chunk_kda_finalize`, so
    padding / dtype / final-state handling is shared with the sequential
    paths.

    Args:
        Same as :func:`naive_chunk_kda`.
    """
    prep = _chunk_kda_prepare(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        chunk_size, g_clamp_min, state_dtype, fn_name='assoc_scan_chunk_kda',
        recompute_attn=False,
    )
    if prep['early']:
        return prep['o'], prep['S']
    qq, kk = prep['q'], prep['k']
    u, gg, w = prep['u'], prep['g'], prep['w']
    Aqk, S0 = prep['Aqk'], prep['S']
    if Aqk is None:  # pragma: no cover - prepare(recompute_attn=False)
        raise RuntimeError(
            'assoc_scan_chunk_kda requires the cached Aqk matrix; '
            'recompute_attn=True is not supported by the scan path.')
    # kexp[c] = k_c * exp(g_last - g_c) with PER-CHANNEL log-decay, so the
    # diagonal part of M_n is Diag(exp(g_last)) (anisotropic across K).
    kexp = kk * (gg[..., -1:, :] - gg).exp()
    kexp_t = kexp.transpose(-1, -2)
    M = torch.diag_embed(gg[..., -1, :].exp()) - kexp_t @ w
    b = kexp_t @ u
    S_in, S_final, _levels = _assoc_scan_chunk_states(M, b, S0)
    C = qq * gg.exp() - Aqk @ w
    a = Aqk @ u
    o_all = C @ S_in + a
    chunk_outs = list(o_all.unbind(2))
    return _chunk_kda_finalize(
        chunk_outs, S_final,
        dtype=prep['dtype'], compute_dtype=prep['compute_dtype'],
        state_dtype=state_dtype, output_final_state=output_final_state,
        original_T=prep['original_T'], fn_name='assoc_scan_chunk_kda',
    )


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
    recompute_attn: bool = False,
):
    """``naive_chunk_kda`` with a ``torch.jit.script``-compiled inner loop.

    ``recompute_attn`` is forwarded to :func:`naive_chunk_kda`; the
    recompute inner loop is TorchScript-compiled too (with the same
    graceful fallback to eager).
    """
    if not use_script:
        return naive_chunk_kda(
            q, k, v, g, beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            g_clamp_min=g_clamp_min,
            state_dtype=state_dtype,
            recompute_attn=recompute_attn,
        )

    prep = _chunk_kda_prepare(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        chunk_size, g_clamp_min, state_dtype, fn_name='scripted_chunk_kda',
        recompute_attn=recompute_attn,
    )
    if prep['early']:
        return prep['o'], prep['S']

    inner_eager = (_chunk_kda_inner_loop_recompute if recompute_attn
                   else _chunk_kda_inner_loop)
    inner_args = (
        (prep['q'], prep['k'], prep['u'], prep['g'], prep['w'], prep['S'],
         prep['upper_mask'], prep['clamp_g_diff'], prep['NT'])
        if recompute_attn else
        (prep['q'], prep['k'], prep['u'], prep['g'], prep['w'], prep['S'],
         prep['Aqk'], prep['NT'])
    )
    try:
        scripted_inner = torch.jit.script(inner_eager)
        chunk_outs, S = scripted_inner(*inner_args)
    except Exception as exc:
        _warnings.warn(
            f"scripted_chunk_kda: torch.jit.script failed ({type(exc).__name__}: "
            f"{exc}); falling back to the eager inner loop.",
            stacklevel=2,
        )
        chunk_outs, S = inner_eager(*inner_args)

    return _chunk_kda_finalize(
        chunk_outs, S,
        dtype=prep['dtype'], compute_dtype=prep['compute_dtype'],
        state_dtype=state_dtype, output_final_state=output_final_state,
        original_T=prep['original_T'], fn_name='scripted_chunk_kda',
    )
