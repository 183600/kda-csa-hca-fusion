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

import math as _math
import os as _os
import sys
import warnings as _warnings
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from ops_cache import normal_tensor_scope

# Non-finite-output guard. The check is a full-output reduction, so
# latency-bound callers (e.g. per-token decoding loops) can opt out via
# ``KDA_NONFINITE_WARN=0`` (or by setting this module flag to False)
# without losing the default warning contract.
_KDA_NONFINITE_WARN = _os.environ.get(
    'KDA_NONFINITE_WARN', '1').strip().lower() not in ('0', 'false', 'off', 'no')


def _is_finite_real(value) -> bool:
    """True for a non-bool real representable as a finite float.

    ``float(10**400)`` raises OverflowError; a JSON-sized integer is a
    malformed configuration value, not a crash.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return _math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def per_channel_log_decay_bias(
    n_heads: int,
    head_dim: int,
    decay_scale: float,
    *,
    min_half_life: float = 1.0,
    max_half_life: float = 1024.0,
) -> torch.Tensor:
    """Build a zero-parameter RWKV-style multi-timescale KDA gate bias.

    KDA's log-decay gate is ``g = -softplus(u) * decay_scale``. At
    initialization ``u`` is zero-mean, so every channel starts with the
    same expected decay (``softplus(0) * 0.1 ~= 0.069``). RWKV-7 instead
    initializes a deterministic per-channel decay spectrum spanning fast
    and slow channels. This helper returns an additive bias for ``u``
    that places channel ``c``'s expected per-step log-decay at

    .. math::

        \\lambda_c = \\ln(2) / T_{1/2,c},

    where ``T_{1/2,c}`` is log-spaced between ``min_half_life`` and
    ``max_half_life`` tokens. Because ``u`` is offset by
    ``softplus^{-1}(\\lambda_c / decay_scale)``, the expected decay is
    ``exp(-\\lambda_c)``.

    The spectrum is identical for every value head and is returned as a
    flat ``[n_heads * head_dim]`` tensor (the gate's channel order). It is
    intended as a non-persistent initialization buffer, not a parameter:
    the low-rank gate remains fully trainable, and the default (zero)
    bias reproduces the historical KDA initialization exactly.

    Args:
        n_heads: number of value heads (``HV``).
        head_dim: key width ``K``.
        decay_scale: the KDA ``kda_decay_scale`` multiplier.
        min_half_life: half-life (tokens) of the fastest channel; must be
            finite and positive.
        max_half_life: half-life (tokens) of the slowest channel; must be
            finite, positive, and at least ``min_half_life``.

    Returns:
        ``torch.Tensor`` of shape ``[n_heads * head_dim]`` in float32.
    """
    for name, value in (('n_heads', n_heads), ('head_dim', head_dim)):
        if (not isinstance(value, int) or isinstance(value, bool)
                or value < 1):
            raise ValueError(
                f"per_channel_log_decay_bias: {name}={value!r} must be a "
                f"positive int.")
    if (not isinstance(decay_scale, (int, float))
            or isinstance(decay_scale, bool)
            or not _is_finite_real(decay_scale)
            or decay_scale < 0):
        raise ValueError(
            f"per_channel_log_decay_bias: decay_scale={decay_scale!r} must "
            f"be a finite non-negative number.")
    for name, value in (('min_half_life', min_half_life),
                        ('max_half_life', max_half_life)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not _is_finite_real(value)
                or value <= 0):
            raise ValueError(
                f"per_channel_log_decay_bias: {name}={value!r} must be a "
                f"finite positive number.")
    if min_half_life > max_half_life:
        raise ValueError(
            f"per_channel_log_decay_bias: min_half_life={min_half_life} "
            f"must be <= max_half_life={max_half_life}.")

    bias = torch.zeros(n_heads * head_dim, dtype=torch.float32)
    if decay_scale == 0:
        return bias
    if head_dim == 1:
        fractions = torch.zeros(1, dtype=torch.float64)
    else:
        fractions = torch.arange(head_dim, dtype=torch.float64) / (head_dim - 1)
    log_span = torch.log(torch.tensor(
        float(max_half_life) / float(min_half_life), dtype=torch.float64))
    half_lives = float(min_half_life) * torch.exp(fractions * log_span)
    target = (torch.log(torch.tensor(2.0, dtype=torch.float64))
              / half_lives / float(decay_scale))
    inverse = target + torch.log(-torch.expm1(-target))
    channel_bias = inverse.to(torch.float32)
    return channel_bias.repeat(n_heads)


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
    # q and k share the query/key head count and key width; letting torch
    # broadcast a mismatched k would silently change the recurrence
    # geometry (or fail later with an opaque matmul error).
    if k.shape != q.shape:
        raise ValueError(
            f"{fn_name}: k.shape={tuple(k.shape)} must equal "
            f"q.shape={tuple(q.shape)} ([B, T, H, K]).")
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


def _validate_kda_shapes(q, k, v, g, beta, fn_name, chunk_size=None,
                         initial_state=None):
    """Validate KDA shape dimensions.

    Args:
        chunk_size: if not None, also validates chunk_size >= 1 (chunk path
            only).
        initial_state: optional ``[B, HV, K, V]`` recurrent state. Exact
            shape is required: a broadcastable slice would silently use a
            smaller state than the caller intended.
    """
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    if H < 1:
        raise ValueError(
            f"{fn_name}: H={H} must be >= 1 "
            f"(would cause ZeroDivisionError in HV // H)")
    if HV < 1:
        raise ValueError(f"{fn_name}: HV={HV} must be >= 1")
    if K < 1:
        raise ValueError(f"{fn_name}: K={K} must be >= 1")
    if V < 1:
        raise ValueError(f"{fn_name}: V={V} must be >= 1")
    if chunk_size is not None:
        # An exact int is required: ``True`` would silently mean 1, a
        # float/string reaches ``T // chunk_size`` / ``F.pad`` and fails
        # later with an opaque TypeError instead of naming the parameter.
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError(
                f"{fn_name}: chunk_size={chunk_size!r} must be an int, got "
                f"{type(chunk_size).__name__}.")
        if chunk_size < 1:
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
    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError(
                f"{fn_name}: initial_state must be a torch.Tensor or None, "
                f"got {type(initial_state).__name__}.")
        expected = (B, HV, K, V)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                f"{fn_name}: initial_state.shape="
                f"{tuple(initial_state.shape)} does not match the expected "
                f"{expected} ([B, HV, K, V]).")


def _validate_g_clamp_min(value, fn_name):
    """Validate the optional lower bound on the log-decay gate.

    A real number or the ``-inf`` sentinel (clamp disabled) is accepted.
    ``+inf`` is not a meaningful lower bound: ``g.clamp(min=+inf)`` makes
    every gate ``+inf``, so the recurrent state becomes non-finite instead
    of the caller getting an actionable error. NaN is rejected for the same
    reason (its comparisons silently produce unspecified results).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{fn_name}: g_clamp_min={value!r} must be a real number or "
            f"-inf (clamp disabled).")
    # ``value != value`` is the NaN predicate. Unlike ``math.isnan(float(...))``
    # it is a Python comparison on a constant, so a ``torch.compile``
    # fullgraph trace can fold it instead of emitting a non-Tensor ``isnan``
    # call (which Dynamo rejects).
    if value != value:
        raise ValueError(f"{fn_name}: g_clamp_min must not be NaN.")
    if value == float('inf'):
        raise ValueError(
            f"{fn_name}: g_clamp_min=+inf is not a valid lower bound; pass "
            f"a finite value or -inf to disable clamping.")
    try:
        float(value)
    except OverflowError:
        # JSON integers are unbounded; a bound that cannot be represented as
        # a float would surface as OverflowError from ``g.clamp`` later.
        raise ValueError(
            f"{fn_name}: g_clamp_min={value!r} must be representable as a "
            f"finite float (or -inf to disable clamping).") from None


def _validate_scale(value, fn_name):
    """Validate the optional softmax temperature.

    ``None`` selects the documented ``1/sqrt(K)`` default and finite stress
    values are accepted. A JSON-sized int is not representable as a float,
    so it must fail here rather than inside ``q * scale``.
    """
    if value is None or isinstance(value, torch.Tensor):
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{fn_name}: scale={value!r} must be a real number or None.")
    try:
        float(value)
    except OverflowError:
        raise ValueError(
            f"{fn_name}: scale={value!r} must be representable as a finite "
            f"float.") from None


def _compute_dtype(dtype):
    """Pick the compute dtype for KDA math."""
    return torch.float64 if dtype == torch.float64 else torch.float


def _to_contig_compute(x, compute_dtype):
    """Cast to ``compute_dtype`` in one contiguous copy.

    The chunk-prep rearrange views are non-contiguous, so ``.to(dtype)``
    would produce a strided copy that a following ``.contiguous()`` copies a
    second time. Requesting the contiguous memory format lets the cast
    allocate the final layout directly (elementwise cast, same values).
    """
    if x.dtype == compute_dtype:
        return x
    return x.to(compute_dtype, memory_format=torch.contiguous_format)


def _is_compiling_safely() -> bool:
    """Return True if we are inside a torch.compile / torch.export trace."""
    # Only probe ``torch._dynamo`` when it has ALREADY been imported: a
    # live dynamo trace implies the module is present in ``sys.modules``,
    # so the semantics below are unchanged.
    _dm = sys.modules.get('torch._dynamo')
    if _dm is not None:
        try:
            # Probe torch._dynamo FIRST. ``torch._dynamo.is_compiling()`` is
            # tracable inside a ``fullgraph=True`` compile on every supported
            # torch release (2.2+), whereas attribute access on
            # ``torch.compiler.is_compiling`` raises InternalTorchDynamoError on
            # torch 2.2 (the attribute exists only from 2.3, and touching it in a
            # traced function aborts the compile).
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
    if o.numel() == 0:
        return
    # ``aminmax`` propagates NaN and +/-Inf into the two scalars, so testing
    # those scalars is equivalent to ``torch.isfinite(o).all()``.
    _lo, _hi = torch.aminmax(o)
    if not bool(torch.isfinite(_lo) & torch.isfinite(_hi)):
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


def _recurrent_kda_loop_eager(q, k_col, v, g_exp_col, beta_k_col, S):
    """One sequential pass of the KDA recurrence (inner-loop body).

    ``q`` is ``[B, T, HV, 1, K]`` (already scaled, with the singleton query
    dim pre-applied), ``v`` is ``[B, T, HV, V]``, and the three factor
    tensors are ``[B, HV, T, K, 1]`` so that ``unbind`` yields the exact
    ``[B, HV, K, 1]`` broadcast shape the loop consumes.
    """
    outs = []
    for i, (q_i, k_i, v_i, g_exp_i, beta_k_i) in enumerate(zip(
            q.unbind(1), k_col.unbind(2), v.unbind(1),
            g_exp_col.unbind(2), beta_k_col.unbind(2))):
        S = S * g_exp_i
        delta = v_i - (k_i * S).sum(-2)
        # ``beta_k_i`` is ``beta_i * k_i`` (precomputed by the caller); the
        # update is the delta-rule ``S + beta_i ⊗ delta`` in fp association.
        S = torch.addcmul(S, beta_k_i, delta[..., None, :])
        # Keep the singleton head dim until the single post-loop squeeze:
        # append-and-stack as ``[B, H, 1, V]`` then squeeze(-2) is
        # elementwise-identical to squeezing each step before the stack.
        outs.append(q_i @ S)
    return torch.stack(outs, 1).squeeze(-2), S


def _recurrent_kda_loop_inplace_eager(q, k_col, v, g_exp_col, beta_k_col, S):
    """No-grad variant of :func:`_recurrent_kda_loop_eager` updating the
    recurrent state IN PLACE.

    ``S * g_exp`` and ``addcmul(S, beta_k, delta)`` are evaluated as
    ``S.mul_(g_exp)`` and ``S.addcmul_(beta_k, delta)`` — the same
    products/sums in the same order (bit-identical values). The caller
    must pass a freshly allocated state (this function mutates its ``S``
    argument) and autograd must be disabled — an in-place update of a
    graph-tracked state would trip the version counter.
    """
    # Collect the per-step readouts and stack once at the end: the single
    # batched copy replaces T tiny per-step copies (elementwise-identical
    # values; the stack squeezes the singleton head dim only once).
    outs: list[torch.Tensor] = []
    for i, (q_i, k_i, v_i, g_exp_i, beta_k_i) in enumerate(zip(
            q.unbind(1), k_col.unbind(2), v.unbind(1),
            g_exp_col.unbind(2), beta_k_col.unbind(2))):
        S = S.mul_(g_exp_i)
        delta = v_i - (k_i * S).sum(-2)
        # ``beta_k_i`` is ``beta_i * k_i`` (precomputed by the caller); the
        # update is the delta-rule ``S + beta_i ⊗ delta`` in fp association.
        S = S.addcmul_(beta_k_i, delta[..., None, :])
        outs.append(q_i @ S)
    return torch.stack(outs, 1).squeeze(-2), S


_RECURRENT_LOOP_SCRIPTED = None
_RECURRENT_LOOP_SCRIPTED_FAILED = False
_RECURRENT_LOOP_INPLACE_SCRIPTED = None
_RECURRENT_LOOP_INPLACE_SCRIPTED_FAILED = False


def _get_scripted_recurrent_loop():
    """Lazily ``torch.jit.script`` the loop body, with eager fallback.

    The scripted function executes the same operations as
    :func:`_recurrent_kda_loop_eager`. If scripting fails on the installed
    torch build, the eager loop is used.
    """
    global _RECURRENT_LOOP_SCRIPTED, _RECURRENT_LOOP_SCRIPTED_FAILED
    if _RECURRENT_LOOP_SCRIPTED is None and not _RECURRENT_LOOP_SCRIPTED_FAILED:
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore', FutureWarning)
                _RECURRENT_LOOP_SCRIPTED = torch.jit.script(
                    _recurrent_kda_loop_eager)
        except Exception:
            _RECURRENT_LOOP_SCRIPTED_FAILED = True
    return _RECURRENT_LOOP_SCRIPTED


def _get_scripted_recurrent_loop_inplace():
    """Scripted no-grad loop body, with eager fallback (see
    :func:`_recurrent_kda_loop_inplace_eager`)."""
    global _RECURRENT_LOOP_INPLACE_SCRIPTED
    global _RECURRENT_LOOP_INPLACE_SCRIPTED_FAILED
    if (_RECURRENT_LOOP_INPLACE_SCRIPTED is None
            and not _RECURRENT_LOOP_INPLACE_SCRIPTED_FAILED):
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore', FutureWarning)
                _RECURRENT_LOOP_INPLACE_SCRIPTED = torch.jit.script(
                    _recurrent_kda_loop_inplace_eager)
        except Exception:
            _RECURRENT_LOOP_INPLACE_SCRIPTED_FAILED = True
    return _RECURRENT_LOOP_INPLACE_SCRIPTED


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

        This is a **Python-loop reference implementation**. The loop body is
        run through ``torch.jit.script`` when available, with an eager
        fallback that computes the same result.
    """
    dtype = v.dtype
    _validate_g_clamp_min(g_clamp_min, 'naive_recurrent_kda')
    _validate_scale(scale, 'naive_recurrent_kda')
    _validate_kda_inputs(q, k, v, g, beta, fn_name='naive_recurrent_kda')
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    _validate_kda_shapes(q, k, v, g, beta, fn_name='naive_recurrent_kda',
                         initial_state=initial_state)
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    compute_dtype = _compute_dtype(dtype)
    q, k, v, g, beta = map(lambda x: x.to(compute_dtype), [q, k, v, g, beta])
    if g_clamp_min > -float('inf'):
        g = g.clamp(min=float(g_clamp_min))
    # GVA expansion. With G == 1 (H == HV) the repeat_interleave is an
    # identity operation; values are unchanged.
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
    if T == 1:
        # Single-step fast path: the exact same per-step op chain as the
        # scripted loop body (``S.mul_(g_exp)`` / ``v - (k * S).sum(-2)`` /
        # ``addcmul_(beta_k, delta)`` / ``q_i @ S``) with the same operand
        # association, just without materializing the ``[B, HV, 1, K, 1]``
        # factor columns and without the single-element ``unbind``/``stack``
        # round trip — the values are bit-identical. The ``transpose`` +
        # ``contiguous`` reproduces the loop's ``[B, T, HV, V]`` shape and
        # contiguity (the stack inserts the time dim at position 1).
        # ``contiguous()`` mirrors the loop's factor-column layout: the
        # ``(k_i * S).sum(-2)`` reduction then sees the same strides and
        # reduces in the same order, keeping the result bit-identical for
        # non-contiguous inputs too.
        g_exp_i = g.exp()[:, 0].unsqueeze(-1).contiguous()
        k_i = k[:, 0].unsqueeze(-1).contiguous()
        beta_k_i = (beta[:, 0].unsqueeze(-1) * k[:, 0]).unsqueeze(
            -1).contiguous()
        v_i = v[:, 0]
        q_i = q[:, 0].unsqueeze(-2)
        if not torch.is_grad_enabled():
            S = S.mul_(g_exp_i)
            delta = v_i - (k_i * S).sum(-2)
            S = S.addcmul_(beta_k_i, delta[..., None, :])
        else:
            S = S * g_exp_i
            delta = v_i - (k_i * S).sum(-2)
            S = torch.addcmul(S, beta_k_i, delta[..., None, :])
        o = (q_i @ S).transpose(1, 2).contiguous()
        if not output_final_state:
            S = None
        else:
            target_state_dtype = (
                state_dtype if state_dtype is not None else compute_dtype)
            S = S.to(target_state_dtype)
        o = o.to(dtype)
        _warn_if_nonfinite(o, 'naive_recurrent_kda')
        return o, S
    # Per-step factors laid out so ``unbind(2)`` yields the ``[B, H, K, 1]``
    # broadcast shape the loop consumes; ``q`` carries its singleton query
    # dim from the start (a view, so the values are unchanged).
    q_col = q.unsqueeze(-2)
    if not torch.is_grad_enabled():
        # ``exp`` / ``mul`` are per-element maps, so computing them on the
        # permuted ``[B,HV,T,K,1]`` view writes exactly the same values as
        # computing in the ``[B,T,HV,K]`` layout and permuting the result
        # (bit-identical), while the whole factor is materialized once in
        # its final layout instead of twice. ``empty_like(..., contiguous)``
        # guarantees a fresh buffer even when the permuted view happens to
        # be stride-contiguous (e.g. singleton T/HV dims), so the ``out=``
        # writes can never touch the caller's tensors.
        g_m = g.movedim(1, 2).unsqueeze(-1)
        g_exp_col = torch.empty_like(
            g_m, memory_format=torch.contiguous_format)
        torch.exp(g_m, out=g_exp_col)
        k_m = k.movedim(1, 2).unsqueeze(-1)
        # ``k_col`` is only read by the loop, and the loop's reductions
        # must see the same contiguous strides as before.
        k_col = k_m.contiguous()
        beta_m = beta.movedim(1, 2).unsqueeze(-1).unsqueeze(-1)
        beta_k_col = torch.empty_like(
            k_m, memory_format=torch.contiguous_format)
        # ``beta * k``: multiplication is commutative and bit-identical in
        # either operand order.
        torch.mul(k_m, beta_m, out=beta_k_col)
    else:
        g_exp_col = g.exp().movedim(1, 2).unsqueeze(-1).contiguous()
        k_col = k.movedim(1, 2).unsqueeze(-1).contiguous()
        beta_k_col = (beta[..., None] * k).movedim(
            1, 2).unsqueeze(-1).contiguous()
    # Autograd-safe out-of-place loop while a graph is being recorded; the
    # in-place loop (same values) otherwise. ``S`` is always freshly
    # allocated above (``new_zeros`` or the ``+ initial_state`` sum), so
    # mutating it never touches caller tensors.
    if not torch.is_grad_enabled():
        loop_i = _get_scripted_recurrent_loop_inplace()
        fn = (loop_i if loop_i is not None
              else _recurrent_kda_loop_inplace_eager)
        o, S = fn(q_col, k_col, v, g_exp_col, beta_k_col, S)
    else:
        loop = _get_scripted_recurrent_loop()
        if loop is not None:
            o, S = loop(q_col, k_col, v, g_exp_col, beta_k_col, S)
        else:
            o, S = _recurrent_kda_loop_eager(
                q_col, k_col, v, g_exp_col, beta_k_col, S)
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


# Number of query rows whose intra-chunk attention entries are built per
# vectorized step. The per-row elementwise/reduction math is unchanged (each
# ``(c, j)`` entry applies the exact same sub/clamp/exp/multiply/sum chain);
# only the loop granularity changes, which bounds the transient
# ``[..., R, c1, K]`` block. The count is selected per work/key-width regime
# by :func:`_attn_block_rows`.
_ATTN_BLOCK_ROWS_MAX = 8


def _attn_block_rows(B: int, HV: int, NT: int, BT: int, K: int) -> int:
    """Rows per vectorized attention-block step (>= 1, <= the cap).

    The count trades fewer loop iterations (larger ``rows``) against
    keeping the transient ``[..., rows, c1, K]`` block small (smaller
    ``rows``): the per-iteration temporary is proportional to
    ``B * HV * NT * rows * BT * K`` elements.

    There are two bands: a single-sequence band (``B == 1``) and a batched
    band (``B >= 2``). The batched band keeps the pre-existing heuristic,
    keyed on the key width ``K`` and chunk count ``NT``, plus an override
    for the multi-batch/multi-head regime where the transient for
    ``rows > 1`` can exceed the cache: once the total attention-matrix
    footprint ``B * HV * NT * BT * BT`` is large enough, one row per
    iteration is measurably faster despite the extra Python iterations.
    The single-sequence band uses the same ``K``/``NT`` keying but is tuned
    separately for ``B == 1``, where the per-iteration transient budget is
    larger and wider blocks win (it decides the standalone ``kda_chunk``
    rows of ``run_benchmark.py``, which run at ``B == 1``).

    ``rows`` only changes the Python loop granularity, never a value (see
    :func:`_chunk_kda_attn_loop_eager`). The result is capped by
    ``_ATTN_BLOCK_ROWS_MAX`` in the batched band and by ``BT`` in both.
    """
    if B >= 2:
        if HV >= 2:
            a_elems = B * HV * NT * BT * BT
            if a_elems >= 262144:
                if NT == 1 and BT >= 256:
                    # One enormous chunk per (batch, head) pair: the loop
                    # count is tiny, so the wide blocks amortize the
                    # per-iteration overhead instead of thrashing a cache
                    # the other way.
                    rows = 32 if K <= 64 else 16
                    return max(1, min(rows, _ATTN_BLOCK_ROWS_MAX, BT))
                rows = 1
                return max(1, min(rows, _ATTN_BLOCK_ROWS_MAX, BT))
            if a_elems >= 131072 and (NT >= 4 or BT <= 64 or K >= 64):
                return 1
        if K <= 16:
            rows = 8 if NT <= 32 else 4
        elif K <= 32:
            rows = 8 if NT <= 32 else (2 if NT <= 64 else 1)
        elif K <= 64:
            rows = 4 if NT <= 16 else (2 if NT <= 32 else 1)
        elif K <= 128:
            rows = 4 if NT <= 8 else (2 if NT <= 16 else 1)
        else:
            rows = (8 if NT <= 2 else 4 if NT <= 8
                    else 2 if NT <= 32 else 1)
        return max(1, min(rows, _ATTN_BLOCK_ROWS_MAX, BT))
    if K <= 16:
        rows = (32 if NT <= 2 else 16 if NT <= 16
                else 8 if NT <= 32 else 4)
    elif K <= 32:
        rows = (16 if NT <= 4 else 8 if NT <= 32
                else 2 if NT <= 64 else 1)
    elif K <= 64:
        rows = (16 if NT <= 2 else 8 if NT <= 8 else 4 if NT <= 32
                else 2 if NT <= 64 else 1)
    elif K <= 128:
        rows = (16 if NT <= 2 else 8 if NT <= 4 else 4 if NT <= 8
                else 2 if NT <= 16 else 1)
    else:
        rows = (8 if NT <= 2 else 4 if NT <= 8
                else 2 if NT <= 32 else 1)
    return max(1, min(rows, BT))


def _chunk_kda_attn_loop_eager(
    g: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    A: torch.Tensor,
    Aqk: torch.Tensor,
    BT: int,
    clamp_g_diff: bool,
    recompute_attn: bool,
    upper_mask: torch.Tensor,
    block_rows: int = 1,
):
    """Block-wise build of the intra-chunk attention matrices ``A`` / ``Aqk``.

    Computes, for every row ``c`` and column ``j <= c``,
    ``sum_k k[c,k] * exp(g[c,k] - g[j,k]) * k[j,k]`` (and the q-weighted
    variant for ``Aqk``), writing each row's ``[:c+1]`` prefix. Rows are
    processed in blocks of ``block_rows``: entry ``(c, j)`` evaluates the
    identical elementwise sub/clamp/exp/multiply chain followed by a
    reduction over the last dim (the same mathematical contraction) as the
    one-row-at-a-time form, so the values agree up to the reduction
    kernel's floating-point association. Entries with ``j > c`` are
    computed by the block write and then zeroed with ``upper_mask`` (the
    one-row form left them at their ``torch.zeros`` contents, i.e. exactly
    ``+0.0``). The Python-level iteration is optionally TorchScript-compiled
    via :func:`_get_scripted_prep_attn_loop`.
    """
    for c0 in range(0, BT, block_rows):
        c1 = min(BT, c0 + block_rows)
        # ``[..., c0:c1, :].unsqueeze(-2)`` broadcasts identically to
        # ``[..., c, :].unsqueeze(-2)`` for each row ``c`` in the block.
        g_c = g[..., c0:c1, :].unsqueeze(-2)
        g_diff = g_c - g[..., :c1, :].unsqueeze(-3)
        if clamp_g_diff:
            g_diff = g_diff.clamp(max=50.0)
        # ``t = exp(g_diff) * k`` is shared by both intra-chunk matrices.
        # The sub/clamp produce fresh tensors, so the exp may run in place;
        # the product stays out-of-place while a graph is recorded because
        # autograd saves the exp output for the multiply's backward. With
        # autograd disabled the product runs in place on the fresh exp
        # output (identical per-element multiply; one fewer block-sized
        # temporary).
        g_diff.exp_()
        if torch.is_grad_enabled():
            t = g_diff * k[..., :c1, :].unsqueeze(-3)
        else:
            t = g_diff.mul_(k[..., :c1, :].unsqueeze(-3))
        A[..., c0:c1, :c1] = torch.einsum(
            '...rck,...rk->...rc', t, k[..., c0:c1, :])
        if not recompute_attn:
            Aqk[..., c0:c1, :c1] = torch.einsum(
                '...rck,...rk->...rc', t, q[..., c0:c1, :])
    if block_rows > 1:
        # Multi-row blocks write within-block entries with ``j > c`` as well,
        # which must be zeroed. With ``block_rows == 1`` every row writes
        # exactly its ``[:c+1]`` prefix, so the strict upper triangle keeps
        # the ``+0.0`` from the ``torch.zeros`` allocation and only the
        # diagonal (zeroed by the caller) overlaps the writes.
        #
        # The only invalid entries a block writes are its own strict upper
        # triangle, i.e. the diagonals at offsets 1..block_rows-1: for row
        # ``c`` in ``[c0, c1)`` the written columns stop at ``c1 - 1``, so a
        # written ``j > c`` satisfies ``j - c <= c1 - 1 - c0 = rows - 1``,
        # while any diagonal entry outside a block was never written and is
        # already ``+0.0``/untouched (zeroing it is a no-op). ``Aqk`` is
        # untouched in ``recompute_attn`` mode (it is rebuilt per chunk by
        # the inner loop).
        for off in range(1, block_rows):
            A.diagonal(offset=off, dim1=-2, dim2=-1).zero_()
            if not recompute_attn:
                Aqk.diagonal(offset=off, dim1=-2, dim2=-1).zero_()
    return A, Aqk


_PREP_ATTN_LOOP_SCRIPTED = None
_PREP_ATTN_LOOP_SCRIPTED_FAILED = False


def _get_scripted_prep_attn_loop():
    """Lazily ``torch.jit.script`` the ``A`` / ``Aqk`` build loop.

    The loop is shape/dtype-generic; on builds where scripting fails, the
    eager loop is used instead.
    """
    global _PREP_ATTN_LOOP_SCRIPTED, _PREP_ATTN_LOOP_SCRIPTED_FAILED
    if _PREP_ATTN_LOOP_SCRIPTED is None and not _PREP_ATTN_LOOP_SCRIPTED_FAILED:
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore', FutureWarning)
                _PREP_ATTN_LOOP_SCRIPTED = torch.jit.script(
                    _chunk_kda_attn_loop_eager)
        except Exception:
            _PREP_ATTN_LOOP_SCRIPTED_FAILED = True
    return _PREP_ATTN_LOOP_SCRIPTED


# Shape-keyed cache for the two constant ``[BT, BT]`` matrices consumed by
# ``_chunk_kda_prepare``: the identity used by the unitriangular solve
# (``eye - A`` / ``A + eye``) and the strict-upper triangular mask returned
# to the recompute inner loop. Both depend only on (BT, dtype, device) and
# are treated as immutable by every consumer (subtract / add / masked_fill
# read-only).
_CHUNK_CONST_CACHE: dict = {}


def _chunk_constants(BT: int, dtype: torch.dtype, device):
    """Cached ``(eye, upper_mask)`` pair for the chunk-size ``BT``."""
    key = (BT, dtype, str(device))
    entry = _CHUNK_CONST_CACHE.get(key)
    if entry is None:
        with normal_tensor_scope():
            eye = torch.eye(BT, dtype=dtype, device=device)
            upper_mask = torch.triu(
                torch.ones(BT, BT, dtype=torch.bool, device=device),
                diagonal=1)
        if len(_CHUNK_CONST_CACHE) >= 64:
            _CHUNK_CONST_CACHE.clear()
        entry = (eye, upper_mask)
        _CHUNK_CONST_CACHE[key] = entry
    return entry


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

    ``recompute_attn=True`` selects the *recompute* policy: the
    q-weighted intra-chunk attention matrices are NOT cached for the whole
    sequence here; :func:`_chunk_kda_inner_loop_recompute` rebuilds each
    chunk's matrix from ``q`` / ``k`` / the cumsum gates inside the
    sequential loop (see that function's docstring). The math is unchanged.
    """
    dtype = v.dtype
    _validate_g_clamp_min(g_clamp_min, fn_name)
    _validate_scale(scale, fn_name)
    _validate_kda_inputs(q, k, v, g, beta, fn_name=fn_name)
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    _validate_kda_shapes(q, k, v, g, beta, fn_name=fn_name,
                         chunk_size=chunk_size, initial_state=initial_state)
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
    # Clamp the REAL gates before padding. The padded steps carry g=0
    # (an identity decay); clamping zeros up to a positive ``g_clamp_min``
    # would instead make every padded step decay the state by
    # ``exp(g_clamp_min) > 1``, corrupting the returned final state when
    # ``T % chunk_size != 0``. The clamp is out of place so the caller's
    # ``g`` is never mutated.
    if g_clamp_min > -float('inf'):
        g = g.clamp(min=float(g_clamp_min))
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
    q, k = [_to_contig_compute(
        rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT), compute_dtype)
        for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(compute_dtype) for x in [v, g, beta]]
    # The rearranged views feed several downstream matmuls, which need
    # contiguous q / k tensors; with G == 1 the GVA expansion is a pure
    # identity.
    if G == 1:
        q = q.contiguous()
        # Always scale out of place. A non-contiguous input can rearrange to
        # a CONTIGUOUS view of the caller's storage (e.g. a transposed
        # tensor whose original layout matches the chunk-major order), so
        # ``contiguous()`` returning a view does not prove ownership.
        # Testing the input's contiguity is likewise insufficient: the
        # rearranged result may alias regardless of the input strides. The
        # out-of-place product is value-identical and, unlike a storage
        # pointer check, is traceable by ``torch.compile(fullgraph=True)``.
        q = q * scale
        k = k.contiguous()
    else:
        q = q.repeat_interleave(G, dim=1)
        if torch.is_grad_enabled():
            q = q * scale
        else:
            q.mul_(scale)
        k = k.repeat_interleave(G, dim=1)
    g = g.cumsum(-2)

    # NOTE: the main diagonal AND the strict upper triangle of the raw
    # ``A`` must be zeroed (``diagonal=0`` semantics, NOT ``diagonal=1``) to
    # match the FLA-referenced chunkwise-parallel KDA math
    # (fla/ops/kda/naive.py). Because the loop below writes exactly the lower
    # triangle of a zero-initialized ``A``, only the written diagonal needs
    # an explicit zero afterwards; the strict upper triangle is already zero.
    # ``eye`` / ``upper_mask`` are cached per (BT, dtype, device) — both are
    # read-only constants (see :func:`_chunk_constants`).
    eye, upper_mask = _chunk_constants(BT, compute_dtype, q.device)
    # ``A``'s columns are written only for rows ``c >= i`` below; the
    # never-written upper triangle must be finite zeros (not ``empty``
    # garbage), because ``A`` is multiplied by ``beta`` afterwards and
    # autograd's backward through that multiply would turn uninitialized
    # inf/NaN garbage into ``0 * inf = NaN`` gradients for ``beta``.
    A = torch.zeros(*g.shape[:-1], BT, dtype=compute_dtype, device=q.device)
    # ``Aqk`` is the q-weighted within-chunk attention matrix consumed by
    # the inner loop. In ``recompute_attn`` mode the cache is dropped and
    # the inner loop rebuilds each chunk's matrix from the ``q`` / ``k``
    # and cumsum ``g`` already returned above.
    # ``torch.zeros`` (not ``empty``) keeps the never-written strictly-upper
    # triangle at exactly zero; the lower triangle is written below.
    Aqk = None if recompute_attn else torch.zeros(
        *g.shape[:-1], BT, dtype=compute_dtype, device=q.device)
    # ``clamp(max=50.0)`` is ALWAYS applied. The old code first reduced the
    # gate range to a host scalar and skipped the clamp when every pair
    # difference was <= 50; that decision is a data-dependent device sync
    # (``bool(tensor)``) and breaks ``torch.compile(fullgraph=True)``.
    # Applying the clamp unconditionally is value-identical:
    #   * if max(g) - min(g) <= 50, every individual difference is <= 50,
    #     so ``clamp(max=50)`` is the identity for every entry;
    #   * if any difference exceeds 50 (or is NaN/+inf), the old code
    #     clamped all entries anyway.
    # The metadata key stays ``True`` so the eager / TorchScript inner loops
    # and the recompute policy keep making the same (now unconditional)
    # choice without a second range reduction.
    clamp_g_diff = True
    # Only the entries with row ``c >= j`` (column ``j``) are materialized;
    # the strict upper triangle stays at the zeros from the allocation above
    # and the diagonal is explicitly zeroed right after the loop. The loop
    # walks rows rather than columns; entry ``(c, j)`` computes
    # ``sum_k k[c,k] * exp(g[c,k]-g[j,k]) * k[j,k]`` as one contraction.
    # The row loop itself is shared with ``_chunk_kda_attn_loop_eager`` (see
    # :func:`_get_scripted_prep_attn_loop`); the eager fallback runs the same
    # loop. In ``recompute_attn`` mode ``Aqk`` is None, and the loop receives
    # ``A`` as an unused placeholder tensor.
    _attn_loop = _get_scripted_prep_attn_loop()
    _Aqk_arg = A if Aqk is None else Aqk
    _block_rows = _attn_block_rows(*g.shape[:-2], BT, g.shape[-1])
    if _attn_loop is not None:
        A, _ = _attn_loop(g, k, q, A, _Aqk_arg, BT,
                          clamp_g_diff, recompute_attn, upper_mask,
                          _block_rows)
    else:
        A, _ = _chunk_kda_attn_loop_eager(
            g, k, q, A, _Aqk_arg, BT, clamp_g_diff, recompute_attn,
            upper_mask, _block_rows)
    A.mul_(beta[..., None])
    # Only the diagonal needs zeroing: ``A`` was allocated with
    # ``torch.zeros`` and the loop above wrote exactly its lower triangle
    # (including the diagonal), so the strict upper triangle is already
    # +0.0.
    A.diagonal(dim1=-2, dim2=-1).zero_()
    A.neg_()
    # ``eye - A`` is unit lower triangular (the diagonal zeroing above left
    # A's diagonal at 0), so the triangular solve uses
    # ``unitriangular=True``.
    A = torch.linalg.solve_triangular(
        eye - A, A, upper=False, unitriangular=True
    )
    if torch.is_grad_enabled():
        # Autograd keeps the out-of-place form (the two broadcasts are saved
        # graph nodes).
        A = (A + eye) * beta[..., None, :]
    else:
        # Same elementwise add-then-multiply per entry; in place it needs no
        # second [B, HV, NT, BT, BT] temporary.
        A = A.add_(eye).mul_(beta[..., None, :])

    # ``exp(g)`` is consumed here (the ``w`` map) and again by the inner
    # loops (the per-chunk gate map and the last-position decay); compute
    # it once on the shared, already-clamped cumsum gate and hand it to
    # both.
    gexp = g.exp()
    w = A @ (gexp * k)
    u = A @ v

    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state.to(device=S.device, dtype=compute_dtype)

    return {
        'early': False,
        'q': q, 'k': k, 'u': u, 'g': g, 'gexp': gexp, 'w': w, 'S': S,
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
    """Shared tail for the chunk-parallel KDA path.

    ``chunk_outs`` is either the already-assembled ``[B, HV, NT, BT, V]``
    tensor (the no-grad inner loops write straight into a preallocated
    buffer; the associative scan already holds its full result) or the
    historical per-chunk list, which is stacked into that layout.
    """
    o = (chunk_outs if isinstance(chunk_outs, torch.Tensor)
         else torch.stack(chunk_outs, dim=2))
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
            ``_chunk_kda_prepare``. When True the cache is dropped and
            each chunk's matrix is recomputed inside the sequential loop.
            Both settings perform the same per-element operations in the
            same order and produce the same values.
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
            prep['q'], prep['k'], prep['u'], prep['g'], prep['gexp'],
            prep['w'], prep['S'],
            prep['upper_mask'], prep['clamp_g_diff'], prep['NT'],
            _recompute_block_rows(prep['q'].shape[0], prep['q'].shape[1],
                                  prep['NT'], prep['q'].shape[3],
                                  prep['q'].shape[-1]),
        )
    else:
        chunk_outs, S = _chunk_kda_inner_loop(
            prep['q'], prep['k'], prep['u'], prep['g'], prep['gexp'],
            prep['w'], prep['S'],
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
    gexp: torch.Tensor,
    w: torch.Tensor,
    S: torch.Tensor,
    Aqk: torch.Tensor,
    NT: int,
) -> tuple:
    """TorchScript-compatible inner loop of ``naive_chunk_kda``.

    ``Aqk`` (the masked q-weighted within-chunk attention matrix precomputed
    by :func:`_chunk_kda_prepare`) is sliced per chunk below, and
    ``gexp`` is the ``exp`` of the cumsum gate map (also precomputed by
    :func:`_chunk_kda_prepare`, which consumes the same tensor for the
    ``w`` map).

    Returns the assembled ``[B, HV, NT, BT, V]`` output tensor. With
    autograd disabled the loop writes each chunk straight into a
    preallocated buffer; while a graph is being recorded the rows are
    accumulated and stacked at the end (the same values / graph ops as the
    historical list-plus-stack form).
    """
    B, HV, BT = q.shape[0], q.shape[1], q.shape[3]
    V = u.shape[-1]
    o: Optional[torch.Tensor] = None
    chunk_outs = []
    if not torch.is_grad_enabled():
        o = q.new_empty(B, HV, NT, BT, V)
    # The per-chunk gate maps (``exp(g)``, the last-position decay, and the
    # decay-difference weighted keys) are pure elementwise functions of
    # ``g`` / ``k``, evaluated for all chunks up front; each chunk then
    # consumes its own slice.
    gexp_all = gexp
    glast_all = gexp_all[..., -1, :].unsqueeze(-1)
    grad_enabled = torch.is_grad_enabled()
    q_chunks = q.unbind(2)
    u_chunks = u.unbind(2)
    w_chunks = w.unbind(2)
    k_chunks = k.unbind(2)
    gexp_chunks = gexp_all.unbind(2)
    g_chunks = g.unbind(2)
    glast_chunks = glast_all.unbind(2)
    Aqk_chunks = Aqk.unbind(2)
    for i, (q_i, u_i, w_i, k_i, gexp_i, g_i, glast_i, Aqk_i) in enumerate(
            zip(q_chunks, u_chunks, w_chunks, k_chunks, gexp_chunks,
                g_chunks, glast_chunks, Aqk_chunks)):
        # Each chunk of ``u`` is consumed exactly once, so outside autograd
        # the subtraction runs on its slice (one fewer chunk-sized
        # allocation; identical values). While a graph is recorded the
        # out-of-place form is kept.
        if grad_enabled:
            v_i = u_i - w_i @ S
            row = (q_i * gexp_i) @ S + Aqk_i @ v_i
        else:
            v_i = u_i.sub_(w_i @ S)
            row = (q_i * gexp_i) @ S
            row.add_(Aqk_i @ v_i)
        if o is not None:
            o[:, :, i] = row
        else:
            chunk_outs.append(row)
        # NOTE: decay direction is ``g_last - g_pos`` (NOT ``g_pos - g_last``).
        # Inside a chunk, position ``c`` contributes to the carried state via
        # ``exp(g_cumsum[last] - g_cumsum[c]) * k_c ⊗ v_c``, where
        # ``g_cumsum`` is the running sum of the per-channel log-decay: the
        # recurrence multiplies the state by ``exp(g_t)`` AFTER step ``t``,
        # so a token written at position ``c`` is decayed by the sum of the
        # gates from ``c + 1`` through the chunk's last position. The sign
        # matches the upstream FLA reference (fla/ops/kda/naive.py).
        # The contraction over positions computes ``sum_c kexp[c, :] ⊗
        # v[c, :]`` as a matmul.
        #
        # ``kexp`` is built lazily per chunk (same per-element subtraction /
        # exp / product as the former whole-sequence tensor, evaluated on the
        # identical values) so only one ``[B, HV, BT, K]`` temporary is live
        # instead of the full ``[B, HV, NT, BT, K]`` map.
        kexp_i = g_i[..., -1:, :] - g_i
        kexp_i.exp_()
        if grad_enabled:
            # ``exp`` backward saves its output, so the multiply stays out
            # of place while a graph is recorded; same per-element product.
            kexp_i = kexp_i * k_i
        else:
            kexp_i.mul_(k_i)
        if grad_enabled:
            S = S * glast_i + kexp_i.transpose(-1, -2) @ v_i
        else:
            # ``S`` is this call's freshly-allocated running state; the
            # in-place multiply/add spell the identical elementwise chain.
            S = S.mul_(glast_i).add_(
                kexp_i.transpose(-1, -2) @ v_i)
    if o is not None:
        return o, S
    return torch.stack(chunk_outs, dim=2), S


def _recompute_block_rows(B: int, HV: int, NT: int, BT: int, K: int) -> int:
    """Column-block size for the recompute variant's ``Aqk`` rebuild.

    Independent of ``_attn_block_rows`` (which serves the ``prepare``
    loop that materializes BOTH ``A`` and ``Aqk``): the recompute path
    writes a single matrix per chunk and uses wider blocks. The values do
    not depend on the block size at all, only the Python loop granularity
    does.
    """
    return max(1, min(32 if K <= 32 else 16, BT))


def _chunk_kda_inner_loop_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    gexp: torch.Tensor,
    w: torch.Tensor,
    S: torch.Tensor,
    upper_mask: torch.Tensor,
    clamp_g_diff: bool,
    NT: int,
    block_rows: int,
) -> tuple:
    """TorchScript-compatible recompute variant of ``_chunk_kda_inner_loop``.

    The default chunk path caches every chunk's q-weighted attention matrix
    ``Aqk`` in ``_chunk_kda_prepare``. This variant rebuilds ``Aqk`` for
    the current chunk only, from the already-materialized ``q`` / ``k`` and
    the cumsum gates ``g``.

    The column update mirrors ``_chunk_kda_prepare``'s loop body
    (``diff = g - g_col``, ``(q * exp(diff)) * k_col`` reduced over the
    contiguous last dim), so both variants perform the same per-element
    operations and contraction.

    ``clamp_g_diff`` is the flag returned by ``_chunk_kda_prepare``. It is
    now unconditionally ``True`` there (the clamp is an identity when the
    gate range is below 50, and the old range check forced a device sync),
    so both variants keep the identical clamp chain.

    ``block_rows`` sets the column-block granularity of the rebuild
    (see :func:`_recompute_block_rows`): entry ``(c, j)`` still evaluates
    the same elementwise sub / clamp / exp / multiply chain and reduction
    over the last dim (``(q * exp(diff)) * k``); only the Python loop
    granularity differs. Entries with ``j > c`` are zeroed by
    ``upper_mask``.

    Returns the assembled ``[B, HV, NT, BT, V]`` output tensor, with the
    same no-grad preallocated-buffer / grad list-plus-stack split as
    :func:`_chunk_kda_inner_loop`.
    """
    B, HV, BT = q.shape[0], q.shape[1], q.shape[3]
    V = u.shape[-1]
    o: Optional[torch.Tensor] = None
    chunk_outs = []
    Aqk_buf: Optional[torch.Tensor] = None
    if not torch.is_grad_enabled():
        o = q.new_empty(B, HV, NT, BT, V)
        # Reused per-chunk attention-matrix buffer: every entry is rewritten
        # on each iteration (the lower triangle is stored, everything else
        # is masked to exactly zero), so no stale value can survive.
        Aqk_buf = q.new_empty(B, HV, BT, BT)
    # Same gate-map layout as ``_chunk_kda_inner_loop`` (pure elementwise
    # functions of ``g`` / ``k``); ``gexp`` is the ``exp`` map precomputed
    # by ``_chunk_kda_prepare``.
    gexp_all = gexp
    glast_all = gexp_all[..., -1, :].unsqueeze(-1)
    grad_enabled = torch.is_grad_enabled()
    _block_rows = block_rows
    for i in range(NT):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i = u[:, :, i]
        g_i = g[:, :, i]
        w_i = w[:, :, i]
        if Aqk_buf is None:
            Aqk_i = q_i.new_empty(B, HV, BT, BT)
        else:
            Aqk_i = Aqk_buf
        # Only the rows ``c >= j`` survive ``upper_mask``.
        for j0 in range(0, BT, _block_rows):
            j1 = min(BT, j0 + _block_rows)
            diff = (g_i[..., j0:, :].unsqueeze(-2)
                    - g_i[..., j0:j1, :].unsqueeze(-3))
            if clamp_g_diff:
                diff = diff.clamp(max=50.0)
            diff.exp_()
            qe = q_i[..., j0:, :].unsqueeze(-2) * diff
            Aqk_i[..., j0:, j0:j1] = torch.einsum(
                '...rck,...ck->...rc', qe, k_i[..., j0:j1, :])
        Aqk_i.masked_fill_(upper_mask, 0)
        if grad_enabled:
            v_i = u_i - w_i @ S
            row = (q_i * gexp_all[:, :, i]) @ S + Aqk_i @ v_i
        else:
            # Same in-place reuse as ``_chunk_kda_inner_loop`` (each ``u``
            # chunk is consumed once; the add's operands are fresh).
            v_i = u_i.sub_(w_i @ S)
            row = (q_i * gexp_all[:, :, i]) @ S
            row.add_(Aqk_i @ v_i)
        if o is not None:
            o[:, :, i] = row
        else:
            chunk_outs.append(row)
        # State carry mirrors ``_chunk_kda_inner_loop`` (same per-element
        # operations in the same order); ``kexp`` is rebuilt lazily for this
        # chunk (same values as the former whole-sequence map, one chunk-sized
        # temporary instead of the full ``[B, HV, NT, BT, K]`` tensor).
        kexp_i = g_i[..., -1:, :] - g_i
        kexp_i.exp_()
        if grad_enabled:
            kexp_i = kexp_i * k_i
            S = S * glast_all[:, :, i] + kexp_i.transpose(-1, -2) @ v_i
        else:
            kexp_i.mul_(k_i)
            S = S.mul_(glast_all[:, :, i]).add_(
                kexp_i.transpose(-1, -2) @ v_i)
    if o is not None:
        return o, S
    return torch.stack(chunk_outs, dim=2), S


def _assoc_scan_chunk_states(
    M: torch.Tensor,
    b: torch.Tensor,
    S0: torch.Tensor,
    _owns_inputs: bool = False,
) -> tuple:
    """TorchScript-compatible associative (prefix) scan of affine chunk maps.

    Each chunk ``n`` carries an affine transition on the recurrent state::

        S_{n+1} = M_n S_n + b_n

    with ``M_n`` of shape ``[B, HV, K, K]`` and ``b_n`` of shape
    ``[B, HV, K, V]``. Affine maps compose associatively::

        (M_2, b_2) ∘ (M_1, b_1) = (M_2 M_1, M_2 b_1 + b_2)

    so the per-chunk *incoming* states are produced by a prefix scan whose
    combine is a ``K×K`` matmul plus a ``K×V`` matmul. This implementation
    uses a Hillis–Steele (doubling) scan: ``ceil(log2(NT))`` levels, each
    level one batched ``matmul`` over ``[B, HV, NT, ...]``.

    Returns:
        ``(S_in, S_final, levels)`` where ``S_in`` has shape
        ``[B, HV, NT, K, V]`` (the state entering each chunk, i.e.
        ``S_in[:, :, 0] == S0``), ``S_final`` is the state after the last
        chunk, and ``levels`` is the number of doubling steps performed
        (``ceil(log2(NT))``, 0 for ``NT == 1``).
    """
    B, HV, NT, K = M.shape[0], M.shape[1], M.shape[2], M.shape[3]
    V = b.shape[-1]
    # Inclusive prefix scan: E[:, :, n] / F[:, :, n] describe the composed
    # map of chunks 0..n (applied to S0).
    E = M
    Fv = b
    if not torch.is_grad_enabled() and not _owns_inputs:
        # The doubling levels below overwrite the tail in place; the
        # caller's tensors must not be touched (the eager/scripted parity
        # test reuses the same inputs), so work on dedicated copies.
        # A caller that passes freshly-owned tensors it never reads again
        # (``assoc_scan_chunk_kda``: both are fresh matmul outputs) opts
        # into consuming them via ``_owns_inputs``.
        E = M.clone()
        Fv = b.clone()
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
        if torch.is_grad_enabled():
            # Autograd keeps the out-of-place form: the cat produces a
            # fresh tensor per level, so no saved operand is mutated.
            E = torch.cat([E[:, :, :off], new_E_tail], dim=2)
            Fv = torch.cat([Fv[:, :, :off], new_F_tail], dim=2)
        else:
            # Prefix unchanged, tail replaced by the composed values
            # computed above from the OLD tail: identical to the cat.
            E[:, :, off:] = new_E_tail
            Fv[:, :, off:] = new_F_tail
        off = off * 2
    # Exclusive form: S_in[:, :, 0] = S0 (identity map, zero source), while
    # chunk n >= 1 consumes the inclusive prefix of chunks 0..n-1.
    # Row 0's map is the identity by construction, so its state is S0
    # itself; applying the identity matmul to it only re-associates the
    # (exactly-zero) summations, and slicing it off lets the prefix
    # matmul run over the NT-1 composed rows only instead of materializing
    # a broadcast identity row and contracting all NT rows.
    tail = E[:, :, :-1] @ S0.unsqueeze(2) + Fv[:, :, :-1]
    S_in = torch.cat([S0.unsqueeze(2), tail], dim=2)
    # NOTE: keep both operands 4-D here: ``E[:, :, -1] @ S0.unsqueeze(2)``
    # would broadcast the batch dims asymmetrically ((B,HV) vs (B,HV,1)).
    S_final = E[:, :, -1] @ S0 + Fv[:, :, -1]
    return S_in, S_final, levels


def _blocked_chunk_states(
    kexp: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    glast: torch.Tensor,
    S0: torch.Tensor,
    block_size: int,
) -> tuple:
    """Two-phase (block-reduce / block-propagate) prefix scan for KDA.

    A work-efficient alternative to the full Hillis–Steele scan over all
    ``NT`` chunk maps.  The chunk sequence is partitioned into blocks of
    ``block_size``; each block's affine map is composed sequentially
    (the "producer"/reduce phase), the ``NT / block_size`` block maps are
    prefix-scanned with the existing doubling core (the "scanner"), and
    the per-chunk incoming states are then expanded inside each block
    from the DPLR factors (the "consumer"/propagate phase).

    This keeps only ``ceil(NT/block_size)`` dense ``K x K`` maps alive at
    once instead of ``NT`` of them, and performs
    ``NT + (NT/block_size) log2(NT/block_size)`` map compositions instead
    of the full scan's ``NT log2(NT)``.  The per-chunk maps are rebuilt
    from the DPLR factors in the expansion phase rather than cached, which
    trades a small amount of arithmetic for the memory bound.

    The inputs are the prepared KDA factor tensors
    (``kexp``/``w``/``u`` from :func:`_chunk_kda_prepare` and
    ``glast = gexp[..., -1, :]``).  No input is mutated and the function is
    autograd-safe (out-of-place ops plus a final ``stack``).

    Returns:
        ``(S_in, S_final, meta)`` with ``S_in`` ``[B, HV, NT, K, V]``,
        ``S_final`` ``[B, HV, K, V]`` and a JSON-safe ``meta`` dict
        recording the composition counts.  The math is identical to
        :func:`_assoc_scan_chunk_states`; only the association order
        differs.
    """
    B, HV, NT = kexp.shape[:3]
    V = u.shape[-1]
    n_blocks = (NT + block_size - 1) // block_size

    def _chunk_map(n: int) -> tuple:
        # M_n = Diag(glast) - kexp_n^T w_n,  b_n = kexp_n^T u_n.
        m = torch.diag_embed(glast[:, :, n]) - \
            kexp[:, :, n].transpose(-1, -2) @ w[:, :, n]
        src = kexp[:, :, n].transpose(-1, -2) @ u[:, :, n]
        return m, src

    block_maps = []
    block_sources = []
    within_compositions = 0
    for nb in range(n_blocks):
        # A block shorter than ``block_size`` (only the final block can be)
        # simply stops early.  Seeding from the first chunk map avoids one
        # identity composition per block; the remaining chunks compose as
        # M_c @ (M_{c-1} ... M_first).
        n0 = nb * block_size
        mb, fb = _chunk_map(n0)
        for c in range(1, block_size):
            n = n0 + c
            if n >= NT:
                break
            m, src = _chunk_map(n)
            fb = m @ fb + src
            mb = m @ mb
            within_compositions += 1
        block_maps.append(mb)
        block_sources.append(fb)
    MB = torch.stack(block_maps, dim=2)
    FB = torch.stack(block_sources, dim=2)
    block_in, S_final, block_levels = _assoc_scan_chunk_states(MB, FB, S0)

    # Propagate: replay each block from its scanned incoming state.  Chunk
    # maps are rebuilt here (they were not cached above), which bounds the
    # live dense-map set to one block.
    states = []
    for nb in range(n_blocks):
        S = block_in[:, :, nb]
        for c in range(block_size):
            n = nb * block_size + c
            if n >= NT:
                break
            states.append(S)
            m, src = _chunk_map(n)
            S = m @ S + src
    S_in = torch.stack(states, dim=2)

    hillis = 0
    off = 1
    while off < NT:
        hillis += NT - off
        off *= 2
    block_scan = 0
    off = 1
    while off < n_blocks:
        block_scan += n_blocks - off
        off *= 2
    total = within_compositions + block_scan
    meta = {
        'n_chunks': int(NT),
        'block_size': int(block_size),
        'n_blocks': int(n_blocks),
        'block_scan_levels': int(block_levels),
        'within_block_compositions': int(within_compositions),
        'block_scan_compositions': int(block_scan),
        'total_compositions': int(total),
        'hillis_steele_compositions': int(hillis),
        'composition_savings_factor': (
            float(hillis / total) if total else 1.0),
    }
    return S_in, S_final, meta


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
    scan_block_size: int = 0,
):
    """Chunkwise KDA via an **associative scan over chunks** (reference).

    Same math as :func:`naive_chunk_kda`, with the state recurrence
    expressed as per-chunk *affine maps* and resolved by a parallel prefix
    (associative) scan. For chunk ``n`` let ``gexp = exp(g_cumsum)`` and
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
    ``ceil(log2 NT)`` batched levels (Hillis–Steele). Numerically it agrees
    with ``naive_chunk_kda`` to fp round-off; it is NOT bit-exact, because
    the composition order of the chunk prefix differs from the running
    multiplication in the loop.

    This path uses the cached ``Aqk`` matrices from
    :func:`_chunk_kda_prepare`; ``recompute_attn`` is incompatible here
    because the readout terms ``C_n`` / ``a_n`` are computed from the full
    ``Aqk`` tensor.

    The scan core is TorchScript-compatible; the surrounding wrapper stays
    eager. The output is assembled by :func:`_chunk_kda_finalize`, so
    padding / dtype / final-state handling is shared with the sequential
    paths.

    ``scan_block_size`` selects a work-efficient **two-phase block scan**
    (``>= 2``).  Chunks are grouped into blocks; each block's affine map is
    composed sequentially (producer/reduce phase), only the
    ``ceil(NT / b)`` block maps are prefix-scanned, and the per-chunk
    incoming states are expanded inside each block from the DPLR factors
    (consumer/propagate phase).  Relative to ``0`` (full Hillis–Steele)
    this performs ``NT + (NT/b) log2(NT/b)`` map compositions instead of
    ``NT log2(NT)`` and keeps ``ceil(NT/b)`` dense ``K x K`` maps live
    instead of ``NT``.  The result is the same up to fp association.

    Args:
        Same as :func:`naive_chunk_kda`, plus ``scan_block_size`` (0 =
            full scan, >=2 = blocked two-phase scan; 1 is treated as 0).
    """
    if isinstance(scan_block_size, bool) or not isinstance(
            scan_block_size, int) or scan_block_size < 0:
        raise ValueError(
            f"assoc_scan_chunk_kda: scan_block_size={scan_block_size!r} must "
            f"be a non-negative int (0 = full Hillis-Steele scan).")
    prep = _chunk_kda_prepare(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        chunk_size, g_clamp_min, state_dtype, fn_name='assoc_scan_chunk_kda',
        recompute_attn=False,
    )
    if prep['early']:
        return prep['o'], prep['S']
    qq, kk = prep['q'], prep['k']
    u, gg, ggx, w = prep['u'], prep['g'], prep['gexp'], prep['w']
    Aqk, S0 = prep['Aqk'], prep['S']
    if Aqk is None:  # pragma: no cover - prepare(recompute_attn=False)
        raise RuntimeError(
            'assoc_scan_chunk_kda requires the cached Aqk matrix; '
            'recompute_attn=True is not supported by the scan path.')
    # kexp[c] = k_c * exp(g_last - g_c) with PER-CHANNEL log-decay, so the
    # diagonal part of M_n is Diag(exp(g_last)) (anisotropic across K).
    # ``ggx`` is the ``exp`` of the cumsum gate map precomputed by
    # ``_chunk_kda_prepare``; ``ggx[..., -1, :]`` is elementwise-identical
    # to ``gg[..., -1, :].exp()`` and ``ggx`` to ``gg.exp()``.
    # The subtraction is fresh, so its ``exp`` runs in place; the multiply
    # stays out of place only where autograd must keep the ``exp`` output
    # unmodified for its backward.
    kexp = gg[..., -1:, :] - gg
    kexp.exp_()
    if torch.is_grad_enabled():
        kexp = kk * kexp
    else:
        kexp.mul_(kk)
    if scan_block_size >= 2:
        # Two-phase scan: the dense per-chunk maps are never materialised
        # for the whole sequence, and the prefix scan runs on block maps.
        S_in, S_final, _meta = _blocked_chunk_states(
            kexp, w, u, ggx[..., -1, :], S0, scan_block_size)
    else:
        kexp_t = kexp.transpose(-1, -2)
        if torch.is_grad_enabled():
            M = torch.diag_embed(ggx[..., -1, :]) - kexp_t @ w
        else:
            # ``diag(d) - X`` == ``diag(d) + (-X)`` elementwise (identical
            # for signed zeros / NaN too): negate the fresh matmul output
            # and add the diagonal in place instead of materialising
            # ``diag_embed``.
            M = kexp_t @ w
            M.neg_()
            M.diagonal(dim1=-2, dim2=-1).add_(ggx[..., -1, :])
        b = kexp_t @ u
        # ``M`` and ``b`` are fresh matmul outputs with no other readers,
        # so the no-grad scan may consume (overwrite) them instead of
        # cloning.
        S_in, S_final, _levels = _assoc_scan_chunk_states(
            M, b, S0, _owns_inputs=True)
    C = qq * ggx - Aqk @ w
    a = Aqk @ u
    o_all = C @ S_in + a
    return _chunk_kda_finalize(
        o_all, S_final,
        dtype=prep['dtype'], compute_dtype=prep['compute_dtype'],
        state_dtype=state_dtype, output_final_state=output_final_state,
        original_T=prep['original_T'], fn_name='assoc_scan_chunk_kda',
    )


_CHUNK_INNER_SCRIPTED: dict = {}
_CHUNK_INNER_SCRIPT_FAILED: dict = {}


def _get_scripted_chunk_inner(inner_eager, recompute_attn: bool):
    """Lazily ``torch.jit.script`` the chunk-path inner loop.

    The inner loop is one of two module-level functions selected by
    ``recompute_attn``; its compiled form is shape/dtype-generic.
    Returns ``(scripted_inner_or_None, error_str_or_None)``; a cached error
    string records a broken ``torch.jit`` build.
    """
    key = bool(recompute_attn)
    entry = _CHUNK_INNER_SCRIPTED.get(key)
    if entry is not None:
        return entry
    failed = _CHUNK_INNER_SCRIPT_FAILED.get(key)
    if failed is not None:
        return None, failed
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore', FutureWarning)
            fn = torch.jit.script(inner_eager)
        _CHUNK_INNER_SCRIPTED[key] = (fn, None)
        return fn, None
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        _CHUNK_INNER_SCRIPT_FAILED[key] = err
        return None, err


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
        (prep['q'], prep['k'], prep['u'], prep['g'], prep['gexp'],
         prep['w'], prep['S'],
         prep['upper_mask'], prep['clamp_g_diff'], prep['NT'],
         _recompute_block_rows(prep['q'].shape[0], prep['q'].shape[1],
                               prep['NT'], prep['q'].shape[3],
                               prep['q'].shape[-1]))
        if recompute_attn else
        (prep['q'], prep['k'], prep['u'], prep['g'], prep['gexp'],
         prep['w'], prep['S'],
         prep['Aqk'], prep['NT'])
    )
    scripted_inner, script_err = _get_scripted_chunk_inner(
        inner_eager, recompute_attn)
    if scripted_inner is not None:
        chunk_outs, S = scripted_inner(*inner_args)
    else:
        _warnings.warn(
            f"scripted_chunk_kda: torch.jit.script failed ({script_err}); "
            f"falling back to the eager inner loop.",
            stacklevel=2,
        )
        chunk_outs, S = inner_eager(*inner_args)

    return _chunk_kda_finalize(
        chunk_outs, S,
        dtype=prep['dtype'], compute_dtype=prep['compute_dtype'],
        state_dtype=state_dtype, output_final_state=output_final_state,
        original_T=prep['original_T'], fn_name='scripted_chunk_kda',
    )


# ---------------------------------------------------------------------------
# torch.compile wrapper for the chunked KDA path
# ---------------------------------------------------------------------------
_COMPILED_CHUNK_KDA_CACHE: '_collections.OrderedDict' = _collections.OrderedDict()
_COMPILED_CHUNK_CACHE_MAX = 32


def compiled_chunk_kda(
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
    dynamic: bool = False,
    fullgraph: bool = False,
    mode: str | None = None,
):
    """``torch.compile``-wrapped :func:`naive_chunk_kda`.

    The chunked path is the training / prefill workhorse, but its setup step
    (per-chunk decay map, intra-chunk attention matrices, triangular solve)
    is a long elementwise + batched-GEMM chain that eager PyTorch executes
    as many small kernels. Compiling the whole operator lets Inductor fuse
    the per-channel decay arithmetic and the chunk preparation while keeping
    the exact same math. ``recompute_attn=True`` selects the memory-bounded
    policy (one ``Aqk`` chunk instead of the full ``[B, HV, NT, BT, BT]``
    cache); combining it with compile is the intended
    "memory-efficient attention" configuration.

    The compiled callable is cached per (shape, dtype, grad mode, chunk
    policy, compile options) signature, mirroring
    :func:`compiled_recurrent_kda`. ``T == 0`` and invalid inputs delegate
    to :func:`naive_chunk_kda` / its validators directly so the degenerate
    path never pays a compile or hides a shape error.

    ``fullgraph=False`` is the default because the setup code contains
    Python-level branches (padding / GVA / optional state). With the
    branch-free decay clamp in :func:`_chunk_kda_prepare`, ``fullgraph=True``
    is supported for the standard shape regime and is what the regression
    test pins.
    """
    _validate_kda_inputs(q, k, v, g, beta, fn_name='compiled_chunk_kda')
    if q.shape[1] == 0:
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
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    state_key = (None if initial_state is None
                 else (tuple(initial_state.shape), initial_state.dtype))
    cache_key = (
        B, T, H, K, HV, V,
        q.dtype, q.device.type,
        q.requires_grad or k.requires_grad or v.requires_grad
        or g.requires_grad or beta.requires_grad,
        int(chunk_size), bool(recompute_attn),
        bool(dynamic), bool(fullgraph), mode,
        bool(output_final_state), state_key,
        str(state_dtype), repr(scale), repr(g_clamp_min),
    )
    compiled_fn = _COMPILED_CHUNK_KDA_CACHE.get(cache_key)
    if compiled_fn is None:
        if len(_COMPILED_CHUNK_KDA_CACHE) >= _COMPILED_CHUNK_CACHE_MAX:
            _COMPILED_CHUNK_KDA_CACHE.popitem(last=False)

        def _chunk_kernel(
            q_arg, k_arg, v_arg, g_arg, beta_arg,
            scale_val, g_clamp_min_val, initial_state_arg,
        ):
            return naive_chunk_kda(
                q_arg, k_arg, v_arg, g_arg, beta_arg,
                scale=scale_val,
                initial_state=initial_state_arg,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
                g_clamp_min=g_clamp_min_val,
                state_dtype=state_dtype,
                recompute_attn=recompute_attn,
            )

        compile_kwargs = {"dynamic": dynamic, "fullgraph": fullgraph}
        if mode is not None:
            compile_kwargs["mode"] = mode
        compiled_fn = torch.compile(_chunk_kernel, **compile_kwargs)
        _COMPILED_CHUNK_KDA_CACHE[cache_key] = compiled_fn
    else:
        _COMPILED_CHUNK_KDA_CACHE.move_to_end(cache_key)

    return compiled_fn(
        q, k, v, g, beta,
        scale, g_clamp_min,
        initial_state,
    )
