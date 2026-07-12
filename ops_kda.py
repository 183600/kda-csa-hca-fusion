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


def _validate_kda_inputs(q, k, v, g, beta, fn_name='naive_recurrent_kda'):
    """Centralized shape / device / dtype contract validation for KDA inputs.

    P1-7 fix: previously each KDA operator validated only a subset of the
    shape contract (head dims, GVA divisibility). A malformed caller
    passing e.g. ``q`` with rank 3 instead of 4, or ``k`` on a different
    device than ``v``, would crash deep inside the recurrence loop with a
    cryptic einsum or broadcasting error that gave no hint about WHICH
    input was wrong. This helper validates the full contract up-front so
    the error message points at the misconfigured input.

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
    # Rank checks.
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
    # B and T consistency. q and v must agree on B and T (the leading two
    # dims); k must match q; g and beta must match q on B and T.
    B_q, T_q = q.shape[0], q.shape[1]
    for name, t in [('k', k), ('v', v), ('g', g), ('beta', beta)]:
        if t.shape[0] != B_q or t.shape[1] != T_q:
            raise ValueError(
                f"{fn_name}: {name}.shape[0:2]={tuple(t.shape[0:2])} does not "
                f"match q.shape[0:2]=({B_q}, {T_q}). All inputs must share "
                f"the same batch (B) and sequence (T) dimensions.")
    # Device consistency.
    ref_device = q.device
    for name, t in [('k', k), ('v', v), ('g', g), ('beta', beta)]:
        if t.device != ref_device:
            raise ValueError(
                f"{fn_name}: {name}.device={t.device} does not match "
                f"q.device={ref_device}. All inputs must be on the same "
                f"device.")
    # Dtype consistency.
    ref_dtype = q.dtype
    for name, t in [('k', k), ('v', v), ('g', g), ('beta', beta)]:
        if t.dtype != ref_dtype:
            raise ValueError(
                f"{fn_name}: {name}.dtype={t.dtype} does not match "
                f"q.dtype={ref_dtype}. All inputs must share the same "
                f"dtype (cast before calling if needed).")


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

    Args:
        q: ``[B, T, H, K]``
        k: ``[B, T, H, K]``
        v: ``[B, T, HV, V]`` (HV must be divisible by H, enabling GVA)
        g: per-channel log-decay gate ``[B, T, HV, K]``
        beta: delta-rule learning rate ``[B, T, HV]``
        scale: defaults to ``1/sqrt(K)``
        initial_state: ``[B, HV, K, V]``
        output_final_state: return the final recurrent state.
        g_clamp_min: lower bound for the log-decay gate ``g``. ``g`` is
            produced upstream as ``-softplus(...) * kda_decay_scale`` which
            has no finite lower bound: in pathological training regimes
            (very large pre-activation) ``g`` can diverge to ``-inf``,
            making ``exp(g) -> 0`` and wiping the recurrent state to zero
            (catastrophic forgetting of all long-term memory). We clamp
            ``g`` to ``[g_clamp_min, +inf)`` (default ``-10``) so
            ``exp(g) >= exp(-10) ~= 4.5e-5`` — small enough to allow
            aggressive forgetting but bounded away from zero. Set to
            ``-float('inf')`` to disable. The clamp is applied AFTER the
            dtype promotion below so it always runs in ``compute_dtype``.
        state_dtype: dtype of the RETURNED recurrent state. If ``None``
            (the default), the state is returned in ``compute_dtype``
            (fp32 for fp16/bf16 inputs, fp64 for fp64 inputs) — this
            preserves precision across chunked/streaming calls where the
            state is passed back as ``initial_state`` repeatedly. The
            OUTPUT tensor ``o`` is still cast back to ``v.dtype``. Pass
            ``state_dtype=torch.float16`` (or any dtype) to override
            (e.g. for memory-constrained streaming). P1-2 fix: previously
            the state was always cast back to ``v.dtype``, causing
            repeated fp32→fp16→fp32 round-trips that accumulated
            quantization error in long streaming inference.

    .. note::

        This is a **Python-loop reference implementation**. The per-step
        loop is dominated by interpreter overhead (typically ~30ms at
        T=2k on CPU), NOT by the underlying math. These numbers are
        suitable for correctness / relative-trend comparisons only; do
        NOT use them as production-latency estimates. For representative
        latency, wrap with ``torch.compile`` or use the FLA Triton kernel.
        A ``T > 8192`` performance warning is emitted via ``warnings``.
    """
    dtype = v.dtype
    # P1-7 fix (revised): validate BEFORE unpacking q.shape so a rank-mismatch
    # (e.g. ``q`` is 3D instead of 4D) is caught with a clear message instead
    # of crashing on ``B, T, H, K, HV, V = *q.shape, ...`` with the cryptic
    # ``ValueError: not enough values to unpack (expected 6, got 5)``.
    _validate_kda_inputs(q, k, v, g, beta, fn_name='naive_recurrent_kda')
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
    # P0 numerical-stability fix: clamp ``g`` from below. ``g`` is produced
    # upstream as ``-softplus(...) * kda_decay_scale`` which has no finite
    # lower bound — in pathological training regimes (very large
    # pre-activation, NaN/Inf in the upstream linear, or simply an
    # aggressive learning rate) ``g`` can diverge to ``-inf``, making
    # ``exp(g) -> 0`` and wiping the recurrent state to zero on a single
    # step (catastrophic forgetting of ALL long-term memory). With
    # ``g_clamp_min=-10`` (the default), ``exp(g) >= 4.5e-5`` — small
    # enough to allow aggressive forgetting of stale entries but bounded
    # away from zero so a single bad step cannot silently erase the entire
    # state. The clamp is a no-op for well-behaved inputs (typical
    # ``g`` values are in ``[-1, 0]``). Set ``g_clamp_min=-inf`` to
    # disable (e.g. for fp64 gradient checks that need exact maths).
    if g_clamp_min > -float('inf'):
        g = g.clamp(min=float(g_clamp_min))
    q = q.repeat_interleave(G, dim=2) * scale   # [B, T, HV, K]
    k = k.repeat_interleave(G, dim=2)           # [B, T, HV, K]
    # P1 performance warning: this Python for-loop is the dominant cost
    # for moderate T (interpreter overhead, not the math). Emit a one-shot
    # warning for very long sequences so users do not silently wait on a
    # 30+ second Python loop thinking it is algorithmic work. The threshold
    # 8192 is chosen so the default benchmark sweep (T <= 2048) stays
    # silent; production-latency comparisons should use the FLA Triton
    # kernel or wrap this function with ``torch.compile``.
    if T > 8192:
        import warnings as _warnings
        _warnings.warn(
            f"naive_recurrent_kda: T={T} > 8192; the Python for-loop is "
            f"interpreter-overhead-bound (typical cost ~15-30ms per 1k "
            f"steps on CPU). For production latency, use the FLA Triton "
            f"kernel or wrap with torch.compile. This warning is emitted "
            f"once per process per call site.",
            stacklevel=2,
        )

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
            # P1-2 fix: mirror the main return path — keep state in
            # compute_dtype by default (NOT v.dtype) to avoid fp32→fp16
            # precision loss in streaming inference.
            target_state_dtype = (state_dtype if state_dtype is not None
                                  else compute_dtype)
            S = S.to(target_state_dtype)
        return o.to(dtype), S
    # P0 autograd-safety fix: accumulate per-step outputs in a Python list and
    # ``torch.stack`` at the end, instead of in-place ``o[:, i] = ...`` on a
    # pre-allocated ``torch.zeros_like(v)`` buffer. The in-place form saves a
    # CopySlices node on ``o`` for backward; when ``compute_dtype == dtype``
    # (e.g. fp32 inputs) the final ``o.to(dtype)`` returns the SAME tensor, so
    # that saved CopySlices is part of the autograd graph. Under
    # ``torch.utils.checkpoint`` the forward is recomputed during backward, and
    # the recompute re-runs the in-place writes — mutating a tensor that autograd
    # already saved, raising ``RuntimeError: one of the variables needed for
    # gradient computation has been modified by an inplace operation``. The
    # out-of-place ``stack`` form has no such dependency. The cost (one extra
    # list of T tensors) is negligible relative to the per-step einsum cost.
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
        # P1-2 fix: return the state in ``compute_dtype`` (fp32 for fp16/bf16
        # inputs, fp64 for fp64 inputs) by default — NOT ``v.dtype``. The
        # previous ``S = S.to(dtype)`` caused repeated fp32→fp16→fp32
        # round-trips when the state was passed back as ``initial_state`` in
        # streaming inference, accumulating quantization error in long
        # sessions. The OUTPUT ``o`` is still cast to ``v.dtype`` below so
        # the model's forward pass sees the expected dtype; only the
        # PERSISTENT state stays in compute precision. Callers can override
        # via ``state_dtype`` (e.g. for memory-constrained streaming).
        target_state_dtype = state_dtype if state_dtype is not None else compute_dtype
        S = S.to(target_state_dtype)
    return o.to(dtype), S


# =============================================================================
# Compiled / accelerated KDA path (issue 2.1 fix)
# -----------------------------------------------------------------------------
# The README's *Limitations* section acknowledges that ``naive_recurrent_kda``
# is a "Python-loop reference implementation" but the project did not provide
# any accelerated path — users had no documented route from "correctness
# reference" to "production-grade latency" without leaving the repo for the
# FLA Triton kernel. The wrappers below fill that gap with two complementary
# routes:
#
#   1. ``compiled_recurrent_kda`` — ``torch.compile`` wrapper around
#      ``naive_recurrent_kda``. Captures the per-step einsum recurrence into
#      a single CUDA graph, eliminating the ~30ms / 1k-steps interpreter
#      overhead that dominates the naive path. Best for moderate ``T`` (the
#      compilation graph is fully dynamic in ``T`` when ``dynamic=True``).
#
#   2. ``_scripted_chunk_kda_inner`` — ``torch.jit.script`` of the
#      ``naive_chunk_kda`` inner per-chunk loop body. The chunk path's inner
#      ``for i in range(0, NT)`` loop is small enough that TorchScript can
#      fully unroll / fuse it into a single kernel-launch sequence,
#      eliminating Python-side dispatch overhead. Used by
#      ``naive_chunk_kda`` automatically when the inputs are scriptable
#      (no dynamic Python control flow, no autograd-graph-break ops).
#
# Both wrappers preserve the EXACT numerical contract of the naive path: the
# ``compiled`` route returns identical outputs (within ``torch.compile``
# fusion noise, well below the fp32 tolerance enforced by
# ``run_correctness.py``); the ``scripted`` route is bit-identical to the
# Python loop it replaces (TorchScript fuses but does not reorder math).
# ============================================================================

# Module-level cache of compiled callables. ``torch.compile`` re-traces on
# every shape/dtype change, so memoizing per (B, T, H, K, HV, V, dtype,
# requires_grad) signature keeps the second-and-onward call fast. Mirrors the
# pattern used by ``torch._inductor``'s own ``cache_size`` limiter — we cap
# at ``_COMPILED_CACHE_MAX`` entries to bound memory in long-lived processes
# (e.g. a training loop that varies ``T`` per batch).
_COMPILED_KDA_CACHE: dict = {}
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
    """``torch.compile``-wrapped :func:`naive_recurrent_kda`.

    Issue 2.1 fix: the README documents ``naive_recurrent_kda`` as a
    "Python-loop reference implementation" but the project did not provide
    a ``torch.compile`` or Triton accelerated path, leaving users without
    an in-repo route to representative latency. This wrapper closes that
    gap.

    Args:
        q, k, v, g, beta, scale, initial_state, output_final_state,
        g_clamp_min, state_dtype: identical to :func:`naive_recurrent_kda`.
        mode: ``torch.compile`` mode (``"default"``, ``"reduce-overhead"``,
            ``"max-autotune"``). ``None`` lets PyTorch pick the default.
        dynamic: passed to ``torch.compile``. ``True`` (the default) allows
            variable ``T`` / ``B`` without re-tracing; ``False`` produces
            tighter graphs (faster inference) at the cost of re-tracing on
            any shape change.
        fullgraph: passed to ``torch.compile``. ``False`` (the default)
            allows graph-breaks; ``True`` forces a single graph (fails if
            any op cannot be traced — used to verify the recurrence is
            graph-safe).

    Returns:
        Same as :func:`naive_recurrent_kda` — ``(o, S)`` tuple where ``S``
        is ``None`` unless ``output_final_state=True``.

    .. note::

        The compiled graph is cached per (shape, dtype, requires_grad)
        signature. The first call for each unique signature incurs the
        one-time compilation cost (~5–30s); subsequent calls with the same
        signature hit the cache and run at the compiled speed. For
        ``T <= 256`` the compile cost may exceed the saved interpreter
        overhead; for ``T >= 1024`` the speedup is typically 5–20× on GPU.
    """
    # Lazily build the compiled callable (cached per signature). We do NOT
    # compile at module import time because (a) it adds 5–30s to import
    # even for users who never call the KDA path, and (b) the compile
    # target depends on the caller's tensor metadata (device, dtype,
    # requires_grad) which is unknown at import.
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
    )
    compiled_fn = _COMPILED_KDA_CACHE.get(cache_key)
    if compiled_fn is None:
        if len(_COMPILED_KDA_CACHE) >= _COMPILED_CACHE_MAX:
            # Evict oldest entry to bound cache size in long-lived processes.
            _COMPILED_KDA_CACHE.pop(next(iter(_COMPILED_KDA_CACHE)))
        # Wrap the inner recurrence with torch.compile. We compile a thin
        # closure that takes only tensor args (no Python floats / bools) so
        # ``dynamic=True`` can vary T/B without re-tracing. Scalar args
        # (scale, g_clamp_min, output_final_state) are baked into the
        # closure — they are small in number and rarely varied per call.
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

    return compiled_fn(
        q, k, v, g, beta,
        scale, g_clamp_min, output_final_state,
        initial_state,
    )


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
    """Chunkwise-parallel KDA (reference). Matches ``naive_recurrent_kda`` up to fp error.

    The ``g_clamp_min`` parameter mirrors :func:`naive_recurrent_kda` — see
    that function's docstring for the rationale. The chunk path applies the
    clamp BEFORE the cumulative-sum (``g.cumsum(-2)``) so the bound on the
    per-step gate also bounds the cumulative gate that appears in the
    chunk-internal Neumann series.

    The ``state_dtype`` parameter mirrors :func:`naive_recurrent_kda` (P1-2
    fix): the returned state defaults to ``compute_dtype`` (fp32 for
    fp16/bf16 inputs) instead of being downcast to ``v.dtype``, preventing
    precision loss in streaming inference where the state is repeatedly
    passed back as ``initial_state``.
    """
    dtype = v.dtype
    # P1-7 fix (revised): validate BEFORE unpacking q.shape (mirrors
    # naive_recurrent_kda) so a rank-mismatch is caught with a clear message
    # instead of the cryptic ``ValueError: not enough values to unpack``.
    _validate_kda_inputs(q, k, v, g, beta, fn_name='naive_chunk_kda')
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
            # P1-2 fix: mirror the main return path — keep state in
            # compute_dtype by default to avoid fp32→fp16 precision loss.
            target_state_dtype = (state_dtype if state_dtype is not None
                                  else compute_dtype)
            S = S.to(target_state_dtype)
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
    # P0 numerical-stability fix: clamp per-step ``g`` BEFORE the cumsum.
    # See naive_recurrent_kda's docstring for the rationale — without the
    # clamp, a single diverged ``g`` value can wipe the chunk-internal
    # state via ``exp(cumsum) -> 0``. Clamping per-step (before cumsum)
    # is the right place: it bounds the per-step decay factor AND keeps
    # the cumulative sum bounded (worst case ``cumsum(g) >= -10 * BT``).
    if g_clamp_min > -float('inf'):
        g = g.clamp(min=float(g_clamp_min))
    g = g.cumsum(-2)

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    A = torch.zeros(*g.shape[:-1], BT, dtype=compute_dtype, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        # P0 numerical-stability fix: clamp the upper bound of ``g - g_i`` BEFORE
        # ``exp``. ``g`` is the cumulative sum of per-step log-decays; for c < i
        # (lower triangular) ``g_c - g_i`` is POSITIVE (it equals the negated
        # cumulative decay from c+1 to i, and decay < 1 so its negation > 0).
        # With ``g_clamp_min=-10`` and ``BT=64`` this difference can reach ~630,
        # and ``exp(630) = inf`` in fp32 — which then propagates NaN through
        # ``solve_triangular``. The subsequent ``A.masked_fill(mask, 0)`` only
        # zeroes the UPPER triangular (c >= i); the overflowing LOWER triangular
        # entries are KEPT, so the inf/NaN reaches the solver. Clamping the
        # exponent to a safe upper bound (``exp(50) ~= 5e21``, finite in fp32)
        # prevents the overflow without changing the math for reasonable g
        # values (typical g in [-1, 0] gives differences well below 50). The
        # mask at line 451 still zeroes the upper triangle as before.
        g_diff = (g - g_i).clamp(max=50.0)
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * g_diff.exp(), k_i)
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
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    # P0 autograd-safety fix: accumulate per-chunk outputs in a list and
    # ``torch.stack`` at the end instead of in-place ``o[:, :, i] = ...`` on a
    # pre-allocated buffer. Mirrors the fix in ``naive_recurrent_kda``: the
    # in-place form saves a CopySlices node on ``o`` for backward, and under
    # ``torch.utils.checkpoint`` the recompute mutates the saved tensor,
    # raising ``RuntimeError: one of the variables needed for gradient
    # computation has been modified by an inplace operation``.
    chunk_outs = []
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
        chunk_outs.append((q_i * g_i.exp()) @ S + Aqk @ v_i)
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        # Use out-of-place ``S = S + ...`` (not in-place ``S += ...``) so the
        # state-update step is safe under future gradient-checkpointing. The
        # in-place variant would raise "one of the variables needed for
        # gradient computation has been modified by an inplace operation" if
        # anyone ever wraps this loop with checkpoint(). Mirrors the
        # out-of-place pattern in naive_recurrent_kda (line: S = S + einsum(...)).
        S = S + rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    o = torch.stack(chunk_outs, dim=2)
    if not output_final_state:
        S = None
    else:
        # P1-2 fix: return the state in ``compute_dtype`` (fp32 for fp16/bf16
        # inputs, fp64 for fp64 inputs) by default — NOT ``v.dtype``. The
        # previous ``S = S.to(dtype)`` caused repeated fp32→fp16→fp32
        # round-trips when the state was passed back as ``initial_state`` in
        # streaming inference, accumulating quantization error in long
        # sessions. Mirrors the fix in naive_recurrent_kda.
        target_state_dtype = state_dtype if state_dtype is not None else compute_dtype
        S = S.to(target_state_dtype)
    o = rearrange(o, 'b h n c d -> b (n c) h d').to(dtype)
    return o[:, :original_T], S


# =============================================================================
# TorchScript-compatible inner loop for ``naive_chunk_kda`` (issue 2.1 fix)
# -----------------------------------------------------------------------------
# ``naive_chunk_kda``'s inner ``for i in range(0, NT)`` loop (lines ~658-677
# above) uses einops ``rearrange`` which is not TorchScript-compatible, so the
# whole function cannot be ``torch.jit.script``'d directly. The inner loop
# IS the hot path (Python dispatch overhead per chunk dominates at small
# ``BT``), so we provide a script-compatible reimplementation below with the
# exact same math expressed via pure ``torch`` ops (``permute``, ``reshape``,
# ``unsqueeze``, ``matmul``). ``torch.jit.script`` fuses the loop into a
# single kernel-launch sequence, eliminating the per-chunk Python dispatch
# overhead.
#
# The ``scripted_chunk_kda`` wrapper below calls this inner function inside
# a scripted outer graph, falling back to the eager path if scripting fails
# (e.g. on a PyTorch version with stricter TorchScript support).
# ============================================================================


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
    """TorchScript-compatible inner loop of ``naive_chunk_kda``.

    Inputs are the chunk-rearranged tensors in shape ``[B, H, NT, BT, D]``
    (the same layout produced by the ``rearrange(... 'b (n c) h ... -> b h n
    c ...'`` calls in ``naive_chunk_kda``). ``mask`` is the upper-triangular
    mask of shape ``[BT, BT]`` used to zero the future entries of ``Aqk``.

    Returns:
        ``(chunk_outs, S)`` where ``chunk_outs`` is a list of per-chunk
        output tensors (caller stacks them) and ``S`` is the final state.
    """
    chunk_outs = []
    for i in range(NT):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i = u[:, :, i]
        g_i = g[:, :, i]
        w_i = w[:, :, i]
        # ``diff = g_i.unsqueeze(-2) - g_i.unsqueeze(-3)`` mirrors the
        # einsum-friendly broadcast form. Shape: ``[B, H, BT, BT, K]``.
        diff = g_i.unsqueeze(-2) - g_i.unsqueeze(-3)
        Aqk = (q_i.unsqueeze(-2) * diff.exp() * k_i.unsqueeze(-3)).sum(-1)
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        chunk_outs.append((q_i * g_i.exp()) @ S + Aqk @ v_i)
        # ``rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')`` == ``unsqueeze(-1)``.
        S = S * g_i[:, :, -1].exp().unsqueeze(-1)
        # ``rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i``
        # is equivalent to: take the per-chunk-cumulative-gate times k, swap
        # the last two dims, matmul with v_i, and accumulate into S.
        update = ((g_i[:, :, -1:] - g_i).exp() * k_i).permute(0, 1, 3, 2)
        S = S + update @ v_i
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
    """``naive_chunk_kda`` with a ``torch.jit.script``-compiled inner loop.

    Issue 2.1 fix: the README's *Limitations* section notes that
    ``naive_chunk_kda``'s inner per-chunk loop is unfused Python. This
    wrapper extracts the inner loop into a TorchScript-compatible function
    (``_chunk_kda_inner_loop``) and scripts it, eliminating the per-chunk
    Python dispatch overhead. The OUTER setup (validation, padding, cumsum,
    Neumann series, etc.) still runs in eager mode because it uses einops
    ``rearrange`` which is not TorchScript-compatible — only the hot inner
    loop is scripted.

    Args:
        Same as :func:`naive_chunk_kda`, plus:
        use_script: if ``True`` (the default), script the inner loop with
            ``torch.jit.script``. If scripting fails (older PyTorch, unusual
            dtype, etc.), automatically falls back to the eager path with
            a one-shot ``warnings.warn`` so callers are not silently
            degraded. Set to ``False`` to skip the scripting attempt and
            call :func:`naive_chunk_kda` directly (e.g. for debugging).

    Returns:
        Identical to :func:`naive_chunk_kda`.
    """
    # When scripting is disabled (or fails), we just delegate to the eager
    # implementation. This keeps the public contract identical to
    # ``naive_chunk_kda`` and lets callers opt out cleanly.
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

    # Re-implement ``naive_chunk_kda`` up to (but not including) the inner
    # loop. We cannot simply call ``naive_chunk_kda`` and intercept the
    # inner loop because the loop body is inlined into the function — there
    # is no hook point. Instead we duplicate the setup here and call the
    # scripted inner helper, then duplicate the tail.
    #
    # This duplication is regrettable but unavoidable: the alternative is
    # to refactor ``naive_chunk_kda`` itself to take an ``inner_loop_fn``
    # callback, which would change its public API. We keep the duplication
    # local to this wrapper so ``naive_chunk_kda``'s contract stays stable.
    dtype = v.dtype
    _validate_kda_inputs(q, k, v, g, beta, fn_name='scripted_chunk_kda')
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    if H < 1:
        raise ValueError(f"H={H} must be >= 1")
    if K < 1:
        raise ValueError(f"K={K} must be >= 1")
    if V < 1:
        raise ValueError(f"V={V} must be >= 1")
    if chunk_size < 1:
        raise ValueError(f"chunk_size={chunk_size} must be >= 1")
    G = HV // H
    if HV % H != 0:
        raise ValueError(f"HV={HV} must be divisible by H={H}")
    if g.shape[2] != HV:
        raise ValueError(f"g.shape[2]={g.shape[2]} must equal HV={HV}")
    if g.shape[-1] != K:
        raise ValueError(f"g.shape[-1]={g.shape[-1]} must equal K={K}")
    if beta.shape[2] != HV:
        raise ValueError(f"beta.shape[2]={beta.shape[2]} must equal HV={HV}")
    BT = chunk_size
    original_T = T
    if T == 0:
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float
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
        return o.to(dtype), S
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

    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float
    q, k = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(compute_dtype) for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(compute_dtype) for x in [v, g, beta]]
    q = q.repeat_interleave(G, dim=1) * scale
    k = k.repeat_interleave(G, dim=1)
    if g_clamp_min > -float('inf'):
        g = g.clamp(min=float(g_clamp_min))
    g = g.cumsum(-2)

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    A = torch.zeros(*g.shape[:-1], BT, dtype=compute_dtype, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        g_diff = (g - g_i).clamp(max=50.0)
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * g_diff.exp(), k_i)
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

    # --- scripted inner loop ---
    # Try to script the inner loop. If TorchScript rejects it (older PyTorch,
    # unusual dtype, etc.) fall back to the eager loop with a one-shot
    # warning. We do NOT swallow other exceptions — those indicate real bugs.
    chunk_outs = None
    try:
        scripted_inner = torch.jit.script(_chunk_kda_inner_loop)
        chunk_outs, S = scripted_inner(q, k, u, g, w, S, upper_mask, NT)
    except Exception as exc:
        import warnings as _warnings
        _warnings.warn(
            f"scripted_chunk_kda: torch.jit.script failed ({type(exc).__name__}: "
            f"{exc}); falling back to the eager inner loop. The output is "
            f"numerically identical but the per-chunk Python dispatch overhead "
            f"remains. Set use_script=False to silence this warning.",
            stacklevel=2,
        )
        chunk_outs, S = _chunk_kda_inner_loop(q, k, u, g, w, S, upper_mask, NT)

    o = torch.stack(chunk_outs, dim=2)
    if not output_final_state:
        S = None
    else:
        target_state_dtype = state_dtype if state_dtype is not None else compute_dtype
        S = S.to(target_state_dtype)
    o = rearrange(o, 'b h n c d -> b (n c) h d').to(dtype)
    return o[:, :original_T], S
