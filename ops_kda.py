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

import warnings as _warnings  # K7 fix: lifted to module scope

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


def _validate_kda_shapes(q, k, v, g, beta, fn_name, chunk_size=None):
    """Validate KDA shape dimensions (K2 fix — eliminates triplicated code).

    ``_validate_kda_inputs`` checks rank / device / dtype consistency. This
    helper checks the DIMENSION-value contract (H>=1, K>=1, V>=1, HV%H==0,
    g.shape[2]==HV, g.shape[-1]==K, beta.shape[2]==HV) that was previously
    copy-pasted into all three KDA entrypoints (naive_recurrent_kda,
    naive_chunk_kda, scripted_chunk_kda). Any future fix to the validation
    message now lives in ONE place.

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
    """K4 fix: pick the compute dtype for KDA math (centralized helper).

    fp64 inputs stay in fp64 (gradient checks, high-precision correctness
    tests). Everything else (fp16, bf16, fp32) computes in fp32 for
    numerical stability — the recurrence accumulates ``exp(g)`` over T
    steps, which loses too much precision in fp16/bf16 to be safe.
    """
    return torch.float64 if dtype == torch.float64 else torch.float


def _is_compiling_safely() -> bool:
    """Return True if we are inside a torch.compile / torch.export trace.

    P0-2 fix — torch 2.2 compatibility: ``torch.compiler.is_compiling()``
    was introduced in torch 2.3. On torch 2.2 (allowed by pyproject.toml
    ``torch>=2.2,<2.7``) the attribute does not exist and a direct call
    raises ``AttributeError``, crashing ``naive_recurrent_kda`` entirely.
    We try the new API first, then fall back to ``torch._dynamo``'s
    equivalent, then to False. The helper never raises.
    """
    try:
        # torch >=2.3 path (official documented pattern)
        return torch.compiler.is_compiling()  # type: ignore[attr-defined]
    except AttributeError:
        pass
    except Exception:
        # Any other error inside is_compiling (e.g. dynamo not initialized)
        # should be treated as "not compiling" rather than crashing the op.
        return False
    try:
        import torch._dynamo as _dm  # local import to avoid hard dep
        fn = getattr(_dm, "is_compiling", None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    return False


def _warn_if_nonfinite(o, fn_name, stacklevel=3):
    """Review-fix 1.1: surface non-finite KDA outputs with an actionable hint.

    ``naive_recurrent_kda`` / ``naive_chunk_kda`` implement the delta-rule
    recurrence ``S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t
    k_t v_t^T``. This recurrence is only numerically bounded when ``q``/``k``
    are unit-norm (as the KDA paper assumes for eigenvalue stability — every
    correctness test in this repo L2-normalizes ``q``/``k`` before calling
    into KDA) and ``v`` is of moderate magnitude. Neither function enforces
    this contract: nothing prevents a caller from passing un-normalized
    ``q``/``k`` or large-magnitude ``v``.

    When that contract is violated, the state ``S`` can grow without bound
    over a long sequence, and the accumulated per-step outputs silently
    become ``NaN`` (recurrent path) or a mix of ``NaN``/``Inf`` (chunk path
    — the two paths diverge differently because the chunk path's Neumann
    series and ``solve_triangular`` amplify large intermediate values
    differently than the strictly-sequential recurrence). Previously this
    propagated silently into the caller's loss / downstream layers with no
    diagnostic, making the eventual ``NaN`` in a much later part of the
    model very hard to trace back to "un-normalized KDA input" — the true
    root cause is far away, in the projections that produced ``q``/``k``/``v``.

    This helper performs one ``torch.isfinite(...).all()`` pass over the
    finished output (negligible cost relative to the O(T) per-step math it
    follows) and emits a single, actionable ``RuntimeWarning`` pointing at
    the likely cause and the fix, instead of leaving the caller to discover
    a bare ``NaN`` several layers downstream. This does NOT change the
    return value (the non-finite output is still returned as-is — some
    callers, e.g. NaN-tolerant gradient-checking utilities, may want to
    inspect it) and does NOT raise: KDA is a research reference
    implementation, and this repository's contract is "warn loudly, don't
    silently corrupt state" rather than "hard-fail on any exotic input".

    .. note:: review-fix 1.1-a — ``torch.compile`` / ``fullgraph=True``
        compatibility.

        The ``if not torch.isfinite(o).all(): ...`` check below is
        data-dependent Python control flow. When this function runs
        UNCOMPILED (the common case — direct calls to
        ``naive_recurrent_kda`` / ``naive_chunk_kda`` / ``scripted_chunk_kda``)
        this is perfectly fine: it is a single boolean reduction plus an
        ordinary ``if``. But when this function is invoked from INSIDE a
        ``torch.compile``-traced graph — which happens for
        ``compiled_recurrent_kda`` (the ``torch.compile`` wrapper around
        ``naive_recurrent_kda`` added for issue 2.1) — Dynamo cannot trace
        a branch whose condition depends on a tensor's runtime values, and
        raises ``Unsupported: Data-dependent branching``. This was a real
        regression introduced by review-fix 1.1: before that fix,
        ``naive_recurrent_kda``'s return path had no such data-dependent
        branch, so ``compiled_recurrent_kda(..., fullgraph=True)`` compiled
        successfully; after review-fix 1.1 it did not.

        We guard the check with ``torch.compiler.is_compiling()`` — the
        officially documented pattern (see ``torch.compiler.is_compiling``'s
        own docstring example) for "skip this logic while being traced by
        torch.compile / torch.export". ``is_compiling()`` itself is handled
        specially by Dynamo (it is NOT treated as data-dependent branching
        the same way an arbitrary tensor predicate is — Dynamo resolves it
        at trace time to a compile-time constant), so the surrounding
        ``if`` is graph-safe: under ``torch.compile`` the entire block is
        pruned away at trace time (no diagnostic overhead in the compiled
        graph, and no graph break), while EAGER calls (including calls
        made through ``compiled_recurrent_kda`` before the FIRST trace, and
        any direct eager call to the naive functions) still get the
        non-finite check and warning. This restores
        ``compiled_recurrent_kda(..., fullgraph=True)`` compilation while
        keeping the eager-path diagnostic that review-fix 1.1 was written
        to add. See ``test_compiled_recurrent_kda_fullgraph`` and
        ``test_kda_unnormalized_input_warns`` in ``run_correctness.py``
        for regression coverage of both halves of this contract.
    """
    # P0-2 fix — safe wrapper for torch 2.2 compat (see _is_compiling_safely)
    if _is_compiling_safely():
        # Skip the check entirely inside a torch.compile / torch.export
        # trace — see the review-fix 1.1-a note above. The compiled graph
        # therefore never contains this data-dependent branch.
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

        Review-fix 1.1: this function does NOT itself normalize ``q``/``k``,
        but the delta-rule recurrence ``S_t = (I - beta_t k_t k_t^T)
        Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T`` is only numerically
        bounded when ``q``/``k`` are unit-norm along the last (``K``) axis
        — this is the standard KDA/DeltaNet convention (see the paper) and
        is what every regression test in ``run_correctness.py`` does before
        calling into KDA. With un-normalized ``q``/``k`` (e.g. raw
        ``torch.randn`` outputs) or large-magnitude ``v``, the recurrent
        state can grow without bound over a long sequence and the output
        silently becomes non-finite (``NaN``) after enough steps — this was
        empirically confirmed to start around ``T~500`` for standard-normal
        inputs at ``K=64``. The **chunked** path (:func:`naive_chunk_kda`)
        diverges differently once inputs are unnormalized (it can produce a
        mix of ``NaN`` **and** ``Inf`` rather than pure ``NaN``, because the
        chunk-internal Neumann series / ``solve_triangular`` amplify large
        intermediate values differently than the strictly sequential
        recurrence) — the two paths are only guaranteed to numerically
        agree when both remain finite. Callers should ``F.normalize(q,
        dim=-1)`` / ``F.normalize(k, dim=-1)`` before calling this function
        unless they have a specific reason not to. If the returned output
        is non-finite, a one-shot ``RuntimeWarning`` is emitted pointing at
        this contract (see :func:`_warn_if_nonfinite`).

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
    # K2 fix: shape validation centralized in ``_validate_kda_shapes`` so
    # the same contract is enforced identically across naive_recurrent_kda,
    # naive_chunk_kda, and scripted_chunk_kda.
    _validate_kda_shapes(q, k, v, g, beta, fn_name='naive_recurrent_kda')
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    # K4 fix: compute-dtype selection centralized in ``_compute_dtype``.
    compute_dtype = _compute_dtype(dtype)
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
        _warnings.warn(
            f"naive_recurrent_kda: T={T} > 8192; the Python for-loop is "
            f"interpreter-overhead-bound (typical cost ~15-30ms per 1k "
            f"steps on CPU). For production latency, use the FLA Triton "
            f"kernel or wrap with torch.compile. This warning is emitted "
            f"once per process per call site.",
            stacklevel=2,
        )
    elif T > 1024 and q.is_cuda:
        # K8 fix: lower threshold on CUDA — the Python interpreter dispatch
        # dominates the math once T crosses ~1k on GPU, so the user is
        # silently paying a 5-20x overhead vs ``compiled_recurrent_kda``.
        # We do NOT auto-delegate (that would change the function's
        # observable contract and could surprise callers measuring the
        # naive path explicitly), but we do emit a one-shot info-level
        # nudge so users know the faster path exists.
        _warnings.warn(
            f"naive_recurrent_kda: T={T} > 1024 on CUDA — the Python "
            f"for-loop is interpreter-overhead-bound here. Consider "
            f"calling compiled_recurrent_kda(...) for a 5-20x speedup "
            f"(one-time compile cost ~5-30s, then cached).",
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
    o = o.to(dtype)
    # Review-fix 1.1: warn (do not raise) if the recurrence diverged to
    # non-finite values — see the "Input contract" note in this function's
    # docstring and :func:`_warn_if_nonfinite` for the rationale.
    _warn_if_nonfinite(o, 'naive_recurrent_kda')
    return o, S


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
#
# K6 fix: the previous eviction policy was FIFO
# (``pop(next(iter(...)))``). In a training loop that varies ``T`` per
# batch, FIFO can evict a just-compiled signature before its second use,
# forcing a recompile. We now use ``collections.OrderedDict`` and
# ``move_to_end`` on every hit so the eviction is LRU (least-recently-
# used), which keeps hot signatures resident.
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
        # P0-1 fix: state_dtype is baked into the compiled closure (line 443)
        # rather than passed as a runtime argument. If two calls share every
        # other key dimension but differ in state_dtype (e.g. None then fp16),
        # the cache would silently return the wrong-dtype state. Include it
        # in the key so a differing state_dtype forces a fresh compile.
        str(state_dtype),
        # P0-1 fix (2026-07-13): scale and g_clamp_min are also baked into
        # the closure's captured values (via _kda_kernel's closure over
        # state_dtype and the passed scale_val/g_clamp_min_val). If two
        # calls share shape/dtype but differ in scale (e.g. K**-0.5 vs 1.0)
        # they must NOT hit the same cached graph, otherwise the second
        # call would silently use the first call's scale, producing wrong
        # numerical results. Include both in the key.
        repr(scale),
        repr(g_clamp_min),
    )
    compiled_fn = _COMPILED_KDA_CACHE.get(cache_key)
    if compiled_fn is None:
        if len(_COMPILED_KDA_CACHE) >= _COMPILED_CACHE_MAX:
            # K6 fix: LRU eviction (least-recently-used) instead of FIFO.
            # ``popitem(last=False)`` removes the OLDEST entry in insertion
            # order, but because we ``move_to_end`` on every hit below, the
            # oldest insertion-order entry IS the least-recently-used one.
            _COMPILED_KDA_CACHE.popitem(last=False)
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
    else:
        # K6 fix: LRU — promote the just-hit entry to the end so it is the
        # LAST to be evicted by ``popitem(last=False)`` above.
        _COMPILED_KDA_CACHE.move_to_end(cache_key)

    return compiled_fn(
        q, k, v, g, beta,
        scale, g_clamp_min, output_final_state,
        initial_state,
    )


# =============================================================================
# Shared chunk-path setup/finalize helpers (review-fix 1.2)
# -----------------------------------------------------------------------------
# ``naive_chunk_kda`` and ``scripted_chunk_kda`` previously duplicated ~90
# lines of setup code (validation, T==0 early return, right-padding,
# chunk-rearrange, gate clamping/cumsum, building + solving the
# chunk-internal Neumann series, computing ``w``/``u``, seeding ``S`` from
# ``initial_state``) AND ~20 lines of tail code (stacking per-chunk outputs,
# casting the returned state's dtype, rearranging back to ``[B, T, ...]``,
# trimming padding, and the non-finite-output warning). The two copies had
# already drifted apart once in review history (only ``naive_chunk_kda``
# picked up the review-fix 1.1 non-finite warning until this refactor), and
# any future numerical-stability fix to one copy could silently fail to
# propagate to the other.
#
# ``_chunk_kda_prepare`` centralizes the setup: it does everything up to
# (but not including) the per-chunk loop, and returns either an "early"
# result (T == 0) or the pieces the loop needs. ``_chunk_kda_finalize``
# centralizes the tail. Both ``naive_chunk_kda`` and ``scripted_chunk_kda``
# now call these helpers and differ ONLY in which inner-loop implementation
# they invoke (eager Python loop vs. ``torch.jit.script``-compiled loop) —
# the one piece of genuine, unavoidable divergence between the two
# functions. ``_chunk_kda_inner_loop`` (below) is written entirely in
# TorchScript-compatible ops and is numerically identical whether called
# eagerly or under ``torch.jit.script``, so ``naive_chunk_kda`` now calls
# the very same function ``scripted_chunk_kda`` scripts — there is no
# longer a second, hand-duplicated eager loop body to keep in sync.
# ============================================================================


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
    """Shared setup for the chunk-parallel KDA path (review-fix 1.2).

    Performs validation, the ``T == 0`` degenerate case, right-padding to a
    multiple of ``chunk_size``, chunk-rearrangement, gate clamping/cumsum,
    the chunk-internal Neumann series solve, and ``w``/``u``/``S``
    construction — everything ``naive_chunk_kda`` and ``scripted_chunk_kda``
    need before their (potentially different) inner per-chunk loop runs.

    Returns:
        A dict. If ``dict['early']`` is ``True``, the sequence was empty
        (``T == 0``); ``dict['o']`` and ``dict['S']`` are the final return
        values (already cast to the right dtypes) and the caller should
        return them directly without running any inner loop. Otherwise the
        dict contains the keys ``q, k, u, g, w, S, upper_mask, NT, dtype,
        compute_dtype, original_T`` needed to run the inner loop and then
        call :func:`_chunk_kda_finalize`.
    """
    dtype = v.dtype
    # P1-7 fix (revised): validate BEFORE unpacking q.shape (mirrors
    # naive_recurrent_kda) so a rank-mismatch is caught with a clear message
    # instead of the cryptic ``ValueError: not enough values to unpack``.
    _validate_kda_inputs(q, k, v, g, beta, fn_name=fn_name)
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    # K2 fix: shape + chunk_size validation centralized in
    # ``_validate_kda_shapes`` so naive_chunk_kda and scripted_chunk_kda
    # share the same contract.
    _validate_kda_shapes(q, k, v, g, beta, fn_name=fn_name, chunk_size=chunk_size)
    G = HV // H
    BT = chunk_size
    original_T = T
    # Degenerate case: empty sequence. The downstream
    # ``torch.linalg.solve_triangular`` on an empty NT=0 batch raises
    # ``RuntimeError: solve_triangular: A and b must have the same number
    # of rows``. Guard explicitly (mirrors naive_recurrent_kda /
    # naive_csa / naive_hca).
    if T == 0:
        compute_dtype = _compute_dtype(dtype)
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
        return {'early': True, 'o': o.to(dtype), 'S': S}
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

    compute_dtype = _compute_dtype(dtype)
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

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
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
        # zeroes the UPPER triangular (c > i); the overflowing LOWER triangular
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
    """Shared tail for the chunk-parallel KDA path (review-fix 1.2).

    Stacks the per-chunk outputs, casts the returned state to
    ``state_dtype`` (or ``compute_dtype`` by default — P1-2 fix), rearranges
    back to ``[B, T, HV, V]``, trims off the right-padding, and emits the
    review-fix 1.1 non-finite-output warning. Shared by ``naive_chunk_kda``
    and ``scripted_chunk_kda`` so the two functions cannot silently diverge
    on this bookkeeping.
    """
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
    o = o[:, :original_T]
    # Review-fix 1.1: warn (do not raise) if the chunked recurrence
    # diverged to non-finite values — see the "Input contract" note in
    # the caller's docstring and ``_warn_if_nonfinite`` for the rationale.
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
    """Chunkwise-parallel KDA (reference). Matches ``naive_recurrent_kda`` up to fp error.

    .. warning:: **Input contract — q/k should be unit-norm.**

        Review-fix 1.1: see the identical warning in
        :func:`naive_recurrent_kda`'s docstring — the same unit-norm
        ``q``/``k`` contract applies here. This path is MORE sensitive to
        the contract being violated than the recurrent path: the
        chunk-internal Neumann series (``torch.linalg.solve_triangular``)
        and the ``exp(g_diff.clamp(max=50))`` term can amplify large
        intermediate values into ``Inf`` (not just ``NaN``) once ``q``/``k``
        are not unit-norm, whereas the strictly-sequential recurrent path
        tends to decay straight to ``NaN`` instead. **The two paths are
        therefore not expected to agree, and this function is not expected
        to match** :func:`naive_recurrent_kda` **once either output
        contains non-finite values** — the "matches to fp tolerance"
        guarantee below assumes finite, well-conditioned inputs (e.g.
        L2-normalized ``q``/``k``, as used throughout
        ``run_correctness.py``). If the returned output is non-finite, a
        one-shot ``RuntimeWarning`` is emitted (see
        :func:`_warn_if_nonfinite`).

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

    .. note:: review-fix 1.2 — shares its setup/inner-loop/tail code with
        :func:`scripted_chunk_kda` via :func:`_chunk_kda_prepare`,
        :func:`_chunk_kda_inner_loop`, and :func:`_chunk_kda_finalize`
        instead of maintaining two hand-duplicated copies. This function
        runs :func:`_chunk_kda_inner_loop` EAGERLY (no ``torch.jit.script``)
        so it always works regardless of TorchScript compatibility; use
        :func:`scripted_chunk_kda` for the compiled variant.
    """
    prep = _chunk_kda_prepare(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        chunk_size, g_clamp_min, state_dtype, fn_name='naive_chunk_kda',
    )
    if prep['early']:
        return prep['o'], prep['S']
    # P0 autograd-safety fix: ``_chunk_kda_inner_loop`` accumulates
    # per-chunk outputs in a Python list and the caller (``_chunk_kda_finalize``)
    # ``torch.stack``s them at the end, instead of in-place ``o[:, :, i] = ...``
    # on a pre-allocated buffer. The in-place form saves a CopySlices node on
    # ``o`` for backward, and under ``torch.utils.checkpoint`` the recompute
    # mutates the saved tensor, raising ``RuntimeError: one of the variables
    # needed for gradient computation has been modified by an inplace
    # operation``. Running the SAME ``_chunk_kda_inner_loop`` that
    # ``scripted_chunk_kda`` scripts (rather than a hand-duplicated eager
    # loop) is the point of review-fix 1.2: any future numerical fix to the
    # loop body only needs to be made once.
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


# =============================================================================
# TorchScript-compatible inner loop for ``naive_chunk_kda`` (issue 2.1 fix)
# -----------------------------------------------------------------------------
# ``naive_chunk_kda``'s inner per-chunk loop uses einops ``rearrange``-free,
# pure ``torch`` ops (``permute``, ``unsqueeze``, ``matmul``) so it is
# TorchScript-compatible, and (since review-fix 1.2) is the ONLY
# implementation of the inner loop — ``naive_chunk_kda`` calls it eagerly and
# ``scripted_chunk_kda`` calls it through ``torch.jit.script``. Previously
# ``naive_chunk_kda`` had its OWN hand-duplicated eager loop (written with
# ``einops.rearrange`` instead of ``permute``/``unsqueeze``) that had to be
# kept in sync with this one by hand; that duplication has been removed.
#
# The ``scripted_chunk_kda`` wrapper below scripts this function, falling
# back to the eager path if scripting fails (e.g. on a PyTorch version with
# stricter TorchScript support).
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
        # P0 numerical-stability fix (mirrors the clamp at line 844 in
        # ``naive_chunk_kda``'s A-matrix construction). For masked-out
        # upper-triangular entries (c > i), ``g_i[..., c, :] - g_i[..., i, :]``
        # is POSITIVE when ``g < 0`` (the documented contract — g is produced
        # upstream as ``-softplus(...) * kda_decay_scale`` and clamped to
        # ``>= g_clamp_min=-10``). With ``BT=64`` and ``g_clamp_min=-10`` this
        # difference can reach ~630, and ``exp(630) = inf`` in fp32. The
        # subsequent ``Aqk.masked_fill(mask, 0)`` zeroes the forward output
        # (so the forward is fine), but ``exp(diff) * 0 = inf * 0 = NaN`` in
        # the BACKWARD pass, poisoning ``q.grad`` / ``k.grad`` / ``g.grad``.
        # The sibling A-matrix at line 844 was already clamped via
        # ``(g - g_i).clamp(max=50.0)``; the same fix was missing here.
        # Clamping the exponent to a safe upper bound (``exp(50) ~= 5e21``,
        # finite in fp32) prevents the overflow without changing the math
        # for reasonable g values (typical g in [-1, 0] gives differences
        # well below 50). The masked_fill below still zeroes the upper
        # triangle as before; the clamp only changes the (masked-out)
        # entries that would otherwise produce inf/NaN in backward.
        # Verified: forward output bit-identical (max diff 0.0); backward
        # gradients finite for g down to ``g_clamp_min=-10``.
        Aqk = (q_i.unsqueeze(-2) * diff.clamp(max=50.0).exp() * k_i.unsqueeze(-3)).sum(-1)
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
    wrapper scripts the SAME :func:`_chunk_kda_inner_loop` that
    :func:`naive_chunk_kda` runs eagerly, eliminating the per-chunk Python
    dispatch overhead. The OUTER setup (validation, padding, cumsum,
    Neumann series, etc.) still runs in eager mode because it uses einops
    ``rearrange`` which is not TorchScript-compatible — only the hot inner
    loop is scripted.

    .. note:: review-fix 1.2 — this function previously duplicated
        ``naive_chunk_kda``'s entire setup and tail (~90 lines) by hand,
        with a comment acknowledging the duplication was "regrettable but
        unavoidable". It now calls the SAME :func:`_chunk_kda_prepare` /
        :func:`_chunk_kda_finalize` helpers that :func:`naive_chunk_kda`
        uses, and differs from it ONLY in scripting
        :func:`_chunk_kda_inner_loop` before calling it. Any future
        numerical-stability fix to the shared setup/tail code (e.g. the
        ``g_clamp_min`` handling, or the review-fix 1.1 non-finite-output
        warning) now automatically applies to both functions.

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

    prep = _chunk_kda_prepare(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        chunk_size, g_clamp_min, state_dtype, fn_name='scripted_chunk_kda',
    )
    if prep['early']:
        return prep['o'], prep['S']

    # --- scripted inner loop ---
    # Try to script the inner loop. If TorchScript rejects it (older PyTorch,
    # unusual dtype, etc.) fall back to the eager loop with a one-shot
    # warning. We do NOT swallow other exceptions — those indicate real bugs.
    try:
        scripted_inner = torch.jit.script(_chunk_kda_inner_loop)
        chunk_outs, S = scripted_inner(
            prep['q'], prep['k'], prep['u'], prep['g'], prep['w'], prep['S'],
            prep['upper_mask'], prep['NT'],
        )
    except Exception as exc:
        _warnings.warn(
            f"scripted_chunk_kda: torch.jit.script failed ({type(exc).__name__}: "
            f"{exc}); falling back to the eager inner loop. The output is "
            f"numerically identical but the per-chunk Python dispatch overhead "
            f"remains. Set use_script=False to silence this warning.",
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
