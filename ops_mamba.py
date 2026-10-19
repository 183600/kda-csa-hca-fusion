"""Standalone Mamba-2 (SSD) reference implementation and unified mixer module.

This module adds a second *recurrent memory* operator to the hybrid research
framework alongside KDA (``ops_kda.py``). It follows the Mamba-2 /
State-Space-Duality formulation (arXiv:2405.21060, "Transformers are SSMs:
Generalized Models and Efficient Algorithms Through Structured State
Association"):

  * ``d_inner = expand * d_model`` channels are split into ``nheads`` heads of
    ``headdim`` channels each; the SSM state is one ``[headdim, d_state]``
    matrix per head (the "matrix state" that distinguishes Mamba-2 from
    Mamba-1's vector state).
  * Per step t (per head h, with B/C shared inside a group of heads):

        a_t   = exp(dt_t * A_h)                    (A_h < 0, decay in (0, 1))
        S_t   = a_t * S_{t-1} + dt_t * (x_t (x) B_t)
        y_t   = C_t . S_t + D_h * x_t

    This is the same simplified ZOH discretization used by the official
    ``mamba_ssm`` kernels (``deltaB * u`` with ``deltaA = exp(delta * A)``),
    NOT an exact ZOH — matching the official semantics is the whole point of
    the reference path.
  * The full mixer is: ``in_proj -> [x, z]``; causal depthwise conv over
    ``x``; SiLU activation; ``x_proj(x_act) -> [dt, B, C]``; the scan above;
    output ``y * SiLU(z)``; ``out_proj``. No RoPE, no normalization, no
    residual — the surrounding stack (``ops_fused.HybridAttentionStack``)
    owns those, mirroring how the official ``Mamba2`` mixer module is
    wrapped by an outer block.

Parameter names (``in_proj``, ``conv1d``, ``x_proj``, ``dt_proj``, ``A_log``,
``D``, ``out_proj``) intentionally match the official ``mamba_ssm`` ``Mamba2``
module so state_dicts transfer between this reference and the optional
official backend (see ``ops_mamba_backend.py``). The parity test in
``run_correctness.py`` relies on this.

The sequential Python-loop scan is the *correctness reference* used to
validate the official backend and to run CPU/GPU-agnostic experiments.

Correctness chain / SSD duality oracle
--------------------------------------
The recurrent scan above is validated by self-consistency checks (prefill +
decode == full forward) and by the optional official-backend parity test,
which self-skips without CUDA + ``mamba-ssm``. Both chains run the SAME
recurrence, so a shared semantic error (e.g. a decay/indexing convention
mistake) could survive them. :func:`quadratic_mamba2_scan` closes that hole
with a mathematically independent evaluation: the SSD dual form — the same
operator as a masked quadratic "attention" with a semi-separable pairwise
decay (arXiv:2405.21060 Sections 3-4), computed from cumulative log-decays
instead of a sequential state recurrence. It shares the shape/validation
contract by construction (one validator) and is pinned by forward AND
gradient parity tests in
``test_ssd_duality.py``. It is an oracle, never a production path.

:func:`chunked_mamba2_scan` adds a third view: the chunked-SSD
decomposition (quadratic intra-chunk + an associative scan over the
scalar-affine chunk transitions), with every exponent anchored at the
chunk boundary for log-space gating stability. It is a drop-in replacement
for the recurrence inside ``Mamba2Mixer`` (selected by
``HybridConfig.mamba_chunk_size``) and is pinned against both views above
by ``test_chunked_ssd.py``.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops_cache import normal_tensor_scope

__all__ = [
    "Mamba2Mixer",
    "build_mamba2_mixer",
    "chunked_mamba2_scan",
    "naive_mamba2_scan",
    "quadratic_mamba2_scan",
    "resolve_mamba_dt_rank",
    "validate_mamba2_geometry",
]


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------
def resolve_mamba_dt_rank(dt_rank: int | str, d_model: int) -> int:
    """Resolve Mamba-2's ``dt_rank`` following the official ``"auto"`` rule.

    ``dt_rank="auto"`` maps to ``ceil(d_model / 16)`` (the low-rank
    factorization of the input-dependent discretization), matching
    ``mamba_ssm``. An explicit int is validated and returned unchanged.
    """
    if isinstance(dt_rank, str):
        if dt_rank != "auto":
            raise ValueError(
                f"mamba_dt_rank={dt_rank!r} must be 'auto' or a positive int.")
        if not isinstance(d_model, int) or d_model < 1:
            raise ValueError(
                f"d_model={d_model!r} must be a positive int to resolve "
                f"dt_rank='auto'.")
        return max(1, math.ceil(d_model / 16))
    if not isinstance(dt_rank, int) or isinstance(dt_rank, bool) or dt_rank < 1:
        raise ValueError(
            f"mamba_dt_rank={dt_rank!r} must be 'auto' or a positive int.")
    return dt_rank


def validate_mamba2_geometry(
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    headdim: int,
    ngroups: int = 1,
    dt_rank: int | str = "auto",
) -> dict:
    """Validate Mamba-2 dimensional constraints and return derived sizes.

    Constraints (all mirror the official ``mamba_ssm`` ``Mamba2`` module):

      * ``d_inner = expand * d_model`` must be divisible by ``headdim``
        (``nheads = d_inner // headdim``).
      * ``nheads`` must be divisible by ``ngroups`` (heads are assigned to
        B/C groups in contiguous blocks).
      * ``d_conv >= 2`` — a 1-tap "convolution" has no left context, which
        makes the streaming conv-state contract vacuous; reject it up front
        instead of threading an empty-state special case through the code.
      * ``d_state >= 1``, ``expand >= 1``.

    Returns ``{'d_inner', 'nheads', 'heads_per_group', 'dt_rank'}``.
    """
    for name, val in [('d_model', d_model), ('d_state', d_state),
                      ('d_conv', d_conv), ('expand', expand),
                      ('headdim', headdim), ('ngroups', ngroups)]:
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise ValueError(
                f"Mamba-2 {name}={val!r} must be a positive int.")
    if d_conv < 2:
        raise ValueError(
            f"mamba_d_conv={d_conv} must be >= 2. A 1-tap conv carries no "
            f"left context, so the streaming conv-state contract would be "
            f"vacuous; use a real short conv (official default 4).")
    d_inner = expand * d_model
    if d_inner % headdim != 0:
        raise ValueError(
            f"Mamba-2 geometry violated: d_inner = expand * d_model = "
            f"{expand} * {d_model} = {d_inner} must be divisible by "
            f"headdim={headdim}. Adjust mamba_expand or mamba_headdim "
            f"(e.g. d_model=32, expand=2 -> d_inner=64 pairs with "
            f"headdim in {{8, 16, 32, 64}}).")
    nheads = d_inner // headdim
    if nheads % ngroups != 0:
        raise ValueError(
            f"Mamba-2 nheads={nheads} (d_inner={d_inner} / headdim={headdim}) "
            f"must be divisible by ngroups={ngroups}.")
    return {
        'd_inner': d_inner,
        'nheads': nheads,
        'heads_per_group': nheads // ngroups,
        'dt_rank': resolve_mamba_dt_rank(dt_rank, d_model),
    }


# ---------------------------------------------------------------------------
# Core SSD scan (the analogue of ops_kda.naive_recurrent_kda)
# ---------------------------------------------------------------------------
# The scan loop below performs, per step, the decay multiply, the
# ``dt * (x (x) B)`` state update, the ``C . S`` readout and the D skip.
# Every step-varying factor is computed over the whole time axis: the
# group-to-head expanded B/C, the outer products ``x (x) B``, their
# ``dt`` scaling, the per-step decay ``exp(dt * A)`` and the skip term
# ``D * x``. Each such product is a pure elementwise operation, so it
# rounds exactly like the per-step evaluation and the forward values stay
# bit-identical. Only the *accumulation order* of the autograd sums
# feeding the ``A`` / ``dt`` gradients changes (one slab-wide reduction
# instead of per-step accumulation), which is fp round-off level and
# pinned by the SSD-duality gradient tests in fp64. The loop itself
# executes the per-step kernels (decay-multiply, state add, ``C . S``
# contraction, skip add), with the batch/head axes merged so the readout
# is a direct 3D ``bmm`` (the same per-batch GEMM).
_SCAN_LOOP_SCRIPTED = None
_SCAN_LOOP_SCRIPTED_FAILED = False
_SCAN_LOOP_INPLACE_SCRIPTED = None
_SCAN_LOOP_INPLACE_SCRIPTED_FAILED = False
# Cap on the number of elements in one slab of precomputed step-varying
# factors (``[batch * nheads, slab, headdim, d_state]`` for the outer product,
# which is dt-scaled in place); long sequences are processed in several
# slabs, and the recurrent state carries across slab boundaries. The slab
# size never changes the values (each step touches only its own slice of
# the factors). The outer product is the only slab-sized factor array held
# at once, so the cap bounds that single array.
_SCAN_XOUTER_SLAB_ELEMS = 8_000_000

# Slab budget for the sequential reference scan's factor arrays. The slab
# size is value-independent (the state carries across slabs), so it is
# chosen purely for cache behaviour: the per-step working set (one
# ``[BH, D, N]`` state slice plus its factors) stays hot in L2/L3 for the
# smaller slab, while a slab that large spills to DRAM on every step.
# Kept separate from ``_SCAN_XOUTER_SLAB_ELEMS`` so the chunked path's
# fallback decision (a semantic switch, not a perf knob) is unchanged.
_SCAN_NAIVE_SLAB_ELEMS = 1_000_000

# Cap on the number of elements in one slab of the chunked path's
# ``[batch, nheads, slab, chunk, chunk]`` intra-chunk intermediates. The
# per-chunk intra-chunk outputs are independent of every other chunk, so
# evaluating a slab of chunks yields exactly the values the all-chunks
# evaluation would (only the transient peak size changes).
_SCAN_INTRA_SLAB_ELEMS = 8_000_000

# Shape-keyed cache for the strict-lower-triangular causal mask consumed by
# ``chunked_mamba2_scan``. It depends only on (chunk_size, device) and is
# read-only at every use site (``masked_fill(~causal, ...)``). ``chunk_size``
# may legally be as large as ``T``, so a single mask can reach GB size; masks
# beyond the element budget are recomputed per call instead of retained
# (recomputation produces the same bits).
_CAUSAL_CHUNK_MASK_CACHE: dict = {}
_CAUSAL_CHUNK_MASK_MAX_ELEMS = 1 << 22


def _causal_chunk_mask(BT: int, device) -> torch.Tensor:
    """Cached ``[BT, BT]`` lower-triangular boolean mask (True = s <= c)."""
    key = (BT, str(device))
    mask = _CAUSAL_CHUNK_MASK_CACHE.get(key)
    if mask is None:
        with normal_tensor_scope():
            mask = torch.tril(
                torch.ones(BT, BT, dtype=torch.bool, device=device))
        if BT * BT <= _CAUSAL_CHUNK_MASK_MAX_ELEMS:
            if len(_CAUSAL_CHUNK_MASK_CACHE) >= 64:
                _CAUSAL_CHUNK_MASK_CACHE.clear()
            _CAUSAL_CHUNK_MASK_CACHE[key] = mask
    return mask


# Cached negation of ``_causal_chunk_mask`` (True = strictly-above-diagonal,
# the entries masked to ``-inf`` before the decay exp). Read-only at every
# use site, same byte budget as the positive mask.
_NONCAUSAL_CHUNK_MASK_CACHE: dict = {}


def _noncausal_chunk_mask(BT: int, device) -> torch.Tensor:
    """Cached ``~_causal_chunk_mask(BT, device)`` (read-only)."""
    key = (BT, str(device))
    mask = _NONCAUSAL_CHUNK_MASK_CACHE.get(key)
    if mask is None:
        with normal_tensor_scope():
            mask = ~_causal_chunk_mask(BT, device)
        if BT * BT <= _CAUSAL_CHUNK_MASK_MAX_ELEMS:
            if len(_NONCAUSAL_CHUNK_MASK_CACHE) >= 64:
                _NONCAUSAL_CHUNK_MASK_CACHE.clear()
            _NONCAUSAL_CHUNK_MASK_CACHE[key] = mask
    return mask


def _mamba2_scan_loop_eager(x_dt, decay_slab, C_h, Dx_slab, S):
    """Sequential scan body over one slab of precomputed factors.

    The batch and head axes are pre-merged (``BH = batch * nheads``), which
    lets the readout use an explicit ``bmm`` of 3D operands instead of a 4D
    ``matmul``; the two compute the same per-batch GEMM, so the values are
    bit-identical. ``x_dt`` is ``[BH, slab, D, N]`` (the
    ``dt * (x (x) B)`` state-update source), ``decay_slab`` is
    ``[BH, slab, 1, 1]`` (the per-step decay ``exp(dt * A)``), ``C_h`` is
    ``[BH, slab, 1, N]`` (the expanded readout factors) and ``Dx_slab`` is
    ``[BH, slab, 1, D]`` (the ``D * x`` skip term with its singleton head
    dim pre-applied). Returns ``(y_slab, S)`` with ``y_slab`` shaped
    ``[BH, slab, D]``.
    """
    ys: list[torch.Tensor] = []
    T = x_dt.shape[1]
    # ``unbind`` yields storage-sharing views equivalent to ``[:, t]``
    # indexing; the slab tensors carry the singleton broadcast dims
    # explicitly.
    x_steps = x_dt.unbind(1)
    decay_steps = decay_slab.unbind(1)
    C_steps = C_h.unbind(1)
    Dx_steps = Dx_slab.unbind(1)
    # For a contiguous state the readout ``C @ S^T + Dx`` runs as one
    # ``baddbmm`` (the same contraction and sum for fp32/fp64 state
    # operands, see :func:`_mamba2_scan_loop_inplace_eager`); the fresh
    # ``decay * S + x`` state is always contiguous, but the guard keeps
    # the explicit two-step form for any other layout.
    if S.is_contiguous():
        for t in range(T):
            S = decay_steps[t] * S + x_steps[t]
            ys.append(torch.baddbmm(Dx_steps[t], C_steps[t],
                                    S.transpose(-1, -2)))
    else:
        for t in range(T):
            S = decay_steps[t] * S + x_steps[t]
            # ``einsum('bdn,bn->bd', S, C)`` written as the equivalent
            # explicit matmul (the same ``bmm`` contraction over ``n``). The
            # singleton query dim is kept until the single post-loop squeeze
            # (append-and-stack is elementwise-identical to squeeze-then-stack).
            ys.append(torch.bmm(C_steps[t], S.transpose(-1, -2))
                      + Dx_steps[t])
    return torch.stack(ys, dim=1).squeeze(-2), S


def _mamba2_scan_loop_inplace_eager(x_dt, decay_slab, C_h, Dx_slab, S):
    """No-grad scan body that updates the recurrent state IN PLACE.

    ``S = decay * S + x`` is evaluated as ``S.mul_(decay).add_(x)``: the
    same products and sums with the same roundings (bit-identical
    values). The caller must pass a dedicated copy of the state (this
    function mutates its ``S`` argument) and autograd must be disabled —
    an in-place update of a graph-tracked state would trip the version
    counter. See :func:`_mamba2_scan_loop_eager` for the (autograd-safe)
    out-of-place form.

    The in-place updates keep the state's storage (and therefore every
    view of it) fixed for the whole loop, so for a contiguous state the
    transposed readout view ``S^T`` is built once instead of once per
    step, and the readout ``C @ S^T + Dx`` runs as the single
    ``baddbmm`` whose GEMM epilogue folds the skip add into the matmul.
    ``baddbmm`` and ``bmm(...) + Dx`` perform the same contraction and
    sum for a contiguous state operand; a non-contiguous state keeps the
    explicit two-step form so both layouts produce the exact bits they
    produced before the readout was fused.
    """
    x_steps = x_dt.unbind(1)
    decay_steps = decay_slab.unbind(1)
    C_steps = C_h.unbind(1)
    Dx_steps = Dx_slab.unbind(1)
    # Collect the per-step readouts and stack once at the end: the single
    # batched copy replaces T tiny per-step copies (elementwise-identical
    # values; the stack squeezes the singleton head dim only once).
    ys: list[torch.Tensor] = []
    if S.is_contiguous():
        S_T = S.transpose(-1, -2)
        for xs, ds, Cs, Dxs in zip(x_steps, decay_steps, C_steps,
                                   Dx_steps):
            # ``mul_`` then ``add_`` (NOT a fused ``addcmul``): the two
            # roundings match the out-of-place reference path bit-for-bit.
            S = S.mul_(ds).add_(xs)
            ys.append(torch.baddbmm(Dxs, Cs, S_T))
    else:
        for xs, ds, Cs, Dxs in zip(x_steps, decay_steps, C_steps,
                                   Dx_steps):
            S = S.mul_(ds).add_(xs)
            ys.append(torch.bmm(Cs, S.transpose(-1, -2)) + Dxs)
    return torch.stack(ys, dim=1).squeeze(-2), S


def _get_scripted_scan_loop():
    """Scripted scan loop body, with eager fallback.

    The scripted function executes the same operations as
    :func:`_mamba2_scan_loop_eager` (bit-identical values); scripting
    unavailability is recorded so the caller uses the eager function.
    """
    global _SCAN_LOOP_SCRIPTED, _SCAN_LOOP_SCRIPTED_FAILED
    if _SCAN_LOOP_SCRIPTED is None and not _SCAN_LOOP_SCRIPTED_FAILED:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                _SCAN_LOOP_SCRIPTED = torch.jit.script(
                    _mamba2_scan_loop_eager)
        except Exception:
            _SCAN_LOOP_SCRIPTED_FAILED = True
    return _SCAN_LOOP_SCRIPTED


def _get_scripted_scan_loop_inplace():
    """Scripted no-grad scan body, with eager fallback (see
    :func:`_mamba2_scan_loop_inplace_eager`)."""
    global _SCAN_LOOP_INPLACE_SCRIPTED, _SCAN_LOOP_INPLACE_SCRIPTED_FAILED
    if (_SCAN_LOOP_INPLACE_SCRIPTED is None
            and not _SCAN_LOOP_INPLACE_SCRIPTED_FAILED):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                _SCAN_LOOP_INPLACE_SCRIPTED = torch.jit.script(
                    _mamba2_scan_loop_inplace_eager)
        except Exception:
            _SCAN_LOOP_INPLACE_SCRIPTED_FAILED = True
    return _SCAN_LOOP_INPLACE_SCRIPTED


def _validate_mamba2_scan_inputs(
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    initial_state: torch.Tensor | None,
    fn_name: str,
) -> dict:
    """Shared shape/dtype contract for both Mamba-2 scan views.

    The recurrent (:func:`naive_mamba2_scan`) and dual
    (:func:`quadratic_mamba2_scan`) evaluations validate through this ONE
    function, so the oracle cannot silently accept inputs the reference
    path would reject (or vice versa). ``fn_name`` is only used to keep the
    error messages attributable to the caller.

    Returns the derived geometry plus ``initial_state`` normalized to a
    compute-precision tensor (zeros when ``None``).
    """
    if x.dim() != 4:
        raise ValueError(
            f"{fn_name}: x must be [batch, seqlen, nheads, headdim] "
            f"(rank 4), got rank {x.dim()} (shape={tuple(x.shape)}).")
    batch, seqlen, nheads, headdim = x.shape
    for name, t, rank, shape_desc in [
            ('dt', dt, 3, '[batch, seqlen, nheads]'),
            ('B', B, 4, '[batch, seqlen, ngroups, d_state]'),
            ('C', C, 4, '[batch, seqlen, ngroups, d_state]'),
            ('A', A, 1, '[nheads]'),
            ('D', D, 1, '[nheads]')]:
        if t.dim() != rank:
            raise ValueError(
                f"{fn_name}: {name} must have rank {rank} "
                f"({shape_desc}), got rank {t.dim()} (shape={tuple(t.shape)}).")
    if dt.shape[0] != batch or dt.shape[1] != seqlen or dt.shape[2] != nheads:
        raise ValueError(
            f"{fn_name}: dt.shape={tuple(dt.shape)} does not match "
            f"x[:3]={(batch, seqlen, nheads)}.")
    # B/C time and batch dims must match x as well: the scan indexes them
    # per step, so a silent broadcast would produce wrong outputs instead
    # of an error (the dual's einsum would fail with a cryptic message).
    if B.shape[0] != batch or B.shape[1] != seqlen:
        raise ValueError(
            f"{fn_name}: B.shape={tuple(B.shape)} does not match the x "
            f"batch/seqlen dims {(batch, seqlen)}.")
    ngroups, d_state = B.shape[2], B.shape[3]
    if C.shape != B.shape:
        raise ValueError(
            f"{fn_name}: C.shape={tuple(C.shape)} must equal "
            f"B.shape={tuple(B.shape)}.")
    if nheads % ngroups != 0:
        raise ValueError(
            f"{fn_name}: nheads={nheads} must be divisible by "
            f"ngroups={ngroups}.")
    if A.shape[0] != nheads or D.shape[0] != nheads:
        raise ValueError(
            f"{fn_name}: A/D must have {nheads} entries "
            f"(one per head), got A={tuple(A.shape)}, D={tuple(D.shape)}.")

    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    if initial_state is not None:
        if initial_state.dim() != 4:
            raise ValueError(
                f"{fn_name}: initial_state must be "
                f"[batch, nheads, headdim, d_state] (rank 4), got rank "
                f"{initial_state.dim()} (shape={tuple(initial_state.shape)}).")
        expected = (batch, nheads, headdim, d_state)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                f"{fn_name}: initial_state.shape="
                f"{tuple(initial_state.shape)} does not match the expected "
                f"{expected} implied by x/B (a state persisted from a "
                f"different config?).")
        state = initial_state.to(device=x.device, dtype=compute_dtype)
    else:
        state = torch.zeros(batch, nheads, headdim, d_state,
                            device=x.device, dtype=compute_dtype)
    return {
        'batch': batch, 'seqlen': seqlen, 'nheads': nheads,
        'headdim': headdim, 'ngroups': ngroups, 'd_state': d_state,
        'heads_per_group': nheads // ngroups,
        'compute_dtype': compute_dtype, 'state': state,
    }


def _expand_heads(x: torch.Tensor, heads_per_group: int) -> torch.Tensor:
    """Per-head B/C with the ``repeat_interleave(hpg, dim=2)`` value layout.

    Heads map to groups in contiguous blocks (head ``h`` -> group
    ``h // heads_per_group``), which is exactly a reshape of the
    ``(ngroups, heads_per_group)`` pair (or a plain view when the GVA
    factor is 1).
    """
    if heads_per_group == 1:
        return x
    batch, seqlen, ngroups, d_state = x.shape
    return (x.unsqueeze(3)
            .expand(batch, seqlen, ngroups, heads_per_group, d_state)
            .reshape(batch, seqlen, ngroups * heads_per_group, d_state))


def naive_mamba2_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sequential reference scan of the Mamba-2 (SSD) recurrence.

    Shapes:
      * ``x``:   ``[batch, seqlen, nheads, headdim]``  (post-conv, post-SiLU).
      * ``dt``:  ``[batch, seqlen, nheads]`` (post-softplus, all positive).
      * ``B``:   ``[batch, seqlen, ngroups, d_state]``.
      * ``C``:   ``[batch, seqlen, ngroups, d_state]``.
      * ``A``:   ``[nheads]`` NEGATIVE decay rates (``A = -exp(A_log)``).
      * ``D``:   ``[nheads]`` skip connection scales.
      * ``initial_state``: ``[batch, nheads, headdim, d_state]`` or ``None``.

    Returns ``(y, final_state)`` with ``y`` shaped like ``x`` (cast back to
    ``x.dtype``) and ``final_state`` in compute precision (fp32 for
    fp16/bf16/fp32 inputs, fp64 for fp64 inputs — the same state-precision
    contract as ``ops_kda.naive_recurrent_kda``).

    Head-to-group mapping: heads are assigned to B/C groups in CONTIGUOUS
    blocks — head ``h`` uses group ``h // heads_per_group`` — matching the
    ``repeat(..., 'g h')`` expansion in the official chunked kernels. With
    the default ``ngroups=1`` every head shares the same B/C.

    The dual (masked-attention) view of this same computation is
    :func:`quadratic_mamba2_scan`; the two must agree bit-for-bit up to
    reduction-order rounding (pinned by ``test_ssd_duality.py``).
    """
    v = _validate_mamba2_scan_inputs(
        x, dt, B, C, A, D, initial_state, 'naive_mamba2_scan')
    batch, seqlen = v['batch'], v['seqlen']
    nheads, headdim = v['nheads'], v['headdim']
    ngroups, d_state = v['ngroups'], v['d_state']
    heads_per_group = v['heads_per_group']
    compute_dtype = v['compute_dtype']
    S = v['state']

    x_c = x.to(compute_dtype)
    dt_c = dt.to(compute_dtype)
    B_c = B.to(compute_dtype)
    C_c = C.to(compute_dtype)
    A_c = A.to(compute_dtype)
    D_c = D.to(compute_dtype)

    if seqlen == 0:
        y = x.new_zeros(batch, 0, nheads, headdim)
        return y, (S if output_final_state else None)

    # Step-varying factors computed over the whole time axis
    # (elementwise-identical to the per-step evaluation):
    #   * B_h / C_h: the group-level B/C expanded to per-head B/C. Heads are
    #     assigned to groups in CONTIGUOUS blocks (head h -> group
    #     h // heads_per_group); repeat_interleave matches the per-step
    #     ``[:, t].repeat_interleave(heads_per_group, dim=1)``.
    #   * x_outer[bh, t, d, n] = x[b, t, h, d] * B_h[b, t, h, n], the
    #     per-step ``einsum('bhd,bhn->bhdn', x[:, t], B_h)``.
    #   * x_dt = dt * x_outer, decay = exp(dt * A), Dx = D * x: the remaining
    #     per-step products/skip term. The scalar A is broadcast with an
    #     explicit singleton layout so the ``exp`` argument shape matches the
    #     per-step form exactly.
    # The batch and head axes are merged (``bh = b * nheads + h``) throughout
    # the scan: every per-step factor is a pure elementwise map (or a
    # per-batch ``bmm``) over which the merge is exact, the merge is just a
    # permute-copy of the same values, and the per-step readout then runs as
    # a direct 3D ``bmm`` computing the same per-batch GEMM as the
    # equivalent 4D ``matmul``.
    _bh = batch * nheads
    A_col = A_c.repeat(batch).reshape(_bh, 1, 1, 1)          # [bh, 1, 1, 1]
    D_col = D_c.repeat(batch).reshape(_bh, 1, 1)             # [bh, 1, 1]
    S_flat = S.reshape(_bh, headdim, d_state)

    # Slab the precomputed factors for long sequences; the recurrent state
    # carries across slabs, so the values do not depend on the slab size.
    # Only the outer product is held at slab size, so the whole element
    # budget applies to it. The head expansion (permute+reshape, or the
    # ``expand`` for grouped B/C) is built per slab from the 4D inputs
    # instead of flattening the whole sequence up front, so the flattened
    # copies stay slab-bounded as well (identical copies of the same
    # values).
    slab = max(1, _SCAN_NAIVE_SLAB_ELEMS
               // max(1, batch * nheads * headdim * d_state))
    # Autograd-safe out-of-place loop while a graph is being recorded; the
    # in-place loop (same values) otherwise. The in-place loop mutates its
    # state argument, so it gets a dedicated copy and the caller's
    # ``initial_state`` stays untouched.
    use_inplace = not torch.is_grad_enabled()
    loop = None if use_inplace else _get_scripted_scan_loop()
    loop_inplace = _get_scripted_scan_loop_inplace() if use_inplace else None
    # The in-place loop mutates its state argument; only a caller-provided
    # ``initial_state`` needs the defensive copy (a zero state built by the
    # validator is already exclusively owned here).
    if use_inplace and initial_state is not None:
        S_flat = S_flat.clone()

    # Preallocated output: each slab is written straight into its time slice
    # (index assignment is elementwise-identical to the former
    # append-and-cat), so the slab temporaries are released as the loop
    # advances instead of being held until the final concatenation.
    y_flat = torch.empty(_bh, seqlen, headdim,
                         dtype=S_flat.dtype, device=S_flat.device)
    for lo in range(0, seqlen, slab):
        hi = min(seqlen, lo + slab)
        n_slab = hi - lo
        if heads_per_group == 1:
            # No head expansion to do: the group C/B tensors ARE the
            # per-head ones.
            B_h_slab = (B_c[:, lo:hi].permute(0, 2, 1, 3)
                        .reshape(_bh, n_slab, d_state))
            C_h_slab = (C_c[:, lo:hi].permute(0, 2, 1, 3)
                        .reshape(_bh, n_slab, d_state))
        else:
            B_h_slab = (B_c[:, lo:hi].permute(0, 2, 1, 3).unsqueeze(2)
                        .expand(batch, ngroups, heads_per_group, n_slab,
                                d_state)
                        .reshape(_bh, n_slab, d_state))
            C_h_slab = (C_c[:, lo:hi].permute(0, 2, 1, 3).unsqueeze(2)
                        .expand(batch, ngroups, heads_per_group, n_slab,
                                d_state)
                        .reshape(_bh, n_slab, d_state))
        x_flat_slab = (x_c[:, lo:hi].permute(0, 2, 1, 3)
                       .reshape(_bh, n_slab, headdim))
        dt_flat_slab = (dt_c[:, lo:hi].permute(0, 2, 1)
                        .reshape(_bh, n_slab, 1, 1))
        x_outer = (x_flat_slab.unsqueeze(-1)
                   * B_h_slab.unsqueeze(-2))                 # [bh, slab, D, N]
        # ``dt * x_outer`` in place: ``x_outer`` was allocated just above
        # and has no other readers.
        x_dt = x_outer.mul_(dt_flat_slab)                    # [bh, slab, D, N]
        decay_slab = torch.exp(dt_flat_slab * A_col)         # [bh, slab, 1, 1]
        C_slab = C_h_slab.unsqueeze(-2)                      # [bh, slab, 1, N]
        Dx_slab = (D_col * x_flat_slab).unsqueeze(-2)        # [bh, slab, 1, D]
        if use_inplace:
            fn = (loop_inplace if loop_inplace is not None
                  else _mamba2_scan_loop_inplace_eager)
            y_slab, S_flat = fn(x_dt, decay_slab, C_slab, Dx_slab, S_flat)
        elif loop is not None:
            y_slab, S_flat = loop(x_dt, decay_slab, C_slab, Dx_slab, S_flat)
        else:
            y_slab, S_flat = _mamba2_scan_loop_eager(
                x_dt, decay_slab, C_slab, Dx_slab, S_flat)
        y_flat[:, lo:hi] = y_slab
    # ``transpose`` after the reshape yields the target [B, T, H, D] layout
    # as a view; the single ``to`` cast then produces the output in one
    # pass. Values are unchanged; callers that need contiguity can call
    # ``.contiguous()`` themselves.
    y = (y_flat.reshape(batch, nheads, seqlen, headdim)
         .transpose(1, 2).to(x.dtype))
    S = S_flat.reshape(batch, nheads, headdim, d_state)
    return y, (S if output_final_state else None)


def quadratic_mamba2_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dual (masked-attention) evaluation of the same Mamba-2 (SSD) operator.

    State-space duality (arXiv:2405.21060 Sections 3-4): the diagonal-SSM
    recurrence computed by :func:`naive_mamba2_scan`,

        ``a_t = exp(dt_t * A)``, ``S_t = a_t * S_{t-1} + dt_t * (x_t (x) B_t)``,
        ``y_t = C_t . S_t + D * x_t``,

    is mathematically identical to a *causal masked attention* over the
    input with a semi-separable decay mask. With per-head cumulative
    log-decays ``G_t = sum_{r<=t} dt_r * A`` the unrolled output is::

        y_t = sum_{s<=t} exp(G_t - G_s) * dt_s * (C_t . B_s) * x_s
              + exp(G_t) * (C_t . S_0)            [initial-state term]
              + D * x_t                           [skip connection]

    and the final state follows from the same pairwise weights at the
    final position::

        S_T = exp(G_{T-1}) * S_0
              + sum_s exp(G_{T-1} - G_s) * dt_s * (x_s (x) B_s)

    This function evaluates those closed forms directly (cumsum, matmuls,
    one elementwise ``exp``) — a mathematically INDEPENDENT code path from
    the sequential recurrence, sharing only the input contract (the same
    validator). It exists to pin the recurrence's semantics as the
    SSD-duality oracle in ``test_ssd_duality.py`` and as a reference for
    future chunked/parallel implementations; it is
    O(T^2) in both time and memory and is NOT a production path.

    Shapes, validation, dtype contract, and return values are identical to
    :func:`naive_mamba2_scan`; with ``initial_state=None`` the initial
    term is skipped (the state starts at zeros).
    """
    v = _validate_mamba2_scan_inputs(
        x, dt, B, C, A, D, initial_state, 'quadratic_mamba2_scan')
    batch, seqlen = v['batch'], v['seqlen']
    nheads, headdim = v['nheads'], v['headdim']
    heads_per_group = v['heads_per_group']
    compute_dtype = v['compute_dtype']
    S0 = v['state']

    if seqlen == 0:
        y = x.new_zeros(batch, 0, nheads, headdim)
        return y, (S0 if output_final_state else None)

    x_c = x.to(compute_dtype)
    dt_c = dt.to(compute_dtype)
    B_c = B.to(compute_dtype)
    C_c = C.to(compute_dtype)
    A_c = A.to(compute_dtype)
    D_c = D.to(compute_dtype)

    B_h = _expand_heads(B_c, heads_per_group)  # [B, T, H, N]
    C_h = _expand_heads(C_c, heads_per_group)  # [B, T, H, N]

    # Per-head cumulative log-decay: G[b, t, h] = sum_{r<=t} dt_r * A_h.
    # A < 0 and dt >= 0 make G non-increasing, so every exponent used
    # below is <= 0 and the exponentials cannot overflow.
    log_a = dt_c * A_c                                   # [B, T, H]
    G = torch.cumsum(log_a, dim=1)                       # [B, T, H]

    # Pairwise semi-separable decay for s <= t: exp(G_t - G_s). The
    # non-causal entries are masked BEFORE the exp (a positive exponent
    # would overflow to inf and inf * 0 = NaN could then poison the masked
    # entries). Below-diagonal entries become exactly 0.
    diff = G.unsqueeze(2) - G.unsqueeze(1)               # [B, T, T, H]
    noncausal = _noncausal_chunk_mask(seqlen, x.device)
    # [b, t, s, h] = (C_t . B_s) * dt_s; broadcast dt over the readout
    # position t via an extra singleton dim (dim 1).
    cb = torch.einsum('bthn,bshn->btsh', C_h, B_h)
    if torch.is_grad_enabled():
        decay = torch.exp(
            diff.masked_fill(noncausal[None, :, :, None], float('-inf')))
        weights = decay * cb * dt_c.unsqueeze(1)
    else:
        # ``diff`` / ``cb`` are fresh tensors of this call; outside autograd
        # the identical elementwise chain runs in place (same association
        # order for the product).
        decay = diff.masked_fill_(
            noncausal[None, :, :, None], float('-inf')).exp_()
        weights = decay.mul_(cb).mul_(dt_c.unsqueeze(1))
    y = torch.einsum('btsh,bshd->bthd', weights, x_c)
    if initial_state is not None:
        y = y + torch.exp(G).unsqueeze(-1) * torch.einsum(
            'bthn,bhdn->bthd', C_h, S0)
    y = (y + x_c * D_c.view(1, 1, -1, 1)).to(x.dtype)
    if not output_final_state:
        return y, None

    # Final state: the same pairwise weights evaluated at t = T-1 (the
    # last row of ``decay``), accumulated into the per-head outer products.
    w_final = torch.exp(G[:, -1].unsqueeze(1) - G) * dt_c  # [B, T, H]
    S = torch.einsum('bsh,bshd,bshn->bhdn', w_final, x_c, B_h)
    if initial_state is not None:
        S = S + torch.exp(G[:, -1]).unsqueeze(-1).unsqueeze(-1) * S0
    # ``torch.autocast`` intercepts the einsum above (matmul-family ops are
    # autocast-eligible) and would return the state on the reduced-precision
    # grid, while the sequential and chunked paths accumulate the state in
    # compute precision. Cast back so all three views honor the SAME
    # documented state-precision contract (fp32 for fp16/bf16/fp32 inputs,
    # fp64 for fp64 inputs) under explicit dtypes AND under AMP autocast.
    S = S.to(compute_dtype)
    return y, S


# ---------------------------------------------------------------------------
# Chunked SSD path: quadratic intra-chunk + associative scan over chunks
# ---------------------------------------------------------------------------
def _assoc_scan_chunk_transitions(
    a: torch.Tensor,
    b: torch.Tensor,
    S0: torch.Tensor,
    _owns_inputs: bool = False,
) -> tuple:
    """TorchScript-compatible associative scan of scalar chunk transitions.

    Each chunk ``n`` carries an affine transition on the recurrent state::

        S_{n+1} = a_n S_n + b_n

    with a SCALAR ``a_n`` of shape ``[BH, NT]`` (the product of the per-step
    decays ``exp(dt_t * A_h)`` over the chunk) and ``b_n`` of shape
    ``[BH, NT, D, N]`` (the chunk's own source contribution). Scalar-affine
    maps compose associatively::

        (a_2, b_2) ∘ (a_1, b_1) = (a_2 a_1, a_2 b_1 + b_2)

    so the incoming state of every chunk is recovered by a Hillis–Steele
    (doubling) prefix scan: ``ceil(log2 NT)`` levels, each level one
    elementwise multiply plus one fused multiply-add over
    ``[BH, NT, D, N]``-sized tensors. This is the Mamba-2 analogue of
    ``ops_kda._assoc_scan_chunk_states`` (there the transition is a dense
    ``K×K`` matrix; here the SSD decay is scalar-per-head, so the combine
    stays elementwise).

    Returns:
        ``(S_in, S_end, levels)`` with ``S_in`` shaped ``[BH, NT, D, N]``
        (the state entering each chunk, i.e. ``S_in[:, 0] == S0``),
        ``S_end`` the state after the last chunk, and ``levels`` the number
        of doubling steps (``ceil(log2 NT)``, 0 for ``NT == 1``).
    """
    A = a
    BB = b
    if not torch.is_grad_enabled() and not _owns_inputs:
        # The doubling levels below overwrite the tail in place; the
        # caller's tensors must not be touched (the eager/scripted parity
        # tests reuse the same inputs), so work on dedicated copies.
        # ``_owns_inputs=True`` marks the private call site, which passes
        # freshly-built tensors that are never read again afterwards (the
        # in-place tail writes are then pure buffer reuse).
        A = a.clone()
        BB = b.clone()
    off = 1
    levels = 0
    NT = a.shape[1]
    while off < NT:
        levels = levels + 1
        A_tail = A[:, off:]
        A_shift = A[:, :-off]
        BB_tail = BB[:, off:]
        BB_shift = BB[:, :-off]
        # NOTE: compose from the OLD slices, then overwrite (mirrors
        # ``ops_kda._assoc_scan_chunk_states``).
        new_A_tail = A_tail * A_shift
        new_BB_tail = (A_tail.unsqueeze(-1).unsqueeze(-1) * BB_shift
                       + BB_tail)
        if torch.is_grad_enabled():
            # Autograd keeps the out-of-place form: the cat produces a
            # fresh tensor per level, so no saved operand is mutated.
            A = torch.cat([A[:, :off], new_A_tail], dim=1)
            BB = torch.cat([BB[:, :off], new_BB_tail], dim=1)
        else:
            # Prefix unchanged, tail replaced by the composed values
            # computed above from the OLD tail: identical to the cat.
            A[:, off:] = new_A_tail
            BB[:, off:] = new_BB_tail
        off = off * 2
    # Inclusive prefixes after the scan: A[:, n] / BB[:, n] describe the
    # composed map of chunks 0..n. Exclusive form for S_in: chunk 0 sees S0
    # exactly, chunk n >= 1 consumes the prefix of chunks 0..n-1.
    S_end = A[:, -1].unsqueeze(-1).unsqueeze(-1) * S0 + BB[:, -1]
    S_in = torch.cat(
        [S0.unsqueeze(1),
         A[:, :-1].unsqueeze(-1).unsqueeze(-1) * S0.unsqueeze(1)
         + BB[:, :-1]],
        dim=1)
    return S_in, S_end, levels


_CHUNK_TRANSITION_SCAN_SCRIPTED = None
_CHUNK_TRANSITION_SCAN_FAILED = False


def _get_scripted_chunk_transition_scan():
    """Scripted chunk-transition scan core, with eager fallback.

    Mirrors :func:`_get_scripted_scan_loop`: the scripted function executes
    the same operations in the same order (bit-identical values), and
    scripting unavailability is recorded so the eager function is used.
    """
    global _CHUNK_TRANSITION_SCAN_SCRIPTED, _CHUNK_TRANSITION_SCAN_FAILED
    if (_CHUNK_TRANSITION_SCAN_SCRIPTED is None
            and not _CHUNK_TRANSITION_SCAN_FAILED):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                _CHUNK_TRANSITION_SCAN_SCRIPTED = torch.jit.script(
                    _assoc_scan_chunk_transitions)
        except Exception:
            _CHUNK_TRANSITION_SCAN_FAILED = True
    return _CHUNK_TRANSITION_SCAN_SCRIPTED


def chunked_mamba2_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked SSD evaluation: quadratic intra-chunk + scanned inter-chunk.

    A THIRD numerically independent path for the same Mamba-2 (SSD)
    operator as :func:`naive_mamba2_scan` (sequential recurrence) and
    :func:`quadratic_mamba2_scan` (global O(T²) dual form). It splits the
    sequence into chunks and evaluates each term of the duality expansion
    where it is numerically well-conditioned:

      * **intra-chunk outputs** with the masked quadratic form restricted to
        the chunk (``O(C²)`` per chunk instead of ``O(T²)`` globally),
      * **inter-chunk state** via the affine chunk recurrence
        ``S_{n+1} = a_n S_n + b_n`` with the SCALAR per-head transition
        ``a_n = exp(Σ_chunk dt·A)`` and the chunk source ``b_n``, resolved
        by the Hillis–Steele associative scan in
        :func:`_assoc_scan_chunk_transitions` (``ceil(log2 NT)`` batched
        levels).

    Log-space gating
    ----------------
    All exponents are anchored at the *state before the chunk starts*:
    with ``G_t = Σ_{r≤t} dt_r A_h`` (non-increasing because ``A < 0`` and
    ``dt ≥ 0``) the code forms ``g_c = G_{t₀+c} - G_{t₀-1}`` per chunk, so

      * every intra-chunk exponent ``g_c - g_s`` (``s ≤ c``) is ``≤ 0``,
      * the chunk transition is a single ``exp(g_last)``, and
      * the inter-chunk state readout is ``exp(g_c)``.

    Anchoring at the chunk boundary avoids differencing two large negative
    cumulative log-decays (the global dual form computes ``G_t - G_s`` over
    the whole sequence), keeping the absolute magnitude of every ``exp``
    argument bounded by the chunk's own decay budget.

    Equivalence / contract
    ----------------------
    The result matches :func:`naive_mamba2_scan` and
    :func:`quadratic_mamba2_scan` to fp round-off (different summation
    order; pinned at fp64 machine precision by ``test_chunked_ssd.py``).
    Shapes, validation, dtype contract, and return values are identical to
    the other two views (one shared validator). ``T`` not divisible by
    ``chunk_size`` is padded with zero ``dt``/``x`` on the right; the
    padded steps leave the final state unchanged and are sliced off the
    output. The non-increasing-``G`` assumption (``A < 0``, ``dt ≥ 0``) is
    the same contract the dual form documents. When the chunk-transition
    source would exceed the shared slab element budget (e.g.
    ``chunk_size=1`` on a very long sequence) the call falls back to
    :func:`naive_mamba2_scan` — the sequential reference for the same
    operator and the same tolerance contract — instead of materializing
    the full-sequence source.

    Args:
        chunk_size: positions per chunk (must be a positive int). ``1``
            degenerates to a per-token associative scan; ``>= T`` runs one
            quadratic chunk (a numerically reorganized dual form).
    """
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool)
            or chunk_size < 1):
        raise ValueError(
            f"chunked_mamba2_scan: chunk_size={chunk_size!r} must be a "
            f"positive int.")
    v = _validate_mamba2_scan_inputs(
        x, dt, B, C, A, D, initial_state, 'chunked_mamba2_scan')
    batch, seqlen = v['batch'], v['seqlen']
    nheads, headdim = v['nheads'], v['headdim']
    ngroups, d_state = v['ngroups'], v['d_state']
    heads_per_group = v['heads_per_group']
    compute_dtype = v['compute_dtype']
    S0 = v['state']

    if seqlen == 0:
        y = x.new_zeros(batch, 0, nheads, headdim)
        return y, (S0 if output_final_state else None)

    x_c = x.to(compute_dtype)
    dt_c = dt.to(compute_dtype)
    B_c = B.to(compute_dtype)
    C_c = C.to(compute_dtype)
    A_c = A.to(compute_dtype)
    D_c = D.to(compute_dtype)

    B_h = _expand_heads(B_c, heads_per_group)             # [B, T, H, N]
    C_h = _expand_heads(C_c, heads_per_group)

    # Cumulative log-decay G_t = Σ_{r≤t} dt_r A_h (non-increasing).
    log_a = dt_c * A_c                                    # [B, T, H]
    G = torch.cumsum(log_a, dim=1)

    BT = chunk_size
    pad = (-seqlen) % BT
    Tp = seqlen + pad
    NT = Tp // BT
    # The transition source ``b_chunk`` ([B, H, NT, D, N]) and the
    # Hillis–Steele scan buffers ([BH, NT, D, N], 2–3 live copies) are
    # allocated whole; only the intra-chunk term is slabbed. With
    # ``chunk_size=1`` the source degenerates to the full-sequence dual
    # tensor that :func:`naive_mamba2_scan` deliberately slabs, so past
    # the shared slab budget fall back to that sequential reference.
    # Splitting the transition scan across slabs instead would change the
    # Hillis–Steele association order, so the fallback is the only
    # semantic-preserving bound; both paths are the same operator's
    # references (the fp64 tolerance contract pinned by
    # ``test_chunked_ssd.py``).
    if batch * nheads * NT * headdim * d_state > _SCAN_XOUTER_SLAB_ELEMS:
        return naive_mamba2_scan(
            x, dt, B, C, A, D,
            initial_state=initial_state,
            output_final_state=output_final_state)
    if pad:
        # A padded step has dt=0, so the cumulative log-decay must stay
        # CONSTANT (not jump to 0): extend G with a copy of its last value.
        G = torch.cat([G, G[:, -1:].expand(batch, pad, nheads)], dim=1)
        x_c = F.pad(x_c, (0, 0, 0, 0, 0, pad))
        dt_c = F.pad(dt_c, (0, 0, 0, pad))
        B_h = F.pad(B_h, (0, 0, 0, 0, 0, pad))
        C_h = F.pad(C_h, (0, 0, 0, 0, 0, pad))
    # Cumulative log-decay at the START of each chunk (the state level at
    # which chunk n begins): 0 for chunk 0, ``G[:, n*BT - 1]`` otherwise.
    # Only these NT boundary values are read below (``G_prev[:, t0]``), so
    # they are gathered directly instead of materializing the whole
    # ``[B, T, H]`` position-shifted sequence. The gathered entries are the
    # same tensor elements as before.
    G_chunk_start = torch.cat(
        [torch.zeros(batch, 1, nheads, device=x.device, dtype=compute_dtype),
         G[:, BT - 1:Tp - BT:BT, :]], dim=1)                # [B, NT, H]

    # Chunk-major views: [B, H, NT, C, ...] / [B, H, NT, C].
    G_r = G.reshape(batch, NT, BT, nheads).permute(0, 3, 1, 2)
    x_r = x_c.reshape(batch, NT, BT, nheads, headdim).permute(0, 3, 1, 2, 4)
    dt_r = dt_c.reshape(batch, NT, BT, nheads).permute(0, 3, 1, 2)
    Bh_r = B_h.reshape(batch, NT, BT, nheads, d_state).permute(0, 3, 1, 2, 4)
    Ch_r = C_h.reshape(batch, NT, BT, nheads, d_state).permute(0, 3, 1, 2, 4)

    # Chunk-local log-gating: g_c = G_{t0+c} - G_{t0-1} <= 0.
    g = G_r - G_chunk_start.permute(0, 2, 1).unsqueeze(-1)  # [B, H, NT, C]

    # --- intra-chunk quadratic term -----------------------------------
    # decay[t, s] = exp(G_t - G_s) for s <= t inside the chunk. The strict
    # upper triangle is masked to -inf BEFORE the exp (a positive exponent
    # would overflow and inf * 0 = NaN), exactly like the dual form.
    # The [B,H,NT,C,C] intermediates are evaluated in chunk slabs: every
    # chunk's intra-chunk term is independent (the inter-chunk state is
    # applied separately below), so the slab size changes only the transient
    # peak memory, never the values.
    noncausal = _noncausal_chunk_mask(BT, x.device)
    valid = None
    if pad:
        valid = (torch.arange(Tp, device=x.device) < seqlen).reshape(NT, BT)
    slab_n = max(1, _SCAN_INTRA_SLAB_ELEMS
                 // max(1, batch * nheads * BT * BT))
    # Preallocated output written slab-by-slab (elementwise-identical to the
    # former append-and-cat), so only one slab-sized transient is live.
    y_intra = torch.empty(batch, nheads, NT, BT, headdim,
                          dtype=compute_dtype, device=x.device)
    for lo in range(0, NT, slab_n):
        hi = min(NT, lo + slab_n)
        g_slab = g[:, :, lo:hi]
        diff = g_slab.unsqueeze(-1) - g_slab.unsqueeze(-2)  # [B,H,slab,C,C]
        # ``diff`` was allocated just above and has no other readers; the
        # mask writes the same ``-inf`` entries in place (before the ``exp``,
        # as required).
        diff.masked_fill_(noncausal, float('-inf'))
        # cb[c, s] = C_c · B_s (per chunk); the readout is contracted with
        # the source x_s scaled by dt_s (dt at the SOURCE position).
        cb = torch.einsum(
            'bhncN,bhnsN->bhncs', Ch_r[:, :, lo:hi], Bh_r[:, :, lo:hi])
        # ``decay`` / ``cb`` have no other readers. The same products in the
        # same association are evaluated in place to avoid a second
        # slab-sized temporary; in-place chained multiplies are only safe
        # with autograd disabled (a saved operand would be mutated
        # afterwards), so the graph-recording path keeps the out-of-place
        # form.
        if torch.is_grad_enabled():
            decay = torch.exp(diff)
            weights = decay * cb
            if pad:
                weights = weights * valid[None, None, lo:hi, None, :]
            weights = weights * dt_r[:, :, lo:hi].unsqueeze(-2)
        else:
            weights = torch.exp(diff, out=diff)
            weights.mul_(cb)
            if pad:
                weights.mul_(valid[None, None, lo:hi, None, :])
            weights.mul_(dt_r[:, :, lo:hi].unsqueeze(-2))
        y_intra[:, :, lo:hi] = torch.einsum(
            'bhncs,bhnsd->bhncd', weights, x_r[:, :, lo:hi])

    # --- chunk transition (a_n) and source (b_n) ----------------------
    # b_n[d, N] = Σ_s exp(g_last - g_s) dt_s x_s[d] B_h[s, N]
    w_src = torch.exp(g[:, :, :, -1:] - g) * dt_r
    if pad:
        w_src = w_src * valid[None, None]
    b_chunk = torch.einsum('bhnc,bhncd,bhncN->bhndN', w_src, x_r, Bh_r)
    # a_n = Π exp(dt·A) over the chunk = exp(G_last - G_prev[chunk start]).
    a_chunk = torch.exp(g[:, :, :, -1])                   # [B, H, NT]

    BH = batch * nheads
    scan = _get_scripted_chunk_transition_scan()
    # ``_owns_inputs=True``: the reshaped ``a_chunk`` / ``b_chunk`` views are
    # consumed only by the scan, so its no-grad in-place tail writes reuse
    # their storage instead of cloning first.
    scan_args = (a_chunk.reshape(BH, NT),
                 b_chunk.reshape(BH, NT, headdim, d_state),
                 S0.reshape(BH, headdim, d_state), True)
    if scan is not None:
        S_in, S_end, _levels = scan(*scan_args)
    else:
        S_in, S_end, _levels = _assoc_scan_chunk_transitions(*scan_args)
    S_in_r = S_in.reshape(batch, nheads, NT, headdim, d_state)

    # --- inter-chunk readout ------------------------------------------
    # y_inter[t] = exp(g_c) * (C_t · S_in_n): the state entering the chunk
    # (including the initial state and all earlier chunks) decayed to the
    # current position.
    y_inter = torch.exp(g).unsqueeze(-1) * torch.einsum(
        'bhncN,bhndN->bhncd', Ch_r, S_in_r)

    y_r = (y_intra + y_inter
           + x_r * D_c.reshape(1, nheads, 1, 1, 1))
    y = (y_r.permute(0, 2, 3, 1, 4)
         .reshape(batch, Tp, nheads, headdim)[:, :seqlen])
    y = y.to(x.dtype)
    S = S_end.reshape(batch, nheads, headdim, d_state)
    return y, (S if output_final_state else None)


# ---------------------------------------------------------------------------
# Unified mixer module
# ---------------------------------------------------------------------------
class Mamba2Mixer(nn.Module):
    """Reference Mamba-2 mixer with the repository's memory-layer contract.

    The module exposes the SAME interface family as
    ``ops_fused.KDAHybridLayer`` so the hybrid stack can treat KDA and
    Mamba-2 as interchangeable *recurrent memory* layers:

      * ``forward(x, state=None) -> (out, new_state)`` — stateful wrapper
        that persists the short-conv state in ``self._conv_lookback``
        (registered non-persistent buffer, mirrors KDA).
      * ``forward_functional(x, state, conv_state, *, detach_lookback=True)
        -> (out, new_state, new_conv_state)`` — pure path for DDP /
        ``torch.compile`` / checkpointing / concurrent inference.
      * ``reset_conv_state()`` — clear the streaming conv state between
        independent sequences.
      * ``prefill(x) -> (out, state_dict)`` /
        ``step(x_t, state_dict) -> (out_t, state_dict)`` — the framework's
        unified inference protocol. ``state_dict`` is a plain
        ``{'ssm_state': ..., 'conv_state': ...}`` mapping; both entries are
        detached so inference state never retains a training graph. This
        mirrors the spec's Mixer contract (full-sequence ``forward`` /
        prefix ``prefill`` / single-token ``step``) while CSA/HCA keep their
        existing ``ops_decoding_cache`` incremental caches.

    Args:
        cfg: a ``HybridConfig``-like namespace. Required attributes:
            ``d_model``; optional (with defaults shown):
            ``mamba_d_state=64``, ``mamba_d_conv=4``, ``mamba_expand=2``,
            ``mamba_headdim=64``, ``mamba_ngroups=1``,
            ``mamba_dt_rank='auto'``, ``mamba_backend='reference'``,
            ``mamba_chunk_size=0`` (0 = sequential reference scan;
            positive selects :func:`chunked_mamba2_scan`).
            Duck-typing (instead of importing ``HybridConfig``) avoids an
            import cycle: ``ops_fused`` imports this module for the mixer
            factory, so this module must not import ``ops_fused``.
    """

    def __init__(self, cfg):
        super().__init__()
        d_model = getattr(cfg, 'd_model', None)
        if not isinstance(d_model, int) or d_model < 1:
            raise ValueError(
                f"Mamba2Mixer: cfg.d_model={d_model!r} must be a positive int.")
        self.d_model = d_model
        self.d_state = int(getattr(cfg, 'mamba_d_state', 64))
        self.d_conv = int(getattr(cfg, 'mamba_d_conv', 4))
        self.expand = int(getattr(cfg, 'mamba_expand', 2))
        self.headdim = int(getattr(cfg, 'mamba_headdim', 64))
        self.ngroups = int(getattr(cfg, 'mamba_ngroups', 1))
        dt_rank_cfg = getattr(cfg, 'mamba_dt_rank', 'auto')
        # Backend selection is validated eagerly (typo -> immediate error,
        # before any parameters are allocated) but the optional official
        # package import stays lazy — mirroring the KDA backend contract in
        # HybridConfig.__post_init__ / ops_kda_backend.
        from ops_mamba_backend import validate_mamba_backend
        self.backend = validate_mamba_backend(
            getattr(cfg, 'mamba_backend', 'reference'))
        # Chunked-SSD scan knob (analogue of KDA's ``kda_assoc_scan``):
        # 0 (default) keeps the sequential reference recurrence; a positive
        # value selects :func:`chunked_mamba2_scan` with that chunk size
        # (chunk-local log-space gating + a Hillis-Steele scan over the
        # scalar chunk transitions). Both paths match to fp round-off;
        # ``scan_mode`` records which one produced the outputs.
        raw_chunk = getattr(cfg, 'mamba_chunk_size', 0)
        if isinstance(raw_chunk, bool) or not isinstance(raw_chunk, int) \
                or raw_chunk < 0:
            raise ValueError(
                f"Mamba2Mixer: mamba_chunk_size={raw_chunk!r} must be a "
                f"non-negative int (0 = sequential reference scan).")
        self.chunk_size = raw_chunk
        self.scan_mode = 'chunked' if raw_chunk > 0 else 'sequential'

        geo = validate_mamba2_geometry(
            d_model, self.d_state, self.d_conv, self.expand,
            self.headdim, self.ngroups, dt_rank_cfg)
        self.d_inner = geo['d_inner']
        self.nheads = geo['nheads']
        self.heads_per_group = geo['heads_per_group']
        self.dt_rank = geo['dt_rank']
        # Record which implementation actually produced outputs. The
        # reference mixer is always 'reference'; the official wrapper in
        # ops_mamba_backend sets its own value. Experiment runners serialize
        # this field so a result row never silently reports the wrong path.
        self.effective_backend = 'reference'

        # --- parameters (names match the official mamba_ssm Mamba2) -------
        # in_proj produces [x, z]: 2 * d_inner, no bias (official contract).
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # Causal depthwise short conv over the x half. padding=0: the causal
        # left context is applied explicitly in forward (F.pad / concat with
        # the streaming conv state) so position t NEVER sees t+1 — the same
        # explicit-causality convention as KDAHybridLayer's short_conv.
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=self.d_conv,
            padding=0, groups=self.d_inner, bias=True)
        # x_proj maps the activated x to [dt(low-rank), B, C].
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.ngroups * self.d_state,
            bias=False)
        # dt low-rank projection to one scalar per head, with bias. The bias
        # is initialized so softplus(bias) lands in [dt_min, dt_max]; the
        # weight starts at zero (official Mamba2Simple behaviour), so at
        # init every head gets a stable, diverse, input-independent dt.
        self.dt_proj = nn.Linear(self.dt_rank, self.nheads, bias=True)
        # Per-head negative log-decay and skip scales.
        A = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_dt()
        # Persistent conv state: the last (d_conv - 1) in_proj features
        # ("conv lookback"), shape [batch, d_inner, d_conv - 1]. Registered
        # as a non-persistent buffer so .to(device)/.half() move it and
        # state_dict skips it (runtime state, reset via reset_conv_state).
        self.register_buffer('_conv_lookback', None, persistent=False)

    def _init_dt(self) -> None:
        """Initialize ``dt_proj`` following the official Mamba-2 scheme.

        ``dt ~ exp(U(log dt_min, log dt_max))`` clamped to a floor, and the
        bias is set to ``log(expm1(dt))`` so that ``softplus(bias) == dt``
        exactly at initialization. Only the reference module performs this
        init; the official backend initializes itself and receives weights
        via state_dict transfer.
        """
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(self.nheads)
                       * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj.weight.zero_()

    # ------------------------------------------------------------------
    # Streaming-state helpers
    # ------------------------------------------------------------------
    def reset_conv_state(self) -> None:
        """Clear the persistent short-conv state.

        Called between independent sequences (the stack's ``reset_state``
        calls this for every memory layer) so the conv does not "see" tokens
        from the previous sequence as left context.
        """
        self._conv_lookback = None

    def state_nbytes(self, batch: int = 1) -> dict:
        """Analytic per-layer runtime state size in bytes (fp32 states).

        The SSM state is stored in compute precision (fp32 for fp16/bf16
        inputs); the conv state lives in the input dtype. This helper
        reports the fp32 figure used by ``run_kv_cache``'s analytic
        accounting and by the budget-matching experiment design.
        """
        ssm_elems = batch * self.nheads * self.headdim * self.d_state
        conv_elems = batch * self.d_inner * (self.d_conv - 1)
        return {
            'ssm_state_bytes': ssm_elems * 4,
            'conv_state_bytes': conv_elems * 4,
            'total_bytes': (ssm_elems + conv_elems) * 4,
            'ssm_state_elements': ssm_elems,
            'conv_state_elements': conv_elems,
        }

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                state: torch.Tensor | None = None):
        """Stateful forward.

        Reads ``self._conv_lookback`` and writes the new conv state back to
        ``self``. Use :meth:`forward_functional` for the pure path (DDP,
        ``torch.compile``, checkpointing, concurrent inference).
        """
        o, new_state, new_conv = self.forward_functional(
            x, state, self._conv_lookback)
        self._conv_lookback = new_conv
        return o, new_state

    def forward_functional(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        conv_state: torch.Tensor | None,
        *,
        detach_lookback: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Pure functional forward — does NOT mutate ``self``.

        Args:
            x: ``[B, T, d_model]`` input hidden states.
            state: ``[B, nheads, headdim, d_state]`` SSM state from the
                previous call (compute precision), or ``None`` for a fresh
                sequence.
            conv_state: ``[B, d_inner, d_conv - 1]`` short-conv left context
                (the previous call's trailing in_proj features), or ``None``
                for a fresh sequence.
            detach_lookback: if True (default), the returned conv state is
                detached so the autograd graph of the current chunk is not
                retained by the next call. Set False for cross-chunk BPTT
                (mirrors KDAHybridLayer's contract).

        Returns:
            ``(output, new_ssm_state, new_conv_state)``:
              * ``output``: ``[B, T, d_model]`` (input dtype).
              * ``new_ssm_state``: ``[B, nheads, headdim, d_state]`` in
                compute precision (caller-owned; detach policy is the
                caller's, matching the KDA stack contract).
              * ``new_conv_state``: ``[B, d_inner, d_conv - 1]``.
        """
        B, T, d = x.shape
        if d != self.d_model:
            raise ValueError(
                f"Mamba2Mixer: input d_model={d} does not match the "
                f"configured d_model={self.d_model}.")
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        # Degenerate empty-sequence case (mirrors KDAHybridLayer): preserve
        # or initialize the SSM state, leave the conv state untouched, and
        # return a zero-length output.
        if T == 0:
            if state is not None and state.shape[0] == B:
                new_state = state.to(device=x.device, dtype=compute_dtype)
            else:
                new_state = torch.zeros(
                    B, self.nheads, self.headdim, self.d_state,
                    device=x.device, dtype=compute_dtype)
            new_conv = conv_state
            if new_conv is not None:
                if new_conv.shape[0] != B:
                    new_conv = None
                else:
                    new_conv = new_conv.to(device=x.device, dtype=x.dtype)
                    if detach_lookback:
                        new_conv = new_conv.detach()
                    new_conv = new_conv.clone()
            return x.new_zeros(B, 0, d), new_state, new_conv

        # Batch-size / device / dtype guards on the incoming conv state
        # (mirrors KDAHybridLayer's lookback guards).
        lookback = conv_state
        if lookback is not None:
            if lookback.shape[0] != B:
                lookback = None
            elif lookback.device != x.device or lookback.dtype != x.dtype:
                lookback = lookback.to(device=x.device, dtype=x.dtype)
                if detach_lookback:
                    lookback = lookback.detach()
            elif detach_lookback:
                lookback = lookback.detach()

        # in_proj -> [x, z]  (x first, z second — official split order).
        xz = self.in_proj(x)
        xh, z = xz.chunk(2, dim=-1)                    # [B, T, d_inner]

        # Causal depthwise conv over xh with explicit LEFT context.
        xt = xh.transpose(1, 2)                        # [B, d_inner, T]
        if lookback is not None:
            conv_in = torch.cat([lookback, xt], dim=2)
        else:
            conv_in = F.pad(xt, (self.d_conv - 1, 0))
        # conv_in has length (d_conv - 1) + T, so the padding-0 conv returns
        # exactly T positions — the causal window [t-d_conv+1, t] per step.
        conv_out = self.conv1d(conv_in)                # [B, d_inner, T]
        # ``.contiguous()`` after the transpose: the conv returns
        # ``[B, d_inner, T]`` while every downstream consumer (``x_proj``,
        # the scan's per-head view) expects ``[B, T, d_inner]``; materializing
        # the layout ONCE avoids a copy per consumer (values identical — a
        # pure layout change).
        # The conv output is fresh; outside autograd the activation runs in
        # place (identical values), the layout ops below are unchanged.
        if torch.is_grad_enabled():
            x_act = F.silu(conv_out).transpose(1, 2).contiguous()  # [B, T, d_inner]
        else:
            x_act = F.silu(conv_out, inplace=True).transpose(1, 2).contiguous()

        # New conv state: the last (d_conv - 1) conv INPUTS. Deriving it
        # from conv_in (not from xt) automatically handles T < d_conv - 1:
        # the window then still contains older tokens carried in the
        # incoming lookback (or the zero pad of a fresh sequence).
        new_conv = conv_in[:, :, -(self.d_conv - 1):]
        if detach_lookback:
            new_conv = new_conv.detach()
        new_conv = new_conv.clone()

        # dt / B / C from the activated x.
        x_dbl = self.x_proj(x_act)                     # [B, T, dt_rank + 2G*N]
        dt, Bm, Cm = torch.split(
            x_dbl, [self.dt_rank,
                    self.ngroups * self.d_state,
                    self.ngroups * self.d_state], dim=-1)
        dt = self.dt_proj(dt)                          # [B, T, nheads]
        # softplus in compute precision: fp16 softplus of a moderately
        # negative pre-activation can overflow/underflow; the state
        # trajectory must not inherit that noise.
        dt = F.softplus(dt.to(compute_dtype))
        A = -torch.exp(self.A_log.float())             # [nheads], negative

        scan_args = (
            x_act.view(B, T, self.nheads, self.headdim),
            dt,
            Bm.view(B, T, self.ngroups, self.d_state),
            Cm.view(B, T, self.ngroups, self.d_state),
            A, self.D,
        )
        if self.chunk_size > 0:
            y, new_state = chunked_mamba2_scan(
                *scan_args,
                initial_state=state,
                output_final_state=True,
                chunk_size=self.chunk_size,
            )
        else:
            y, new_state = naive_mamba2_scan(
                *scan_args,
                initial_state=state,
                output_final_state=True,
            )
        # Gate with SiLU(z) and project back to d_model. The scan returns
        # y in x.dtype; the gating + out_proj run in the input dtype so the
        # mixer's output dtype always matches its input dtype.
        y = y.reshape(B, T, self.d_inner)
        if torch.is_grad_enabled():
            y = y * F.silu(z)
        else:
            # ``z`` / ``y`` are fresh here (a chunk view of the fresh
            # in-projection output and the fresh scan output); outside
            # autograd the activation and product run in place with the
            # identical values.
            y = y.mul_(F.silu(z, inplace=True))
        out = self.out_proj(y)
        return out, new_state, new_conv

    # ------------------------------------------------------------------
    # Unified inference protocol (prefill / step)
    # ------------------------------------------------------------------
    def prefill(self, x: torch.Tensor):
        """Process a prefix and return ``(output, state_dict)``.

        ``state_dict`` is ``{'ssm_state': ..., 'conv_state': ...}`` with
        both entries detached — the inference protocol never retains a
        training graph (the spec's train/infer state-lifecycle separation).
        Feed ``state_dict`` back into :meth:`step` to decode continuations.
        """
        out, ssm_state, conv_state = self.forward_functional(
            x, None, None, detach_lookback=True)
        return out, {
            'ssm_state': ssm_state.detach() if ssm_state is not None else None,
            'conv_state': conv_state,
        }

    def step(self, x_t: torch.Tensor, state: dict):
        """Process a single token and return ``(output_t, state_dict)``.

        Args:
            x_t: ``[B, 1, d_model]`` (a one-token "chunk"; a T>1 tensor is
                rejected to keep the protocol unambiguous).
            state: the ``state_dict`` returned by a previous ``prefill`` /
                ``step`` call, or ``None`` (treated as a fresh sequence).
        """
        if state is not None and not isinstance(state, dict):
            raise TypeError(
                f"Mamba2Mixer.step: state must be the dict returned by "
                f"prefill/step (or None), got {type(state).__name__}.")
        if x_t.dim() != 3 or x_t.shape[1] != 1:
            raise ValueError(
                f"Mamba2Mixer.step: x_t must be [B, 1, d_model], got shape "
                f"{tuple(x_t.shape)}. Use forward_functional for chunks.")
        state = state or {}
        out, ssm_state, conv_state = self.forward_functional(
            x_t, state.get('ssm_state'), state.get('conv_state'),
            detach_lookback=True)
        return out, {
            'ssm_state': ssm_state.detach() if ssm_state is not None else None,
            'conv_state': conv_state,
        }


# ---------------------------------------------------------------------------
# Backend-dispatching factory
# ---------------------------------------------------------------------------
def build_mamba2_mixer(cfg):
    """Build a Mamba-2 mixer honouring ``cfg.mamba_backend``.

    * ``'reference'`` — :class:`Mamba2Mixer` (default; runs everywhere).
    * ``'official'`` — the ``mamba_ssm``-backed wrapper from
      ``ops_mamba_backend``. Requires the optional dependency AND CUDA;
      raises an actionable error when either is missing (an explicitly
      requested accelerated backend must never silently fall back).
    * ``'auto'`` — official when CUDA is available and ``mamba_ssm`` is
      importable, otherwise the reference path. The resolved backend is
      recorded on ``mixer.effective_backend`` for provenance.
    """
    from ops_mamba_backend import (
        build_official_mamba2_mixer, mamba2_official_available,
        validate_mamba_backend,
    )
    backend = validate_mamba_backend(getattr(cfg, 'mamba_backend', 'reference'))
    if backend == 'reference':
        return Mamba2Mixer(cfg)
    if backend == 'official':
        # Raises ImportError / RuntimeError with an actionable message when
        # the official package or CUDA is unavailable.
        return build_official_mamba2_mixer(cfg)
    # backend == 'auto'
    if torch.cuda.is_available() and mamba2_official_available():
        try:
            return build_official_mamba2_mixer(cfg)
        except (ImportError, RuntimeError):
            # Unavailable in practice (e.g. importable but broken install):
            # fall back to the reference path with a recorded marker. 'auto'
            # is allowed to fall back (mirrors kda_backend='auto'); only the
            # explicit 'official' request is a hard error.
            pass
    mixer = Mamba2Mixer(cfg)
    mixer.effective_backend = 'reference (auto fallback)'
    return mixer
