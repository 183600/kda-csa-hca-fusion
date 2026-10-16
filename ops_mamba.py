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

The sequential Python-loop scan is deliberately naive: this is the
*correctness reference* used to validate the official backend and to run
CPU/GPU-agnostic experiments, not a speed-optimized kernel. Speed comparisons
against KDA must use the official backend rows in ``run_benchmark.py``
(documented there as "current implementation" numbers, never as architecture
intrinsic costs).

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
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Mamba2Mixer",
    "build_mamba2_mixer",
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
# The step-invariant factors (the group-to-head expanded B/C and the outer
# products ``x (x) B``) are computed vectorized over the time axis. Every
# per-element operation is unchanged, so the result is bit-identical to the
# equivalent per-step evaluation. The per-step decay and
# ``dt * (x (x) B)`` product stay inside the loop because evaluating them
# there keeps autograd's reductions over time in order, leaving the
# ``A`` / ``dt`` gradients unchanged.
_SCAN_LOOP_SCRIPTED = None
_SCAN_LOOP_SCRIPTED_FAILED = False
# Cap on the number of elements in one slab of precomputed outer products
# (``[B, slab, nheads, headdim, d_state]``); long sequences are processed in
# several slabs, and the recurrent state carries across slab boundaries.
_SCAN_XOUTER_SLAB_ELEMS = 8_000_000


def _mamba2_scan_loop_eager(x_outer, dt_slab, A_c, C_h, D_c, x_slab, S):
    """Sequential scan body over one slab of precomputed factors.

    ``x_outer`` is ``[B, slab, H, D, N]`` (the ``x (x) B`` outer product),
    ``dt_slab`` is ``[B, slab, H]``, ``A_c`` is ``[H]`` (negative decay) and
    ``C_h`` is ``[B, slab, H, N]``. The per-step decay ``exp(dt * A)`` and
    the ``dt * x_outer`` product are kept inside the loop (not hoisted) so
    that autograd's per-element reduction order — and therefore the
    gradients w.r.t. ``A`` and ``dt`` — are unchanged. Returns
    ``(y_slab, S)`` with ``y_slab`` shaped ``[B, slab, H, D]``.
    """
    ys: list[torch.Tensor] = []
    T = x_outer.shape[1]
    for t in range(T):
        # Per-head decay exp(dt * A) in (0, 1): [batch, nheads]
        decay = torch.exp(dt_slab[:, t] * A_c)
        S = (decay[:, :, None, None] * S
             + dt_slab[:, t, :, None, None] * x_outer[:, t])
        y_t = (torch.einsum('bhdn,bhn->bhd', S, C_h[:, t])
               + D_c[:, None] * x_slab[:, t])
        ys.append(y_t)
    return torch.stack(ys, dim=1), S


def _get_scripted_scan_loop():
    """Lazily ``torch.jit.script`` the scan loop body, with eager fallback.

    The scripted function executes the same operations as
    :func:`_mamba2_scan_loop_eager`; it is cached and reused across calls.
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

    # Step-invariant factors hoisted out of the loop and computed vectorized
    # over the time axis (elementwise-identical to the per-step evaluation):
    #   * B_h / C_h: the group-level B/C expanded to per-head B/C. Heads are
    #     assigned to groups in CONTIGUOUS blocks (head h -> group
    #     h // heads_per_group); repeat_interleave reproduces the original
    #     per-step ``[:, t].repeat_interleave(heads_per_group, dim=1)``.
    #   * x_outer[b, t, h, d, n] = x[b, t, h, d] * B_h[b, t, h, n], the
    #     original per-step ``einsum('bhd,bhn->bhdn', x[:, t], B_h)``.
    # The per-step decay ``exp(dt * A)`` stays inside the loop so its
    # autograd accumulation order (and thus the A gradient) is unchanged.
    B_h_all = B_c.repeat_interleave(heads_per_group, dim=2)  # [B, T, H, N]
    C_h_all = C_c.repeat_interleave(heads_per_group, dim=2)  # [B, T, H, N]

    # Slab the outer products so their memory stays bounded on long
    # sequences; the recurrent state carries across slabs unchanged, so the
    # values do not depend on the slab size.
    slab = max(1, _SCAN_XOUTER_SLAB_ELEMS
               // max(1, batch * nheads * headdim * d_state))
    loop = _get_scripted_scan_loop()
    y_slabs: list[torch.Tensor] = []
    for lo in range(0, seqlen, slab):
        hi = min(seqlen, lo + slab)
        x_outer = (x_c[:, lo:hi].unsqueeze(-1)
                   * B_h_all[:, lo:hi].unsqueeze(-2))       # [B, slab, H, D, N]
        if loop is not None:
            y_slab, S = loop(x_outer, dt_c[:, lo:hi], A_c,
                             C_h_all[:, lo:hi], D_c, x_c[:, lo:hi], S)
        else:
            y_slab, S = _mamba2_scan_loop_eager(
                x_outer, dt_c[:, lo:hi], A_c,
                C_h_all[:, lo:hi], D_c, x_c[:, lo:hi], S)
        y_slabs.append(y_slab)
    y = (y_slabs[0] if len(y_slabs) == 1
         else torch.cat(y_slabs, dim=1)).to(x.dtype)
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

    B_h = B_c.repeat_interleave(heads_per_group, dim=2)  # [B, T, H, N]
    C_h = C_c.repeat_interleave(heads_per_group, dim=2)  # [B, T, H, N]

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
    causal = torch.tril(torch.ones(
        seqlen, seqlen, dtype=torch.bool, device=x.device))
    decay = torch.exp(
        diff.masked_fill(~causal[None, :, :, None], float('-inf')))

    # [b, t, s, h] = (C_t . B_s) * dt_s; broadcast dt over the readout
    # position t via an extra singleton dim (dim 1).
    cb = torch.einsum('bthn,bshn->btsh', C_h, B_h)
    weights = decay * cb * dt_c.unsqueeze(1)
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
    return y, S


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
            ``mamba_dt_rank='auto'``, ``mamba_backend='reference'``.
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
        """Stateful forward (backward-compatible wrapper).

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
        x_act = F.silu(conv_out).transpose(1, 2)       # [B, T, d_inner]

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
        # trajectory must not inherit that noise. The scan internally
        # upcasts everything to compute precision anyway, so passing the
        # already-upcast dt is a no-op there.
        dt = F.softplus(dt.to(compute_dtype))
        A = -torch.exp(self.A_log.float())             # [nheads], negative

        y, new_state = naive_mamba2_scan(
            x_act.view(B, T, self.nheads, self.headdim),
            dt,
            Bm.view(B, T, self.ngroups, self.d_state),
            Cm.view(B, T, self.ngroups, self.d_state),
            A, self.D,
            initial_state=state,
            output_final_state=True,
        )
        # Gate with SiLU(z) and project back to d_model. The scan returns
        # y in x.dtype; the gating + out_proj run in the input dtype so the
        # mixer's output dtype always matches its input dtype.
        y = y.reshape(B, T, self.d_inner)
        y = y * F.silu(z)
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
