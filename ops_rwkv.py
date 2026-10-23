"""RWKV-7 mixer (TimeMix) reference implementation for the hybrid stack.

This module adds **RWKV-7** as the third selectable *recurrent memory*
route (``memory_type='rwkv7'``), alongside KDA and Mamba-2. As with the
Mamba-2 integration, the mixer satisfies the repository's memory-layer
contract so :class:`ops_fused.HybridAttentionStack` can treat the three
memory operators as interchangeable:

  * ``forward(x, state=None) -> (out, new_state)`` — stateful wrapper that
    persists the token-shift state in ``self._shift_state`` (registered
    non-persistent buffer, mirrors KDA's ``_conv_lookback``).
  * ``forward_functional(x, state, conv_state, *, detach_lookback=True)
    -> (out, new_state, new_conv_state)`` — pure path for DDP /
    ``torch.compile`` / checkpointing / concurrent inference.
  * ``reset_conv_state()`` — clear the streaming shift state between
    independent sequences (called by the stack's ``reset_state``).
  * ``prefill(x) -> (out, state_dict)`` / ``step(x_t, state_dict)`` — the
    unified inference protocol (same shape as ``ops_mamba.Mamba2Mixer``).

Upstream bindings (the parameterization and initialization are pinned to
named upstream commits so results are reproducible against a concrete
implementation, per the experiment-design requirement):

  * **Parameterization + initialization**: RWKV-Vibe/rwkv-fla,
    ``fla/layers/rwkv7.py`` (``RWKV7Attention``) at commit
    ``074deaf640284b219f3767de8d1ad007b1049765`` (2025-07-28). The LoRA
    sub-module (zero down-projection / orthogonal up-projection with gain
    ``0.1 * sqrt(out/in)``) follows ``fla/layers/rwkv6.py``'s ``LoRA``
    class at the same commit.
  * **Recurrence semantics (cross-check)**: BlinkDL/RWKV-LM,
    ``RWKV-v7/rwkv_v7_demo.py`` (``RWKV_Tmix_x070`` + the pure-PyTorch
    ``RWKV7_OP`` reference path) at commit
    ``a963e4ad848426a7032fd6f7c54bdf59c11b67d7`` (2025-02-19). The two
    references agree exactly on the state update (see below).

Variant note — **rwkv7-nofirst**. Some RWKV-7 implementations carry a
cross-layer ``v_first`` value residual (the first RWKV block's ``v`` is
blended into later blocks' values through a low-rank gate). This module
deliberately implements the *clearly-labelled* **nofirst** variant: no
``v_first`` and no ``v_lora``, because in a hybrid stack the RWKV layers
are not necessarily adjacent (e.g. layout ``R R R CSA R``) or first (a
CSA/HCA layer may precede every RWKV layer), the residual's semantics do
not transfer across the interleaved CSA/HCA operators, and — most
importantly — a cross-layer tensor must never be mistaken for or stored
as part of the *recurrent* state that persists across generation steps.
``v_first`` is a within-one-forward side channel in upstream; omitting it
keeps the mixer's state exactly ``(matrix_state, shift_state)``.

The RWKV-7 **ChannelMix** is intentionally NOT implemented: the hybrid
stack fixes the outer network structure (pre-norm residual, per-layer
LayerNorm, no FFN in the current stack), and inserting RWKV's own FFN
would change the outer network simultaneously with the mixer swap. The
same rule governs the Mamba-2 route.

Recurrence (per head, ``K`` = key/head dim, ``V`` = value dim)
----------------------------------------------------------------

The per-token state is a matrix ``S ∈ R^{K x V}`` (rows = key dim,
columns = value dim — the **fla kernel layout**, so the optional FLA
backend is a direct pass-through with no transposes). With per-token
``r, k, kk, a, log_w ∈ R^K`` and ``v ∈ R^V``:

.. math::

    S_t = exp(log_w_t) \\odot_K S_{t-1}
          - (kk_t^T S_{t-1}) \\otimes (kk_t \\odot a_t)
          + k_t \\otimes v_t
    y_t = S_t^T r_t

Concretely, in the ``[K, V]`` layout:

.. code-block:: python

    contract = -(kk[:, None] * S).sum(dim=0)      # -kk^T S  -> [V]
    S = exp(log_w)[:, None] * S + (kk * a)[:, None] * contract[None, :]
        + k[:, None] * v[None, :]
    y = (r[:, None] * S).sum(dim=0)               # S^T r    -> [V]

This is algebraically identical to the BlinkDL demo's ``[V, K]`` layout
(``state = state * w + state @ (-kk) \\otimes (kk*a) + vv \\otimes k``;
``out = state @ rr``) transposed, and to the fla Triton kernel
``fused_recurrent_rwkv7_fwd_kernel`` (``b_h = exp(w)[:, None] * b_h +
b_b[:, None] * sum(b_act_a[:, None] * b_h, 0)[None, :] + b_k[:, None] *
b_v[None, :]``).

**Decay convention (the double-transform hazard).** This module's public
contract is: the mixer computes the *log-decay*
``log_w = -0.6065306597126334 * sigmoid(w_lora(xw)) ∈ [-0.60653, 0)``
(so ``decay = exp(log_w) ∈ (0.5453, 1)`` — the RWKV-7 bounded-decay
range) and the scan op applies ``exp()`` exactly once. The two upstream
references differ here and an adapter MUST NOT mix them:

  * the x070 demo passes a *raw* pre-activation ``w`` and transforms it
    inside the op as ``exp(-exp(w))``;
  * the fla ``chunk_rwkv7`` / ``fused_mul_recurrent_rwkv7`` kernels take
    ``log_w`` (``gk``) directly.

``ops_rwkv_backend`` is the ONLY place that performs any conversion, and
each backend adapter documents which convention it feeds. The reference
scan in this module consumes log-decay.

Token shift / state layout
---------------------------

* ``matrix_state``: ``[B, H, K, V]`` recurrent state per layer, in
  compute precision (fp32 for fp16/bf16 inputs, mirroring the KDA /
  Mamba-2 state contract). It is the "M_matrix" of the cache-budget
  model: ``B x L_R x H x D_k x D_v`` elements per sequence.
* ``shift_state`` (the mixer's ``conv_state`` slot): ``[B, 1, d_model]``
  — the previous token's input, needed by the RWKV token shift. It rides
  the stack's per-memory-layer conv-lookback channel (the same channel
  Mamba-2 uses for its conv state), so no new state plumbing exists. It
  is NOT part of the recurrent matrix state and is never mixed into it.
* The KDA-style short convolution is NOT used: the RWKV route replaces
  the KDA layer entirely (including its conv), and RWKV-7's own token
  shift provides the local mixing.

Reference path: the recurrence loop is sequential over ``T`` (inherent
to an RNN state); the pure-PyTorch path is the correctness reference used
by the tests and small-scale experiments. The optional FLA backend
(``rwkv_backend='fla'``) provides the fused-kernel path.
"""

from __future__ import annotations

import math
import warnings as _warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Upstream provenance (recorded in experiment result rows via
# run_ablation / run_quality provenance blocks).
# ---------------------------------------------------------------------------
RWKV7_UPSTREAM_BINDING = {
    'parameterization_init': {
        'repo': 'RWKV-Vibe/rwkv-fla',
        'path': 'fla/layers/rwkv7.py',
        'commit': '074deaf640284b219f3767de8d1ad007b1049765',
        'date': '2025-07-28',
    },
    'lora_module': {
        'repo': 'RWKV-Vibe/rwkv-fla',
        'path': 'fla/layers/rwkv6.py (class LoRA)',
        'commit': '074deaf640284b219f3767de8d1ad007b1049765',
    },
    'recurrence_cross_check': {
        'repo': 'BlinkDL/RWKV-LM',
        'path': 'RWKV-v7/rwkv_v7_demo.py (RWKV_Tmix_x070 reference path)',
        'commit': 'a963e4ad848426a7032fd6f7c54bdf59c11b67d7',
        'date': '2025-02-19',
    },
    'variant': 'rwkv7-nofirst (no cross-layer v_first; see module docstring)',
    'decay_contract': 'mixer emits log-decay; scan applies exp() exactly once',
}

# exp(-1/sqrt(e)) == 0.5453... bounds the per-step decay from below; the
# constant 1/sqrt(e) = 0.60653... is the log-decay magnitude cap used by
# BOTH the fla G1 parameterization (w = -0.60653 * sigmoid(...)) and the
# x070 softplus form (w <= -0.5 -> exp(-exp(w)) >= exp(-exp(-0.5))).
RWKV7_LOG_DECAY_MAGNITUDE = 0.6065306597126334


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------
def validate_rwkv7_geometry(
    d_model: int,
    head_dim: int = 64,
    decay_lora: int = 32,
    a_lora: int = 32,
    gate_lora: int = 64,
) -> dict:
    """Validate the RWKV-7 mixer geometry and derive ``n_heads``.

    RWKV-7 splits ``d_model`` into ``H = d_model // head_dim`` heads of
    ``head_dim`` channels each; the recurrent state is
    ``H x head_dim x head_dim`` per layer. The low-rank (LoRA) widths
    follow the fla defaults evaluated at ``d_model=256`` (the main-table
    geometry: ``head_dim=64 -> H=4``, ``decay/a lora=32``,
    ``gate lora=64``, per the fla formula ``max(32, round(coeff *
    sqrt(d) / 32) * 32)`` with coefficients 2.5 / 2.5 / 5). Small-scale
    experiments override them (e.g. ``run_ablation`` clamps ``head_dim``
    to a divisor of ``d_model``).

    Returns ``{'n_heads': H, 'head_dim': N, 'decay_lora': ...,
    'a_lora': ..., 'gate_lora': ...}`` or raises ``ValueError`` with an
    actionable message (mirrors ``validate_mamba2_geometry``).
    """
    if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model < 1:
        raise ValueError(
            f"rwkv7: d_model={d_model!r} must be a positive int.")
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim < 1:
        raise ValueError(
            f"rwkv7: head_dim={head_dim!r} must be a positive int.")
    if d_model % head_dim != 0:
        raise ValueError(
            f"rwkv7: d_model={d_model} must be divisible by "
            f"head_dim={head_dim} (H = d_model // head_dim heads; "
            f"got remainder {d_model % head_dim}). For small-scale runs "
            f"pass a smaller rwkv_head_dim (e.g. 16 for d_model=32).")
    n_heads = d_model // head_dim
    for name, val in (('decay_lora', decay_lora), ('a_lora', a_lora),
                      ('gate_lora', gate_lora)):
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise ValueError(
                f"rwkv7: {name}={val!r} must be a positive int "
                f"(low-rank width of the corresponding LoRA).")
        if val > d_model:
            raise ValueError(
                f"rwkv7: {name}={val} exceeds d_model={d_model}; a "
                f"'low-rank' projection wider than the model is a "
                f"configuration error (upstream clamps lora widths to "
                f"the hidden size).")
    return {
        'n_heads': n_heads,
        'head_dim': head_dim,
        'decay_lora': decay_lora,
        'a_lora': a_lora,
        'gate_lora': gate_lora,
    }


# ---------------------------------------------------------------------------
# Reference recurrence
# ---------------------------------------------------------------------------
# The sequential loop body is factored out so it can be TorchScript-compiled
# (mirroring the Mamba-2 / KDA scan loops). The eager form runs the same
# per-element operations in the same order; ``w_exp`` receives the
# already-exponentiated decay (see ``rwkv7_reference_scan``).
_RWKV7_SCAN_LOOP_SCRIPTED = None
_RWKV7_SCAN_LOOP_SCRIPTED_FAILED = False
_RWKV7_SCAN_LOOP_INPLACE_SCRIPTED = None
_RWKV7_SCAN_LOOP_INPLACE_SCRIPTED_FAILED = False


def _rwkv7_scan_loop_eager(r, w_exp, k, v, kk, kka, state):
    """One sequential pass of the RWKV-7 recurrence (inner-loop body).

    The caller passes the per-step views already unsqueezed to their
    broadcast shapes (``[B, T, H, K, 1]`` for ``r`` / ``w_exp`` / ``k`` /
    ``kk`` / ``kka`` and ``[B, T, H, 1, V]`` for ``v``); ``unbind(1)`` then
    yields the exact per-step operands the recurrence consumes. ``kka``
    carries ``-(kk * a)``: negating the product is exact, and ``kka * (-c)``
    is bit-for-bit ``(-kka) * c``, so the folded sign does not change any
    value. ``state`` is the ``[B, H, K, V]`` key-by-value matrix. Returns
    ``(o, state)`` with ``o`` shaped ``[B, T, H, V]``.
    """
    outs = []
    for r_t, w_t, k_t, v_t, kk_t, kka_t in zip(
            r.unbind(1), w_exp.unbind(1), k.unbind(1), v.unbind(1),
            kk.unbind(1), kka.unbind(1)):
        # contract[v] = -sum_k kk[k] * S[k, v]   (== -(kk^T S))
        contract = (kk_t * state).sum(dim=-2)                      # [B, H, V]
        # S = exp(log_w)[k] * S[k, v] + (kk*a)[k] * contract[v] + k[k] * v[v]
        state = (w_t * state
                 + kka_t * contract.unsqueeze(-2)
                 + k_t * v_t)
        # y[v] = sum_k r[k] * S[k, v]            (== S^T r)
        outs.append((r_t * state).sum(dim=-2))                     # [B, H, V]
    return torch.stack(outs, dim=1), state


def _rwkv7_scan_loop_inplace_eager(r, w_exp, k, v, kk, kka, state):
    """No-grad variant of :func:`_rwkv7_scan_loop_eager` updating ``state``
    in place.

    The state update is evaluated as ``state.mul_(w).add_(kka*contract)
    .add_(k*v)`` — the same products/sums in the same association order
    as the out-of-place expression (bit-identical values). The caller
    must pass a dedicated copy of the state (this function mutates its
    ``state`` argument) and autograd must be disabled.
    """
    # Collect the per-step readouts and stack once at the end: the single
    # batched copy replaces T tiny per-step copies (elementwise-identical
    # values; the stack writes the singleton-free [B, T, H, V] directly).
    outs: list[torch.Tensor] = []
    for r_t, w_t, k_t, v_t, kk_t, kka_t in zip(
            r.unbind(1), w_exp.unbind(1), k.unbind(1), v.unbind(1),
            kk.unbind(1), kka.unbind(1)):
        # contract[v] = -sum_k kk[k] * S[k, v]   (== -(kk^T S))
        contract = (kk_t * state).sum(dim=-2)                      # [B, H, V]
        # S = exp(log_w)[k] * S[k, v] + (kk*a)[k] * contract[v] + k[k] * v[v]
        state = (state.mul_(w_t)
                 .add_(kka_t * contract.unsqueeze(-2))
                 .add_(k_t * v_t))
        outs.append((r_t * state).sum(dim=-2))                     # [B, H, V]
    return torch.stack(outs, dim=1), state


def _get_scripted_rwkv7_scan_loop():
    """Scripted loop body, with eager fallback.

    The scripted form runs the same operations in the same order as
    :func:`_rwkv7_scan_loop_eager`; on builds where scripting fails, the
    eager loop is used instead.
    """
    global _RWKV7_SCAN_LOOP_SCRIPTED, _RWKV7_SCAN_LOOP_SCRIPTED_FAILED
    if _RWKV7_SCAN_LOOP_SCRIPTED is None and not _RWKV7_SCAN_LOOP_SCRIPTED_FAILED:
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore', FutureWarning)
                _RWKV7_SCAN_LOOP_SCRIPTED = torch.jit.script(
                    _rwkv7_scan_loop_eager)
        except Exception:
            _RWKV7_SCAN_LOOP_SCRIPTED_FAILED = True
    return _RWKV7_SCAN_LOOP_SCRIPTED


def _get_scripted_rwkv7_scan_loop_inplace():
    """Scripted no-grad loop body, with eager fallback (see
    :func:`_rwkv7_scan_loop_inplace_eager`)."""
    global _RWKV7_SCAN_LOOP_INPLACE_SCRIPTED
    global _RWKV7_SCAN_LOOP_INPLACE_SCRIPTED_FAILED
    if (_RWKV7_SCAN_LOOP_INPLACE_SCRIPTED is None
            and not _RWKV7_SCAN_LOOP_INPLACE_SCRIPTED_FAILED):
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore', FutureWarning)
                _RWKV7_SCAN_LOOP_INPLACE_SCRIPTED = torch.jit.script(
                    _rwkv7_scan_loop_inplace_eager)
        except Exception:
            _RWKV7_SCAN_LOOP_INPLACE_SCRIPTED_FAILED = True
    return _RWKV7_SCAN_LOOP_INPLACE_SCRIPTED


def rwkv7_reference_scan(
    r: torch.Tensor,
    log_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure-PyTorch RWKV-7 recurrence (the reference backend).

    Args:
        r:     ``[B, T, H, K]`` receptance (query) — key-dim indexed.
        log_w: ``[B, T, H, K]`` LOG-decay (``decay = exp(log_w)``; the
            caller has ALREADY applied the sigmoid/bound transform — this
            function must not transform it a second time).
        k:     ``[B, T, H, K]`` modified key ``k * (1 + (a-1) * k_a)``.
        v:     ``[B, T, H, V]`` value.
        kk:    ``[B, T, H, K]`` L2-normalized key direction.
        a:     ``[B, T, H, K]`` in-context learning rate (post-sigmoid).
        scale: query scale (RWKV-7 uses 1.0 — ``kk`` is already
            unit-normalized; explicit so all backends share the
            contract).
        initial_state: ``[B, H, K, V]`` recurrent state in compute
            precision, or ``None`` for a fresh (zero) state.
        output_final_state: return the final state (``None`` otherwise).

    Returns:
        ``(o, final_state)`` — ``o`` is ``[B, T, H, V]`` in the input
        dtype; ``final_state`` is ``[B, H, K, V]`` in compute precision
        (fp32 for fp16/bf16 inputs, fp64 for fp64 inputs — the same
        state-dtype contract as the KDA / Mamba-2 reference ops).

    The loop is sequential over ``T`` and differentiable via autograd
    (no custom backward; the FLA backend provides the fused training
    path). The state, ``contract`` and readout einsums run in compute
    precision: fp16/bf16 accumulation over ``T`` steps of a bounded
    (0.545, 1) decay still drifts, and the reference path is the
    numerics anchor the other backends are compared against.
    """
    if not isinstance(scale, torch.Tensor):
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError(
                f"rwkv7_reference_scan: scale={scale!r} must be a real "
                f"number.")
        try:
            float(scale)
        except OverflowError:
            # JSON integers are unbounded; the scale multiplies the
            # receptance in float arithmetic.
            raise ValueError(
                f"rwkv7_reference_scan: scale={scale!r} must be "
                f"representable as a finite float.") from None
    if r.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"rwkv7_reference_scan: r/k/v must be [B, T, H, D] (got "
            f"r.dim={r.dim()}, k.dim={k.dim()}, v.dim={v.dim()}).")
    B, T, H, K = r.shape
    V = v.shape[-1]
    if k.shape != (B, T, H, K) or kk.shape != (B, T, H, K) \
            or a.shape != (B, T, H, K) or log_w.shape != (B, T, H, K):
        raise ValueError(
            f"rwkv7_reference_scan: k/kk/a/log_w must all be "
            f"[B, T, H, K] = [{B}, {T}, {H}, {K}] (got k={tuple(k.shape)}, "
            f"kk={tuple(kk.shape)}, a={tuple(a.shape)}, "
            f"log_w={tuple(log_w.shape)}).")
    if v.shape != (B, T, H, V):
        raise ValueError(
            f"rwkv7_reference_scan: v must be [B, T, H, V] = "
            f"[{B}, {T}, {H}, {V}] (got {tuple(v.shape)}).")
    if initial_state is not None and tuple(initial_state.shape) != (B, H, K, V):
        raise ValueError(
            f"rwkv7_reference_scan: initial_state must be [B, H, K, V] = "
            f"[{B}, {H}, {K}, {V}] (got {tuple(initial_state.shape)}).")

    out_dtype = v.dtype
    compute_dtype = torch.float64 if out_dtype == torch.float64 else torch.float32

    # Upcast the recurrence inputs (they come from the mixer in the input
    # dtype). Note: r/k/kk/a/log_w share [B, T, H, K] so one stacked cast
    # would also work; separate casts keep autograd graphs simple.
    r_c = r.to(compute_dtype)
    k_c = k.to(compute_dtype)
    v_c = v.to(compute_dtype)
    kk_c = kk.to(compute_dtype)
    a_c = a.to(compute_dtype)
    w_c = log_w.to(compute_dtype)

    if initial_state is None:
        state = v_c.new_zeros(B, H, K, V)
    else:
        state = initial_state.to(device=v.device, dtype=compute_dtype)

    if scale != 1.0:
        # Multiplying by exactly 1.0 is an IEEE no-op (pinned by the
        # bit-exactness of every prior test), so the default RWKV-7 scale
        # skips the full [B, T, H, K] multiply.
        r_c = r_c * scale
    # (kk * a) is used at every step. The sign is folded into ``kka`` so
    # the readout contract needs no in-loop negation (exact sign flips
    # only — see ``_rwkv7_scan_loop_eager``). Outside autograd, when the
    # upcast above returned a fresh copy of ``a`` (dtype mismatch), the
    # sign flip is folded into that copy in place: ``kk * (-a)`` is
    # bitwise identical to ``-(kk * a)`` (negation flips the sign bit
    # exactly) and saves one full-size temporary.
    if not torch.is_grad_enabled() and a_c is not a:
        a_c.neg_()
        kka_c = kk_c * a_c
    else:
        kka_c = -(kk_c * a_c)
    # ``exp(log_w)`` is an elementwise map, so evaluating it over the whole
    # time axis computes exactly the same fp values as per-step
    # ``w_t.exp()`` (each element rounds identically). The per-step update
    # below keeps the operand association ``(w*S + kka*contract) + k*v``.
    # The singleton broadcast dims are pre-applied before ``unbind`` as
    # well.
    w_exp_c = w_c.exp().unsqueeze(-1)
    r_c = r_c.unsqueeze(-1)
    k_c = k_c.unsqueeze(-1)
    v_c = v_c.unsqueeze(-2)
    kk_c = kk_c.unsqueeze(-1)
    kka_c = kka_c.unsqueeze(-1)

    if T == 0:
        o = v.new_zeros(B, 0, H, V)
        return o.to(out_dtype), state if output_final_state else None
    # Autograd-safe out-of-place loop while a graph is being recorded; the
    # in-place loop (same values) otherwise. The in-place loop mutates its
    # state argument, so it gets a dedicated copy and the caller's
    # ``initial_state`` stays untouched.
    if not torch.is_grad_enabled():
        loop_i = _get_scripted_rwkv7_scan_loop_inplace()
        fn = (loop_i if loop_i is not None
              else _rwkv7_scan_loop_inplace_eager)
        # The in-place loop mutates its state argument; only a caller-
        # provided ``initial_state`` needs the defensive copy (a zero state
        # built just above is already exclusively owned).
        if initial_state is not None:
            state = state.clone()
        o, state = fn(r_c, w_exp_c, k_c, v_c, kk_c, kka_c, state)
    else:
        loop = _get_scripted_rwkv7_scan_loop()
        if loop is not None:
            o, state = loop(r_c, w_exp_c, k_c, v_c, kk_c, kka_c, state)
        else:
            o, state = _rwkv7_scan_loop_eager(
                r_c, w_exp_c, k_c, v_c, kk_c, kka_c, state)
    return o.to(out_dtype), state if output_final_state else None


# ---------------------------------------------------------------------------
# LoRA sub-module (faithful port of fla/layers/rwkv6.py's LoRA @ 074deaf)
# ---------------------------------------------------------------------------
class Rwkv7LoRA(nn.Module):
    """Low-rank projection used by the decay / a / gate branches.

    Structure (upstream): ``Linear(d, r, bias=False) -> activation ->
    Linear(r, d_out, bias=bias)`` initialized with a **zero**
    down-projection and an **orthogonal** up-projection with gain
    ``0.1 * sqrt(d_out / d)`` (``1 * 0.1`` when ``d_out <= d``), so at
    init the branch output equals its bias exactly and the low-rank part
    learns from zero (the classic asymmetric LoRA init; gradients reach
    the zero matrix through the non-zero one).
    """

    def __init__(self, input_dim: int, output_dim: int, low_rank_dim: int,
                 activation: str | None = 'tanh', bias: bool = True):
        super().__init__()
        if activation is None:
            self._act = nn.Identity()
        elif activation == 'sigmoid':
            self._act = nn.Sigmoid()
        elif activation == 'tanh':
            self._act = nn.Tanh()
        elif activation == 'relu':
            self._act = nn.ReLU()
        else:
            raise ValueError(
                f"Rwkv7LoRA: unsupported activation {activation!r} "
                f"(expected None/'sigmoid'/'tanh'/'relu').")
        self.down = nn.Linear(input_dim, low_rank_dim, bias=False)
        self.up = nn.Linear(low_rank_dim, output_dim, bias=bias)
        with torch.no_grad():
            nn.init.zeros_(self.down.weight)
            weight_fp32 = self.up.weight.float()
            shape = self.up.weight.shape          # [output_dim, low_rank_dim]
            gain = math.sqrt(shape[1] / shape[0]) if shape[1] > shape[0] else 1.0
            nn.init.orthogonal_(weight_fp32, gain=gain * 0.1)
            self.up.weight.copy_(weight_fp32.to(self.up.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self._act(self.down(x)))

    @torch.no_grad()
    def set_bias_value(self, value: torch.Tensor) -> None:
        """Overwrite the output bias (the upstream init recipe hook)."""
        if self.up.bias is None:
            raise RuntimeError(
                "Rwkv7LoRA.set_bias_value: this LoRA was built with "
                "bias=False; the upstream decay/a init recipes require a "
                "biased LoRA.")
        self.up.bias.copy_(value.to(self.up.bias.dtype))


# ---------------------------------------------------------------------------
# The mixer
# ---------------------------------------------------------------------------
class Rwkv7Mixer(nn.Module):
    """RWKV-7 TimeMix with the repository's memory-layer contract.

    See the module docstring for the upstream bindings, the
    ``rwkv7-nofirst`` variant note, the state layout, and the log-decay
    contract. The forward math (bound to the fla layer @ 074deaf):

    .. code-block:: python

        xr..xg = lerp(x, shift(x), x_r..x_g)         # token shift
        r   = r_proj(xr)
        log_w = -0.6065306597126334 * sigmoid(w_lora(xw))
        k   = k_proj(xk)
        v   = v_proj(xv)                             # (no v_first lerp)
        a   = sigmoid(a_lora(xa))                    # 'in-context lr'
        g   = g_lora(xg)                             # output gate
        kk  = l2norm(k * k_k)                        # per head
        k   = k * (1 + (a - 1) * k_a)
        o, S = scan(r, log_w, k, v, kk, a, S_prev)   # [K, V] layout
        o   = GroupNorm(H, C, eps=head_dim*1e-5)(o)
        o   = (o + (r*k*r_k).sum(-1,keepdim) * v) * g
        out = o_proj(o)

    Args:
        cfg: a ``HybridConfig``-like namespace. Required: ``d_model``.
            Optional (defaults shown): ``rwkv_head_dim=64``,
            ``rwkv_decay_lora=32``, ``rwkv_a_lora=32``,
            ``rwkv_gate_lora=64``, ``rwkv_backend='reference'``.
            Duck-typed to avoid an import cycle (``ops_fused`` imports
            this module; this module must not import ``ops_fused``).
        rwkv_layer_idx: index of this layer among the model's RWKV-7
            layers (0-based). The depth-dependent init constants
            (decay slope, lerp powers, GroupNorm gain) follow this
            index exactly as upstream uses ``layer_idx``; in a hybrid
            stack it is the position among the RWKV layers, not the
            global stack position (CSA/HCA positions are irrelevant to
            the RWKV init recipe).
        n_rwkv_layers: total number of RWKV-7 layers in the model
            (``rwkv_layer_idx < n_rwkv_layers``); drives the same
            ratios upstream computes from ``n_layer``.
    """

    def __init__(self, cfg, *, rwkv_layer_idx: int = 0,
                 n_rwkv_layers: int = 1):
        super().__init__()
        d_model = getattr(cfg, 'd_model', None)
        if not isinstance(d_model, int) or isinstance(d_model, bool) \
                or d_model < 1:
            raise ValueError(
                f"Rwkv7Mixer: cfg.d_model={d_model!r} must be a positive int.")
        self.d_model = d_model
        self.head_dim = getattr(cfg, 'rwkv_head_dim', 64)
        self.decay_lora = getattr(cfg, 'rwkv_decay_lora', 32)
        self.a_lora = getattr(cfg, 'rwkv_a_lora', 32)
        self.gate_lora = getattr(cfg, 'rwkv_gate_lora', 64)
        # Backend selection is validated eagerly (typo -> immediate
        # error before any parameters are allocated); the optional FLA
        # import stays lazy inside ops_rwkv_backend (mirrors the
        # KDA / Mamba-2 backend contract).
        from ops_rwkv_backend import validate_rwkv_backend
        self.backend = validate_rwkv_backend(
            getattr(cfg, 'rwkv_backend', 'reference'))

        geo = validate_rwkv7_geometry(
            d_model, self.head_dim, self.decay_lora, self.a_lora,
            self.gate_lora)
        self.head_dim = geo['head_dim']
        self.decay_lora = geo['decay_lora']
        self.a_lora = geo['a_lora']
        self.gate_lora = geo['gate_lora']
        self.n_heads = geo['n_heads']

        if (not isinstance(rwkv_layer_idx, int)
                or isinstance(rwkv_layer_idx, bool) or rwkv_layer_idx < 0):
            raise ValueError(
                f"Rwkv7Mixer: rwkv_layer_idx={rwkv_layer_idx!r} must be a "
                f"non-negative int.")
        if (not isinstance(n_rwkv_layers, int)
                or isinstance(n_rwkv_layers, bool) or n_rwkv_layers < 1
                or rwkv_layer_idx >= n_rwkv_layers):
            raise ValueError(
                f"Rwkv7Mixer: n_rwkv_layers={n_rwkv_layers!r} must be >= 1 "
                f"and rwkv_layer_idx={rwkv_layer_idx} must be < "
                f"n_rwkv_layers.")
        self.rwkv_layer_idx = rwkv_layer_idx
        self.n_rwkv_layers = n_rwkv_layers

        # Record which implementation actually produced outputs
        # ('reference' here; the auto-fallback marker is set by the
        # factory). Experiment runners serialize this field.
        self.effective_backend = 'reference'

        H, N, C = self.n_heads, self.head_dim, d_model

        # --- token-shift lerp ratios (1,1,C), upstream init below -----
        self.x_r = nn.Parameter(torch.zeros(1, 1, C))
        self.x_w = nn.Parameter(torch.zeros(1, 1, C))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.x_v = nn.Parameter(torch.zeros(1, 1, C))
        self.x_a = nn.Parameter(torch.zeros(1, 1, C))
        self.x_g = nn.Parameter(torch.zeros(1, 1, C))

        # --- per-channel key scaling / in-context-lr mixing -----------
        # k_k: key gain before L2 normalization; k_a: how strongly the
        # in-context learning rate a modulates k; r_k: the (r . k)
        # shortcut's per-head gain.
        self.k_k = nn.Parameter(torch.zeros(C))
        self.k_a = nn.Parameter(torch.zeros(C))
        self.r_k = nn.Parameter(torch.zeros(H, N))

        # --- projections -----------------------------------------------
        self.r_proj = nn.Linear(C, C, bias=False)
        self.k_proj = nn.Linear(C, C, bias=False)
        self.v_proj = nn.Linear(C, C, bias=False)
        self.o_proj = nn.Linear(C, C, bias=False)

        # --- low-rank branches ------------------------------------------
        # w_lora: tanh-activated decay branch (biased — the init recipe
        # writes per-channel decay slopes into the bias).
        self.w_lora = Rwkv7LoRA(C, C, self.decay_lora, activation='tanh',
                                bias=True)
        # a_lora: 'in-context learning rate' branch (no activation here;
        # the mixer applies sigmoid).
        self.a_lora = Rwkv7LoRA(C, C, self.a_lora, activation=None,
                                bias=True)
        # g_lora: output gate branch (sigmoid activation inside the LoRA;
        # no bias, matching upstream).
        self.g_lora = Rwkv7LoRA(C, C, self.gate_lora, activation='sigmoid',
                                bias=False)
        # NOTE: upstream's v_lora (the cross-layer v_first residual) is
        # intentionally absent — this is the 'rwkv7-nofirst' variant (see
        # the module docstring).

        # --- output group norm -------------------------------------------
        # eps = head_dim * 1e-5 (64e-5 at head_dim=64) — the upstream
        # value; NOT a typo. GroupNorm over [B*T, H*N] with H groups.
        self.g_norm = nn.GroupNorm(H, C, eps=N * 1e-5)

        self._initialize_weights()

        # Persistent streaming state: the previous token's input, shape
        # [B, 1, d_model]. Registered non-persistent so .to(device) /
        # .half() move it, state_dict skips it, and reset_conv_state()
        # clears it (the stack's reset_state calls that for every memory
        # layer). This is the mixer's 'conv' slot in the hybrid stack's
        # lookback channel.
        self.register_buffer('_shift_state', None, persistent=False)

    # ------------------------------------------------------------------
    # Initialization (upstream recipe, fla @ 074deaf)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _initialize_weights(self) -> None:
        """Port of ``RWKV7Attention._initialize_weights`` (fla @ 074deaf).

        Depth-dependent constants use the RWKV sub-layer index
        (``rwkv_layer_idx / n_rwkv_layers``) — upstream's
        ``layer_idx / num_hidden_layers`` adapted to the hybrid stack.
        """
        i, n = self.rwkv_layer_idx, self.n_rwkv_layers
        C, N = self.d_model, self.head_dim
        # n == 1: a single RWKV layer is 'the first layer' — the 0-to-1
        # ratio degenerates; upstream would divide by zero, we pin it.
        ratio_0_to_1 = (i / (n - 1)) if n > 1 else 0.0
        ratio_1_to_almost0 = 1.0 - (i / n)

        # The per-channel ramps are built with Python-double arithmetic
        # (identical expressions to the former per-element loop) and
        # materialized in ONE tensor construction, which performs the same
        # double -> fp32 rounding the per-element stores did. This replaces
        # C scalar tensor writes with a list comprehension.
        if C > 1:
            frac = [c / (C - 1) for c in range(C)]
        else:
            # 0/0 for C == 1 (degenerate) — guard with 0.0.
            frac = [0.0] * C
        linear = torch.tensor(
            [f - 0.5 for f in frac] if C > 1 else [0.0] * C,
            dtype=torch.float32)
        # zigzag: per-head-position triangular ramp in [-1, 1].
        if N > 1:
            zz = [((c % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                  for c in range(C)]
        else:
            zz = [0.0] * C
        zigzag = torch.tensor([z * abs(z) for z in zz], dtype=torch.float32)
        www = torch.tensor(
            [-6 + 6 * f ** (1 + 1 * ratio_0_to_1 ** 0.3) for f in frac],
            dtype=torch.float32)
        ddd = torch.tensor([c / C for c in range(C)],
                           dtype=torch.float32).view(1, 1, C)

        # Token-shift lerp ratios: earlier tokens lean more on the
        # previous input in deeper layers (powers tuned upstream).
        self.x_r.data.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.x_w.data.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k.data.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v.data.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_a.data.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_g.data.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        # Key gain / lr-mixing / shortcut gains.
        self.k_a.data.fill_(1.02)
        self.r_k.data.fill_(-0.04)
        self.k_k.data.copy_(0.71 - linear * 0.1)

        # Decay / a branch biases. The +0.5 on the decay bias is the
        # softplus offset in the x070 form; the fla G1 form carries it so
        # both conventions land in the same decay range.
        self.w_lora.set_bias_value(www + 0.5 + zigzag * 2.5)
        self.a_lora.set_bias_value(-0.19 + zigzag * 0.3 + linear * 0.4)
        # (v_lora bias — 0.73 - linear*0.4 — is skipped: nofirst variant.)

        # Output group-norm gain grows with depth.
        self.g_norm.weight.data.copy_(((i + 1) / n) ** 0.7)

        # Projections: r/v orthogonal, k orthogonal-and-small, o zero
        # (zero o_proj => the mixer is identity-in-residual at init; the
        # first gradient step lifts it off zero through its non-zero
        # input).
        for proj, gain in ((self.r_proj, 1.0), (self.k_proj, 0.1),
                           (self.v_proj, 1.0)):
            weight_fp32 = proj.weight.float()
            nn.init.orthogonal_(weight_fp32, gain=gain)
            proj.weight.copy_(weight_fp32.to(proj.weight.dtype))
        self.o_proj.weight.zero_()

    # ------------------------------------------------------------------
    # Streaming-state helpers
    # ------------------------------------------------------------------
    def reset_conv_state(self) -> None:
        """Clear the persistent token-shift state.

        Called between independent sequences (the stack's
        ``reset_state`` calls this for every memory layer) so the token
        shift does not "see" the previous sequence's last token as the
        left context of the next one.
        """
        self._shift_state = None

    def state_nbytes(self, batch: int = 1) -> dict:
        """Analytic per-layer runtime state size in bytes (fp32 states).

        The matrix state lives in compute precision (fp32 for
        fp16/bf16 inputs); the shift state lives in the input dtype.
        As with ``Mamba2Mixer.state_nbytes`` this reports the fp32
        figure used by the analytic cache-accounting paths; the
        live-tensor audit measures the ACTUAL dtypes.
        """
        matrix_elems = batch * self.n_heads * self.head_dim * self.head_dim
        shift_elems = batch * self.d_model
        return {
            'matrix_state_bytes': matrix_elems * 4,
            'shift_state_bytes': shift_elems * 4,
            'total_bytes': (matrix_elems + shift_elems) * 4,
            'matrix_state_elements': matrix_elems,
            'shift_state_elements': shift_elems,
        }

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                state: torch.Tensor | None = None):
        """Stateful forward.

        Reads ``self._shift_state`` and writes the new shift state back
        to ``self``. Use :meth:`forward_functional` for the pure path
        (DDP, ``torch.compile``, checkpointing, concurrent inference).
        """
        o, new_state, new_shift = self.forward_functional(
            x, state, self._shift_state)
        self._shift_state = new_shift
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
            state: ``[B, H, K, V]`` recurrent matrix state from the
                previous call (compute precision), or ``None`` for a
                fresh sequence.
            conv_state: ``[B, 1, d_model]`` token-shift state (the
                previous call's LAST input token), or ``None``.
            detach_lookback: if True (default), the returned shift state
                is detached so the autograd graph of the current chunk
                is not retained by the next call. Set False for
                cross-chunk BPTT (mirrors the KDA / Mamba-2 contract).

        Returns:
            ``(output, new_matrix_state, new_shift_state)``:
              * ``output``: ``[B, T, d_model]`` (input dtype).
              * ``new_matrix_state``: ``[B, H, K, V]`` in compute
                precision (caller owns the detach policy).
              * ``new_shift_state``: ``[B, 1, d_model]`` (or ``None``
                if ``T == 0``).
        """
        B, T, d = x.shape
        if d != self.d_model:
            raise ValueError(
                f"Rwkv7Mixer: input d_model={d} does not match the "
                f"configured d_model={self.d_model}.")
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        # Degenerate empty-sequence case (mirrors KDA / Mamba-2):
        # preserve or initialize the matrix state, leave the shift state
        # untouched, return a zero-length output.
        if T == 0:
            if state is not None and state.shape[0] == B:
                expected = (B, self.n_heads, self.head_dim, self.head_dim)
                if tuple(state.shape) != expected:
                    raise ValueError(
                        'Rwkv7Mixer: state.shape='
                        f'{tuple(state.shape)} does not match the expected '
                        f'{expected} ([B, H, K, V]) even though T=0.')
                new_state = state.to(device=x.device, dtype=compute_dtype)
            else:
                new_state = torch.zeros(
                    B, self.n_heads, self.head_dim, self.head_dim,
                    device=x.device, dtype=compute_dtype)
            new_shift = conv_state
            if new_shift is not None:
                if new_shift.shape[0] != B:
                    new_shift = None
                else:
                    expected_shift = (B, 1, d)
                    if tuple(new_shift.shape) != expected_shift:
                        raise ValueError(
                            'Rwkv7Mixer: shift state shape='
                            f'{tuple(new_shift.shape)} does not match the '
                            f'expected {expected_shift} ([B, 1, d]) even '
                            'though T=0.')
                    new_shift = new_shift.to(device=x.device, dtype=x.dtype)
                    if detach_lookback:
                        new_shift = new_shift.detach()
                    new_shift = new_shift.clone()
            return x.new_zeros(B, 0, d), new_state, new_shift

        # Batch-size / device / dtype guards on the incoming shift state
        # (mirrors the KDA lookback guards).
        shift = conv_state
        if shift is not None:
            if shift.dim() != 3 or shift.shape[0] != B:
                # A state from a different batch cannot be reused; drop it.
                shift = None
            elif tuple(shift.shape) != (B, 1, d):
                # Same batch but a malformed geometry is a caller error,
                # not a fresh sequence (mirrors the T=0 path).
                raise ValueError(
                    'Rwkv7Mixer: shift state shape='
                    f'{tuple(shift.shape)} does not match the expected '
                    f'{(B, 1, d)} ([B, 1, d]).')
            elif shift.device != x.device or shift.dtype != x.dtype:
                shift = shift.to(device=x.device, dtype=x.dtype)
                if detach_lookback:
                    shift = shift.detach()
            elif detach_lookback:
                shift = shift.detach()

        # --- token shift: x_prev[t] = x[t-1], x_prev[0] = shift or 0 ---
        if shift is None:
            x_prev = torch.cat(
                [x.new_zeros(B, 1, d), x[:, :-1]], dim=1)
        else:
            x_prev = torch.cat([shift, x[:, :-1]], dim=1)
        # New shift state: this chunk's last token (detached per the
        # lookback contract; the defensive clone lets the caller mutate
        # the returned tensor without aliasing the input slice).
        new_shift = x[:, -1:]
        if detach_lookback:
            new_shift = new_shift.detach()
        new_shift = new_shift.clone()

        # --- lerp inputs (x + (x_prev - x) * ratio, per branch) --------
        xx = x_prev - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.r_proj(xr)                                     # [B,T,C]
        # LOG-decay (the mixer's decay contract): w in [-0.60653, 0).
        log_w = -RWKV7_LOG_DECAY_MAGNITUDE * torch.sigmoid(self.w_lora(xw))
        k = self.k_proj(xk)
        v = self.v_proj(xv)                                     # nofirst: no v_first lerp
        a = torch.sigmoid(self.a_lora(xa))                      # [B,T,C]
        g = self.g_lora(xg)                                     # [B,T,C]

        H, N = self.n_heads, self.head_dim
        B, T, C = k.shape
        # L2-normalize the scaled key per head (kk), then mix the
        # in-context learning rate into k.
        kk = F.normalize((k * self.k_k).view(B, T, H, N), dim=-1, p=2.0)
        k = k * (1 + (a - 1) * self.k_a)

        # Reshape to [B, T, H, D] for the scan (K == N; V == N here —
        # the RWKV-7 value dim equals the key dim per head).
        r4 = r.view(B, T, H, N)
        k4 = k.view(B, T, H, N)
        v4 = v.view(B, T, H, N)
        kk4 = kk
        a4 = a.view(B, T, H, N)
        w4 = log_w.view(B, T, H, N)

        # Dispatch through the backend adapter. The reference path is
        # the in-module scan; 'fla'/'auto' resolve inside
        # ops_rwkv_backend (lazy import). The adapter receives the
        # ALREADY-final log-decay — it must not transform it again.
        from ops_rwkv_backend import rwkv7_forward
        o4, new_state = rwkv7_forward(
            r4, w4, k4, v4, kk4, a4,
            scale=1.0,
            initial_state=state,
            output_final_state=True,
            backend=self.backend,
        )

        # --- readout post-processing (upstream order) -------------------
        # GroupNorm over [B*T, C] with H groups (eps = N * 1e-5).
        o = self.g_norm(o4.reshape(B * T, C)).view(B, T, C)
        # The (r . k) shortcut: per-head dot of r and (post-modulation) k
        # scaled by r_k, broadcasting over the head's value channels.
        rk_dot = (r4 * k4 * self.r_k).sum(dim=-1, keepdim=True)  # [B,T,H,1]
        if torch.is_grad_enabled():
            o = o + (rk_dot * v4).reshape(B, T, C)
            # Gate, then project.
            out = self.o_proj(o * g)
        else:
            # The GroupNorm output is freshly allocated with no other
            # readers, and ``g`` is not read after the gate: fold the
            # add and the gate multiply into it in place
            # (elementwise-identical).
            o.add_((rk_dot * v4).reshape(B, T, C))
            o.mul_(g)
            out = self.o_proj(o)
        return out, new_state, new_shift

    # ------------------------------------------------------------------
    # Backend switching
    # ------------------------------------------------------------------
    def enable_fla_backend(self) -> None:
        """Switch the recurrence dispatch to the FLA kernels.

        Called by :func:`build_rwkv7_mixer` after it has verified the
        ``fla.ops.rwkv7`` API is importable. The parameterization /
        init / output post-processing stay in this module either way —
        ONLY the recurrence dispatch changes, so both backends share
        the exact same weights and decay convention (the mixer always
        emits log-decay; the FLA adapter passes it through as ``gk``).
        """
        from ops_rwkv_backend import rwkv7_fla_available
        if not rwkv7_fla_available():
            raise ImportError(
                "Rwkv7Mixer.enable_fla_backend: fla.ops.rwkv7 is not "
                "importable in this environment.")
        self.backend = 'fla'
        self.effective_backend = 'fla'

    # ------------------------------------------------------------------
    # Unified inference protocol (prefill / step)
    # ------------------------------------------------------------------
    def prefill(self, x: torch.Tensor):
        """Process a prefix and return ``(output, state_dict)``.

        ``state_dict`` is ``{'matrix_state': ..., 'shift_state': ...}``
        with both entries detached — the inference protocol never
        retains a training graph. Feed it back into :meth:`step` to
        decode continuations.
        """
        out, matrix_state, shift_state = self.forward_functional(
            x, None, None, detach_lookback=True)
        return out, {
            'matrix_state': matrix_state.detach() if matrix_state is not None else None,
            'shift_state': shift_state,
        }

    def step(self, x_t: torch.Tensor, state: dict | None):
        """Process a single token and return ``(output_t, state_dict)``.

        Args:
            x_t: ``[B, 1, d_model]`` (a one-token "chunk"; a T>1 tensor
                is rejected to keep the protocol unambiguous).
            state: the ``state_dict`` returned by a previous
                ``prefill`` / ``step`` call, or ``None``.
        """
        if state is not None and not isinstance(state, dict):
            raise TypeError(
                f"Rwkv7Mixer.step: state must be the dict returned by "
                f"prefill/step (or None), got {type(state).__name__}.")
        if x_t.dim() != 3 or x_t.shape[1] != 1:
            raise ValueError(
                f"Rwkv7Mixer.step: x_t must be [B, 1, d_model], got shape "
                f"{tuple(x_t.shape)}. Use forward_functional for chunks.")
        state = state or {}
        out, matrix_state, shift_state = self.forward_functional(
            x_t, state.get('matrix_state'), state.get('shift_state'),
            detach_lookback=True)
        return out, {
            'matrix_state': matrix_state.detach() if matrix_state is not None else None,
            'shift_state': shift_state,
        }


# ---------------------------------------------------------------------------
# Backend-dispatching factory
# ---------------------------------------------------------------------------
def build_rwkv7_mixer(cfg, *, rwkv_layer_idx: int = 0, n_rwkv_layers: int = 1):
    """Build an RWKV-7 mixer honouring ``cfg.rwkv_backend``.

    * ``'reference'`` — :class:`Rwkv7Mixer` with the in-module scan
      (default; runs everywhere).
    * ``'fla'`` — the same mixer with the recurrence dispatched to
      ``fla.ops.rwkv7.chunk_rwkv7`` (training) /
      ``fused_mul_recurrent_rwkv7`` (decode). Requires the optional
      ``fla`` dependency AND CUDA; raises an actionable error when
      either is missing (an explicitly requested accelerated backend
      must never silently fall back). NOTE: the repository's KDA FLA
      backend may pin a different flash-linear-attention version than
      RWKV-FLA needs (both import the ``fla`` namespace); if the two
      cannot coexist, run the RWKV experiments in an isolated
      environment — do not overwrite-install over the KDA pin.
    * ``'auto'`` — fla when CUDA is available and importable, otherwise
      the reference path. The resolved backend is recorded on
      ``mixer.effective_backend`` for provenance.
    """
    from ops_rwkv_backend import rwkv7_fla_available
    backend = getattr(cfg, 'rwkv_backend', 'reference')
    mixer = Rwkv7Mixer(cfg, rwkv_layer_idx=rwkv_layer_idx,
                       n_rwkv_layers=n_rwkv_layers)
    if backend == 'reference':
        return mixer
    if backend == 'fla':
        # Raises with an actionable message when fla / CUDA is missing.
        # (Imported eagerly HERE, not at module import, so a missing
        # optional dependency never breaks the reference path.)
        import torch as _torch
        if not (_torch.cuda.is_available() and rwkv7_fla_available()):
            raise ImportError(
                "RWKV-7 backend 'fla' requires the optional dependency "
                "'flash-linear-attention' (with fla.ops.rwkv7) AND CUDA. "
                "Install it in a compatible environment, or use "
                "rwkv_backend='reference' / 'auto'.")
        mixer.enable_fla_backend()
        return mixer
    # backend == 'auto'
    import torch as _torch
    if _torch.cuda.is_available() and rwkv7_fla_available():
        try:
            mixer.enable_fla_backend()
            return mixer
        except (ImportError, RuntimeError):
            pass
    mixer.effective_backend = 'reference (auto fallback)'
    return mixer
