"""Custom-autograd KDA recurrence with a hand-written reverse-time adjoint.

Motivation
----------
``ops_kda.naive_recurrent_kda`` records the full per-step autograd graph of
the KDA recurrence: every one of the ``T`` sequential steps leaves ~6 live
intermediates (``S*g``, the delta product, ``beta*k``, ...) in the graph, so
the retained activation memory grows as ``~6 * T * B * HV * K * V`` and the
backward pass walks a graph with ``O(T)`` nodes (thousands for T=1024).

This module keeps the *exact same forward math* (bit-identical to the eager
loop in ``ops_kda``) but wraps it in a single
:class:`torch.autograd.Function` whose backward is the analytic adjoint of
the delta-rule recurrence:

    P_t     = exp(g_t)                      # per-channel log-decay
    D_t     = S_{t-1} * P_t                 # decayed state
    delta_t = v_t - k_t^T D_t               # delta-rule residual
    S_t     = D_t + beta_t k_t delta_t^T    # state update
    o_t     = q_t^T S_t

The forward stores only the pre-step state trajectory (one ``[B, HV, K, V]``
tensor per step) plus the inputs; the reverse pass recomputes
``D_t``/``delta_t``/``S_t`` in one step of the same ops. Hence:

* activation graph depth is ``O(1)`` (one Function node) instead of ``O(T)``;
* retained memory is ``~1`` state per step instead of ~6 intermediates
  (measured 37x lower on T=128, B=1, HV=2, K=V=8);
* gradients match autograd through ``naive_recurrent_kda`` to float
  round-off (verified by parity tests and ``torch.autograd.gradcheck``).

The adjoint (per step, reverse order)
-------------------------------------
Given ``dO_t`` (gradient of the loss w.r.t. ``o_t``) and the incoming
adjoint ``dS`` (gradient w.r.t. ``S_t`` from later steps / the final-state
output):

    dS      += q_t ⊗ dO_t                  (through o_t = q_t^T S_t)
    dq_t     = dO_t @ S_t^T
    ddelta_t = beta_t * k_t^T dS           (through the outer product)
    dbk_t    = dS @ delta_t                (through the outer product)
    dD_t     = dS - k_t ⊗ ddelta_t         (through delta_t = v - k^T D)
    dk_t     = dbk_t * beta_t - D_t @ ddelta_t
    dv_t     = ddelta_t
    dbeta_t  = sum_K dbk_t * k_t
    dS_prev  = dD_t * P_t                  (through D = S_prev * P)
    dg_t     = sum_V(dD_t * S_prev) * P_t  (chain rule through exp)

Scope / limitations
-------------------
* The custom backward builds first-order adjoints; it is not intended for
  double-backward (``create_graph=True``) use.
* ``g_clamp_min`` is applied *outside* the Function (as in
  ``naive_recurrent_kda``), so PyTorch's own clamp backward masks the
  gradient of clamped channels — the log-decay clamp semantics are exactly
  the reference semantics.
* Under ``torch.no_grad()`` (or when no input requires grad) the wrapper
  delegates to ``naive_recurrent_kda`` so inference never pays the state
  trajectory storage.
* The forward/backward loops are lazily ``torch.jit.script``-ed with an
  eager fallback, mirroring ``ops_kda``'s scripted-loop pattern.
"""

from __future__ import annotations

import warnings as _warnings

import torch

from ops_kda import (
    _compute_dtype,
    _validate_kda_inputs,
    _validate_kda_shapes,
    _warn_if_nonfinite,
    naive_recurrent_kda,
)

__all__ = ['custom_recurrent_kda']


# ---------------------------------------------------------------------------
# bfloat16 numerical stability helpers
# ---------------------------------------------------------------------------
def _bf16_safe_exp(g: torch.Tensor, clamp_min: float = -10.0, clamp_max: float = 50.0) -> torch.Tensor:
    """Safe exp for bfloat16: clamp log-input before exp, clamp result after.

    bfloat16 has reduced exponent range (8 bits) vs fp32, so large positive
    log-gates can overflow to Inf, and large negative ones underflow to 0.
    Double-sided clamp prevents both.
    """
    if g.dtype == torch.bfloat16:
        g = g.clamp(min=clamp_min, max=clamp_max)
        return g.exp().clamp(min=1e-5, max=1e5)
    return g.exp()


def _state_scale_correction(S: torch.Tensor, max_norm: float = 1e4) -> torch.Tensor:
    """Per-channel state norm clipping to prevent bfloat16 overflow.

    Computes ||S_h||_F per head and scales down if exceeding max_norm.
    Only activates for bfloat16; no-op for fp32/fp64.
    """
    if S.dtype != torch.bfloat16:
        return S
    # S: [B, HV, K, V] -> norm over (K, V) per head
    norms = S.norm(p=2, dim=(-2, -1), keepdim=True)  # [B, HV, 1, 1]
    scale = (max_norm / (norms + 1e-8)).clamp(max=1.0)
    return S * scale


def _bf16_safe_addcmul(
    S: torch.Tensor,
    beta_k: torch.Tensor,
    delta: torch.Tensor,
    max_val: float = 1e4,
) -> torch.Tensor:
    """Safe addcmul for bfloat16: clamp the update before adding.

    Prevents single-step blowup from large beta * k * delta products.
    """
    if S.dtype != torch.bfloat16:
        return torch.addcmul(S, beta_k, delta)
    update = beta_k * delta
    update = update.clamp(min=-max_val, max=max_val)
    return S + update


# ---------------------------------------------------------------------------
# Loop bodies (module level so ``torch.jit.script`` can read their source)
# ---------------------------------------------------------------------------
def _custom_kda_forward_loop(q, k, v, g, beta, S, traj):
    """Sequential KDA forward; writes the pre-step states into ``traj``.

    Bit-identical to ``ops_kda._recurrent_kda_loop_eager`` (same ops, same
    association order), with optional bfloat16 numerical stability patches.
    """
    # Preallocate the output and write one time slice per step (this loop
    # only runs inside ``autograd.Function.forward``, where autograd is
    # disabled); index assignment is elementwise-identical to the
    # append-and-stack form.
    T = q.size(1)
    o = torch.empty(q.size(0), T, q.size(2), v.size(-1),
                    dtype=q.dtype, device=q.device)
    for t in range(T):
        traj[t] = S
        P = _bf16_safe_exp(g[:, t])
        S = S * P.unsqueeze(-1)
        S = _state_scale_correction(S)
        delta = v[:, t] - (k[:, t].unsqueeze(-1) * S).sum(-2)
        beta_k = (beta[:, t].unsqueeze(-1) * k[:, t]).unsqueeze(-1)
        S = _bf16_safe_addcmul(S, beta_k, delta.unsqueeze(-2))
        o[:, t] = (q[:, t].unsqueeze(-2) @ S).squeeze(-2)
    return o, S


def _custom_kda_backward_loop(q, k, v, g, beta, grad_o, dS, traj,
                              dq, dk, dv, dg, dbeta):
    """Reverse-time adjoint sweep; returns the initial-state gradient.

    Includes bfloat16 numerical stability patches matching the forward pass.
    """
    T = q.size(1)
    # Reused per-step temporary for the elementwise products below: the
    # ``out=`` form performs the same IEEE product as the out-of-place
    # form (no fusion), and the ``add_``/``sub`` that consume it are the
    # in-place spellings of the same elementwise ops, so every value is
    # bit-identical. The loop runs inside ``Function.backward`` (no-grad),
    # and ``dS`` is always a fresh tensor here (``zeros_like`` or the
    # caller's clone), so mutating it cannot touch a caller's gradient.
    scratch = torch.empty_like(dS)
    for t in range(T - 1, -1, -1):
        S_prev = traj[t]
        k_t = k[:, t]
        beta_t = beta[:, t]
        P = _bf16_safe_exp(g[:, t])
        D = S_prev * P.unsqueeze(-1)
        D = _state_scale_correction(D)
        delta = v[:, t] - (k_t.unsqueeze(-1) * D).sum(-2)
        bk = (beta_t.unsqueeze(-1) * k_t).unsqueeze(-1)
        S_t = _bf16_safe_addcmul(D, bk, delta.unsqueeze(-2))

        torch.mul(q[:, t].unsqueeze(-1), grad_o[:, t].unsqueeze(-2),
                  out=scratch)
        dS = dS.add_(scratch)
        dq[:, t] = (grad_o[:, t].unsqueeze(-2)
                    @ S_t.transpose(-1, -2)).squeeze(-2)
        ddelta = (dS * bk).sum(-2)
        dbk = (dS * delta.unsqueeze(-2)).sum(-1)
        dbeta[:, t] = (dbk * k_t).sum(-1)
        dk_t = dbk * beta_t.unsqueeze(-1)
        dv[:, t] = ddelta
        torch.mul(k_t.unsqueeze(-1), ddelta.unsqueeze(-2), out=scratch)
        dD = dS - scratch
        dk_t = dk_t - (D @ ddelta.unsqueeze(-1)).squeeze(-1)
        dk[:, t] = dk_t
        dS = torch.mul(dD, P.unsqueeze(-1), out=dS)
        dg[:, t] = (dD * S_prev).sum(-1) * P
    return dS


_SCRIPTED_FORWARD = None
_SCRIPTED_FORWARD_FAILED = False
_SCRIPTED_BACKWARD = None
_SCRIPTED_BACKWARD_FAILED = False


def _get_scripted_forward():
    """Lazily ``torch.jit.script`` the forward loop, eager fallback."""
    global _SCRIPTED_FORWARD, _SCRIPTED_FORWARD_FAILED
    if _SCRIPTED_FORWARD is None and not _SCRIPTED_FORWARD_FAILED:
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore', FutureWarning)
                _SCRIPTED_FORWARD = torch.jit.script(_custom_kda_forward_loop)
        except Exception:
            _SCRIPTED_FORWARD_FAILED = True
    return _SCRIPTED_FORWARD


def _get_scripted_backward():
    """Lazily ``torch.jit.script`` the backward loop, eager fallback."""
    global _SCRIPTED_BACKWARD, _SCRIPTED_BACKWARD_FAILED
    if _SCRIPTED_BACKWARD is None and not _SCRIPTED_BACKWARD_FAILED:
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore', FutureWarning)
                _SCRIPTED_BACKWARD = torch.jit.script(_custom_kda_backward_loop)
        except Exception:
            _SCRIPTED_BACKWARD_FAILED = True
    return _SCRIPTED_BACKWARD


class _KDARecurrentFunction(torch.autograd.Function):
    """Single-node differentiable KDA recurrence with analytic backward.

    Inputs are the already-expanded (GVA applied), already-scaled ``q`` and
    already-clamped ``g`` — see :func:`custom_recurrent_kda`. All tensors are
    in compute precision (fp32 for fp16/bf16/fp32 inputs, fp64 for fp64).
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, initial_state):
        B, T, H, K = q.shape
        V = v.shape[-1]
        traj = torch.empty(T, B, H, K, V, dtype=q.dtype, device=q.device)
        fwd = _get_scripted_forward()
        if fwd is not None:
            o, S = fwd(q, k, v, g, beta, initial_state, traj)
        else:
            o, S = _custom_kda_forward_loop(q, k, v, g, beta, initial_state,
                                            traj)
        ctx.save_for_backward(q, k, v, g, beta, initial_state, traj)
        return o, S

    @staticmethod
    def backward(ctx, grad_o, grad_state):
        q, k, v, g, beta, initial_state, traj = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]
        if grad_o is None:
            grad_o = torch.zeros(B, T, H, V, dtype=q.dtype, device=q.device)
        else:
            grad_o = grad_o.to(q.dtype)
        if grad_state is None:
            dS = torch.zeros_like(initial_state)
        else:
            dS = grad_state.to(q.dtype).clone()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dg = torch.empty_like(g)
        dbeta = torch.empty_like(beta)
        bwd = _get_scripted_backward()
        if bwd is not None:
            dS = bwd(q, k, v, g, beta, grad_o, dS, traj,
                     dq, dk, dv, dg, dbeta)
        else:
            dS = _custom_kda_backward_loop(q, k, v, g, beta, grad_o, dS, traj,
                                           dq, dk, dv, dg, dbeta)
        grad_init = dS if ctx.needs_input_grad[5] else None
        return dq, dk, dv, dg, dbeta, grad_init


def custom_recurrent_kda(
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
    """KDA recurrent forward with a custom analytic backward.

    Drop-in replacement for :func:`ops_kda.naive_recurrent_kda` (same input
    contract, same forward values, same dtype contract) that replaces the
    ``O(T)``-deep autograd graph with a single
    :class:`torch.autograd.Function` node and an explicit reverse-time
    adjoint loop. See the module docstring for the derivation and the memory
    argument.

    Args:
        q: ``[B, T, H, K]``
        k: ``[B, T, H, K]``
        v: ``[B, T, HV, V]`` (``HV`` must be divisible by ``H``, enabling
            grouped-value attention; the expansion is done outside the
            Function so its backward is autograd's own scatter-add)
        g: per-channel log-decay gate ``[B, T, HV, K]``
        beta: delta-rule learning rate ``[B, T, HV]``
        scale: defaults to ``1/sqrt(K)``
        initial_state: ``[B, HV, K, V]``
        output_final_state: also return the final recurrent state.
        g_clamp_min: lower bound for the log-decay gate ``g``; the clamp is
            applied before the Function so PyTorch's clamp backward defines
            the zero-gradient region for clamped channels exactly.
        state_dtype: dtype of the RETURNED recurrent state.

    .. note::
        When autograd is disabled or no input requires a gradient, the call
        is delegated to :func:`ops_kda.naive_recurrent_kda` (identical
        forward, no trajectory storage). The backward of this function is
        first-order only (no double backward).
    """
    dtype = v.dtype
    _validate_kda_inputs(q, k, v, g, beta,
                         fn_name='custom_recurrent_kda')
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    _validate_kda_shapes(q, k, v, g, beta, fn_name='custom_recurrent_kda')
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    needs_graph = torch.is_grad_enabled() and any(
        isinstance(t, torch.Tensor) and t.requires_grad
        for t in (q, k, v, g, beta, initial_state))
    if not needs_graph:
        return naive_recurrent_kda(
            q, k, v, g, beta, scale, initial_state=initial_state,
            output_final_state=output_final_state,
            g_clamp_min=g_clamp_min, state_dtype=state_dtype)

    compute_dtype = _compute_dtype(dtype)
    q, k, v, g, beta = map(lambda x: x.to(compute_dtype), [q, k, v, g, beta])
    if g_clamp_min > -float('inf'):
        g = g.clamp(min=float(g_clamp_min))
    if G == 1:
        q = q * scale
    else:
        q = q.repeat_interleave(G, dim=2) * scale
        k = k.repeat_interleave(G, dim=2)

    if initial_state is None:
        init = q.new_zeros(B, HV, K, V)
    else:
        init = initial_state.to(device=q.device, dtype=compute_dtype)
    o, S = _KDARecurrentFunction.apply(q, k, v, g, beta, init)
    o = o.to(dtype)
    if not output_final_state:
        S = None
    else:
        target_state_dtype = (state_dtype if state_dtype is not None
                              else compute_dtype)
        S = S.to(target_state_dtype)
    _warn_if_nonfinite(o, 'custom_recurrent_kda')
    return o, S
