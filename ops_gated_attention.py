"""Gated Attention Mixin — unified gating interface for KDA/CSA/HCA.

Provides optional per-head or per-channel gating for any attention operator.
The gate is learned via nn.Parameter, initialized near 1.0 (identity mapping).

Gate types:
  - 'none': no gating (identity)
  - 'per_head': [H] or [HV] gate shared across K/V channels
  - 'per_channel': [H, K] or [HV, K] gate per key/value channel
"""

from __future__ import annotations

import math as _math
import warnings as _warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMixin:
    """Mixin that adds optional gating to attention operators.

    Usage:
        class MyAttention(GatedAttentionMixin, nn.Module):
            def __init__(self, ..., gate_type='per_head', gate_dim=64):
                super().__init__()
                self._init_gate(gate_type, gate_dim)

            def forward(self, q, k, v, ...):
                # Apply gating to keys/values before attention
                k = self.apply_gate(k, dim=-1)
                v = self.apply_gate(v, dim=-1)
                ...
    """

    def _init_gate(self, gate_type: str, gate_dim: int, num_heads: int = 1):
        """Initialize gating parameters.

        Args:
            gate_type: 'none' | 'per_head' | 'per_channel'
            gate_dim: dimension of the gated axis (K for keys, V for values)
            num_heads: number of heads (H for Q/K, HV for V)
        """
        if gate_type not in ('none', 'per_head', 'per_channel'):
            raise ValueError(
                f"gate_type={gate_type!r} must be 'none', 'per_head', or "
                f"'per_channel'")
        # 'none' has no gate tensor, so its geometry arguments are unused
        # (mirroring ``create_gate``'s early return). A real gate must have
        # positive integer dimensions: zero/negative entries created a
        # zero-length parameter that only failed later, deep inside
        # ``apply_gate``.
        if gate_type != 'none':
            gate_dim = _require_positive_int('gate_dim', gate_dim)
            num_heads = _require_positive_int('num_heads', num_heads)
        self.gate_type = gate_type
        self.gate_dim = gate_dim
        self.num_heads = num_heads

        if gate_type == 'none':
            self.register_buffer('_gate', None)
        elif gate_type == 'per_head':
            # Shape: [num_heads] -> broadcast to [num_heads, gate_dim]
            self.register_parameter(
                '_gate', nn.Parameter(torch.ones(num_heads))
            )
        elif gate_type == 'per_channel':
            # Shape: [num_heads, gate_dim]
            self.register_parameter(
                '_gate', nn.Parameter(torch.ones(num_heads, gate_dim))
            )
        else:  # pragma: no cover - validated above
            raise ValueError(
                f"gate_type={gate_type!r} must be 'none', 'per_head', or "
                f"'per_channel'"
            )

    def apply_gate(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Apply learned gate to tensor along specified dimension.

        Args:
            x: input tensor [..., num_heads, gate_dim] or [..., gate_dim]
            dim: feature dimension to gate (must match gate_dim); the head
                axis, when present, is the axis immediately before it.

        Returns:
            Gated tensor of same shape as x.
        """
        if self.gate_type == 'none' or self._gate is None:
            return x
        gate = _broadcast_gate(self._gate, x, dim)
        return x * gate.to(dtype=x.dtype, device=x.device)

    def get_gate_values(self) -> torch.Tensor | None:
        """Return current gate values (for logging/analysis)."""
        return self._gate.detach() if self._gate is not None else None

    def extra_repr(self) -> str:
        return f"gate_type={self.gate_type}, gate_dim={self.gate_dim}, num_heads={self.num_heads}"


def _require_positive_int(name: str, value) -> int:
    """Exact positive integer geometry (bool is not a dimension)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name}={value!r} must be a positive int.')
    return value


def _require_finite_real(name: str, value) -> float:
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not _math.isfinite(float(value))):
        raise ValueError(f'{name}={value!r} must be a finite number.')
    return float(value)


def _require_gate_clamp(name: str, value, *, allow_negative_inf: bool) -> float:
    """Validate a multiplicative-gate clamp bound.

    ``g_clamp_min=-inf`` is a meaningful way to disable the lower clamp;
    ``+inf`` is not a bound (``exp(+inf)`` poisons the state) and is
    rejected on either side. NaN and non-real values are rejected too.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name}={value!r} must be a real number.')
    if value != value:  # NaN
        raise ValueError(f'{name}={value!r} must not be NaN.')
    if value == float('inf'):
        raise ValueError(
            f'{name}=+inf is not a valid clamp bound (exp(+inf) has no '
            f'finite meaning).')
    if not allow_negative_inf and value == float('-inf'):
        raise ValueError(f'{name}=-inf is not a valid upper clamp bound.')
    return float(value)


def create_gate(
    gate_type: str,
    num_heads: int,
    gate_dim: int,
    init_value: float = 1.0,
    init_std: float = 0.02,
    device=None,
    dtype=None,
) -> torch.Tensor | None:
    """Factory for standalone gate tensors (non-parameter use cases).

    Args:
        gate_type: 'none' | 'per_head' | 'per_channel'
        num_heads: number of heads
        gate_dim: dimension per head
        init_value: initial gate value (near 1.0 for identity)
        init_std: std for normal initialization around init_value
        device: target device
        dtype: target dtype

    Returns:
        Gate tensor or None if 'none'.
    """
    if gate_type == 'none':
        return None
    num_heads = _require_positive_int('num_heads', num_heads)
    gate_dim = _require_positive_int('gate_dim', gate_dim)
    init_value = _require_finite_real('init_value', init_value)
    init_std = _require_finite_real('init_std', init_std)
    if init_std < 0.0:
        raise ValueError(f'init_std={init_std!r} must be >= 0.')
    if gate_type == 'per_head':
        gate = torch.normal(init_value, init_std, (num_heads,), device=device, dtype=dtype)
    elif gate_type == 'per_channel':
        gate = torch.normal(init_value, init_std, (num_heads, gate_dim), device=device, dtype=dtype)
    else:
        raise ValueError(f"gate_type={gate_type!r} must be 'none', 'per_head', or 'per_channel'")
    return gate


def _broadcast_gate(gate: torch.Tensor, x: torch.Tensor,
                    dim: int) -> torch.Tensor:
    """Reshape ``gate`` so its feature axis lands on ``dim`` of ``x``."""
    if gate.dim() not in (1, 2):
        raise RuntimeError(f"Unexpected gate shape: {gate.shape}")
    if x.dim() == 0:
        raise ValueError('apply_gate: x must have at least one dimension.')
    axis = dim if dim >= 0 else x.dim() + dim
    if axis < 0 or axis >= x.dim():
        raise ValueError(
            f'apply_gate: dim={dim} is out of range for x with '
            f'{x.dim()} dimensions.')
    n_heads = gate.shape[0]
    head_axis = axis - 1
    if gate.dim() == 1:
        if n_heads == 1:
            return gate.reshape(())
        if head_axis < 0 or x.shape[head_axis] not in (1, n_heads):
            raise ValueError(
                f'apply_gate: gate has {n_heads} heads but the axis before '
                f'dim={dim} of x has size '
                f'{None if head_axis < 0 else x.shape[head_axis]}.')
        shape = [1] * x.dim()
        shape[head_axis] = n_heads
        return gate.view(shape)
    gate_dim = gate.shape[-1]
    if x.shape[axis] != gate_dim:
        raise ValueError(
            f'apply_gate: x.shape[{axis}]={x.shape[axis]} does not match '
            f'gate feature dim {gate_dim}.')
    if n_heads != 1 and (
            head_axis < 0 or x.shape[head_axis] not in (1, n_heads)):
        raise ValueError(
            f'apply_gate: gate has {n_heads} heads but the axis before '
            f'dim={dim} of x has size '
            f'{None if head_axis < 0 else x.shape[head_axis]}.')
    shape = [1] * x.dim()
    shape[axis] = gate_dim
    if n_heads != 1:
        shape[head_axis] = n_heads
    return gate.view(shape)


def apply_gate_fn(
    x: torch.Tensor,
    gate: torch.Tensor | None,
    dim: int = -1,
) -> torch.Tensor:
    """Functional form of gate application (no module state).

    Args:
        x: [..., num_heads, gate_dim] or [..., gate_dim]
        gate: [num_heads] or [num_heads, gate_dim] or None
        dim: feature dimension to gate

    Returns:
        Gated tensor.
    """
    if gate is None:
        return x
    gate = _broadcast_gate(gate, x, dim)
    return x * gate.to(dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Sequential loop bodies (TorchScript-compiled with an eager fallback, the
# same pattern the KDA / RWKV-7 / Mamba-2 reference scans use). ``gate_state``
# is the pre-broadcast ``exp(clamp(g))`` view, ``beta_k`` the precomputed
# ``beta * k`` product (identical per-element values); both are consumed
# with the exact operand association of the plain-Python loop, so the values
# are bit-identical.
# ---------------------------------------------------------------------------
_GATED_DELTA_LOOP_SCRIPTED = None
_GATED_DELTA_LOOP_SCRIPTED_FAILED = False
_GATED_DELTA_LOOP_INPLACE_SCRIPTED = None
_GATED_DELTA_LOOP_INPLACE_SCRIPTED_FAILED = False


def _gated_delta_loop_eager(q, k, v, gate_state, beta_k, S):
    outs = []
    for q_t, k_t, v_t, bk_t in zip(
            q.unbind(1), k.unbind(1), v.unbind(1), beta_k.unbind(1)):
        S = S * gate_state
        delta = v_t - (k_t.unsqueeze(-1) * S).sum(dim=-2)
        S = torch.addcmul(S, bk_t.unsqueeze(-1), delta.unsqueeze(-2))
        outs.append(q_t.unsqueeze(-2) @ S)
    return torch.stack(outs, dim=1).squeeze(-2), S


def _gated_delta_loop_inplace_eager(q, k, v, gate_state, beta_k, S,
                                    first_gated: bool):
    """No-grad loop body updating ``S`` in place.

    ``first_gated=True`` means the caller has already applied the step-0
    gate multiplication out of place (a caller-provided ``initial_state``
    must not be mutated); the loop then applies the in-place gate at every
    later step. With ``first_gated=False`` the in-place gate runs at every
    step (the caller's zero-initialized state is exclusively owned, so the
    in-place multiply keeps the state's dtype exactly as before).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    # Collect the per-step readouts and stack once at the end: the single
    # batched copy replaces T tiny per-step copies (elementwise-identical
    # values; the stack writes the [B, T, H, V] directly).
    outs: list[torch.Tensor] = []
    for t, (q_t, k_t, v_t, bk_t) in enumerate(zip(
            q.unbind(1), k.unbind(1), v.unbind(1), beta_k.unbind(1))):
        if t > 0 or not first_gated:
            S = S.mul_(gate_state)
        delta = v_t - (k_t.unsqueeze(-1) * S).sum(dim=-2)
        S = S.addcmul_(bk_t.unsqueeze(-1), delta.unsqueeze(-2))
        outs.append((q_t.unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(outs, dim=1), S


def _get_scripted_gated_delta_loop():
    global _GATED_DELTA_LOOP_SCRIPTED, _GATED_DELTA_LOOP_SCRIPTED_FAILED
    if _GATED_DELTA_LOOP_SCRIPTED is None and not _GATED_DELTA_LOOP_SCRIPTED_FAILED:
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore', FutureWarning)
                _GATED_DELTA_LOOP_SCRIPTED = torch.jit.script(
                    _gated_delta_loop_eager)
        except Exception:
            _GATED_DELTA_LOOP_SCRIPTED_FAILED = True
    return _GATED_DELTA_LOOP_SCRIPTED


def _get_scripted_gated_delta_loop_inplace():
    global _GATED_DELTA_LOOP_INPLACE_SCRIPTED
    global _GATED_DELTA_LOOP_INPLACE_SCRIPTED_FAILED
    if (_GATED_DELTA_LOOP_INPLACE_SCRIPTED is None
            and not _GATED_DELTA_LOOP_INPLACE_SCRIPTED_FAILED):
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore', FutureWarning)
                _GATED_DELTA_LOOP_INPLACE_SCRIPTED = torch.jit.script(
                    _gated_delta_loop_inplace_eager)
        except Exception:
            _GATED_DELTA_LOOP_INPLACE_SCRIPTED_FAILED = True
    return _GATED_DELTA_LOOP_INPLACE_SCRIPTED


class GatedDeltaRule(nn.Module):
    """Fused gated delta-rule layer: combines per-channel gating with delta update.

    This generalizes KDA's gating mechanism (log-decay gate `g`) into a
    reusable module that can enhance CSA/HCA or serve as a standalone
    gated associative memory.

    Forward:
        S_t = S_{t-1} * gate_t + beta_t * k_t * (v_t - k_t^T S_{t-1})^T
        o_t = q_t^T S_t

    Where gate_t = exp(g_t) with g_t being the learned log-gate.
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        gate_type: str = 'per_channel',
        beta_init: float = 0.1,
        g_init: float = -2.0,  # exp(-2) ≈ 0.135, moderate decay
        g_clamp_min: float = -10.0,
        g_clamp_max: float = 50.0,
    ):
        super().__init__()
        self.num_heads = _require_positive_int('num_heads', num_heads)
        self.key_dim = _require_positive_int('key_dim', key_dim)
        self.value_dim = _require_positive_int('value_dim', value_dim)
        self.gate_type = gate_type
        # The gate/beta initial values are baked into nn.Parameters and the
        # clamps are applied on every forward. Reject non-finite values
        # up-front: a NaN ``g_init`` silently makes every output NaN, while
        # ``g_clamp_max=+inf`` can turn the state multiplicative update into
        # an overflow instead of an actionable configuration error.
        self.g_init = _require_finite_real('g_init', g_init)
        self.beta_init = _require_finite_real('beta_init', beta_init)
        self.g_clamp_min = _require_gate_clamp(
            'g_clamp_min', g_clamp_min, allow_negative_inf=True)
        self.g_clamp_max = _require_gate_clamp(
            'g_clamp_max', g_clamp_max, allow_negative_inf=False)
        if self.g_clamp_min > self.g_clamp_max:
            raise ValueError(
                f'g_clamp_min={self.g_clamp_min!r} must be <= '
                f'g_clamp_max={self.g_clamp_max!r}.')

        # Learnable log-gate (g) and beta
        if gate_type == 'per_channel':
            self.g = nn.Parameter(
                torch.full((num_heads, key_dim), self.g_init))
            self.beta = nn.Parameter(
                torch.full((num_heads,), self.beta_init))
        elif gate_type == 'per_head':
            self.g = nn.Parameter(torch.full((num_heads,), self.g_init))
            self.beta = nn.Parameter(
                torch.full((num_heads,), self.beta_init))
        else:
            raise ValueError(f"gate_type={gate_type!r} must be 'per_head' or 'per_channel'")

    def forward(
        self,
        q: torch.Tensor,     # [B, T, H, K]
        k: torch.Tensor,     # [B, T, H, K]
        v: torch.Tensor,     # [B, T, H, V]
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Gated delta-rule recurrence.

        Args:
            q: queries [B, T, H, K]
            k: keys [B, T, H, K]
            v: values [B, T, H, V]
            initial_state: optional initial state [B, H, K, V]

        Returns:
            outputs [B, T, H, V], final_state [B, H, K, V] or None
        """
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                'GatedDeltaRule: q/k/v must be 4-D [B, T, H, *]; got '
                f'{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}.')
        B, T, H, K = q.shape
        if tuple(k.shape) != (B, T, H, K):
            raise ValueError(
                'GatedDeltaRule: k must share q shape '
                f'{tuple(q.shape)}; got {tuple(k.shape)}. The recurrence '
                'zips the time axis, so a shorter k would silently truncate '
                'the output.')
        if tuple(v.shape[:3]) != (B, T, H):
            raise ValueError(
                'GatedDeltaRule: v must be [B, T, H, V] matching q[:3]='
                f'{(B, T, H)}; got {tuple(v.shape)}.')
        V = v.shape[-1]
        # NOTE: use ``raise ValueError`` (NOT ``assert``) so the check
        # survives ``python -O`` / ``PYTHONOPTIMIZE=1``; ``assert``
        # statements are silently stripped under optimization, which would
        # turn a geometry mismatch into an opaque low-level error from
        # inside the recurrence.
        if H != self.num_heads:
            raise ValueError(
                f'GatedDeltaRule: query head count H={H} does not match '
                f'num_heads={self.num_heads}.')
        if K != self.key_dim:
            raise ValueError(
                f'GatedDeltaRule: key dim K={K} does not match '
                f'key_dim={self.key_dim}.')
        if V != self.value_dim:
            raise ValueError(
                f'GatedDeltaRule: value dim V={V} does not match '
                f'value_dim={self.value_dim}.')

        # The state geometry is part of the operator contract. Accepting a
        # broadcastable slice (e.g. [B, H, K, 1] or [1, H, K, V]) would
        # silently run a smaller state and return a differently shaped one.
        if initial_state is not None:
            if not isinstance(initial_state, torch.Tensor):
                raise TypeError(
                    'GatedDeltaRule: initial_state must be a torch.Tensor or '
                    f'None, got {type(initial_state).__name__}.')
            expected = (B, H, K, V)
            if tuple(initial_state.shape) != expected:
                raise ValueError(
                    'GatedDeltaRule: initial_state.shape='
                    f'{tuple(initial_state.shape)} does not match the '
                    f'expected {expected} ([B, H, K, V]).')

        dtype = q.dtype
        device = q.device

        # Compute gate = exp(clamped(g))
        g_clamped = self.g.clamp(min=self.g_clamp_min, max=self.g_clamp_max)
        gate = g_clamped.exp()  # [H, K] or [H]

        # Expand gate for broadcasting: match S shape [B, H, K, V]
        if self.gate_type == 'per_channel':
            gate_state = gate.view(1, H, K, 1)  # for S [B, H, K, V]
        else:  # per_head
            gate_state = gate.view(1, H, 1, 1)

        beta = self.beta.view(1, H, 1, 1)  # [1, H, 1, 1]

        # Initialize state
        if initial_state is None:
            S = torch.zeros(B, H, K, V, dtype=dtype, device=device)
        else:
            S = initial_state

        # Parameters default to float32, and a caller-provided initial
        # state may differ from the activation dtype. Promote every operand
        # (including the state) to one work dtype: multiplying by a
        # float32 gate would otherwise promote ``S`` and then fail the
        # ``q @ S`` matmul on mixed dtypes. When all inputs already share a
        # dtype this is a no-op and preserves the original output dtype.
        work_dtype = torch.promote_types(
            torch.promote_types(q.dtype, k.dtype),
            torch.promote_types(v.dtype, S.dtype))
        if q.dtype != work_dtype:
            q = q.to(work_dtype)
        if k.dtype != work_dtype:
            k = k.to(work_dtype)
        if v.dtype != work_dtype:
            v = v.to(work_dtype)
        if S.dtype != work_dtype:
            S = S.to(work_dtype)

        # Empty sequence: no steps to take, but the result must follow the
        # same dtype contract as the non-empty path. In particular a
        # mixed-dtype ``initial_state`` promotes both the returned state
        # and the (empty) output; returning the raw initial state / q dtype
        # would make T=0 disagree with T=1.
        if T == 0:
            out_dtype = torch.promote_types(dtype, S.dtype)
            return (torch.zeros(B, 0, H, V, dtype=out_dtype,
                                device=device), S)

        gate_state = gate_state.to(dtype=work_dtype, device=device)
        beta = beta.to(dtype=work_dtype, device=device)

        # ``beta * k`` is a pure elementwise broadcast of one scalar per
        # head onto each key element; hoisting the whole-sequence product
        # out of the loop computes the identical per-element values and
        # removes one loop-body allocation per step. ``beta`` is reshaped
        # so the head scalar broadcasts onto the key axis exactly as the
        # per-step ``beta * k_t`` did.
        beta_k_all = beta.view(1, 1, H, 1) * k

        # No-grad inference may update ``S`` in place after the first
        # iteration, when ``S`` is guaranteed to be owned by this call (a
        # fresh zero state, or the out-of-place product below); a
        # caller-provided ``initial_state`` is never mutated. The output is
        # preallocated in the same branch: index assignment is
        # elementwise-identical to the append-and-stack form (the promoted
        # dtype of ``q @ S`` fixes the output dtype up front; a provided
        # ``initial_state`` may promote it beyond ``q.dtype``). While a
        # graph is recorded the append-and-stack form is kept. The loop body
        # is TorchScript-compiled when available (same ops, same order).
        inplace = not torch.is_grad_enabled()
        if inplace:
            o_dtype = torch.promote_types(dtype, S.dtype)
            first_gated = initial_state is not None
            if first_gated:
                # First-step multiply stays out of place (the caller's
                # initial state must not be mutated); remaining steps run
                # in place on the now-owned state.
                S = S * gate_state
            loop_i = _get_scripted_gated_delta_loop_inplace()
            if loop_i is not None:
                o, S = loop_i(q, k, v, gate_state, beta_k_all, S,
                              first_gated)
            else:
                o, S = _gated_delta_loop_inplace_eager(
                    q, k, v, gate_state, beta_k_all, S, first_gated)
            if o.dtype != o_dtype:
                o = o.to(o_dtype)
            return o, S
        loop = _get_scripted_gated_delta_loop()
        if loop is not None:
            o, S = loop(q, k, v, gate_state, beta_k_all, S)
        else:
            o, S = _gated_delta_loop_eager(q, k, v, gate_state, beta_k_all, S)
        return o, S


# Convenience function for functional use
def gated_delta_rule_step(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    gate: torch.Tensor,     # [H, K] or [H]
    beta: torch.Tensor,     # [H]
    gate_type: str = 'per_channel',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single step of gated delta-rule (functional, no module state).

    Args:
        S: [B, H, K, V] recurrent state
        q_t: [B, H, K] query
        k_t: [B, H, K] key
        v_t: [B, H, V] value
        gate: [H, K] or [H] gate values (already exp()'ed)
        beta: [H] learning rates
        gate_type: 'per_channel' | 'per_head'

    Returns:
        output [B, H, V], new_state [B, H, K, V]
    """
    if gate_type not in ('per_channel', 'per_head'):
        raise ValueError(
            f"gate_type={gate_type!r} must be 'per_channel' or 'per_head'")
    if S.dim() != 4:
        raise ValueError(
            f'gated_delta_rule_step: S must be [B, H, K, V], got '
            f'{tuple(S.shape)}.')
    B, H, K, V = S.shape
    if q_t.shape != (B, H, K):
        raise ValueError(
            f'gated_delta_rule_step: q_t.shape={tuple(q_t.shape)} must be '
            f'{(B, H, K)}.')
    if k_t.shape != (B, H, K):
        raise ValueError(
            f'gated_delta_rule_step: k_t.shape={tuple(k_t.shape)} must be '
            f'{(B, H, K)}.')
    if v_t.shape != (B, H, V):
        raise ValueError(
            f'gated_delta_rule_step: v_t.shape={tuple(v_t.shape)} must be '
            f'{(B, H, V)}.')
    if beta.shape != (H,):
        raise ValueError(
            f'gated_delta_rule_step: beta.shape={tuple(beta.shape)} must be '
            f'{(H,)}.')
    expected_gate = (H, K) if gate_type == 'per_channel' else (H,)
    if gate.shape != expected_gate:
        raise ValueError(
            f'gated_delta_rule_step: gate.shape={tuple(gate.shape)} must be '
            f'{expected_gate} for gate_type={gate_type!r}.')

    # Match ``GatedDeltaRule.forward``: promote every operand (state, q/k/v,
    # gate, beta) to one work dtype before the recurrence. Otherwise a
    # float64 state with float32 activations fails the ``q_t @ S`` matmul
    # even though the module-level operator accepts the same combination.
    work_dtype = torch.promote_types(
        torch.promote_types(q_t.dtype, k_t.dtype),
        torch.promote_types(v_t.dtype, S.dtype))
    work_dtype = torch.promote_types(work_dtype, gate.dtype)
    work_dtype = torch.promote_types(work_dtype, beta.dtype)
    if S.dtype != work_dtype:
        S = S.to(work_dtype)
    if q_t.dtype != work_dtype:
        q_t = q_t.to(work_dtype)
    if k_t.dtype != work_dtype:
        k_t = k_t.to(work_dtype)
    if v_t.dtype != work_dtype:
        v_t = v_t.to(work_dtype)
    if gate.dtype != work_dtype:
        gate = gate.to(work_dtype)
    if beta.dtype != work_dtype:
        beta = beta.to(work_dtype)

    if gate_type == 'per_channel':
        gate_state = gate.view(1, H, K, 1)
    else:
        gate_state = gate.view(1, H, 1, 1)

    beta = beta.view(1, H, 1, 1)  # [1, H, 1, 1]
    beta_k = beta * k_t.unsqueeze(-1)  # [B, H, K, 1]

    # Decay state
    S = S * gate_state

    # Delta rule update
    kTS = (k_t.unsqueeze(-1) * S).sum(dim=-2)  # [B, H, V]
    delta = v_t - kTS
    S = torch.addcmul(S, beta_k, delta.unsqueeze(-2))

    # Output
    o_t = (q_t.unsqueeze(-2) @ S).squeeze(-2)
    return o_t, S