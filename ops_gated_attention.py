"""Gated Attention Mixin — unified gating interface for KDA/CSA/HCA.

Provides optional per-head or per-channel gating for any attention operator.
The gate is learned via nn.Parameter, initialized near 1.0 (identity mapping).

Gate types:
  - 'none': no gating (identity)
  - 'per_head': [H] or [HV] gate shared across K/V channels
  - 'per_channel': [H, K] or [HV, K] gate per key/value channel
"""

from __future__ import annotations

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
        else:
            raise ValueError(
                f"gate_type={gate_type!r} must be 'none', 'per_head', or 'per_channel'"
            )

    def apply_gate(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Apply learned gate to tensor along specified dimension.

        Args:
            x: input tensor [..., num_heads, gate_dim] or [..., gate_dim]
            dim: dimension to gate (must match gate_dim)

        Returns:
            Gated tensor of same shape as x.
        """
        if self.gate_type == 'none' or self._gate is None:
            return x

        gate = self._gate
        # Expand gate to match x's head dimension if needed
        if gate.dim() == 1:
            # per_head: [H] -> [..., H, 1]
            gate = gate.view(*([1] * (x.dim() - 2)), gate.shape[0], 1)
        elif gate.dim() == 2:
            # per_channel: [H, D] -> [..., H, D]
            gate = gate.view(*([1] * (x.dim() - 2)), gate.shape[0], gate.shape[1])
        else:
            raise RuntimeError(f"Unexpected gate shape: {gate.shape}")

        # Broadcast and multiply
        return x * gate.to(x.dtype).to(x.device)

    def get_gate_values(self) -> torch.Tensor | None:
        """Return current gate values (for logging/analysis)."""
        return self._gate.detach() if self._gate is not None else None

    def extra_repr(self) -> str:
        return f"gate_type={self.gate_type}, gate_dim={self.gate_dim}, num_heads={self.num_heads}"


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
    if gate_type == 'per_head':
        gate = torch.normal(init_value, init_std, (num_heads,), device=device, dtype=dtype)
    elif gate_type == 'per_channel':
        gate = torch.normal(init_value, init_std, (num_heads, gate_dim), device=device, dtype=dtype)
    else:
        raise ValueError(f"gate_type={gate_type!r} must be 'none', 'per_head', or 'per_channel'")
    return gate


def apply_gate_fn(
    x: torch.Tensor,
    gate: torch.Tensor | None,
    dim: int = -1,
) -> torch.Tensor:
    """Functional form of gate application (no module state).

    Args:
        x: [..., num_heads, gate_dim] or [..., gate_dim]
        gate: [num_heads] or [num_heads, gate_dim] or None
        dim: dimension to gate

    Returns:
        Gated tensor.
    """
    if gate is None:
        return x

    # Align gate dimensions with x
    if gate.dim() == 1:
        gate = gate.view(*([1] * (x.dim() - 2)), gate.shape[0], 1)
    elif gate.dim() == 2:
        gate = gate.view(*([1] * (x.dim() - 2)), gate.shape[0], gate.shape[1])

    return x * gate.to(x.dtype).to(x.device)


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
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.gate_type = gate_type
        self.g_clamp_min = g_clamp_min
        self.g_clamp_max = g_clamp_max

        # Learnable log-gate (g) and beta
        if gate_type == 'per_channel':
            self.g = nn.Parameter(torch.full((num_heads, key_dim), g_init))
            self.beta = nn.Parameter(torch.full((num_heads,), beta_init))
        elif gate_type == 'per_head':
            self.g = nn.Parameter(torch.full((num_heads,), g_init))
            self.beta = nn.Parameter(torch.full((num_heads,), beta_init))
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
        B, T, H, K = q.shape
        V = v.shape[-1]
        assert H == self.num_heads
        assert K == self.key_dim
        assert V == self.value_dim

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

        outputs = []
        for t in range(T):
            # S = S * gate
            S = S * gate_state

            # delta = v_t - k_t^T S
            k_t = k[:, t].unsqueeze(-1)      # [B, H, K, 1]
            v_t = v[:, t]                    # [B, H, V]
            kTS = (k_t * S).sum(dim=-2)      # [B, H, V]
            delta = v_t - kTS                # [B, H, V]

            # S = S + beta * k_t * delta^T
            beta_k = beta * k_t              # [B, H, K, 1]
            S = torch.addcmul(S, beta_k, delta.unsqueeze(-2))  # [B, H, K, V]

            # o_t = q_t^T S
            q_t = q[:, t].unsqueeze(-2)      # [B, H, 1, K]
            o_t = (q_t @ S).squeeze(-2)      # [B, H, V]
            outputs.append(o_t)

        o = torch.stack(outputs, dim=1)      # [B, T, H, V]
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
    B, H, K, V = S.shape

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