"""Gradient-checkpointing policies + retained-activation accounting (Exp 14).

Why this module?
----------------
Training-memory planning for the hybrid stack has two knobs that interact
and are nowhere quantified together in the repo:

* the memory route (KDA / Mamba-2 / RWKV-7) fixes the operator internals;
* **gradient checkpointing** trades one extra forward pass for dropping the
  layer-internal activations retained by autograd.

``ops_fused`` documents that checkpointing REQUIRES the pure functional
forward (``HybridAttentionStack.forward_functional`` / the per-layer
``forward_functional``), because the stateful ``forward`` mutates
``_kda_state`` / ``_conv_lookback`` and a recompute would observe the
mutated state. This module turns that requirement into a working policy
engine:

    policy 'none'      -- no layer is checkpointed (plain 3x training FLOPs)
    policy 'all'       -- every layer is checkpointed
    policy 'memory'    -- only the recurrent memory layers (kda/mamba2/rwkv7)
    policy 'retrieval' -- only the aggregation layers (csa/hca)

:func:`forward_with_policy` mirrors ``HybridAttentionStack.forward_functional``
exactly (same residual/norm plumbing, same state/lookback threading) and
wraps selected layers in ``torch.utils.checkpoint.checkpoint(...,
use_reentrant=False)``. The tests pin three contracts:

1. **value parity** — outputs, states, and lookbacks equal
   ``forward_functional`` for every policy (deterministic forward);
2. **gradient parity** — parameter gradients match the uncheckpointed
   baseline (this is what checkpointing must NOT change);
3. **bytes contract** — :class:`SavedBytesProbe` measures the bytes the
   autograd graph retains after the forward via
   ``torch.autograd.graph.saved_tensors_hooks``; checkpointed policies
   retain strictly fewer than ``'none'``.

Measurement honesty
-------------------
The probe counts tensors **saved for backward during the forward**. During
the backward recompute, checkpointing saves tensors again (transiently)
before freeing them; those are NOT counted, so the probe measures
*retained activation footprint*, which is exactly what checkpointing
optimizes — not total autograd traffic. The per-layer tag is attributed
from the policy loop (a tensor saved inside a layer's forward is tagged
with that layer's kind), which is valid because the stack forward is
strictly sequential.

The analytic overhead is a one-line model: training FLOPs are
``3 * forward`` (fwd + bwd) plus ``1 * forward`` per checkpointed layer
(:func:`checkpoint_overhead_model`). :func:`minimum_retained_bytes` is the
per-layer-input floor both policies must retain — the measured probe sits
above it and the software gap between the two policies is the deliverable.
"""
from __future__ import annotations

import contextlib

import torch
import torch.utils.checkpoint as torch_checkpoint

from ops_fused import MEMORY_LAYER_KINDS

#: Selectable checkpointing policies (stable order for tables).
CHECKPOINT_POLICIES = ('none', 'all', 'memory', 'retrieval')

#: Layer kinds served by the retrieval / global-aggregation operators.
RETRIEVAL_LAYER_KINDS = ('csa', 'hca')


def parse_policy(policy) -> str:
    """Validate and return a checkpointing policy name."""
    if not isinstance(policy, str):
        raise TypeError(
            f"checkpoint policy must be a str, got {type(policy).__name__}.")
    if policy not in CHECKPOINT_POLICIES:
        raise ValueError(
            f"checkpoint policy {policy!r} must be one of "
            f"{list(CHECKPOINT_POLICIES)}.")
    return policy


def should_checkpoint(kind: str, policy: str) -> bool:
    """Whether a layer of ``kind`` is checkpointed under ``policy``."""
    policy = parse_policy(policy)
    if policy == 'none':
        return False
    if policy == 'all':
        return True
    if policy == 'memory':
        return kind in MEMORY_LAYER_KINDS
    # 'retrieval'
    return kind in RETRIEVAL_LAYER_KINDS


def layout_of(stack) -> list[str]:
    """Layer-kind list of a stack (accepts the stack or a raw sequence)."""
    layout = getattr(stack, 'layout', stack)
    if isinstance(layout, str):
        raise TypeError(
            "layout_of: pass the stack or a sequence of kinds, not a "
            "dash-joined string.")
    return list(layout)


def checkpoint_overhead_model(stack_or_layout, policy: str) -> dict:
    """Analytic recompute cost of a checkpointing policy.

    Model (documented in the module docstring): one forward is ``1``;
    training without checkpointing is ``3`` (fwd + bwd = 2 fwd-equivalents);
    each checkpointed layer adds one more forward during backward. The
    returned factor is therefore ``(3n + c) / (3n)`` with ``n`` layers and
    ``c`` checkpointed ones.
    """
    policy = parse_policy(policy)
    layout = layout_of(stack_or_layout)
    checked = [k for k in layout if should_checkpoint(k, policy)]
    n = len(layout)
    c = len(checked)
    base = 3.0 * n if n else 1.0
    return {
        'policy': policy,
        'n_layers': n,
        'n_layers_checked': c,
        'checked_kinds': checked,
        'extra_forward_fraction': (c / n) if n else 0.0,
        'training_flops_relative_to_unchecked': (
            (3.0 * n + c) / base if n else 1.0),
    }


def minimum_retained_bytes(n_layers: int, batch: int, seq_len: int,
                           d_model: int, *, itemsize: int = 4) -> int:
    """Lower bound (bytes) on activations retained by any policy.

    Every layer retains at least its own pre-norm input (one
    ``[B, T, d_model]`` tensor) for the residual backward, checkpointed or
    not; layer internals dominate in practice and are what checkpointing
    drops. This floor keeps the probe numbers interpretable.
    """
    for name, value in (('n_layers', n_layers), ('batch', batch),
                        ('seq_len', seq_len), ('d_model', d_model)):
        if isinstance(value, bool) or not isinstance(value, int) \
                or value < 0:
            raise ValueError(f"{name}={value!r} must be a non-negative int.")
    if isinstance(itemsize, bool) or not isinstance(itemsize, int) \
            or itemsize < 1:
        raise ValueError(f"itemsize={itemsize!r} must be a positive int.")
    return int(n_layers) * int(batch) * int(seq_len) * int(d_model) * int(
        itemsize)


class SavedBytesProbe:
    """Count bytes retained for backward via saved-tensor hooks.

    Usage::

        probe = SavedBytesProbe()
        with probe.measure():
            out, state, lookbacks = gc.forward_with_policy(
                stack, x, policy=..., probe=probe)
        loss.backward()
        probe.total_bytes()          # retained during that forward

    ``measure()`` installs ``torch.autograd.graph.saved_tensors_hooks`` and
    enables counting for the duration of the block only; backward (and the
    checkpoint recompute inside it) runs after the block and is not
    counted. ``tagged(name)`` attributes saves made inside its block to
    ``name`` (the policy loop tags by layer kind).
    """

    def __init__(self) -> None:
        self._bytes: dict[str, int] = {}
        self._tensors: dict[str, int] = {}
        self._tag: str | None = None
        self._enabled = False
        self._unattributed = '__unattributed__'

    def _pack(self, tensor: torch.Tensor):
        if self._enabled:
            tag = self._tag or self._unattributed
            nbytes = int(tensor.numel()) * int(tensor.element_size())
            self._bytes[tag] = self._bytes.get(tag, 0) + nbytes
            self._tensors[tag] = self._tensors.get(tag, 0) + 1
        return tensor

    @staticmethod
    def _unpack(packed):
        return packed

    @contextlib.contextmanager
    def measure(self):
        """Install the hooks and enable counting for this block."""
        self._enabled = True
        try:
            with torch.autograd.graph.saved_tensors_hooks(
                    self._pack, self._unpack):
                yield self
        finally:
            self._enabled = False

    @contextlib.contextmanager
    def tagged(self, name: str):
        """Attribute saves inside this block to ``name``."""
        if not isinstance(name, str):
            raise TypeError(f"tag must be a str, got {type(name).__name__}.")
        prev = self._tag
        self._tag = name
        try:
            yield
        finally:
            self._tag = prev

    def bytes_by_tag(self) -> dict[str, int]:
        return dict(self._bytes)

    def tensors_by_tag(self) -> dict[str, int]:
        return dict(self._tensors)

    def total_bytes(self) -> int:
        return int(sum(self._bytes.values()))

    def total_tensors(self) -> int:
        return int(sum(self._tensors.values()))

    def reset(self) -> None:
        """Drop all tallies (allows re-measuring with the same probe)."""
        self._bytes.clear()
        self._tensors.clear()


def _memory_layer_call(layer, x, state, lookback, detach_lookback: bool):
    """Functional memory-layer call (pure; safe to recompute)."""
    return layer.forward_functional(
        x, state, lookback, detach_lookback=detach_lookback)


def _stateless_layer_call(layer, x):
    """CSA / HCA call returning the output tensor only."""
    out, _ = layer(x, None)
    return out


def _tagged(probe, kind):
    return probe.tagged(kind) if probe is not None else contextlib.nullcontext()


def forward_with_policy(stack, x: torch.Tensor, *, policy: str = 'none',
                        state: torch.Tensor | None = None,
                        lookbacks: list | None = None,
                        probe: SavedBytesProbe | None = None,
                        detach_state: bool = True,
                        detach_lookback: bool = True,
                        preserve_rng_state: bool = False,
                        ) -> tuple[torch.Tensor, torch.Tensor | None, list]:
    """Stack forward identical to ``forward_functional`` modulo checkpointing.

    Mirrors ``HybridAttentionStack.forward_functional``'s loop exactly
    (residual + LayerNorm, functional memory layers, stateless CSA/HCA,
    restacked state) so ``policy='none'`` must match it value-for-value.
    Layers selected by the policy are wrapped in non-reentrant
    ``torch.utils.checkpoint``; the pure functional layer calls make the
    recompute safe (no module-state mutation).

    ``detach_state`` mirrors the stack argument of the same name: the
    incoming stacked memory state is detached by default and is dropped when
    its batch size no longer matches ``x``. Pass ``detach_state=False`` to
    carry the autograd graph across chunks (cross-chunk BPTT).
    """
    policy = parse_policy(policy)
    n_mem = stack.n_memory_layers
    if state is None:
        states = [None] * n_mem
    else:
        if state.shape[0] != n_mem:
            raise ValueError(
                f"forward_with_policy: state has {state.shape[0]} layers "
                f"but the stack has {n_mem} memory layers.")
        if state.shape[1] != x.shape[0]:
            # Batch size changed between calls: the per-sequence state is
            # invalid. Mirror HybridAttentionStack.forward_functional and
            # fall back to a fresh sequence rather than mixing rows.
            states = [None] * n_mem
        else:
            if state.device != x.device:
                state = state.to(device=x.device)
            if detach_state:
                state = state.detach()
            states = list(state.unbind(0))
    if lookbacks is None:
        lookback_list = [None] * n_mem
    else:
        lookback_list = list(lookbacks)
        if len(lookback_list) != n_mem:
            raise ValueError(
                f"forward_with_policy: lookbacks has length "
                f"{len(lookback_list)} but the stack has {n_mem} memory "
                f"layers.")

    mem_idx = 0
    new_lookbacks: list = []
    for layer, norm, kind in zip(stack.layers, stack.norms, stack.layout):
        residual = x
        use_ckpt = should_checkpoint(kind, policy)
        with _tagged(probe, kind):
            x = norm(x)
            if kind in MEMORY_LAYER_KINDS:
                if use_ckpt:
                    o, new_state, new_lb = torch_checkpoint.checkpoint(
                        _memory_layer_call, layer, x, states[mem_idx],
                        lookback_list[mem_idx], detach_lookback,
                        use_reentrant=False,
                        preserve_rng_state=preserve_rng_state)
                else:
                    o, new_state, new_lb = _memory_layer_call(
                        layer, x, states[mem_idx], lookback_list[mem_idx],
                        detach_lookback)
                states[mem_idx] = new_state
                new_lookbacks.append(new_lb)
                mem_idx += 1
            else:
                if use_ckpt:
                    o = torch_checkpoint.checkpoint(
                        _stateless_layer_call, layer, x,
                        use_reentrant=False,
                        preserve_rng_state=preserve_rng_state)
                else:
                    o = _stateless_layer_call(layer, x)
        x = residual + o

    if n_mem > 0:
        if any(s is None for s in states):
            raise RuntimeError(
                "forward_with_policy: a memory layer returned new_state=None "
                "despite output_final_state=True. Refusing to persist a "
                "partial state stack.")
        ref_shape = tuple(states[0].shape)
        for i, s in enumerate(states):
            if tuple(s.shape) != ref_shape:
                raise RuntimeError(
                    f"forward_with_policy: memory layer {i} returned "
                    f"new_state with shape {tuple(s.shape)} but layer 0 "
                    f"returned shape {ref_shape}.")
        new_state = torch.stack(states, dim=0)
    else:
        new_state = None
    return x, new_state, new_lookbacks
